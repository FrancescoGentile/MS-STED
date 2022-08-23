##
##
##

import math
from typing import Tuple, Optional
import logging
import os 
import csv
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
from prettytable import PrettyTable
import random
import os
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchmetrics

from ..model.dropout import DropHeadScheduler
from ..model.embeddings import Embeddings
from ..model.model import Decoder, Discriminator, Encoder, Reconstructor, ReconstructorDiscriminatorModel
from ..dataset.dataset import Dataset
from ..dataset.skeleton import SkeletonGraph
from .. import utils
from .config import PretrainingConfig
from . import metrics
        
class PretrainingProcessor: 
    
    def __init__(self, config: PretrainingConfig, logger: logging.Logger, writer: SummaryWriter) -> None:
        self._config = config
        self._logger = logger
        self._writer = writer
        
        self._scaler = GradScaler()
        self._init_environment()
        self._set_device()
        self._load_checkpoint()
        
        skeleton = self._config.dataset.to_skeleton_graph()
        train_dataset, eval_dataset = self._get_datasets(skeleton)
        self._set_loaders(train_dataset, eval_dataset)
        self._set_model(skeleton, train_dataset.channels)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_dropout_scheduler()
        self._init_metrics_file()
        self._set_metrics()
        
    def _init_environment(self):
        torch.set_default_dtype(torch.float32)
        np.random.seed(self._config.seed)
        random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        torch.cuda.manual_seed(self._config.seed)
        torch.backends.cudnn.enabled = True
        
        if self._config.debug:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        else:
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
            try:
                del os.environ['CUBLAS_WORKSPACE_CONFIG']
                os.unsetenv('CUBLAS_WORKSPACE_CONFIG') # to be sure
            except:
                pass

    def _load_checkpoint(self):
        try:
            self._checkpoint = None
            if self._config.resume_checkpoint is not None:
                self._checkpoint = torch.load(self._config.resume_checkpoint, map_location=self._device)
                self._logger.info('Successfully loaded checkpoint.')
        except:
            raise ValueError(f'{self._config.resume_checkpoint} is not  a valid checkpoint')
    
    def _save_checkpoint(self, current_epoch: int, best_results: dict):
        ckp = {
            'model': self._model.module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(), 
            'start_epoch': current_epoch + 1,
            'best_results': best_results
        }
        
        torch.save(ckp, self._config.checkpoint_file(current_epoch))
        
    def _save_results(self, current_epoch: int, rloss: float, dacc: float):     
        with open(self._config.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, rloss, dacc])
           
    def _set_device(self):
        gpu_id = self._config.distributed.get_gpu_id()
        if gpu_id is None:
            self._device = torch.device('cpu')
            self._logger.info('Using CPU')
        else:
            self._device = torch.device(gpu_id)
            self._logger.info('Using GPU')
    
    def _get_datasets(self, skeleton: SkeletonGraph) -> Tuple[Dataset, Dataset]:
        train_dataset = self._config.dataset.to_dataset(skeleton, train_set=True, pretrain=True)
        eval_dataset = self._config.dataset.to_dataset(skeleton, train_set=False, pretrain=True)
        self._logger.info(f'Pretraining on dataset {train_dataset.name}')
        
        return train_dataset, eval_dataset
    
    def _set_loaders(self, train_dataset: Dataset, eval_dataset: Dataset):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self._config.distributed.world_size,
            shuffle=True, 
            seed=self._config.seed, 
            drop_last=True)
        eval_sampler = DistributedSampler(
            eval_dataset, 
            num_replicas=self._config.distributed.world_size,
            shuffle=False,
            seed=self._config.seed,
            drop_last=False)
        
        train_batch_size = self._config.train_batch_size // self._config.distributed.world_size        
        eval_batch_size = self._config.eval_batch_size // self._config.distributed.world_size
        
        self._train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=train_batch_size // self._config.accumulation_steps, 
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler)

        self._eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=eval_batch_size // self._config.accumulation_steps, 
            num_workers=4,
            pin_memory=True, 
            sampler=eval_sampler)

        self._logger.info(f'Training batch size: (device) {train_batch_size} - '
                          f'(total) {self._config.train_batch_size}')
        self._logger.info(f'Eval batch size: (device) {eval_batch_size} - '
                          f'(total) {self._config.eval_batch_size}')
        self._logger.info(f'Accumulation steps: {self._config.accumulation_steps}')

    def _set_model(self, skeleton: SkeletonGraph, channels: int):
        model_cfg = self._config.model
        embeddings = Embeddings(model_cfg.embeddings, channels, skeleton)
        encoder = Encoder(model_cfg.encoder, True)
        decoder = Decoder(model_cfg.decoder, self._config.model.encoder)
        reconstructor = Reconstructor(model_cfg.decoder.out_channels, channels, model_cfg.dropout)
        discriminator = Discriminator(model_cfg.decoder.out_channels, model_cfg.dropout)
        model = ReconstructorDiscriminatorModel(embeddings, encoder, decoder, reconstructor, discriminator)
        
        self._logger.info(f'Model: {self._config.model.name}')
        self._save_model_description(model)
        
        if self._checkpoint is not None: 
            model.load_state_dict(self._checkpoint['model'])
            self._logger.info('Successfully loaded model state from checkpoint')
        
        model = model.to(self._device)
        gpu_id = self._config.distributed.get_gpu_id()
        devices_ids = [gpu_id] if gpu_id is not None else None
        self._model = DDP(model, device_ids=devices_ids, find_unused_parameters=self._config.model.dropout.layer > 0)
    
    def _save_model_description(self, model: nn.Module):
        if self._config.distributed.is_local_master():
            with open(self._config.model_description_file, 'w') as f:
                print(model, file=f)
            
            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: 
                    continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params+=params
        
            with open(self._config.model_parameters_file, 'w') as f:
                print(table, file=f)
                print(f'Total number of parameters: {total_params}', file=f)
            
            self._logger.info(f'Model profile: {total_params/1e6:.2f}M parameters ({total_params})')
    
    def _set_optimizer(self):
        optimizer = self._config.optimizer.to_optimizer(self._model.parameters())
        self._logger.info(f'Optimizer: {self._config.optimizer}')
        if self._checkpoint is not None:
            optimizer.load_state_dict(self._checkpoint['optimizer'])
            self._logger.info('Successfully loaded optimizer state from checkpoint')
        
        self._optimizer = optimizer
    
    def _set_lr_scheduler(self):
        lr_scheduler = self._config.lr_scheduler.to_lr_scheduler(self._optimizer)
        self._logger.info(f'LR scheduler: {self._config.lr_scheduler}')
        if self._checkpoint is not None:
            lr_scheduler.load_state_dict(self._checkpoint['lr_scheduler'])
            self._logger.info('Successfully loaded lr scheduler state from checkpoint')
        
        self._lr_scheduler = lr_scheduler
    
    def _set_dropout_scheduler(self):
        num_steps = (len(self._train_loader) // self._config.accumulation_steps) * self._config.num_epochs
        self._dropout_scheduler = DropHeadScheduler(
            modules=self._model.modules(),
            p=self._config.model.dropout.head,
            num_steps=num_steps
        )
        
    def _init_metrics_file(self):
        with open(self._config.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'reconstruction_loss', 'discrimination_accuracy'])
        
    def _set_metrics(self):
        self._train_metrics = {
            'recon_loss': torchmetrics.MeanSquaredError().to(self._device),
            'disc_loss': metrics.BCEWithLogitsLoss().to(self._device),
            'total_loss': metrics.TotalLoss(
                self._config.reconstruction_lambda, 
                self._config.discrimination_lamba
                ).to(self._device),
            'disc_acc': torchmetrics.Accuracy(threshold=0.5).to(self._device)
        }
        
        self._eval_metrics = {
            'recon_loss': torchmetrics.MeanSquaredError().to(self._device),
            'disc_loss': metrics.BCEWithLogitsLoss().to(self._device),
            'total_loss': metrics.TotalLoss(
                self._config.reconstruction_lambda, 
                self._config.discrimination_lamba
                ).to(self._device),
            'disc_acc': torchmetrics.Accuracy(threshold=0.5).to(self._device)
        }
        
        self._logger.info(f'Reconstruction loss function used: MeanSquaredError')
        self._logger.info(f'Discriminate loss function used: BinaryCrossEntropyLossWithLogits')
        
        if self._config.distributed.is_master():
            wandb.run.define_metric('pretrain/train/epoch')
            wandb.run.define_metric('pretrain/train/learning_rate', step_metric='pretrain/train/epoch', summary='none')
            wandb.run.define_metric('pretrain/train/reconstruction_loss', step_metric='pretrain/train/epoch', summary='min')
            wandb.run.define_metric('pretrain/train/discrimination_loss', step_metric='pretrain/train/epoch', summary='min')
            wandb.run.define_metric('pretrain/train/total_loss', step_metric='pretrain/train/epoch', summary='min')
            wandb.run.define_metric('pretrain/train/discrimination_acc', step_metric='pretrain/train/epoch', summary='max')
            
            wandb.run.define_metric('pretrain/eval/epoch')
            wandb.run.define_metric('pretrain/eval/learning_rate', step_metric='pretrain/eval/epoch', summary='none')
            wandb.run.define_metric('pretrain/eval/reconstruction_loss', step_metric='pretrain/eval/epoch', summary='min')
            wandb.run.define_metric('pretrain/eval/discrimination_loss', step_metric='pretrain/eval/epoch', summary='min')
            wandb.run.define_metric('pretrain/eval/total_loss', step_metric='pretrain/eval/epoch', summary='min')
            wandb.run.define_metric('pretrain/eval/discrimination_acc', step_metric='pretrain/eval/epoch', summary='max')
    
    def _log_metrics(self, epoch: int, time: float, speed: float, train: bool, lr: Optional[Tuple[float, float]] = None):
        metrics_dict = self._train_metrics if train else self._eval_metrics
            
        recon_loss = metrics_dict['recon_loss'].compute()
        disc_loss = metrics_dict['disc_loss'].compute()
        total_loss = metrics_dict['total_loss'].compute()
        disc_acc = metrics_dict['disc_acc'].compute()
        
        # Log metrics
        if train:
            self._logger.info(f'Training epoch {epoch}')
            self._logger.info(f'\tTraining time: {time:.2f}s - Speed: {speed:.2f} samples/(second * GPU)')
            self._logger.info(f'\tLearning rate: (before) {lr[0]:.5f} -> (after) {lr[1]:.5f}')
        else:
            self._logger.info(f'Eval epoch {epoch}')
            self._logger.info(f'\tEval time: {time:.2f}s - Speed: {speed:.2f} samples/(second * GPU)')
        
        self._logger.info(f'\tReconstruction loss: {recon_loss:.5f}')
        self._logger.info(f'\tDiscrimination loss: {disc_loss:.5f}')
        self._logger.info(f'\tTotal loss: {total_loss:.5f}')
        self._logger.info(f'\tDiscrimination accuracy: {disc_acc:.2%}')
        
        prefix = 'train' if train else 'eval'
        
        # Add metrics to tensorboard
        if self._config.distributed.is_local_master():
            if lr is not None:
                self._writer.add_scalar(f'pretrain/{prefix}/learning_rate', lr[0], epoch)
            self._writer.add_scalar(f'pretrain/{prefix}/reconstruction_loss', recon_loss, epoch)
            self._writer.add_scalar(f'pretrain/{prefix}/discrimination_loss', disc_loss, epoch)
            self._writer.add_scalar(f'pretrain/{prefix}/total_loss', total_loss, epoch)
            self._writer.add_scalar(f'pretrain/{prefix}/discrimination_acc', disc_acc, epoch)
        
        # Add metrics to wandb
        if self._config.distributed.is_master():
            if lr is not None:
                wandb.log({'classification/train/learning_rate': lr[0]}, commit=False)
            wandb.log({
                f'pretrain/{prefix}/epoch': epoch,
                f'pretrain/{prefix}/reconstruction_loss': recon_loss,
                f'pretrain/{prefix}/discrimination_loss': disc_loss,
                f'pretrain/{prefix}/total_loss': total_loss,
                f'pretrain/{prefix}/discrimination_acc': disc_acc})
        
        # Add metrics to file
        with open(self._config.metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, recon_loss, disc_acc])
    
    def _train(self, epoch: int):
        self._model.train()
        self._optimizer.zero_grad()
        
        before_epoch_lr_rate = self._lr_scheduler.get_lr()
        
        counter = tqdm(
            desc=f'train-epoch-{epoch}', 
            total=len(self._train_loader) // self._config.accumulation_steps,
            dynamic_ncols=True)
        
        start_time = timer()
        for idx, (jm, jo, jc, bm, bo, bc) in enumerate(self._train_loader):
            
            jm: torch.Tensor = jm.float().to(self._device)
            jo: torch.Tensor = jo.float().to(self._device)
            jc: torch.Tensor = jc.to(self._device, dtype=torch.bool)
            
            bm: torch.Tensor = jm.float().to(self._device)
            bo: torch.Tensor = jo.float().to(self._device)
            bc: torch.Tensor = jc.to(self._device, dtype=torch.bool)
            
            original = torch.cat((jo, bo), dim=3)
            changed = torch.cat((jc, bc), dim=2)
            
            # Computing loss
            with autocast():
                jr, jd, br, bd = self._model(jm, bm)
                
                reconstructed = torch.cat((jr, br), dim=3)
                mask = changed.unsqueeze(1).expand(reconstructed.shape)
                reconstructed[mask] = original[mask].half()
                discriminated = torch.cat((jd, bd), dim=2)
                
                rloss = self._train_metrics['recon_loss'](reconstructed, original)
                dloss = self._train_metrics['disc_loss'](discriminated, changed.float())
                total_loss = self._train_metrics['total_loss'](rloss, dloss)
                total_loss = total_loss / self._config.accumulation_steps
            
            self._scaler.scale(total_loss).backward()
            
            if (idx + 1) % self._config.accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
            
                self._lr_scheduler.step(after_batch=True)
                self._dropout_scheduler.step()
                counter.update(1)
            
            with torch.no_grad():
                discriminated = torch.sigmoid(discriminated)
                self._train_metrics['disc_acc'](discriminated, changed)
        
        self._lr_scheduler.step(after_batch=False)
        after_epoch_lr_rate = self._lr_scheduler.get_lr()
        
        # Computing time metrics
        end_time = timer()
        counter.close()
        time = end_time - start_time
        num_samples = len(self._train_loader) * self._train_loader.batch_size * self._config.distributed.world_size
        speed = (num_samples / time) / self._config.distributed.world_size
        
        # Log metrics
        self._log_metrics(epoch, time, speed, True, (before_epoch_lr_rate, after_epoch_lr_rate))
        for m in self._train_metrics.values():
            m.reset()
    
    def _eval(self, epoch: int) -> Tuple[float, float]:
        self._model.eval()
        
        counter = tqdm(
            desc=f'eval-epoch-{epoch}', 
            total=len(self._eval_loader) // self._config.accumulation_steps,
            dynamic_ncols=True)
        
        with torch.no_grad():
            start_time = timer()
            for idx, (jm, jo, jc, bm, bo, bc) in enumerate(self._eval_loader):
            
                jm: torch.Tensor = jm.float().to(self._device)
                jo: torch.Tensor = jo.float().to(self._device)
                jc: torch.Tensor = jc.to(self._device, dtype=torch.bool)
            
                bm: torch.Tensor = jm.float().to(self._device)
                bo: torch.Tensor = jo.float().to(self._device)
                bc: torch.Tensor = jc.to(self._device, dtype=torch.bool)
                
                original = torch.cat((jo, bo), dim=3)
                changed = torch.cat((jc, bc), dim=2)
            
                # Computing logits
                with autocast():
                    jr, jd, br, bd = self._model(jm, bm)
                
                    reconstructed = torch.cat((jr, br), dim=3)
                    mask = changed.unsqueeze(1).expand(reconstructed.shape)
                    reconstructed[mask] = original[mask].half()
                    discriminated = torch.cat((jd, bd), dim=2)
                
                    rloss = self._eval_metrics['recon_loss'](reconstructed, original)
                    dloss = self._eval_metrics['disc_loss'](discriminated, changed.float())
                    total_loss = self._eval_metrics['total_loss'](rloss, dloss)
                
                if (idx + 1) % self._config.accumulation_steps == 0:
                    counter.update(1)
            
                discriminated = torch.sigmoid(discriminated)
                self._eval_metrics['disc_acc'](discriminated, changed)
                
        # Computing time metrics
        end_time = timer()
        counter.close()
        time = end_time - start_time
        num_samples = len(self._eval_loader) * self._eval_loader.batch_size * self._config.distributed.world_size
        speed = (num_samples / time) / self._config.distributed.world_size
        
        # Log metrics
        self._log_metrics(epoch, time, speed, False)
        recon_loss = self._eval_metrics['recon_loss'].compute()
        disc_acc = self._eval_metrics['disc_acc'].compute()
        for m in self._eval_metrics.values():
            m.reset()
        
        return recon_loss, disc_acc
    
    def start(self):
        try:
            self._logger.info('Starting training ...')
        
            if self._checkpoint is not None:
                start_epoch = self._checkpoint['start_epoch']
                best_results = self._checkpoint['best_results']
            else: 
                start_epoch = 0
                best_results = {'recon_loss': math.inf, 'disc_acc': 0.0, 'epoch': -1}
            
            for epoch in range(start_epoch, self._config.num_epochs):
                self._train_loader.sampler.set_epoch(epoch)
            
                torch.cuda.empty_cache()
                self._train(epoch)
                torch.cuda.empty_cache()
                rloss, dacc = self._eval(epoch)
            
                self._save_results(epoch, rloss, dacc)
            
                best_epoch = rloss < best_results['recon_loss'] and dacc > best_results['disc_acc']
                if best_epoch:
                    best_results['recon_loss'] = rloss
                    best_results['disc_acc'] = dacc
                    best_results['epoch'] = epoch
                    # Save model weights
                    torch.save(self._model.module.state_dict(), self._config.best_weights_file)
                
                if (epoch + 1) % self._config.save_interleave == 0:
                    self._save_checkpoint(epoch, best_results)
            
            self._logger.info('Finished training ...')
            self._logger.info('Best results:')
            self._logger.info(f'\tepoch: {best_results["epoch"]}')
            self._logger.info(f'\treconstruction loss: {best_results["recon_loss"]}')
            self._logger.info(f'\tdiscrimination accuracy: {best_results["disc_acc"]}')
        except Exception as e:
            self._logger.error('Training failed with the following exception:')
            if self._config.debug:
                self._logger.exception(e)
            else:
                self._logger.error(e)
            raise e
