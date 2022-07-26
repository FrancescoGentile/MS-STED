##
##
##

import math
from typing import Tuple
import logging
import os 
import csv
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
from prettytable import PrettyTable
import random
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ..model.embeddings import Embeddings
from ..model.model import Decoder, Discriminator, Encoder, Reconstructor, ReconstructorDiscriminatorModel
from ..dataset.dataset import Dataset
from ..dataset.skeleton import SkeletonGraph
from .. import utils
from .config import TrainingConfig
        
class PretrainingProcessor: 
    
    def __init__(self, config: TrainingConfig) -> None:
        self._tconfig = config
        self._pconfig = config.pretraining
        
        self._scaler = GradScaler()
        self._init_environment()
        self._init_losses_file()
        self._set_loggers()
        self._load_checkpoint()
        self._set_device()
        
        skeleton = self._tconfig.dataset.to_skeleton_graph()
        train_dataset, eval_dataset = self._get_datasets(skeleton)
        self._set_loaders(train_dataset, eval_dataset)
        self._set_model(skeleton, train_dataset.channels)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_loss()
        
    def _init_environment(self):
        torch.set_default_dtype(torch.float32)
        np.random.seed(self._pconfig.seed)
        random.seed(self._pconfig.seed)
        torch.manual_seed(self._pconfig.seed)
        torch.cuda.manual_seed(self._pconfig.seed)
        torch.backends.cudnn.enabled = True
        
        if self._tconfig.debug:
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
            
    def _init_losses_file(self):
        with open(self._pconfig.accuracy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'reconstruction-loss', 'discrimination-accuracy'])
    
    def _set_loggers(self):
        self._logger = utils.init_logger('pretrain', level=logging.INFO, file=self._pconfig.log_file)
        self._writer = SummaryWriter(log_dir=self._pconfig.log_dir)

    def _load_checkpoint(self):
        try:
            if not self._pconfig.resume:
                self._checkpoint = None
            else:
                self._checkpoint = torch.load(self._pconfig.checkpoint_file, map_location=torch.device('cpu'))
                self._logger.info('Successfully loaded checkpoint.')
        except:
            self._logger.error(f'Checkpoint not found in {self._pconfig.checkpoint_file}')
            raise ValueError(f'Checkpoint not found in {self._pconfig.checkpoint_file}')
    
    def _save_checkpoint(self, current_epoch: int, best_results: dict):
        ckp = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(), 
            'start_epoch': current_epoch + 1,
            'best_results': best_results
        }
        
        torch.save(ckp, self._pconfig.checkpoint_file(current_epoch))
        
    def _save_results(self, current_epoch: int, rloss: float, dacc: float):     
        with open(self._pconfig.accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, rloss, dacc])
           
    def _set_device(self):
        if len(self._pconfig.gpus) > 0 and torch.cuda.is_available():
            self._logger.info('Using GPU')
            self._device = torch.device(f'cuda:{self._pconfig.gpus[0]}')
        else: 
            self._logger.info('Using CPU')
            self._device = torch.device('cpu')
    
    def _get_datasets(self, skeleton: SkeletonGraph) -> Tuple[Dataset, Dataset]:
        train_dataset = self._tconfig.dataset.to_dataset(skeleton, train_set=True, pretrain=True)
        eval_dataset = self._tconfig.dataset.to_dataset(skeleton, train_set=False, pretrain=True)
        self._logger.info(f'Pretraining on dataset {train_dataset.name}')
        
        return train_dataset, eval_dataset
    
    def _set_loaders(self, train_dataset: Dataset, eval_dataset: Dataset):
        self._train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self._pconfig.train_batch_size // self._pconfig.accumulation_steps, 
            num_workers=4 * len(self._pconfig.gpus),
            pin_memory=True,
            shuffle=True, 
            drop_last=True)
        
        self._eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self._pconfig.eval_batch_size // self._pconfig.accumulation_steps, 
            num_workers=1,
            pin_memory=True, 
            shuffle=False, 
            drop_last=False)

        self._logger.info(f'Training batch size: {self._pconfig.train_batch_size}')
        self._logger.info(f'Eval batch size: {self._pconfig.eval_batch_size}')
        self._logger.info(f'Accumulation steps: {self._pconfig.accumulation_steps}')

    def _set_model(self, skeleton: SkeletonGraph, channels: int):
        embeddings = Embeddings(self._tconfig.model.embeddings, channels, skeleton)
        encoder = Encoder(self._tconfig.model.encoder, True)
        decoder = Decoder(self._tconfig.model.decoder, self._tconfig.model.encoder)
        reconstructor = Reconstructor(self._tconfig.model.decoder.out_channels, channels)
        discriminator = Discriminator(self._tconfig.model.decoder.out_channels)
        model = ReconstructorDiscriminatorModel(embeddings, encoder, decoder, reconstructor, discriminator)
        
        self._logger.info(f'Model: {self._tconfig.model.name}')
        
        with open(self._pconfig.model_description_file, 'w') as f:
            print(model, file=f)
            
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: 
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        
        with open(self._pconfig.model_parameters_file, 'w') as f:
            print(table, file=f)
            print(f'Total number of parameters: {total_params}', file=f)
            
        self._logger.info(f'Model profile: {total_params/1e6:.2f}M parameters ({total_params})')
        
        if self._checkpoint is not None: 
            model.load_state_dict(self._checkpoint['model'])
            self._logger.info('Successfully loaded model state from checkpoint')
        
        self._model = model.to(self._device)
    
    def _set_optimizer(self):
        optimizer = self._pconfig.optimizer.to_optimizer(self._model.parameters())
        self._logger.info(f'Optimizer: {self._pconfig.optimizer}')
        if self._checkpoint is not None:
            optimizer.load_state_dict(self._checkpoint['optimizer'])
            self._logger.info('Successfully optimizer state from checkpoint')
        
        self._optimizer = optimizer
    
    def _set_lr_scheduler(self):
        lr_scheduler = self._pconfig.lr_scheduler.to_lr_scheduler(self._optimizer)
        self._logger.info(f'LR scheduler: {self._pconfig.lr_scheduler}')
        if self._checkpoint is not None:
            lr_scheduler.load_state_dict(self._checkpoint['lr_scheduler'])
            self._logger.info('Successfully lr scheduler state from checkpoint')
        
        self._lr_scheduler = lr_scheduler
    
    def _set_loss(self):
        self._reconstruction_loss = nn.BCEWithLogitsLoss(reduction='mean').to(self._device)
        self._discriminate_loss = nn.MSELoss(reduction='mean')
        
        self._logger.info(f'Reconstruction loss function used: {self._reconstruction_loss.__class__.__name__}')
        self._logger.info(f'Discriminate loss function used: {self._discriminate_loss.__class__.__name__}')
    
    def _get_discrimination_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            yp = torch.sigmoid(logits)
            yp[yp >= 0.5] = 1.0
            yp[yp < 0.5] = 0.0
            eq = y == yp
            N, T, V, M = eq.shape
            return (torch.sum(eq) / (N * T * V * M)).detach().item()
    
    def _train(self, epoch: int):
        self._model.train()
        self._optimizer.zero_grad()
        
        num_samples = 0
        daccs = []
        rlosses, dlosses, losses = [], [], []
        
        before_epoch_lr_rate = self._lr_scheduler.get_lr()
        
        start_time = timer()
        for idx, (jm, jo, jc, bm, bo, bc) in enumerate(tqdm(self._train_loader)):
            
            jm: torch.Tensor = jm.float().to(self._device)
            jo: torch.Tensor = jo.float().to(self._device)
            jc: torch.Tensor = jc.float().to(self._device)
            
            bm: torch.Tensor = jm.float().to(self._device)
            bo: torch.Tensor = jo.float().to(self._device)
            bc: torch.Tensor = jc.float().to(self._device)
            
            # Computing logits
            with autocast():
                jr, jd, br, bd = self._model(jm, bm)
                
                rloss: torch.Tensor = (self._reconstruction_loss(jr, jo) + self._reconstruction_loss(br, bo)) / 2
                dloss: torch.Tensor = (self._discriminate_loss(jd, jc) + self._discriminate_loss(bd, bc)) / 2
                loss = self._pconfig.reconstruction_lambda * rloss + self._pconfig.discrimination_lamba * dloss
                
                rlosses.append(rloss.detach().item())
                dlosses.append(dloss.detach().item())
                losses.append(loss.detach().item())
                
                loss = loss / self._pconfig.accumulation_steps
            
            self._scaler.scale(loss).backward()
            
            if (idx + 1) % self._pconfig.accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
            
                self._lr_scheduler.step(after_batch=True)
            
            num_samples += jm.size(0)
            jdacc = self._get_discrimination_accuracy(jd, jc)
            bdacc = self._get_discrimination_accuracy(bd, bc)
            daccs.append(jdacc)
            daccs.append(bdacc)
        
        end_time = timer()
        
        self._lr_scheduler.step(after_batch=False)
        
        # Computing statistics
        train_time = end_time - start_time
        train_speed = (num_samples / train_time) / len(self._pconfig.gpus)
        
        loss = np.mean(losses)
        rloss = np.mean(rlosses)
        dloss = np.mean(dlosses)
        dacc = np.mean(daccs)
        
        after_epoch_lr_rate = self._lr_scheduler.get_lr()
        # Log statistics
        self._logger.info(f'Training epoch {epoch}')
        self._logger.info(f'\tTraining time: {train_time:.2f}s - Speed: {train_speed:.2f} samples/(second * GPU)')
        self._logger.info(f'\tLearning rate: (before) {before_epoch_lr_rate:.5f} -> (after) {after_epoch_lr_rate:.5f}')
        self._logger.info(f'\tMean loss: (total) {loss:.5f} - (reconstruction) {rloss:.5f} - (discrimination) {dloss:.5f}')
        self._logger.info(f'\tMean discrimination accuracy: {dacc:.5f}')
        
        # Add statistics on tensorboard
        self._writer.add_scalar('train/loss', loss, epoch)
        self._writer.add_scalar('train/reconstruction-loss', rloss, epoch)
        self._writer.add_scalar('train/discrimination-loss', dloss, epoch)
        self._writer.add_scalar('train/learning_rate', before_epoch_lr_rate, epoch)
        self._writer.add_scalar('train/discrimination-accuracy', dacc, epoch)
    
    def _eval(self, epoch: int) -> Tuple[float, float]:
        self._model.eval()
        
        with torch.no_grad():
            num_samples = 0
            daccs = []
            rlosses, dlosses, losses = [], [], []
        
            start_time = timer()
            for jm, jo, jc, bm, bo, bc in tqdm(self._eval_loader):
            
                jm: torch.Tensor = jm.float().to(self._device)
                jo: torch.Tensor = jo.float().to(self._device)
                jc: torch.Tensor = jc.float().to(self._device)
            
                bm: torch.Tensor = jm.float().to(self._device)
                bo: torch.Tensor = jo.float().to(self._device)
                bc: torch.Tensor = jc.float().to(self._device)
            
                # Computing logits
                with autocast():
                    jr, jd, br, bd = self._model(jm, bm)
                
                    rloss: torch.Tensor = (self._reconstruction_loss(jr, jo) + self._reconstruction_loss(br, bo)) / 2
                    dloss: torch.Tensor = (self._discriminate_loss(jd, jc) + self._discriminate_loss(bd, bc)) / 2
                    loss = self._pconfig.reconstruction_lambda * rloss + self._pconfig.discrimination_lamba * dloss
                
                    rlosses.append(rloss.detach().item())
                    dlosses.append(dloss.detach().item())
                    losses.append(loss.detach().item())
                
                    loss = loss / self._pconfig.accumulation_steps
            
                num_samples += jm.size(0)
                jdacc = self._get_discrimination_accuracy(jd, jc)
                bdacc = self._get_discrimination_accuracy(bd, bc)
                daccs.append(jdacc)
                daccs.append(bdacc)
        
        end_time = timer()
        
        self._lr_scheduler.step(after_batch=False)
        
        # Computing statistics
        eval_time = end_time - start_time
        eval_speed = (num_samples / eval_time) / len(self._pconfig.gpus)
        
        loss = np.mean(losses)
        rloss = np.mean(rlosses)
        dloss = np.mean(dlosses)
        dacc = np.mean(daccs)
        
        # Log statistics
        self._logger.info(f'Evaluating epoch {epoch}')
        self._logger.info(f'\tEvaluating time: {eval_time:.2f}s - Speed: {eval_speed:.2f} samples/(second * GPU)')
        self._logger.info(f'\tMean loss: (total) {loss:.5f} - (reconstruction) {rloss:.5f} - (discrimination) {dloss:.5f}')
        self._logger.info(f'\tMean discrimination accuracy: {dacc:.5f}')
        
        # Add statistics on tensorboard
        self._writer.add_scalar('eval/loss', loss, epoch)
        self._writer.add_scalar('eval/reconstruction-loss', rloss, epoch)
        self._writer.add_scalar('eva;/discrimination-loss', dloss, epoch)
        self._writer.add_scalar('eval/discrimination-accuracy', dacc, epoch)
        
        return rloss, dacc
    
    def start(self):
        self._logger.info('Starting pretraining ...')
        
        if self._checkpoint is not None:
            start_epoch = self._checkpoint['start_epoch']
            best_results = self._checkpoint['best_results']
        else: 
            start_epoch = 0
            best_results = {'recon_loss': math.inf, 'disc_acc': 0.0}
            
        for epoch in range(start_epoch, self._pconfig.num_epochs):
            torch.cuda.empty_cache()
            self._train(epoch)
            
            torch.cuda.empty_cache()
            rloss, dacc = self._eval(epoch)
            
            self._save_results(epoch, rloss, dacc)
            
            best_epoch = rloss < best_results['recon_loss'] and dacc > best_results['disc_acc']
            if best_epoch:
                best_results['recon_loss'] = rloss
                best_results['disc_acc'] = dacc
                # Save model weights
                torch.save(self._model.state_dict(), self._pconfig.best_weights_file)
                
            if epoch % self._pconfig.save_interleave == 0:
                self._save_checkpoint(epoch, best_results)
            
        self._logger('Finished pretraining ...')
