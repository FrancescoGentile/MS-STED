##
##
##

import math
from typing import Optional, Tuple
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
from ..model.model import Classifier, Encoder, ClassificationModel, Decoder, Reconstructor, Discriminator, ReconstructorDiscriminatorModel
from ..dataset.dataset import Dataset
from ..dataset.skeleton import SkeletonGraph
from .config import ClassificationConfig
from . import metrics
        
class ClassificationProcessor: 
    
    def __init__(self, config: ClassificationConfig, logger: logging.Logger, writer: Optional[SummaryWriter]) -> None:
        self._config = config
        self._logger = logger
        self._writer = writer
        
        self._scaler = GradScaler()
        self._init_environment()
        self._set_device()
        self._load_checkpoint()
        
        skeleton = self._config.dataset.to_skeleton_graph()
        self._set_datasets(skeleton)
        self._set_loaders()
        self._set_model(skeleton)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_dropout_scheduler()
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
            torch.use_deterministic_algorithms(True, warn_only=True)
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
        if self._config.distributed.is_local_master():
            ckp = {
                'model': self._model.module.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'lr_scheduler': self._lr_scheduler.state_dict(), 
                'start_epoch': current_epoch + 1,
                'best_results': best_results
            }
        
            torch.save(ckp, self._config.checkpoint_file(current_epoch))
           
    def _set_device(self):
        gpu_id = self._config.distributed.get_gpu_id()
        if gpu_id is None:
            self._device = torch.device('cpu')
            self._logger.info('Using CPU')
        else:
            self._device = torch.device(gpu_id)
            self._logger.info('Using GPU')
    
    def _set_datasets(self, skeleton: SkeletonGraph):
        self._train_dataset = self._config.dataset.to_dataset(skeleton, train_set=True, pretrain=False)
        self._eval_dataset = self._config.dataset.to_dataset(skeleton, train_set=False, pretrain=False)
        
        self._logger.info(f'Training on dataset {self._train_dataset.name}')
    
    def _set_loaders(self):
        train_sampler = DistributedSampler(
            self._train_dataset,
            num_replicas=self._config.distributed.world_size,
            shuffle=True, 
            seed=self._config.seed, 
            drop_last=True)
        eval_sampler = DistributedSampler(
            self._eval_dataset, 
            num_replicas=self._config.distributed.world_size,
            shuffle=False,
            seed=self._config.seed,
            drop_last=False)
        
        train_batch_size = self._config.train_batch_size // self._config.distributed.world_size        
        eval_batch_size = self._config.eval_batch_size // self._config.distributed.world_size
        
        self._train_loader = DataLoader(
            dataset=self._train_dataset, 
            batch_size=train_batch_size // self._config.accumulation_steps, 
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler)

        self._eval_loader = DataLoader(
            dataset=self._eval_dataset, 
            batch_size=eval_batch_size // self._config.accumulation_steps, 
            num_workers=4,
            pin_memory=True, 
            sampler=eval_sampler)

        self._logger.info(f'Training batch size: (device) {train_batch_size} - '
                          f'(total) {self._config.train_batch_size}')
        self._logger.info(f'Eval batch size: (device) {eval_batch_size} - '
                          f'(total) {self._config.eval_batch_size}')
        self._logger.info(f'Accumulation steps: {self._config.accumulation_steps}')
    
    def _set_model(self, skeleton: SkeletonGraph):
        channels = self._train_dataset.channels
        num_classes = self._train_dataset.num_classes
        num_frames = self._train_dataset.num_frames
        model_cfg = self._config.model
        
        embeddings = Embeddings(model_cfg.embeddings, channels, num_frames, skeleton)
        encoder = Encoder(model_cfg.encoder, skeleton, False)
        if self._config.pretrain_weights is not None:
            decoder = Decoder(self._config.model.decoder, self._config.model.encoder)
            reconstructor = Reconstructor(model_cfg.decoder.out_channels, channels, model_cfg.dropout)
            discriminator = Discriminator(model_cfg.decoder.out_channels, model_cfg.dropout)
            model = ReconstructorDiscriminatorModel(embeddings, encoder, decoder, reconstructor, discriminator)
            
            state_dict = torch.load(self._config.pretrain_weights, map_location=self._device)
            model.load_state_dict(state_dict)
        
        classifier = Classifier(model_cfg.encoder.out_channels, num_classes, model_cfg.dropout) 
        model = ClassificationModel(embeddings, encoder, classifier)
        
        self._logger.info(f'Model: {model_cfg.name}')
        self._save_model_description(model)
        
        if self._checkpoint is not None: 
            model.load_state_dict(self._checkpoint['model'])
            self._logger.info('Successfully loaded model state from checkpoint')
        
        model = model.to(self._device)
        gpu_id = self._config.distributed.get_gpu_id()
        devices_ids = [gpu_id] if gpu_id is not None else None
        self._model = DDP(model, device_ids=devices_ids)
    
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
        start_epoch = 0
        if self._checkpoint is not None:
            start_epoch = self._checkpoint['start_epoch']
        
        self._dropout_scheduler = DropHeadScheduler(
            modules=self._model.modules(),
            config=self._config.model.dropout.head,
            num_epochs=self._config.num_epochs,
            steps_per_epoch=self._get_steps_per_epoch(eval=False),
            start_epoch=start_epoch
        )
    
    def _init_metrics_file(self):
        if self._config.distributed.is_local_master():
            with open(self._config.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'top1_acc', 'top5_acc', 'precision', 'recall', 'f1_score'])

    def _init_wandb_metrics(self):
        if self._config.distributed.is_master():
            for metr, phase in [(self._train_metrics, 'train'), (self._eval_metrics, 'eval')]:
                epoch = f'classification/{phase}/epoch'
                wandb.run.define_metric(epoch)
                
                if phase == 'train':
                    wandb.run.define_metric(f'classification/{phase}/learning_rate', step_metric=epoch, summary='none')
                
                # f1_score, loss, precision, recall, top1_acc, top5_acc
                summaries = ['max', 'min', 'max', 'max', 'max', 'max', 'max']
                for name, summary in zip(metr.keys(), summaries):
                    wandb.run.define_metric(f'classification/{phase}/{name}', step_metric=epoch, summary=summary)
    
    def _set_metrics(self):
        num_classes = self._train_dataset.num_classes
        self._train_metrics = torchmetrics.MetricCollection({
            'loss': metrics.CrossEntropyLoss(self._config.label_smoothing),
            'top1_acc': torchmetrics.Accuracy(num_classes=num_classes, average='micro', top_k=1),
            'top5_acc': torchmetrics.Accuracy(num_classes=num_classes, average='micro', top_k=5),
            'precision': torchmetrics.Precision(num_classes=num_classes, average='micro', top_k=1),
            'recall': torchmetrics.Recall(num_classes=num_classes, average='micro', top_k=1),
            'f1_score': torchmetrics.F1Score(num_classes=num_classes, average='micro', top_k=1)
        }).to(self._device)
        
        self._eval_metrics = torchmetrics.MetricCollection({
            'loss': metrics.CrossEntropyLoss(self._config.label_smoothing),
            'top1_acc': torchmetrics.Accuracy(num_classes=num_classes, average='micro', top_k=1),
            'top5_acc': torchmetrics.Accuracy(num_classes=num_classes, average='micro', top_k=5),
            'precision': torchmetrics.Precision(num_classes=num_classes, average='micro', top_k=1),
            'recall': torchmetrics.Recall(num_classes=num_classes, average='micro', top_k=1),
            'f1_score': torchmetrics.F1Score(num_classes=num_classes, average='micro', top_k=1),
        }).to(self._device)
        
        self._logger.info('Loss function used: CrossEntropyLoss')
        
        self._init_metrics_file()
        self._init_wandb_metrics()
        
        # Confusion matrix does not have to be logged in wandb
        self._eval_metrics.add_metrics({
            'confusion': torchmetrics.ConfusionMatrix(num_classes=num_classes).to(self._device)})
        
    def _log_metrics(self, epoch: int, time: float, speed: float, train: bool, lr: Optional[Tuple[float, float]] = None):
        metrics_dict = self._train_metrics.compute() \
            if train else self._eval_metrics.compute()
            
        loss = metrics_dict['loss']
        top1_acc = metrics_dict['top1_acc']
        top5_acc = metrics_dict['top5_acc']
        precision = metrics_dict['precision']
        recall = metrics_dict['recall']
        f1_score = metrics_dict['f1_score']
        
        # Log metrics
        if train:
            self._logger.info(f'Training epoch {epoch}')
            self._logger.info(f'\tTraining time: {time:.2f}s - Speed: {speed:.2f} samples/(second * GPU)')
            self._logger.info(f'\tLearning rate: (before) {lr[0]:.5f} -> (after) {lr[1]:.5f}')
        else:
            self._logger.info(f'Eval epoch {epoch}')
            self._logger.info(f'\tEval time: {time:.2f}s - Speed: {speed:.2f} samples/(second * GPU)')
        
        self._logger.info(f'\tMean loss: {loss:.5f}')
        self._logger.info(f'\tTop-1 accuracy: {top1_acc:.2%}')
        self._logger.info(f'\tTop-5 accuracy: {top5_acc:.2%}')
        self._logger.info(f'\tPrecision: {precision:.2%}')
        self._logger.info(f'\tRecall: {recall:.2%}')
        self._logger.info(f'\tF1 score: {f1_score:.2%}')
        
        phase = 'train' if train else 'eval'
        
        # Add metrics to tensorboard
        if self._config.distributed.is_local_master():
            if lr is not None:
                self._writer.add_scalar(f'classification/{phase}/learning_rate', lr[0], epoch)
            self._writer.add_scalar(f'classification/{phase}/loss', loss, epoch)
            self._writer.add_scalar(f'classification/{phase}/top1_accuracy', top1_acc, epoch)
            self._writer.add_scalar(f'classification/{phase}/top5_accuracy', top5_acc, epoch)
            self._writer.add_scalar(f'classification/{phase}/precision', precision, epoch)
            self._writer.add_scalar(f'classification/{phase}/recall', recall, epoch)
            self._writer.add_scalar(f'classification/{phase}/f1_score', f1_score, epoch)
            
            if not train:
                conf_matrix = metrics_dict['confusion'].cpu().detach().numpy()
                np.save(self._config.confusion_matrix_file(epoch), conf_matrix)
        
        # Add metrics to wandb
        if self._config.distributed.is_master():
            if lr is not None:
                wandb.log({f'classification/{phase}/learning_rate': lr[0]}, commit=False)
            wandb.log({
                f'classification/{phase}/epoch': epoch,
                f'classification/{phase}/loss': loss,
                f'classification/{phase}/top1_acc': top1_acc,
                f'classification/{phase}/top5_acc': top5_acc,
                f'classification/{phase}/precision': precision,
                f'classification/{phase}/recall': recall,
                f'classification/{phase}/f1_score': f1_score}, commit=True)
        
        # Add metrics to file
        with open(self._config.metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, top1_acc, top5_acc, precision, recall, f1_score])
    
    def _get_steps_per_epoch(self, eval: bool) -> float:
        if eval:
            return math.ceil(len(self._eval_loader) / self._config.accumulation_steps)
        else:
            return math.floor(len(self._train_loader) / self._config.accumulation_steps)
        
    def _train(self, epoch: int):
        self._model.train()
        self._optimizer.zero_grad()
        
        before_epoch_lr_rate = self._lr_scheduler.get_lr()
        
        counter = tqdm(
            desc=f'train-epoch-{epoch}', 
            total=self._get_steps_per_epoch(eval=False),
            dynamic_ncols=True)
        
        start_time = timer()
        for idx, (j, b, y) in enumerate(self._train_loader):
            if counter.n == counter.total:
                break
            
            j: torch.Tensor = j.float().to(self._device)
            b: torch.Tensor = b.float().to(self._device)
            y: torch.Tensor = y.long().to(self._device)
            
            # Computing logits
            with autocast():
                logits = self._model(j, b)
                metrics_dict = self._train_metrics(logits, y)
                loss = metrics_dict['loss'] / self._config.accumulation_steps
            
            self._scaler.scale(loss).backward()
            
            if (idx + 1) % self._config.accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
            
                self._lr_scheduler.step(after_batch=True)
                self._dropout_scheduler.step()
                counter.update(1)
        
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
        self._train_metrics.reset()
            
    def _eval(self, epoch: int) -> float:
        self._model.eval()
        
        counter = tqdm(
            desc=f'eval-epoch-{epoch}', 
            total=self._get_steps_per_epoch(eval=True),
            dynamic_ncols=True)
        
        with torch.no_grad():
            start_time = timer()
            for idx, (j, b, y) in enumerate(self._eval_loader):
                j: torch.Tensor = j.float().to(self._device)
                b: torch.Tensor = b.float().to(self._device)
                y: torch.Tensor = y.long().to(self._device)
                
                # Computing logits
                with torch.cuda.amp.autocast():
                    logits = self._model(j, b)
                    _ = self._eval_metrics(logits, y)
                
                if (idx + 1) % self._config.accumulation_steps == 0:
                    counter.update(1)
        
        if counter.n < counter.total:
            counter.update(counter.total - counter.n)
        
        # Computing time metrics
        end_time = timer()
        counter.close()
        time = end_time - start_time
        num_samples = len(self._eval_loader) * self._eval_loader.batch_size * self._config.distributed.world_size
        speed = (num_samples / time) / self._config.distributed.world_size
        
        # Log metrics
        self._log_metrics(epoch, time, speed, False)
        top1_acc = self._eval_metrics.compute()['top1_acc']
        self._eval_metrics.reset()
           
        return top1_acc
    
    def start(self):
        try:
            self._logger.info('Starting training ...')
        
            if self._checkpoint is not None:
                start_epoch = self._checkpoint['start_epoch']
                best_results = self._checkpoint['best_results']
            else: 
                start_epoch = 0
                best_results = { 'top1_acc': 0, 'epoch': -1}
            
            for epoch in range(start_epoch, self._config.num_epochs):
                self._train_loader.sampler.set_epoch(epoch)
            
                torch.cuda.empty_cache()
                self._train(epoch)
                torch.cuda.empty_cache()
                top1_acc = self._eval(epoch)
            
                best_epoch = top1_acc > best_results['top1_acc']
                if best_epoch:
                    best_results['top1_acc'] = top1_acc
                    best_results['epoch'] = epoch
                    if self._config.distributed.is_local_master():
                        torch.save(self._model.module.state_dict(), self._config.best_weights_file)
                
                if (epoch + 1) % self._config.save_interleave == 0:
                    self._save_checkpoint(epoch, best_results)
   
            self._logger.info('Finished training ...')
            self._logger.info('Best results:')
            self._logger.info(f'\tepoch: {best_results["epoch"]}')
            self._logger.info(f'\ttop1 accuracy: {best_results["top1_acc"]}')
        except Exception as e:
            self._logger.error('Training failed with the following exception:')
            if self._config.debug:
                self._logger.exception(e)
            else:
                self._logger.error(e)
            raise e
