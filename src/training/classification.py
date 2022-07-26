##
##
##

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
from ..model.model import Classifier, Encoder, ClassificationModel, Decoder, Reconstructor, Discriminator, ReconstructorDiscriminatorModel
from ..dataset.dataset import Dataset
from ..dataset.skeleton import SkeletonGraph
from .. import utils
from .config import TrainingConfig
        
class ClassificationProcessor: 
    
    def __init__(self, config: TrainingConfig) -> None:
        self._tconfig = config
        self._cconfig = config.classification
        
        self._scaler = GradScaler()
        self._init_environment()
        self._init_accuracy_file()
        self._set_loggers()
        self._load_checkpoint()
        self._set_device()
        
        skeleton = self._tconfig.dataset.to_skeleton_graph()
        train_dataset, eval_dataset = self._get_datasets(skeleton)
        self._num_classes = train_dataset.num_classes
        self._set_loaders(train_dataset, eval_dataset)
        self._set_model(skeleton, train_dataset.channels, train_dataset.num_classes)
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_loss()
        
    def _init_environment(self):
        torch.set_default_dtype(torch.float32)
        np.random.seed(self._cconfig.seed)
        random.seed(self._cconfig.seed)
        torch.manual_seed(self._cconfig.seed)
        torch.cuda.manual_seed(self._cconfig.seed)
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
        
    def _init_accuracy_file(self):
        with open(self._cconfig.accuracy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'top1_accuracy', 'top5_accuracy'])
    
    def _set_loggers(self):
        self._logger = utils.init_logger('classification', level=logging.INFO, file=self._cconfig.log_file)
        self._writer = SummaryWriter(log_dir=self._cconfig.log_dir)

    def _load_checkpoint(self):
        try:
            if not self._cconfig.resume:
                self._checkpoint = None
            else:
                self._checkpoint = torch.load(self._cconfig.checkpoint_file, map_location=torch.device('cpu'))
                self._logger.info('Successfully loaded checkpoint.')
        except:
            self._logger.error(f'Checkpoint not found in {self._cconfig.checkpoint_file}')
            raise ValueError(f'Checkpoint not found in {self._cconfig.checkpoint_file}')
    
    def _save_checkpoint(self, current_epoch: int, best_results: dict):
        ckp = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(), 
            'start_epoch': current_epoch + 1,
            'best_results': best_results
        }
        
        torch.save(ckp, self._cconfig.checkpoint_file(current_epoch))
        
    def _save_results(self, current_epoch: int, top1_acc: float, top5_acc: float, cm: np.ndarray):
        """_summary_

        Args:
            current_epoch (int): 0-based epoch number
            top1_acc (float): _description_
            top5_acc (float): _description_
            cm (np.ndarray): _description_
        """
        np.save(self._cconfig.confusion_matrix_file(current_epoch), cm)
        
        with open(self._cconfig.accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, top1_acc, top5_acc])
           
    def _set_device(self):
        if len(self._cconfig.gpus) > 0 and torch.cuda.is_available():
            self._logger.info('Using GPU')
            self._device = torch.device(f'cuda:{self._cconfig.gpus[0]}')
        else: 
            self._logger.info('Using CPU')
            self._device = torch.device('cpu')
    
    def _get_datasets(self, skeleton: SkeletonGraph) -> Tuple[Dataset, Dataset]:
        train_dataset = self._tconfig.dataset.to_dataset(skeleton, train_set=True, pretrain=False)
        eval_dataset = self._tconfig.dataset.to_dataset(skeleton, train_set=False, pretrain=False)
        
        self._logger.info(f'Training on dataset {train_dataset.name}')
        
        return train_dataset, eval_dataset
    
    def _set_loaders(self, train_dataset: Dataset, eval_dataset: Dataset):
        self._train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self._cconfig.train_batch_size // self._cconfig.accumulation_steps, 
            num_workers=4 * len(self._cconfig.gpus),
            pin_memory=True,
            shuffle=True, 
            drop_last=True)

        self._eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self._cconfig.eval_batch_size // self._cconfig.accumulation_steps, 
            num_workers=4 * len(self._cconfig.gpus),
            pin_memory=True, 
            shuffle=False, 
            drop_last=False)

        self._logger.info(f'Training batch size: {self._cconfig.train_batch_size}')
        self._logger.info(f'Eval batch size: {self._cconfig.eval_batch_size}')
        self._logger.info(f'Accumulation steps: {self._cconfig.accumulation_steps}')
    
    def _set_model(self, skeleton: SkeletonGraph, channels: int, num_classes: int):
        
        embeddings = Embeddings(self._tconfig.model.embeddings, channels, skeleton)
        encoder = Encoder(self._tconfig.model.encoder, False)
        if self._tconfig.pretraining is not None:
            decoder = Decoder(self._tconfig.model.decoder, self._tconfig.model.encoder)
            reconstructor = Reconstructor(self._tconfig.model.decoder.out_channels, channels)
            discriminator = Discriminator(self._tconfig.model.decoder.out_channels)
            model = ReconstructorDiscriminatorModel(embeddings, encoder, decoder, reconstructor, discriminator)
            
            state_dict = torch.load(self._tconfig.pretraining.best_weights_file, map_location='cpu')
            model.load_state_dict(state_dict)
        
        classifier = Classifier(self._tconfig.model.encoder.out_channels, num_classes) 
        model = ClassificationModel(embeddings, encoder, classifier)
        
        self._logger.info(f'Model: {self._tconfig.model.name}')
        
        with open(self._cconfig.model_description_file, 'w') as f:
            print(model, file=f)
            
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: 
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        
        with open(self._cconfig.model_parameters_file, 'w') as f:
            print(table, file=f)
            print(f'Total number of parameters: {total_params}', file=f)
            
        self._logger.info(f'Model profile: {total_params/1e6:.2f}M parameters ({total_params})')
        
        if self._checkpoint is not None: 
            model.load_state_dict(self._checkpoint['model'])
            self._logger.info('Successfully loaded model state from checkpoint')
        
        self._model = model.to(self._device)
    
    def _set_optimizer(self):
        optimizer = self._cconfig.optimizer.to_optimizer(self._model.parameters())
        self._logger.info(f'Optimizer: {self._cconfig.optimizer}')
        if self._checkpoint is not None:
            optimizer.load_state_dict(self._checkpoint['optimizer'])
            self._logger.info('Successfully optimizer state from checkpoint')
        
        self._optimizer = optimizer
    
    def _set_lr_scheduler(self):
        lr_scheduler = self._cconfig.lr_scheduler.to_lr_scheduler(self._optimizer)
        self._logger.info(f'LR scheduler: {self._cconfig.lr_scheduler}')
        if self._checkpoint is not None:
            lr_scheduler.load_state_dict(self._checkpoint['lr_scheduler'])
            self._logger.info('Successfully lr scheduler state from checkpoint')
        
        self._lr_scheduler = lr_scheduler
    
    def _set_loss(self):
        self._loss_func = nn.CrossEntropyLoss(label_smoothing=self._cconfig.label_smoothing).to(self._device)
        self._logger.info(f'Loss function used: {self._loss_func.__class__.__name__}')
    
    def _train(self, epoch: int):
        self._model.train()
        self._optimizer.zero_grad()
        
        num_samples, num_top1, num_top5 = 0, 0, 0
        train_losses = []
        
        before_epoch_lr_rate = self._lr_scheduler.get_lr()
        
        start_time = timer()
        for idx, (j, b, y) in enumerate(tqdm(self._train_loader)):
            
            j: torch.Tensor = j.float().to(self._device)
            b: torch.Tensor = b.float().to(self._device)
            y: torch.Tensor = y.long().to(self._device)
            
            # Computing logits
            with autocast():
                logits = self._model(j, b)
                loss: torch.Tensor = self._loss_func(logits, y)
                train_losses.append(loss.detach().item())
                loss = loss / self._cconfig.accumulation_steps
            
            self._scaler.scale(loss).backward()
            
            if (idx + 1) % self._cconfig.accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
            
                self._lr_scheduler.step(after_batch=True)
            
            num_samples += j.size(0)
             # Computing top1
            _, predicted_top1 = torch.max(logits, dim=1)
            num_top1 += predicted_top1.eq(y).sum().item()
            # Computing top5
            _, predicted_top5 = torch.topk(logits, 5)
            num_top5 += sum([y[n] in predicted_top5[n, :] for n in range(j.size(0))])
            
            del j
            del b
            del y
        
        end_time = timer()
        
        self._lr_scheduler.step(after_batch=False)
        
        # Computing statistics
        train_time = end_time - start_time
        train_speed = (num_samples / train_time) / len(self._cconfig.gpus)
        
        top1_acc = num_top1 / num_samples
        top5_acc = num_top5 / num_samples
        train_loss = np.mean(train_losses)
        
        after_epoch_lr_rate = self._lr_scheduler.get_lr()
        # Log statistics
        self._logger.info(f'Training epoch {epoch}')
        self._logger.info(f'\tTraining time: {train_time:.2f}s - Speed: {train_speed:.2f} samples/(second * GPU)')
        self._logger.info(f'\tLearning rate: (before) {before_epoch_lr_rate:.5f} -> (after) {after_epoch_lr_rate:.5f}')
        self._logger.info(f'\tTop-1 accuracy {num_top1:d}/{num_samples:d} ({top1_acc:.2%})')
        self._logger.info(f'\tTop-5 accuracy {num_top5:d}/{num_samples:d} ({top5_acc:.2%})')
        self._logger.info(f'\tMean loss: {train_loss:.5f}')
        
        # Add statistics on tensorboard
        self._writer.add_scalar('train/top1_accuracy', top1_acc, epoch)
        self._writer.add_scalar('train/top5_accuracy', top5_acc, epoch)
        self._writer.add_scalar('train/loss', train_loss, epoch)
        self._writer.add_scalar('train/learning_rate', before_epoch_lr_rate, epoch)
            
    def _eval(self, epoch: int) -> Tuple[float, float, np.ndarray]:
        self._model.eval()
        
        with torch.no_grad():
            num_samples, num_top1, num_top5 = 0, 0, 0
            eval_losses = []
            cm = np.zeros((self._num_classes, self._num_classes))
            
            start_time = timer()
            for j, b, y in tqdm(self._eval_loader):
                j: torch.Tensor = j.float().to(self._device)
                b: torch.Tensor = b.float().to(self._device)
                y: torch.Tensor = y.long().to(self._device)
                
                # Computing logits
                with torch.cuda.amp.autocast():
                    logits = self._model(j, b)
                    loss: torch.Tensor = self._loss_func(logits, y)
                
                eval_losses.append(loss.detach().item())
                
                num_samples += j.size(0)
                # Computing top1
                _, predicted_top1 = torch.max(logits, dim=1)
                num_top1 += predicted_top1.eq(y).sum().item()
                # Computing top5
                _, predicted_top5 = torch.topk(logits, 5)
                num_top5 += sum([y[n] in predicted_top5[n, :] for n in range(j.size(0))])
                
                # Computing confusion matrix
                for i in range(j.size(0)):
                    cm[y[i], predicted_top1[i]] += 1
                
                del j 
                del b 
                del y
        
        end_time = timer()
        
        # Computing statistics
        eval_time = end_time - start_time
        eval_speed = (num_samples / eval_time) / len(self._cconfig.gpus)
        
        top1_acc = num_top1 / num_samples
        top5_acc = num_top5 / num_samples
        eval_loss = np.mean(eval_losses)
        
        # Log statistics
        self._logger.info(f'Evaluating epoch {epoch}')
        self._logger.info(f'\tEvaluating time: {eval_time:.2f}s - Speed: {eval_speed:.2f} samples/(second * GPU)')
        self._logger.info(f'\tTop-1 accuracy {num_top1:d}/{num_samples:d} ({top1_acc:.2%})')
        self._logger.info(f'\tTop-5 accuracy {num_top5:d}/{num_samples:d} ({top5_acc:.2%})')
        self._logger.info(f'\tMean loss: {eval_loss:.5f}')
        
        # Add statistics on tensorboard
        self._writer.add_scalar('eval/top1_accuracy', top1_acc, epoch)
        self._writer.add_scalar('eval/top5_accuracy', top5_acc, epoch)
        self._writer.add_scalar('eval/loss', eval_loss, epoch)
                
        return top1_acc, top5_acc, cm
    
    def start(self):
        self._logger.info('Starting training ...')
        
        if self._checkpoint is not None:
            start_epoch = self._checkpoint['start_epoch']
            best_results = self._checkpoint['best_results']
        else: 
            start_epoch = 0
            best_results = {'top1_acc': 0, 'top5_acc': 0, 'cm': 0}
            
        for epoch in range(start_epoch, self._cconfig.num_epochs):
            torch.cuda.empty_cache()
            self._train(epoch)
            
            torch.cuda.empty_cache()
            top1_acc, top5_acc, cm = self._eval(epoch)
            
            self._save_results(epoch, top1_acc, top5_acc, cm)
            
            best_epoch = top1_acc > best_results['top1_acc']
            if best_epoch:
                best_results['top1_acc'] = top1_acc
                best_results['top5_acc'] = top5_acc
                best_results['cm'] = cm
                # Save model weights
                torch.save(self._model.state_dict(), self._cconfig.best_weights_file)
                
            if (epoch + 1) % self._cconfig.save_interleave == 0:
                self._save_checkpoint(epoch, best_results)

            
        self._logger('Finished training ...')
            
            
        