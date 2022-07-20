##
##
##

from typing import Tuple, Union
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
from ..model.model import Classifier, Encoder, EncoderClassifier
from ..dataset.dataset import Dataset
from ..dataset.skeleton import SkeletonGraph
from .. import utils
from .config import TrainingConfig
        
class TrainingProcessor: 
    
    def __init__(self, 
                 config: TrainingConfig) -> None:
        self._config = config
        
        self._init_environment()
        self._init_accuracy_file()
        self._logger, self._writer = self._get_loggers()
        self._checkpoint = self._load_checkpoint()
        self._device = self._get_device()
        
        skeleton = self._config.dataset_config.to_skeleton_graph()
        train_dataset, eval_dataset = self._get_datasets(skeleton)
        self._num_classes = train_dataset.num_classes
        self._train_loader, self._eval_loader =\
            self._get_loaders(train_dataset, eval_dataset)
            
        self._model: EncoderClassifier = self._get_model(
            skeleton, train_dataset.channels, train_dataset.num_classes)
        
        self._optimizer = self._get_optimizer()
        self._lr_scheduler = self._get_lr_scheduler()
        self._loss_func = self._get_loss()
        
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
        
        self._scaler = GradScaler()
        
    def _init_accuracy_file(self):
        with open(self._config.accuracy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'top1_accuracy', 'top5_accuracy'])
    
    def _get_loggers(self) -> Tuple[logging.Logger, SummaryWriter]:
        logger = utils.init_logger('train', 
                                   level=logging.INFO, 
                                   file=self._config.log_file)
        writer = SummaryWriter(log_dir=self._config.log_dir)
        
        return logger, writer

    def _load_checkpoint(self) -> Union[dict, None]:
        if not self._config.resume:
            return None
        
        try:
            checkpoint = torch.load(
                self._config.checkpoint_file, map_location=torch.device('cpu'))
            self._logger.info('Successfully loaded checkpoint.')
        except:
            self._logger.error(f'Checkpoint not found in {self._config.checkpoint_file}')
            raise ValueError(f'Checkpoint not found in {self._config.checkpoint_file}')
        
        return checkpoint
    
    def _save_checkpoint(self, next_epoch: int, best_results: dict, best_epoch: bool):
        ckp = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(), 
            'start_epoch': next_epoch,
            'best_results': best_results
        }
        
        torch.save(ckp, self._config.checkpoint_file)
        
        if best_epoch:
            torch.save(self._model.state_dict(), self._config.weights_file)
        
    def _save_results(self, current_epoch: int, top1_acc: float, top5_acc: float, cm: np.ndarray):
        cm_file = os.path.join(self._config.results_dir, f'cm-epoch-{current_epoch}.npy')
        np.save(cm_file, cm)
        
        with open(self._config.accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, top1_acc, top5_acc])
           
    def _get_device(self) -> torch.device:
        if len(self._config.gpus) > 0 and torch.cuda.is_available():
            self._logger.info('Using GPU')
            device = torch.device(f'cuda:{self._config.gpus[0]}')
        else: 
            self._logger.info('Using CPU')
            device = torch.device('cpu')
        
        return device
    
    def _get_datasets(self, skeleton: SkeletonGraph) -> Tuple[Dataset, Dataset]:
        cfg = self._config
        train_dataset = cfg.dataset_config.to_dataset(skeleton, True, self._logger)
        eval_dataset = cfg.dataset_config.to_dataset(skeleton, True, self._logger)
        
        self._logger.info(f'Training on dataset {train_dataset.name}')
        
        return train_dataset, eval_dataset
    
    def _get_loaders(self, train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=self._config.train_batch_size, 
                                  num_workers=4 * len(self._config.gpus),
                                  pin_memory=True,
                                  shuffle=True, 
                                  drop_last=True)

        eval_loader = DataLoader(dataset=eval_dataset, 
                                 batch_size=self._config.eval_batch_size, 
                                 num_workers=1,
                                 pin_memory=True, 
                                 shuffle=False, 
                                 drop_last=False)

        self._logger.info(f'Training batch size: {self._config.train_batch_size}')
        self._logger.info(f'Eval batch size: {self._config.eval_batch_size}')
        
        return train_loader, eval_loader
    
    def _get_model(self, skeleton: SkeletonGraph, channels: int, num_classes: int) -> EncoderClassifier:
        
        embeddings = Embeddings(self._config.model_config.embeddings, channels, skeleton)
        encoder = Encoder(self._config.model_config.encoder, False)
        classifier = Classifier(self._config.model_config.encoder.out_channels, num_classes) 
        model = EncoderClassifier(embeddings, encoder, classifier)
        
        self._logger.info(f'Model: {self._config.model_config.name}')
        
        with open(self._config.model_file, 'w') as f:
            print(model, file=f)
            
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: 
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        
        with open(os.path.join(self._config.work_dir, 'parameters.txt'), 'w') as f:
            print(table, file=f)
            print(f'Total number of parameters: {total_params}', file=f)
            
        self._logger.info(f'Model profile: {total_params/1e6:.2f}M parameters ({total_params})')
        
        if self._checkpoint is not None: 
            model.load_state_dict(self._checkpoint['model'])
            self._logger.info('Successfully loaded model state from checkpoint')
        
        model = model.to(self._device)
        return model
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self._config._optimizer_config.to_optimizer(self._model.parameters())
        self._logger.info(f'Optimizer: {self._config._optimizer_config}')
        if self._checkpoint is not None:
            optimizer.load_state_dict(self._checkpoint['optimizer'])
            self._logger.info('Successfully optimizer state from checkpoint')
        return optimizer
    
    def _get_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        lr_scheduler = self._config.lr_scheduler_config.to_lr_scheduler(self._optimizer)
        self._logger.info(f'LR scheduler: {self._config.lr_scheduler_config}')
        if self._checkpoint is not None:
            lr_scheduler.load_state_dict(self._checkpoint['lr_scheduler'])
            self._logger.info('Successfully lr scheduler state from checkpoint')
        
        return lr_scheduler
    
    def _get_loss(self):
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(self._device)
        self._logger.info(f'Loss function used: {loss_func.__class__.__name__}')
        
        return loss_func
    
    def _train(self, epoch: int):
        self._model.train()
        
        num_samples, num_top1, num_top5 = 0, 0, 0
        train_losses = []
        
        lr_scheduler_after_batch = self._config.lr_scheduler_config.after_batch
        
        train_iter = tqdm(self._train_loader)
        start_time = timer()
        for j, b, y in train_iter:
            self._optimizer.zero_grad()
            
            j: torch.Tensor = j.float().to(self._device)
            b: torch.Tensor = b.float().to(self._device)
            y: torch.Tensor = y.long().to(self._device)
            
            # Computing logits
            #with autocast():
            logits = self._model(j, b)
            loss: torch.Tensor = self._loss_func(logits, y)
                
            train_losses.append(loss.detach().item())
            
            #'''
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 5.0)
            self._scaler.step(self._optimizer)
            self._scaler.update()
            
            '''
            
            # Updating weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
            self._optimizer.step()
            
            '''
            
            if lr_scheduler_after_batch:
                self._lr_scheduler.step()
            
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
        
        if not lr_scheduler_after_batch:
            self._lr_scheduler.step()
        
        # Computing statistics
        train_time = end_time - start_time
        train_speed = (num_samples / train_time) / len(self._config.gpus)
        
        top1_acc = num_top1 / num_samples
        top5_acc = num_top5 / num_samples
        train_loss = np.mean(train_losses)
        
        last_lr_rate = self._lr_scheduler.get_last_lr()[-1]
        # Log statistics
        self._logger.info(f'Train epoch {epoch}')
        self._logger.info(f'\tTraining time: {train_time:.2f}s - Speed: {train_speed:.2f} samples/(second * GPU')
        self._logger.info(f'\tLearning rate: {last_lr_rate:.5f}')
        self._logger.info(f'\tTop-1 accuracy {num_top1:d}/{num_samples:d} ({top1_acc:.2%})')
        self._logger.info(f'\tTop-5 accuracy {num_top5:d}/{num_samples:d} ({top5_acc:.2%})')
        self._logger.info(f'\tMean loss: {train_loss:.5f}')
        
        # Add statistics on tensorboard
        self._writer.add_scalar('train/top1_accuracy', top1_acc, epoch)
        self._writer.add_scalar('train/top5_accuracy', top5_acc, epoch)
        self._writer.add_scalar('train/loss', train_loss, epoch)
        self._writer.add_scalar('train/learning_rate', last_lr_rate, epoch)
            
    def _eval(self, epoch: int) -> Tuple[float, float, np.ndarray]:
        self._model.eval()
        
        with torch.no_grad():
            num_samples, num_top1, num_top5 = 0, 0, 0
            eval_losses = []
            cm = np.zeros((self._num_classes, self._num_classes))
            
            eval_iter = tqdm(self._eval_loader)
            
            start_time = timer()
            for j, b, y in eval_iter:
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
        eval_speed = (num_samples / eval_time) / len(self._config.gpus)
        
        top1_acc = num_top1 / num_samples
        top5_acc = num_top5 / num_samples
        eval_loss = np.mean(eval_losses)
        
        # Log statistics
        self._logger.info(f'Evaluate epoch {epoch}')
        self._logger.info(f'\tEvaluating time: {eval_time:.2f}s - Speed: {eval_speed:.2f} samples/(second * GPU')
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
            start_epoch = 1
            best_results = {'top1_acc': 0, 'top5_acc': 0, 'cm': 0}
            
        for epoch in range(start_epoch, self._config.max_epoch + 1):
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
                
            self._save_checkpoint(epoch + 1, best_results, best_epoch)
            
        self._logger('Finished training ...')
            
            
        