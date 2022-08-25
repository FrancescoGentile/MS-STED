##
##
##

from __future__ import annotations
from typing import List, Optional
import os
import re

from .. import utils
from ..dataset.config import DatasetConfig
from ..model.config import ModelConfig
from .lr_scheduler import LRSchedulerConfig
from .optimizer import OptimizerConfig
from ..distributed import DistributedConfig

class TrainingConfig:
    def __init__(self, 
                 cfg: dict, 
                 datasets: List[DatasetConfig], 
                 models: List[ModelConfig],
                 optimizers: List[OptimizerConfig],
                 lr_schedulers: List[LRSchedulerConfig]) -> None:
        
        # Set dataset
        dataset_idx = cfg.dataset
        if dataset_idx is None or dataset_idx not in range(len(datasets)):
            raise ValueError(f'No configuration was given for dataset with index {dataset_idx}')
        self._dataset = datasets[dataset_idx]
        
        # Set model
        model_idx = cfg.model
        if model_idx is None or model_idx not in range(len(models)):
            raise ValueError(f'No configuration was given for model with index {model_idx}')
        self._model = models[model_idx]     
        
        # Other
        self._debug = cfg.debug
        self._seed = cfg.seed
        
        # Configuration files
        self._work_dir = cfg.work_dir
        utils.check_and_create_dir(self.work_dir)
        self._config_file = os.path.join(self._work_dir, 'config.yaml')
        self._model_file = os.path.join(self._work_dir, 'model.yaml')
        
        # Log files
        self._log_dir = os.path.join(self._work_dir, 'log')
        utils.check_and_create_dir(self._log_dir)
        self._log_file = os.path.join(self._log_dir, 'training.log')
        self._pretrain_log_file = os.path.join(self._log_dir, 'pretrain.log')
        self._classification_log_file = os.path.join(self._log_dir, 'classification.log')
        
        # Other configs
        self._distributed = cfg.distributed
        self._set_pretraining(cfg.pretraining_args, optimizers, lr_schedulers)
        self._set_classification(cfg.classification_args, optimizers, lr_schedulers)
        self._set_resume(cfg.resume)
    
    def _set_pretraining(
        self,
        cfg: Optional[dict],
        optimizers: List[OptimizerConfig],
        lr_schedulers: List[LRSchedulerConfig]):
        
        self._pretraining = None
        if cfg is not None:
            cfg.debug = self._debug
            cfg.work_dir = os.path.join(self.work_dir, 'pretrain')
            cfg.seed = self._seed
            cfg.distributed = self._distributed
            cfg.dataset = self._dataset
            cfg.model = self._model
            
            self._pretraining = PretrainingConfig(cfg, optimizers, lr_schedulers)
            
    def _set_classification(
        self, 
        cfg: Optional[dict], 
        optimizers: List[OptimizerConfig], 
        lr_schedulers: List[LRSchedulerConfig]):
        
        self._classification = None
        if cfg is not None:
            cfg.debug = self._debug
            cfg.work_dir = os.path.join(self.work_dir, 'classification')
            cfg.seed = self._seed
            cfg.distributed = self._distributed
            cfg.dataset = self._dataset
            cfg.model = self._model
            cfg.pretrain_weights = self._pretraining.best_weights_file \
                if self._pretraining is not None else None
            
            self._classification = ClassificationConfig(cfg, optimizers, lr_schedulers)
        
    def _set_resume(self, resume: Optional[str]):
        self._resume = resume
        if resume is None:
            self._process_pretraining = self._pretraining is not None
            self._process_classification = self._classification is not None
            return
        
        regex = r'^[cp]-\d+$'
        if re.search(regex, resume) is None:
            raise ValueError(f'Resume field in training config must be a string matching the regex {regex}')
        
        part = resume[0]
        epoch = resume[2:]
        if part == 'p':
            if self._pretraining is None:
                raise ValueError(f'Cannot resume from {resume} since pretraining args are missing')
            else:
                self._pretraining.resume_checkpoint = epoch
                self._process_pretraining = True
                self._process_classification = self._classification is not None
        else:
            if self._classification is None:
                raise ValueError(f'Cannot resume from {resume} since classification args are missing')
            else:
                self._classification.resume_checkpoint = epoch
                self._process_pretraining = False
                self._process_classification = True
    
    def to_dict(self, architecture: bool = False) -> dict:
        model = self._model.to_dict(architecture)
        if not architecture:
            model.update({'architecture': self._model_file})
        
        d = {'dataset': self._dataset.to_dict(generate=False, training=True),
             'model': model}
        
        if self._resume is not None:
            d.update({'resume': self._resume})
        
        if self._pretraining is not None:
            d.update({'pretraining': self._pretraining.to_dict()})
            
        if self._classification is not None:
            d.update({'classification': self._classification.to_dict()})
        
        return d
    
    @property
    def debug(self) -> bool:
        return self._debug
    
    @property
    def work_dir(self) -> str:
        return self._work_dir
    
    @property
    def log_dir(self) -> str:
        return self._log_dir
    
    @property
    def log_file(self) -> str:
        return self._log_file
    
    @property
    def pretrain_log_file(self) -> str:
        return self._pretrain_log_file
    
    @property
    def classification_log_file(self) -> str:
        return self._classification_log_file
    
    @property
    def config_file(self) -> str:
        return self._config_file
    
    @property
    def model_file(self) -> str:
        return self._model_file
    
    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset
    
    @property
    def model(self) -> ModelConfig:
        return self._model
    
    @property
    def pretraining(self) -> Optional[PretrainingConfig]:
        return self._pretraining
    
    @property
    def classification(self) -> Optional[ClassificationConfig]:
        return self._classification
    
    @property
    def distributed(self) -> DistributedConfig:
        return self._distributed
    
    @property
    def process_pretraining(self) -> bool:
        return self._process_pretraining
    
    @property
    def process_classification(self) -> bool:
        return self._process_classification


class ClassificationConfig:
    def __init__(self, cfg: dict, optimizers: List[OptimizerConfig], lr_schedulers: List[LRSchedulerConfig]) -> None:
        
        # Setup working directory
        self._work_dir = cfg.work_dir
        utils.check_and_create_dir(self._work_dir)
        self._metrics_file = os.path.join(self._work_dir, 'metrics.csv')
        
        # Setup files about model
        self._model_file = os.path.join(self._work_dir, 'model.txt')
        self._parameters_file = os.path.join(self._work_dir, 'parameters.txt')
        self._best_weights_file = os.path.join(self._work_dir, 'best_weights.pth')
        
        # Setup checkpoints
        self._confusion_dir = os.path.join(self._work_dir, 'confusion_matrix')
        utils.check_and_create_dir(self._confusion_dir)
        self._checkpoints_dir = os.path.join(self._work_dir, 'checkpoints')
        utils.check_and_create_dir(self._checkpoints_dir)
        self._resume_checkpoint = None
        self._save_interleave = cfg.save_interleave
        if self._save_interleave is None:
            self._save_interleave = 1

        # Setup batch size
        self._train_batch_size = cfg.train_batch_size
        self._eval_batch_size = cfg.eval_batch_size
        self._accumulation_steps = cfg.accumulation_steps
        self._num_epochs = cfg.num_epochs
        if self.num_epochs is None:
            raise ValueError(f'Number of epochs must be set.')
        
        # Other
        self._debug = cfg.debug
        self._seed = cfg.seed if cfg.seed is not None else 0
        self._distributed = cfg.distributed
        self._dataset = cfg.dataset
        self._model = cfg.model
        self._pretrain_weights = cfg.pretrain_weights
        
        # Set optimizer
        optim_idx = cfg.optimizer
        if optim_idx is None or optim_idx not in range(len(optimizers)):
            raise ValueError(f'No configuration was given for optimizer with index {optim_idx}')
        self._optimizer = optimizers[optim_idx]
        
        # Set LR scheduler
        scheduler_idx = cfg.scheduler
        if scheduler_idx is None or scheduler_idx not in range(len(lr_schedulers)):
            raise ValueError(f'No configuration was given for lr scheduler with index {scheduler_idx}')
        self._lr_scheduler = lr_schedulers[scheduler_idx]
        
        self._label_smoothing = cfg.label_smoothing
        if self._label_smoothing is None:
            self._label_smoothing = 0.0
    
    def to_dict(self) -> dict:
        d = {'seed': self._seed, 
             'train-batch-size': self._train_batch_size, 
             'eval-batch-size': self._eval_batch_size, 
             'accumulation-steps': self._accumulation_steps, 
             'save-interleave': self._save_interleave,
             'label-smoothing': self._label_smoothing,
             'optimizer': self._optimizer.to_dict(), 
             'lr-scheduler': self._lr_scheduler.to_dict()}
        
        return d
    
    @property
    def debug(self) -> bool:
        return self._debug
    
    @property
    def resume_checkpoint(self) -> Optional[str]:
        return self._resume_checkpoint
    
    @resume_checkpoint.setter
    def resume_checkpoint(self, epoch: int):
        file = self.checkpoint_file(epoch)
        if os.path.isfile(file):
            self._resume_checkpoint = file
        else:
            raise ValueError(f'No classification checkpoint exists for epoch {epoch}')
        
    @property
    def work_dir(self) -> str:
        return self._work_dir

    @property
    def model_description_file(self) -> str:
        return self._model_file
    
    @property
    def model_parameters_file(self) -> str:
        return self._parameters_file
    
    def checkpoint_file(self, epoch: int) -> str:
        return os.path.join(self._checkpoints_dir, f'epoch-{epoch}.tar')
    
    def confusion_matrix_file(self, epoch: int) -> str:
        return os.path.join(self._confusion_dir, f'epoch-{epoch}.npy')
    
    @property
    def save_interleave(self) -> int:
        return self._save_interleave
    
    @property
    def best_weights_file(self) -> str:
        return self._best_weights_file
    
    @property
    def metrics_file(self) -> str:
        return self._metrics_file
    
    @property
    def train_batch_size(self) -> int:
        return self._train_batch_size
    
    @property
    def eval_batch_size(self) -> int:
        return self._eval_batch_size
    
    @property
    def accumulation_steps(self) -> int:
        return self._accumulation_steps
    
    @property
    def num_epochs(self) -> int:
        return self._num_epochs
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @property
    def distributed(self) -> DistributedConfig:
        return self._distributed
    
    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset
    
    @property
    def model(self) -> ModelConfig:
        return self._model
    
    @property
    def optimizer(self) -> OptimizerConfig:
        return self._optimizer
    
    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        return self._lr_scheduler
    
    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
    
    @property
    def pretrain_weights(self) -> Optional[str]:
        return self._pretrain_weights
    
class PretrainingConfig:
    def __init__(self, cfg: dict, optimizers: List[OptimizerConfig], lr_schedulers: List[LRSchedulerConfig]) -> None:
        
        # Setup working directory
        self._work_dir = cfg.work_dir
        utils.check_and_create_dir(self._work_dir)
        self._metrics_file = os.path.join(self._work_dir, 'metrics.csv')
        
        # Setup files about model
        self._model_file = os.path.join(self.work_dir, 'model.txt')
        self._parameters_file = os.path.join(self.work_dir, 'parameters.txt')
        self._best_weights_file = os.path.join(self.work_dir, 'best_weights.pth')
        
        # Setup checkpoints
        self._checkpoints_dir = os.path.join(self._work_dir, 'checkpoints')
        utils.check_and_create_dir(self._checkpoints_dir)
        self._resume_checkpoint = None
        self._save_interleave = cfg.save_interleave
        if self._save_interleave is None:
            self._save_interleave = 1

        # Setup batch size
        self._train_batch_size = cfg.train_batch_size
        self._eval_batch_size = cfg.eval_batch_size
        self._accumulation_steps = cfg.accumulation_steps
        self._num_epochs = cfg.num_epochs
        if self.num_epochs is None:
            raise ValueError(f'Max epoch must be set.')
        
        # Other
        self._debug = cfg.debug
        self._seed = cfg.seed if cfg.seed is not None else 0
        self._distributed = cfg.distributed
        self._dataset = cfg.dataset
        self._model = cfg.model
        
        self._reconstruction_lambda = cfg.reconstruction_lambda
        if self._reconstruction_lambda is None:
            self._reconstruction_lambda = 1.0
            
        self._discrimination_lambda = cfg.discrimination_lambda
        if self._discrimination_lambda is None:
            self._discrimination_lambda = 1.0
        
        # Set optimizer
        optim_idx = cfg.optimizer
        if optim_idx is None or optim_idx not in range(len(optimizers)):
            raise ValueError(f'No configuration was given for optimizer with index {optim_idx}')
        self._optimizer = optimizers[optim_idx]
        
        # Set LR scheduler
        scheduler_idx = cfg.scheduler
        if scheduler_idx is None or scheduler_idx not in range(len(lr_schedulers)):
            raise ValueError(f'No configuration was given for lr scheduler with index {scheduler_idx}')
        self._lr_scheduler = lr_schedulers[scheduler_idx]
        
    def to_dict(self) -> dict:
        d = {'seed': self._seed,
             'train-batch-size': self._train_batch_size,
             'eval-batch-size': self._eval_batch_size,
             'accumulation-steps': self._accumulation_steps, 
             'save-interleave': self._save_interleave,
             'discrimination-lambda': self._discrimination_lambda,
             'reconstruction-lambda': self._reconstruction_lambda,
             'optimizer': self._optimizer.to_dict(), 
             'lr-scheduler': self._lr_scheduler.to_dict()}
        
        return d

    @property
    def debug(self) -> bool:
        return self._debug
    
    @property
    def resume_checkpoint(self) -> Optional[str]:
        return self._resume_checkpoint
    
    @resume_checkpoint.setter
    def resume_checkpoint(self, epoch: int):
        file = self.checkpoint_file(epoch)
        if os.path.isfile(file):
            self._resume_checkpoint = file
        else:
            raise ValueError(f'No classification checkpoint exists for epoch {epoch}')
        
    @property
    def work_dir(self) -> str:
        return self._work_dir

    @property
    def model_description_file(self) -> str:
        return self._model_file
    
    @property
    def model_parameters_file(self) -> str:
        return self._parameters_file
    
    def checkpoint_file(self, epoch: int) -> str:
        return os.path.join(self._checkpoints_dir, f'epoch-{epoch}.tar')
    
    @property
    def save_interleave(self) -> int:
        return self._save_interleave
    
    @property
    def best_weights_file(self) -> str:
        return self._best_weights_file
    
    @property
    def metrics_file(self) -> str:
        return self._metrics_file
    
    @property
    def train_batch_size(self) -> int:
        return self._train_batch_size
    
    @property
    def eval_batch_size(self) -> int:
        return self._eval_batch_size
    
    @property
    def accumulation_steps(self) -> int:
        return self._accumulation_steps
    
    @property
    def num_epochs(self) -> int:
        return self._num_epochs
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @property
    def distributed(self) -> DistributedConfig:
        return self._distributed
    
    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset
    
    @property
    def model(self) -> ModelConfig:
        return self._model
    
    @property
    def optimizer(self) -> OptimizerConfig:
        return self._optimizer
    
    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        return self._lr_scheduler
    
    @property
    def reconstruction_lambda(self) -> float:
        return self._reconstruction_lambda
    
    @property
    def discrimination_lamba(self) -> float:
        return self._discrimination_lambda
    