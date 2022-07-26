##
##
##

from __future__ import annotations
from collections import OrderedDict
from typing import List, Optional
import os

from .. import utils
from ..dataset.config import DatasetConfig
from ..model.config import ModelConfig
from .lr_scheduler import LRSchedulerConfig
from .optimizer import OptimizerConfig

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
        
        self._work_dir = cfg.work_dir
        self._log_file = os.path.join(self._work_dir, 'train.log')
        self._config_file = os.path.join(self._work_dir, 'config.yaml')
        self._model_file = os.path.join(self._work_dir, 'model.yaml')
        utils.check_and_create_dir(self.work_dir)
        
        self._pretraining = None
        if cfg.pretraining_args is not None:
            cfg.pretraining_args.work_dir = self.work_dir
            cfg.pretraining_args.gpus = cfg.gpus
            cfg.pretraining_args.seed = cfg.seed
            self._pretraining = PretrainingConfig(cfg.pretraining_args, optimizers, lr_schedulers)
        
        self._classification = None
        if cfg.classification_args is not None:
            cfg.classification_args.work_dir = self.work_dir
            cfg.classification_args.gpus = cfg.gpus
            cfg.classification_args.seed = cfg.seed
            self._classification = ClassificationConfig(cfg.classification_args, optimizers, lr_schedulers)
    
    def to_dict(self) -> dict:
        model = self._model.to_dict(False)
        model.update({'architecture': self._model_file})
        
        d = {'dataset': self._dataset.to_dict(generate=False, training=True),
             'model': model }
        
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
    def log_file(self) -> str:
        return self._log_file
    
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

class ClassificationConfig:
    def __init__(self, cfg: dict, optimizers: List[OptimizerConfig], lr_schedulers: List[LRSchedulerConfig]) -> None:
        
        # Setup working directory
        work_dir = cfg.work_dir
        self._work_dir = os.path.join(work_dir, 'classification')
        utils.check_and_create_dir(self.work_dir)
        
        # Setup log directory
        self._log_dir = os.path.join(self.work_dir, 'log')
        self._log_file = os.path.join(self.log_dir, 'log.txt')
        utils.check_and_create_dir(self.log_dir)
        
        # Setup results directory
        self._results_dir = os.path.join(self.work_dir, 'results')
        self._accuracy_file = os.path.join(self.results_dir, 'accuracy.csv')
        utils.check_and_create_dir(self.results_dir)
        
        # Setup files about model
        self._model_file = os.path.join(self.work_dir, 'model.txt')
        self._parameters_file = os.path.join(self.work_dir, 'parameters.txt')
        self._best_weights_file = os.path.join(self.work_dir, 'weights.pth')
        self._save_interleave = cfg.save_interleave
        if self._save_interleave is None:
            self._save_interleave = 1
        
        # Setup gpus
        self._gpus = [0]
        if cfg.gpus is not None:
            if type(cfg.gpus) == list and len(cfg.gpus) != 1:
                raise ValueError(f'At the moment only one gpu supported')
            elif type(cfg.gpus) == list:
                self._gpus = cfg.gpus
            else: 
                self._gpus = [cfg.gpus]

        # Setup batch size
        self._train_batch_size = cfg.train_batch_size
        self._eval_batch_size = cfg.eval_batch_size
        self._accumulation_steps = cfg.accumulation_steps
        self._num_epochs = cfg.num_epochs
        if self.num_epochs is None:
            raise ValueError(f'Max epoch must be set.')
        
        # Other
        self._seed = cfg.seed if cfg.seed is not None else 0
        
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
             'gpus': self._gpus,
             'train-batch-size': self._train_batch_size, 
             'eval-batch-size': self._eval_batch_size, 
             'accumulation-steps': self._accumulation_steps, 
             'save-interleave': self._save_interleave,
             'label-smoothing': self._label_smoothing,
             'optimizer': self._optimizer.to_dict(), 
             'lr-scheduler': self._lr_scheduler.to_dict()}
        
        return d
    
    @property
    def resume(self) -> bool:
        return False # TODO
        
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
    def model_description_file(self) -> str:
        return self._model_file
    
    @property
    def model_parameters_file(self) -> str:
        return self._parameters_file
    
    def checkpoint_file(self, epoch: int) -> str:
        return os.path.join(self._work_dir, f'checkpoint-{epoch}.tar')
    
    @property
    def save_interleave(self) -> int:
        return self._save_interleave
    
    @property
    def best_weights_file(self) -> str:
        return self._best_weights_file
    
    @property
    def results_dir(self) -> str:
        return self._results_dir
    
    @property
    def accuracy_file(self) -> str:
        return self._accuracy_file
    
    def confusion_matrix_file(self, epoch: int) -> str:
        return os.path.join(self.results_dir, f'confusion-matrix-{epoch}.npy')
    
    @property
    def gpus(self) -> List[int]:
        return self._gpus
    
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
    def optimizer(self) -> OptimizerConfig:
        return self._optimizer
    
    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        return self._lr_scheduler
    
    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
    
class PretrainingConfig:
    def __init__(self, cfg: dict, optimizers: List[OptimizerConfig], lr_schedulers: List[LRSchedulerConfig]) -> None:
        
        # Setup working directory
        work_dir = cfg.work_dir
        self._work_dir = os.path.join(work_dir, 'pretraining')
        utils.check_and_create_dir(self.work_dir)
        
        # Setup log directory
        self._log_dir = os.path.join(self.work_dir, 'log')
        self._log_file = os.path.join(self.log_dir, 'log.txt')
        utils.check_and_create_dir(self.log_dir)
        
        # Setup results directory
        results_dir = os.path.join(self.work_dir, 'results')
        self._accuracy_file = os.path.join(results_dir, 'accuracy.csv')
        utils.check_and_create_dir(results_dir)
        
        # Setup files about model
        self._model_file = os.path.join(self.work_dir, 'model.txt')
        self._parameters_file = os.path.join(self.work_dir, 'parameters.txt')
        self._best_weights_file = os.path.join(self.work_dir, 'weights.pth')
        self._save_interleave = cfg.save_interleave
        if self._save_interleave is None:
            self._save_interleave = 1
        
        # Setup gpus
        self._gpus = [0]
        if cfg.gpus is not None:
            if type(cfg.gpus) == list and len(cfg.gpus) != 1:
                raise ValueError(f'At the moment only one gpu supported')
            elif type(cfg.gpus) == list:
                self._gpus = cfg.gpus
            else: 
                self._gpus = [cfg.gpus]

        # Setup batch size
        self._train_batch_size = cfg.train_batch_size
        self._eval_batch_size = cfg.eval_batch_size
        self._accumulation_steps = cfg.accumulation_steps
        self._num_epochs = cfg.num_epochs
        if self.num_epochs is None:
            raise ValueError(f'Max epoch must be set.')
        
        # Other
        self._seed = cfg.seed if cfg.seed is not None else 0
        
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
             'gpus': self._gpus,
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
    def resume(self) -> bool:
        return False # TODO
        
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
    def model_description_file(self) -> str:
        return self._model_file
    
    @property
    def model_parameters_file(self) -> str:
        return self._parameters_file
    
    def checkpoint_file(self, epoch: int) -> str:
        return os.path.join(self._work_dir, f'checkpoint-{epoch}.tar')
    
    @property
    def save_interleave(self) -> int:
        return self._save_interleave
    
    @property
    def best_weights_file(self) -> str:
        return self._best_weights_file
    
    @property
    def accuracy_file(self) -> str:
        return self._accuracy_file
    
    @property
    def gpus(self) -> List[int]:
        return self._gpus
    
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
    