##
##
##

from time import strftime
from typing import List
import os

from .. import utils
from ..dataset.config import DatasetConfig
from ..model.config import ModelConfig
from .lr_scheduler import LRSchedulerConfig
from .optimizer import OptimizerConfig

class TrainingConfig:
    def __init__(self, 
                 options: dict, 
                 datasets: List[DatasetConfig], 
                 models: List[ModelConfig],
                 optimizers: List[OptimizerConfig],
                 lr_schedulers: List[LRSchedulerConfig],
                 index: int) -> None:
        
        # Set optimizer
        optim_idx = options.optimizer
        if optim_idx is None or optim_idx not in range(len(optimizers)):
            raise ValueError(f'No configuration was given for optimizer with index {model_idx}')
        self._optimizer_config = optimizers[optim_idx]
        
        # Set LR scheduler
        scheduler_idx = options.scheduler
        if scheduler_idx is None or scheduler_idx not in range(len(lr_schedulers)):
            raise ValueError(f'No configuration was given for lr scheduler with index {model_idx}')
        self._lr_scheduler_config = lr_schedulers[scheduler_idx]
        
        # Set dataset
        dataset_idx = options.dataset
        if dataset_idx is None or dataset_idx not in range(len(datasets)):
            raise ValueError(f'No configuration was given for dataset with index {dataset_idx}')
        self._dataset_config = datasets[dataset_idx]
        
        # Set model
        model_idx = options.model
        if model_idx is None or model_idx not in range(len(models)):
            raise ValueError(f'No configuration was given for model with index {model_idx}')
        self._model_config = models[model_idx]     
        
        # Setup working directory
        work_dir = options.work_dir
        self._work_dir = os.path.join(work_dir, f'training-{index}')
        utils.check_and_create_dir(self._work_dir)
        
        # Setup log directory
        self._log_dir = os.path.join(self._work_dir, 'log')
        self._log_file = os.path.join(self._log_dir, 'log.txt')
        utils.check_and_create_dir(self._log_dir)
        
        # Results directory and file
        self._results_dir = os.path.join(self._work_dir, 'results')
        self._accuracy_file = os.path.join(self._results_dir, 'accuracy.csv')
        utils.check_and_create_dir(self._results_dir)
        
        self._model_file = os.path.join(self.work_dir, 'model.txt')
        
        # Checkpoints file
        self.checkpoint_file = os.path.join(self._work_dir, 'checkpoint.tar') 
        self.weights_file = os.path.join(self._work_dir, 'weights.pth')
        
        self._debug = options.debug 
        
        self.gpus = [0]
        if options.gpus is not None:
            if type(options.gpus) == list and len(options.gpus) != 1:
                raise ValueError(f'At the moment only one gpu supported')
            elif type(options.gpus) == list:
                self.gpus = options.gpus
            else: 
                self.gpus = [options.gpus] 
        
        self.train_batch_size = options.train_batch_size
        self.eval_batch_size = options.eval_batch_size
        self.seed = options.seed if options.seed is not None else 0
        self.max_epoch = options.max_epoch
        if self.max_epoch is None:
            raise ValueError(f'Max epoch must be set.')
        
        self.resume = options.resume if options.resume is not None else False
        
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
    def model_file(self) -> str:
        return self._model_file
    
    @property
    def results_dir(self) -> str:
        return self._results_dir
    
    @property
    def accuracy_file(self) -> str:
        return self._accuracy_file
    
    @property
    def optimizer_config(self) -> OptimizerConfig:
        return self._optimizer_config    
    
    @property 
    def lr_scheduler_config(self) -> LRSchedulerConfig:
        return self._lr_scheduler_config
    
    @property
    def dataset_config(self) -> DatasetConfig:
        return self._dataset_config
    
    @property
    def model_config(self) -> ModelConfig:
        return self._model_config