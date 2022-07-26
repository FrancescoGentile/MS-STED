##
##
##

import os
from time import strftime
from typing import List

from . import utils
from .dataset.config import DatasetConfig, DatasetConfigBuilder
from .model.config import ModelConfig
from .training.optimizer import OptimizerConfig
from .training.lr_scheduler import LRSchedulerConfig
from .test.config import TestConfig
from .training.config import TrainingConfig

class Config:
    
    def __init__(self, 
                 config_file: str,
                 generate: bool,
                 train: bool, 
                 test: bool, 
                 debug: bool) -> None:
        
        cfg = utils.load_config_file(config_file)
        
        self.debug = debug
        
        self.work_dir = cfg.work_dir
        if self.work_dir is None:
            self.work_dir = './work_dir'
            
        self.work_dir = os.path.join(self.work_dir, strftime('%Y-%m-%d-%H-%M-%S'))
        utils.check_and_create_dir(self.work_dir)
        
        config_name = os.path.splitext(config_file)[0]
        config_name = os.path.basename(config_name)
        self.log_file = os.path.join(self.work_dir, f'{config_name}.log')
        
        self.gpus = cfg.gpus
        self.seed = cfg.seed
        
        self.datasets_config = self._get_datasets(cfg, generate)
        self.models_config = Config._get_models(cfg, train, test)
        self.optimizers_config = Config._get_optimizers(cfg, train)
        self.lr_schedulers_config = Config._get_lr_schedulers(cfg, train)
        self.trainings_config = self._get_trainings(cfg, train)
        self.tests_config = self._get_tests(cfg, test) 

    def _get_datasets(self, options: dict, generate: bool) -> List[DatasetConfig]:
        dataset_options = options.datasets
        if dataset_options is None: 
            raise ValueError('No config options for datasets were provided.')
        
        if type(dataset_options) != list: 
            raise ValueError('Configurations for datasets must be a list.')
        
        datasets = []
        for opt in dataset_options:
            opt.debug = self.debug
            if opt.seed is None:
                opt.seed = self.seed
            datasets.append(DatasetConfigBuilder.build(opt, generate))
        
        return datasets
    
    @staticmethod
    def _get_models(options: dict, train: bool, test: bool) -> List[ModelConfig]:
        if not train and not test:
            return []
        
        model_options = options.models
        if model_options is None:
            raise ValueError('No config options for models were provided.')

        if type(model_options) != list: 
            raise ValueError('Configurations for models must be a list.')
        
        models = []
        for opt in model_options:
            models.append(ModelConfig(opt))
        
        return models

    @staticmethod
    def _get_optimizers(options: dict, train: bool) -> List[OptimizerConfig]:
        if not train:
            return []
        
        optimzer_options = options.optimizers
        if optimzer_options is None:
            raise ValueError('No config options for optimizers were provided.')

        if type(optimzer_options) != list: 
            raise ValueError('Configurations for optimizers must be a list.')
        
        optimizers = []
        for opt in optimzer_options:
            optimizers.append(OptimizerConfig(opt))
        
        return optimizers

    @staticmethod
    def _get_lr_schedulers(options: dict, train: bool) -> List[LRSchedulerConfig]:
        if not train:
            return []
        
        scheduler_options = options.lr_schedulers
        if scheduler_options is None:
            raise ValueError('No config options for lr schedulers were provided.')

        if type(scheduler_options) != list: 
            raise ValueError('Configurations for lr schdulers must be a list.')
        
        schedulers = []
        for opt in scheduler_options:
            schedulers.append(LRSchedulerConfig(opt))
        
        return schedulers
    
    def _get_trainings(self, options: dict, train: bool) -> List[TrainingConfig]:
        if not train: 
            return []

        training_options = options.trainings
        if training_options is None: 
            raise ValueError('No config options for trainings were provided.')

        if type(training_options) != list: 
            raise ValueError('Configurations for trainings must be a list.')

        trainings = []
        for idx, opt in enumerate(training_options):
            opt.debug = self.debug
            if opt.work_dir is None:
                opt.work_dir = os.path.join(self.work_dir, f'training-{idx}')
            if opt.gpus is None:
                opt.gpus = self.gpus
            if opt.seed is None: 
                opt.seed = self.seed
            opt.debug = self.debug
            trainings.append(TrainingConfig(
                opt, 
                self.datasets_config, 
                self.models_config,
                self.optimizers_config,
                self.lr_schedulers_config))
        
        return trainings

    def _get_tests(self, options: dict, test: bool) -> List[TestConfig]:
        if not test:
            return []

        test_options = options.tests
        if test_options is None: 
            raise ValueError('No config options for tests were provided.')

        if type(test_options) != list: 
            raise ValueError('Configurations for tests must be a list.')

        tests = []
        for idx, opt in enumerate(test_options): 
            if opt.work_dir is None:
                opt.work_dir = os.path.join(self.work_dir, f'test-{idx}')
            if opt.gpus is None:
                opt.gpus = self.gpus
            if opt.seed is None: 
                opt.seed = self.seed
            opt.debug = self.debug
            tests.append(TestConfig(
                opt,
                self.datasets_config,
                self.models_config))
        
        return tests
    