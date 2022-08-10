##
##
##

import logging
import yaml
import wandb
from torch.utils.tensorboard import SummaryWriter

from .. import utils
from .config import TrainingConfig
from .pretraining import PretrainingProcessor
from .classification import ClassificationProcessor

class TrainingProcessor:
    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._logger = utils.init_logger(
            'train', 
            level=logging.INFO, 
            file=config.log_file, 
            local_master=config.distributed.is_local_master())
        
        if self._config.distributed.is_local_master():
            self._writer = SummaryWriter(log_dir=self._config.log_dir)
    
    def _pretrain(self):
        if self._config.process_pretraining:
            logger = utils.init_logger(
                name='pretrain', 
                level=logging.INFO, 
                file=self._config.pretrain_log_file, 
                local_master=self._config.distributed.is_local_master())
            
            try:
                self._logger.info('Starting pretraining')
                p = PretrainingProcessor(self._config.pretraining, logger, self._writer)
                p.start()
                self._logger.info('Pretraining finished')
            except Exception as e:
                self._logger.error('Pretraining failed with the following exception:')
                if self._config.debug:
                    self._logger.exception(e)
                else: 
                    self._logger.error(e)
                raise e
    
    def _classification(self):
        if self._config.process_classification:
            logger = utils.init_logger(
                name='classification', 
                level=logging.INFO, 
                file=self._config.classification_log_file, 
                local_master=self._config.distributed.is_local_master())
            
            try:
                self._logger.info('Starting classification')
                p = ClassificationProcessor(self._config.classification, logger, self._writer)
                p.start()
                self._logger.info('Classification finished')
            except Exception as e:
                self._logger.error('Classification failed with the following exception:')
                if self._config.debug:
                    self._logger.exception(e)
                else: 
                    self._logger.error(e)
                raise e
    
    def _save_config(self):
        if self._config.distributed.is_local_master():
            with open(self._config.config_file, 'w', newline='') as f:
                yaml.dump(self._config.to_dict(), f, default_flow_style=False, sort_keys=False, Dumper=utils.NoAliasDumper)
            
            with open(self._config.model_file, 'w', newline='') as f:
                yaml.safe_dump(self._config.model.to_dict(True), f, default_flow_style=False, sort_keys=False)
    
    def start(self):
        if self._config.distributed.is_master():
            run = wandb.init(
                job_type='training',
                dir=self._config.log_dir,
                config=self._config.to_dict(architecture=True),
                project='Skeleton-based Action Recognition',
                tags=['skeleton', 'action-recognition'],
                reinit=True,
                resume=True)
        
        self._save_config()
                
        self._pretrain()
        self._classification()
        
        if self._config.distributed.is_local_master():
            self._writer.close()
        if self._config.distributed.is_master():
            run.finish()