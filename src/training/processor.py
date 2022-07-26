##
##
##

import yaml

from .. import utils
from .config import TrainingConfig
from .pretraining import PretrainingProcessor
from .classification import ClassificationProcessor

class TrainingProcessor:
    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._logger = utils.init_logger('train', file=config.log_file)
    
    def _pretrain(self):
        if self._config.pretraining is None:
            self._logger.info('Pretraining not present')
        else:
            try:
                self._logger.info('Starting pretraining')
                p = PretrainingProcessor(self._config)
                p.start()
                self._logger.info('Finished pretraining')
            except Exception as e:
                self._logger.error('Pretraining failed with the following exception:')
                if self._config.debug:
                    self._logger.exception(e)
                else: 
                    self._logger.error(e)
                raise 
    
    def _classify(self):
        if self._config.classification is None:
            self._logger.info('Training not present')
        else:
            try:
                self._logger.info('Starting training')
                p = ClassificationProcessor(self._config)
                p.start()
                self._logger.info('Finished training')
            except Exception as e:
                self._logger.error('Training failed with the following exception:')
                if self._config.debug:
                    self._logger.exception(e)
                else: 
                    self._logger.error(e)
                raise 
    
    def _save_config(self):
        with open(self._config.config_file, 'w', newline='') as f:
            yaml.dump(self._config.to_dict(), f, default_flow_style=False, sort_keys=False, Dumper=utils.NoAliasDumper)
            
        with open(self._config.model_file, 'w', newline='') as f:
            yaml.safe_dump(self._config.model.to_dict(True), f, default_flow_style=False, sort_keys=False)
    
    def start(self):
        self._save_config()
        self._pretrain()
        self._classify()