##
##
##

from typing import List
import logging

from .config import TrainingConfig
from .processor import TrainingProcessor

def start_trainings(configs: List[TrainingConfig], logger: logging.Logger, debug: bool):
    for idx, config in enumerate(configs):
        try:
            logger.info(f'Starting training process {idx}')
            p = TrainingProcessor(config)
            p.start()
            logger.info(f'Finished training process {idx}')
        except Exception as e:
            logger.error(f'Training process {idx} failed with the following exception:')
            if debug:
                logger.exception(e)
            else: 
                logger.error(e)