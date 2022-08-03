##
##
##

from typing import List
import logging
import torch.distributed as dist

from .config import DatasetConfig

def start_generations(configs: List[DatasetConfig], logger: logging.Logger, debug: bool):
    for idx, config in enumerate(configs):
        try:
            logger.info(f'Starting generation process {idx}')
            gp = config.to_generator()
            gp.start()
            logger.info(f'Finished generation process {idx}')
        except Exception as e:
            logger.error(f'Generation process {idx} failed with the following exception:')
            if debug:
                logger.exception(e)
            else: 
                logger.error(e)