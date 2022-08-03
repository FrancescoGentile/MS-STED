##
##
##

import argparse
import logging
import traceback
import torch.distributed

from src.config import Config
import src.distributed as dist
import src.utils as utils
import src.training as training
import src.dataset as dataset

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MS-STA for skeleton-based action recognition')
    
    parser.add_argument('--config', type=str, required=True, 
                        help='path to the config file')
    parser.add_argument('--generate', default=False, action='store_true', 
                        help='whether or not to start data generation')
    parser.add_argument('--train', default=False, action='store_true', 
                        help='whether or not to start trainings')
    parser.add_argument('--test', default=False, action='store_true',
                        help='whether or not to start tests')
    parser.add_argument('--debug', default=False, action='store_true', 
                        help='whether or not to set debug options')
    parser.add_argument('--distributed', default=None, type=str, 
                        help='path to the distributed config')
    parser.add_argument('--rank', default=None, type=int,
                        help='rank of this node')

    return parser

def start(config: Config):
    utils.init_logging()
    logger = utils.init_logger('main', logging.INFO, config.log_file, config.distributed.is_local_master())
    
    if config.distributed.is_local_master() and config.generate:
        dataset.start_generations(config.datasets_config, logger, config.debug)
    
    torch.distributed.barrier()

    if config.train:
        training.start_trainings(config.trainings_config, logger, config.debug)


def main(): 
    parser = init_parser()
    args = parser.parse_args()
    
    try:
        config = Config(
            args.config,
            args.distributed,
            args.rank,
            args.generate, 
            args.train, 
            args.test, 
            args.debug)
    except Exception as e:
        message = traceback.format_exc().strip('\n') if args.debug else e
        print(message)
        exit(1)
        
    dist.run(config.distributed, start, config)

if __name__ == '__main__':
    main()