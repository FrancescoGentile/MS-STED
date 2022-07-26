##
##
##

import argparse
import logging
import traceback

from src.config import Config
import src.utils as utils
import src.training as training

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

    return parser

def generate(configs: Config, logger: logging.Logger):
    for idx, config in enumerate(configs.datasets_config):
        try:
            logger.info(f'Starting generation process {idx}')
            gp = config.to_generator()
            gp.start()
            logger.info(f'Finished generation process {idx}')
        except Exception as e:
            logger.error(f'Generation process {idx} failed with the following exception:')
            if configs.debug:
                logger.exception(e)
            else: 
                logger.error(e)

def main(): 
    parser = init_parser()
    args = parser.parse_args()
    
    try:
        config = Config(args.config, args.generate, args.train, args.test, args.debug)
    except Exception as e:
        message = traceback.format_exc().strip('\n') if args.debug else e
        print(message)
        exit(1)
    
    utils.init_logging()
    logger = utils.init_logger('main', logging.INFO, config.log_file)
    
    if args.generate: 
        generate(config, logger)

    if args.train:
        training.start_trainings(config.trainings_config, logger, config.debug)

if __name__ == '__main__':
    main()