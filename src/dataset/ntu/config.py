##
##
##

from __future__ import annotations
import logging
import os
from typing import Tuple

from ..config import DatasetConfig
from ... import utils

NTU_DATASET_NAMES = ['ntu60-xsub', 'ntu60-xview', 'ntu120-xsub', 'ntu120-xset']

class NTUDatasetConfig(DatasetConfig):
    
    def __init__(self, 
                 options: dict, 
                 generate: bool) -> None:
        self.name = options.name
        if self.name is None or self.name not in NTU_DATASET_NAMES:
            raise ValueError(f'The dataset {self.name} is not valid.')
        
        self.debug = options.debug
        
        self.num_frames = options.generate_args.num_frames
        if self.num_frames is None and generate:
            raise ValueError(f'Missing num_frames config.')
       
        self.num_joints = 25
        self.num_coords = 3
        self.num_people = 2
        
        self.normalize = options.normalize
        if self.normalize is None:
            self.normalize = False
        
        self.ntu60_path, self.ntu120_path, self.ignored_file = \
            self._get_paths(options, generate)
        
    def _get_paths(self, options: dict, generate: None) -> Tuple[str, str, str]:
        self.dataset_path = options.dataset_path
        if self.dataset_path is None: 
            raise ValueError('No path where to save or find the processed dataset was inserted.')
        
        if generate:
            utils.check_and_create_dir(self.dataset_path)
        elif not os.path.isdir(self.dataset_path):
            raise ValueError(f'{self.name} path {self.dataset_path} is not a directory.')

        ntu60_path = None 
        ntu120_path = None
        ignored_file = None
        
        if generate: 
            options = options.generate_args
            ignored_file = options.ignored_file
            ntu60_path = options.ntu60_path
            if ntu60_path is None:
                raise ValueError(f'No path was provided for dataset {self.name}.')
            if not os.path.isdir(ntu60_path):
                raise ValueError(f'The provided path for ntu60 raw data ({ntu60_path}) is not valid.')

            if self.name in NTU_DATASET_NAMES[2:]:
                ntu120_path = options.ntu120_path
                if ntu120_path is None:
                    raise ValueError(f'No path was provided for dataset {self.name}')
                if not os.path.isdir(ntu120_path):
                    raise ValueError(f'The provided path for ntu120 raw data ({ntu60_path}) is not valid.')
        
        return ntu60_path, ntu120_path, ignored_file
    
    def to_skeleton_graph(self) -> NTUSkeletonGraph:
        return NTUSkeletonGraph()
    
    def to_dataset(self, 
                   skeleton: NTUSkeletonGraph,
                   logger: logging.Logger, 
                   train: bool) -> NTUDataset:
        return NTUDataset(self, skeleton, logger, train)
    
    def to_generator(self) -> NTUDatasetGenerator:
        return NTUDatasetGenerator(self)

from .dataset import NTUDataset
from .generator import NTUDatasetGenerator
from .skeleton import NTUSkeletonGraph