##
##
##

from __future__ import annotations
import logging
import os
from typing import Optional, Tuple

from ..config import DatasetConfig
from ... import utils

NTU_DATASET_NAMES = ['ntu60-xsub', 'ntu60-xview', 'ntu120-xsub', 'ntu120-xset']

class NTUDatasetConfig(DatasetConfig):
    
    def __init__(self, 
                 cfg: dict, 
                 generate: bool) -> None:
        self._name = cfg.name
        if self._name is None or self._name not in NTU_DATASET_NAMES:
            raise ValueError(f'The dataset {self._name} is not valid.')
        
        self.debug = cfg.debug
        
        self._dataset_path = cfg.dataset_path
        if self._dataset_path is None: 
            raise ValueError('No path where to save or find the processed dataset was inserted.')
        elif generate:
            utils.check_and_create_dir(self._dataset_path)
        elif not os.path.isdir(self._dataset_path):
            raise ValueError(f'{self.name} path {self._dataset_path} is not a directory.')
       
        self.num_joints = 25
        self.num_coords = 3
        self.num_people = 2
        
        self._parse_training_args(cfg.training_args)
        self._parse_generation_args(cfg.generation_args, generate)

    def _parse_training_args(self, args: Optional[dict]):
        if args is None:
            raise ValueError(f'Missing training-args field in dataset config')
        
        self._normalize = args.normalize
        if self._normalize is None:
            self._normalize = False
        
        self._change_probability = args.change_probability
        if self._change_probability is None:
            self._change_probability = 0.0
        
        self._shuffle_probability = args.shuffle_probability
        if self._shuffle_probability is None:
            self._shuffle_probability = 0.0
        
        self._shuffle_length = args.shuffle_length
        if self._shuffle_length is None:
            self._shuffle_length = 0
            
        self._seed = args.seed
        if self._seed is None:
            self._seed = 0
    
    def _parse_generation_args(self, args: Optional[dict], generate: bool):
        if args is None:
            if generate:
                raise ValueError(f'Missing generate-args field in dataset config')
        if not generate:
            return
            
        self.num_frames = args.num_frames
        if self.num_frames is None:
            raise ValueError(f'Missing num_frames config.')
        
        self._ignored_file = args.ignored_file
        self._ntu60_path = args.ntu60_path
        if self._ntu60_path is None:
            raise ValueError(f'No path was provided for dataset {self.name}.')
        if not os.path.isdir(self._ntu60_path):
            raise ValueError(f'The provided path for ntu60 raw data ({self._ntu60_path}) is not valid.')

        self._ntu120_path = None
        if self._name in NTU_DATASET_NAMES[2:]:
            self._ntu120_path = args.ntu120_path
            if self._ntu120_path is None:
                raise ValueError(f'No path was provided for dataset {self.name}')
            if not os.path.isdir(self._ntu120_path):
                raise ValueError(f'The provided path for ntu120 raw data ({self._ntu120_path}) is not valid.')
            
    def to_dict(self, generate: bool, training: bool) -> dict:
        d = {'name': self._name,
             'dataset-path': self._dataset_path}
         
        if generate:
            dg = {'ignored-file': self._ignored_file,
                  'ntu60-path': self._ntu60_path, 
                  'ntu120-path': self._ntu120_path, 
                  'num-frames': self.num_frames}
            d.update({'generation-args': dg})
        
        if training:
            dt = {'normalize': self._normalize,
                  'change_probability': self._change_probability,
                  'shuffle_probability': self._shuffle_probability,
                  'shuffle_length': self._shuffle_length}
            
            d.update({'training-args': dt})
            
        return d

    def to_skeleton_graph(self) -> NTUSkeletonGraph:
        return NTUSkeletonGraph()
    
    def to_dataset(
        self, 
        skeleton: NTUSkeletonGraph,
        train_set: bool, 
        pretrain: bool) -> NTUDataset:
        return NTUDataset(self, skeleton, train_set, pretrain)
    
    def to_generator(self) -> NTUDatasetGenerator:
        return NTUDatasetGenerator(self)
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def change_probability(self) -> float:
        return self._change_probability
    
    @property
    def shuffle_probability(self) -> float:
        return self._shuffle_probability
    
    @property
    def shuffle_length(self) -> int:
        return self._shuffle_length
    
    @property
    def scale(self) -> Optional[float]:
        return None
    
    @property
    def rotation(self) -> Optional[float]:
        return None
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @property
    def ntu60_path(self) -> str:
        return self._ntu60_path
    
    @property
    def ntu120_path(self) -> Optional[str]:
        return self._ntu120_path
    
    @property
    def ignored_file(self) -> str:
        return self._ignored_file
    
    @property
    def dataset_path(self) -> str:
        return self._dataset_path

from .dataset import NTUDataset
from .generator import NTUDatasetGenerator
from .skeleton import NTUSkeletonGraph