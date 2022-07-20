##
##
##

from __future__ import annotations
import logging

from .skeleton import SkeletonGraph
from .dataset import Dataset
from .generator import DatasetGenerator

class DatasetConfigBuilder:
    @staticmethod
    def build(options, generate: bool) -> DatasetConfig:
        if options.name is None:
            raise ValueError(f'Dataset name not provided.')
        if options.name in NTU_DATASET_NAMES: 
            return NTUDatasetConfig(options, generate)
        else: 
            raise ValueError(f'No dataset {options.name} exists.')


class DatasetConfig:  
    
    def to_skeleton_graph(self) -> SkeletonGraph:
        raise NotImplementedError
          
    def to_dataset(self, 
                   skeleton_graph: SkeletonGraph,
                   train: bool,
                   logger: logging.Logger) -> Dataset:
        raise NotImplementedError()
    
    def to_generator(self) -> DatasetGenerator:
        raise NotImplementedError()
    
from src.dataset.ntu.config import NTUDatasetConfig, NTU_DATASET_NAMES