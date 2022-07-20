##
##
##

import os
import logging
from typing import Tuple
import numpy as np
import torch
import yaml
from munch import Munch

from .skeleton import NTUSkeletonGraph
from .config import NTUDatasetConfig
from ..dataset import Dataset

class NTUDataset(Dataset):
    
    def __init__(self, 
                 config: NTUDatasetConfig, 
                 skeleton: NTUSkeletonGraph,
                 logger: logging.Logger, 
                 train: bool) -> None:
        super().__init__()
        self._config = config
        self._logger = logger
        self._skeleton = skeleton
        self._num_classes = 60 if '60' in self._config.name else 120
        
        phase = 'train' if train else 'test'
        
        self._data, self._labels = self._load_data(phase)
        if self._config.debug: 
            self._data = self._data[:50]
            self._labels = self._labels[:50]
            
        self._jmean, self._jstd, self._bmean, self._bstd = self._get_mean_std(phase)
    
    def _load_data(self, phase: str):
        path = self._config.dataset_path
        data_path = os.path.join(path, f'{phase}_data.npy')
        labels_path = os.path.join(path, f'{phase}_labels.npy')
        
        try:
            data = np.load(data_path, mmap_mode='r')
            labels = np.load(labels_path, mmap_mode='r')
        except Exception as e:
            if self._config.debug:
                self._logger.exception(e)
            else:
                self._logger.error(e)
            raise e

        return data, labels
    
    def _get_mean_std(self, phase: str):
        if not self._config.normalize:
            return None, None, None, None
        
        mean_std_path = os.path.join(self._config.dataset_path, f'{phase}_mean_std.yaml')
        
        try:
            with open(mean_std_path, 'r') as f:
                mean_std = yaml.safe_load(f)
            #mean_std = Munch.fromDict(mean_std)
        except Exception as e:
            if self._config.debug:
                self._logger.exception(e)
            else:
                self._logger.error(e)
            raise e
        
        # joints
        joints_mean = []
        joints_std = []
        for type in ['coordinate', 'velocity', 'distance']:
            for c in ['x', 'y', 'z']:
                joints_mean.append(mean_std['joints'][c][type]['mean'])
                joints_std.append(mean_std['joints'][c][type]['std'])
        
        joints_mean = torch.tensor(joints_mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        joints_std = torch.tensor(joints_std).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # bones
        bones_mean = []
        bones_std = []
        for type in ['coordinate', 'velocity', 'angle']:
            for c in ['x', 'y', 'z']:
                bones_mean.append(mean_std['bones'][c][type]['mean'])
                bones_std.append(mean_std['bones'][c][type]['std'])
        
        bones_mean = torch.tensor(bones_mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        bones_std = torch.tensor(bones_std).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return joints_mean, joints_std, bones_mean, bones_std
    
    def __len__(self): 
        return len(self._labels)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, str]:
        data = torch.from_numpy(np.array(self._data[index]))
        label = torch.tensor(self._labels[index])
        
        C, T, V, M = data.shape
        joints = torch.zeros((C * 3, T, V, M))
        bones = torch.zeros((C * 3, T, V, M))
        
        joints[:C] = data
        joints[C:C*2, :-1] = joints[:C, 1:] - joints[:C, :-1]
        joints[C*2:] = joints[:C] - joints[:C, :, 1].unsqueeze(2)
        
        conn = self._skeleton.joints_connections
        for u, v in conn:
            bones[:C, :, u] = joints[:C, :, u] - joints[:C, :, v]
        bones[C:C*2, :-1] = bones[:C, 1:] - bones[:C, :-1]
        
        bone_length = 0
        for c in range(C):
            bone_length += bones[c] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for c in range(C):
            bones[C*2+c] = np.arccos(bones[c] / bone_length)
            
        if self._config.normalize:
            joints = (joints - self._jmean) / self._jstd
            bones = (bones - self._bmean) / self._bstd
        
        return joints, bones, label
    
    @property
    def name(self) -> str:
        return self._config.name
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def channels(self) -> int:
        return 9
    
    @property
    def num_frames(self) -> int:
        return self._config.num_frames
    