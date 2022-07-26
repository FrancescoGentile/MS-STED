##
##
##

import os
import logging
from typing import Tuple
import numpy as np
import yaml

from .skeleton import NTUSkeletonGraph
from .config import NTUDatasetConfig
from ..dataset import Dataset

class NTUDataset(Dataset):
    
    def __init__(self, 
                 config: NTUDatasetConfig, 
                 skeleton: NTUSkeletonGraph,
                 train_set: bool, 
                 pretrain: bool) -> None:
        super().__init__()
        
        self._config = config
        self._skeleton = skeleton
        self._num_classes = 60 if '60' in self._config.name else 120
        self._rng = np.random.default_rng(self._config.seed)
        
        phase = 'train' if train_set else 'test'
        self._load_data(phase)
        self._parse_mean_std(phase)
        
        self._pretrain = pretrain
    
    def _load_data(self, phase: str):
        path = self._config.dataset_path
        data_path = os.path.join(path, f'{phase}_data.npy')
        labels_path = os.path.join(path, f'{phase}_labels.npy')

        self._data = np.load(data_path, mmap_mode='r')
        self._labels = np.load(labels_path, mmap_mode='r')

        if self._config.debug: 
            self._data = self._data[:64]
            self._labels = self._labels[:64]
    
    def _parse_mean_std(self, phase: str):
        if not self._config.normalize:
            return
        
        mean_std_path = os.path.join(self._config._dataset_path, f'{phase}_mean_std.yaml')
        
        with open(mean_std_path, 'r') as f:
            mean_std = yaml.safe_load(f)
        
        # joints
        joints_mean = []
        joints_std = []
        for type in ['coordinate', 'velocity', 'distance']:
            for c in ['x', 'y', 'z']:
                joints_mean.append(mean_std['joints'][c][type]['mean'])
                joints_std.append(mean_std['joints'][c][type]['std'])
        
        self._jmean = np.reshape(np.array(joints_mean), newshape=(9, 1, 1, 1))
        self._jstd = np.reshape(np.array(joints_std), newshape=(9, 1, 1, 1))
        
        # bones
        bones_mean = []
        bones_std = []
        for type in ['coordinate', 'velocity', 'angle']:
            for c in ['x', 'y', 'z']:
                bones_mean.append(mean_std['bones'][c][type]['mean'])
                bones_std.append(mean_std['bones'][c][type]['std'])
        
        self._bmean = np.reshape(np.array(bones_mean), newshape=(9, 1, 1, 1))
        self._bstd = np.reshape(np.array(bones_std), newshape=(9, 1, 1, 1))
    
    def __len__(self): 
        return len(self._labels)
    
    def _get_transformed(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x: np.ndarray = self._data[idx].copy()
        _, T, V, M = x.shape
        
        files = (np.random.rand(T, V, M) * len(self._data))
        files = np.floor(files).astype(np.int32)
        changed = self._rng.binomial(1, self._config.change_probability, size=(T, V, M)) == 1
        
        shuffle_prob = self._config.shuffle_probability / self._config.shuffle_length
        inverted = np.zeros((T, V, M), dtype=np.bool8)

        for m in range(M):
            for t in range(T):
                for v in range(V):
                    if changed[t, v, m]:
                        if files[t, v, m] == idx:
                            changed[t, v, m] = False
                        else:
                            x[:, t, v, m] = self._data[files[t, v, m], :, t, v, m]
                    elif not inverted[t, v, m] and \
                        self._rng.choice([True, False], p=[shuffle_prob, 1-shuffle_prob]):
                            
                        idxs = [t]
                        offset = 1
                        remaining = self._config.shuffle_length
                        while remaining > 0:
                            if t - offset >= 0:
                                if inverted[t-offset, v, m]:
                                    break
                                idxs.append(t - offset)
                                remaining -= 1
                            elif t + offset < T:
                                if inverted[t+offset, v, m]:
                                    break
                                idxs.append(t + offset)
                                remaining -= 1
                    
                            offset += 1
                
                        if remaining > 0:
                            continue
                
                        permutation = np.random.permutation(idxs)
                        for o, n in zip(idxs, permutation):
                            inverted[o, v, m] = True
                            x[:, o, v, m] = self._data[idx, :, n, v, m]
    
        return x, changed
    
    def _get_joints_bone(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, T, V, M = data.shape
        B = len(self._skeleton.joints_connections)
        joints = np.zeros((C * 3, T, V, M))
        bones = np.zeros((C * 3, T, B, M))
        
        joints[:C] = data
        joints[C:C*2, :-1] = joints[:C, 1:] - joints[:C, :-1]
        joints[C*2:] = joints[:C] - np.expand_dims(joints[:C, :, 1], 2)
        
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
        
        return joints, bones

    def __getitem__(self, index: int):
        
        if self._pretrain:
            data, jc = self._get_transformed(index)
            bc = np.logical_or(
                jc[:, self._skeleton.joints_connections[:, 0]], 
                jc[:, self._skeleton.joints_connections[:, 1]])

            jm, bm = self._get_joints_bone(data)
            jo, bo = self._get_joints_bone(self._data[index])
            
            return jm, jo, jc, bm, bo, bc
        else:
            joints, bones = self._get_joints_bone(self._data[index])
            label = self._labels[index]
            
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
    