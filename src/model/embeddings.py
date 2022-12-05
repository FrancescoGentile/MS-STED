##
##
##

from typing import Tuple
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import scipy
import math
import numpy as np

from .config import EmbeddingsConfig
from ..dataset.skeleton import SkeletonGraph

class Embeddings(nn.Module):
    
    def __init__(self, config: EmbeddingsConfig, in_channels: int, max_len: int, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._in_channels = in_channels
        self._out_channels = config.out_channels
        self._temporal_channels = config.temporal_channels
        self._id_channels = config.id_channels
        self._type_channels = config.type_channels
        self._skeleton = skeleton
        
        id_embed = self._get_id_embeddings()
        self.id_embeddings = Parameter(id_embed, requires_grad=False)
    
        joint_embed, bone_embed = self._get_type_embeddings()
        self.joint_embeddings = Parameter(joint_embed)
        self.bone_embeddings = Parameter(bone_embed)
        
        self.register_buffer(
            'temporal_enc', 
            self._get_temporal_encoding(max_len), 
            persistent=False)
        
        total_channels = in_channels + self._type_channels + 2 * self._id_channels + config.temporal_channels
        self.embed_proj = nn.Conv2d(total_channels, config.out_channels, kernel_size=1)
    
    def _get_id_embeddings(self) -> Tuple[torch.Tensor, int]:
        laplacian = scipy.sparse.csgraph.laplacian(self._skeleton.adjacency_matrix(False), normed=True)
        _, vectors = scipy.linalg.eigh(laplacian, overwrite_a=False)
        vectors: np.ndarray = vectors[:, -self._id_channels:] # (V, C)
        id_embeddings = torch.from_numpy(vectors).float()
        id_embeddings = torch.transpose(id_embeddings, 0, 1) # (C, V)
        id_embeddings = id_embeddings.unsqueeze(0).unsqueeze(2) # (1, C, 1, V)
        
        return id_embeddings
    
    def _get_type_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bound = 1 / math.sqrt(self._type_channels)
        
        joint_embed = torch.empty(self._type_channels)
        nn.init.uniform_(joint_embed, -bound, +bound)
        
        bone_embed = torch.empty(self._type_channels)
        nn.init.uniform_(bone_embed, -bound, +bound)
        
        while torch.all(bone_embed == joint_embed):
            nn.init.uniform_(bone_embed, -bound, +bound)
        
        joint_embed = joint_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #(1, C, 1, 1)
        bone_embed = bone_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1, C, 1, 1)
        
        return joint_embed, bone_embed

    def _get_temporal_encoding(self, max_len: int) -> torch.Tensor:
        te = torch.zeros(max_len, self._temporal_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self._temporal_channels, 2).float() * (-math.log(10000.0) / self._temporal_channels))
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        
        te = te.transpose(0, 1) # (C, T)
        te = te.unsqueeze(0).unsqueeze(-1) # (1, C, T, 1)
        
        return te

    def forward(self, 
                joints: torch.Tensor,
                bones: torch.Tensor) -> torch.Tensor:
        
        N, _, T, J = joints.shape
        _, _, _, B = bones.shape
        
        ide = self.id_embeddings.expand(N, self._id_channels, T, J)
        
        te = self.temporal_enc[:, :, :T, :]
        j_te = te.expand(N, self._temporal_channels, T, J)
        b_te = te.expand(N, self._temporal_channels, T, B)
        
        # joints
        je = self.joint_embeddings.expand(N, self._type_channels, T, J)
        j = torch.cat([joints, ide, ide, je, j_te], dim=1)
        
        # bones
        conn = self._skeleton.bones()
        first = ide[:, :, :, conn[:, 0]]
        second = ide[:, :, :, conn[:, 1]]
        be = self.bone_embeddings.expand(N, self._type_channels, T, B)
        b = torch.cat([bones, first, second, be, b_te], dim=1)
        
        # projection
        concat = torch.cat([j, b], dim=-1)
        output = self.embed_proj(concat)
        
        return output