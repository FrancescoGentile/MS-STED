##
##
##

from typing import Tuple
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
from scipy.linalg import eig
import math

from .config import EmbeddingsConfig
from ..dataset.skeleton import SkeletonGraph, laplacian_adjacency_matrix

class Embeddings(nn.Module):
    
    def __init__(self, config: EmbeddingsConfig, in_channels: int, max_len: int, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._in_channels = in_channels
        self._out_channels = config.out_channels
        self._temporal_channels = config.temporal_channels
        self._skeleton = skeleton
        
        id_embed, id_embed_channels = self._get_id_embeddings()
        self.id_embeddings = Parameter(id_embed, requires_grad=False)
        self._id_channels = id_embed_channels
    
        joint_embed, bone_embed = self._get_type_embeddings(config.type_channels)
        self.joint_embeddings = Parameter(joint_embed)
        self.bone_embeddings = Parameter(bone_embed)
        self._type_channels = config.type_channels
        
        self.register_buffer(
            'temporal_enc', 
            self._get_temporal_encoding(max_len, config.temporal_channels), 
            persistent=False)
        
        total_channels = in_channels + self._type_channels + 2 * self._id_channels + config.temporal_channels
        self.embed_proj = nn.Conv2d(total_channels, config.out_channels, kernel_size=1)
    
    def _get_id_embeddings(self) -> Tuple[torch.Tensor, int]:
        adj = self._skeleton.adjacency_matrix
        laplacian = laplacian_adjacency_matrix(adj)
        
        _, vectors = eig(laplacian, left=False, right=True)
        id_embeddings = torch.from_numpy(vectors).float()
        id_embeddings = id_embeddings.unsqueeze(0).unsqueeze(2) # (1, C, 1, V)
        
        return id_embeddings, id_embeddings.size(1)
    
    def _get_type_embeddings(self, type_channels: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        bound = 1 / math.sqrt(type_channels)
        
        joint_embed = torch.empty(type_channels)
        nn.init.uniform_(joint_embed, -bound, +bound)
        
        bone_embed = torch.empty(type_channels)
        nn.init.uniform_(bone_embed, -bound, +bound)
        
        while torch.all(bone_embed == joint_embed):
            nn.init.uniform_(bone_embed, -bound, +bound)
        
        joint_embed = joint_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #(1, C, 1, 1)
        bone_embed = bone_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1, C, 1, 1)
        
        return joint_embed, bone_embed

    def _get_temporal_encoding(self, max_len: int, temporal_channels: int) -> torch.Tensor:
        te = torch.zeros(max_len, temporal_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, temporal_channels, 2).float() * (-math.log(10000.0) / temporal_channels))
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        
        te = te.transpose(0, 1) # (C, T)
        te = te.unsqueeze(0).unsqueeze(-1) # (1, C, T, 1)
        
        return te

    def forward(self, 
                joints: torch.Tensor,
                bones: torch.Tensor) -> torch.Tensor:
        
        N, _, T, V = joints.shape
        
        if self.training:
            # (V)
            sign_flip = torch.rand(self.id_embeddings.size(-1), device=joints.device)
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            sign_flip = sign_flip.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
            id_embeddings = self.id_embeddings * sign_flip
        else:
            id_embeddings = self.id_embeddings
        
        ide = id_embeddings.expand(N, self._id_channels, T, V)
        te = self.temporal_enc[:, :, :T, :]
        te = te.expand(N, self._temporal_channels, T, V)
        
        # joints
        je = self.joint_embeddings.expand(N, self._type_channels, T, V)
        j = torch.cat([joints, ide, ide, je, te], dim=1)
        
        # bones
        conn = self._skeleton.joints_connections
        first = ide[:, :, :, conn[:, 0]]
        second = ide[:, :, :, conn[:, 1]]
        be = self.bone_embeddings.expand(N, self._type_channels, T, V)
        b = torch.cat([bones, first, second, be, te], dim=1)
        
        # projection
        concat = torch.cat([j, b], dim=-1)
        output = self.embed_proj(concat)
        
        return output
        