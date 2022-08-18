##
##
##

from __future__ import annotations
import math
from typing import Optional
import numpy as np

import torch
import torch.nn as nn

from .config import BlockConfig, CrossViewType, LayerConfig, BranchConfig
from ..dataset.skeleton import SkeletonGraph, normalize_adjacency_matrix

class AttentionPool(nn.Module):
    def __init__(self, k: int, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        assert channels % num_heads == 0, 'Channels must be 0 modulo number of heads.'
        
        self._num_heads = num_heads
        self._k = k
        self.qk_proj = nn.Linear(channels, channels * 2)
        
    def forward(self, x: torch.Tensor):
        N, C, T, V = x.shape
        new_t = math.ceil(T / self._k)
        
        xn = x.permute(0, 2, 1, 3).contiguous().mean(-1)
        qk: torch.Tensor = self.qk_proj(xn)
        qk = qk.view(N, T, self._num_heads, -1)
        qk = qk.permute(0, 2, 1, 3).contiguous()
        q, k = torch.chunk(qk, 2, dim=-1)
        
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = torch.mean(att, dim=1)
        scores = torch.mean(att, dim=-1)
        values, idx = torch.topk(scores, k=new_t, dim=-1)
        
        idx = idx[:, None, :, None]
        idx = idx.expand(N, C, new_t, V)
        
        values = torch.sigmoid(values)
        values = values[:, None, :, None]
        values = values.expand(N, C, new_t, V)
        
        out = torch.gather(x, 2, idx)
        out *= values
        
        return out

class Block(nn.Module):
    
    def __init__(self, config: BlockConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for layer_cfg in config.layers:
            self.layers.append(Layer(layer_cfg, skeleton))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            out = layer(x)
            x = out 
        
        return out

class Layer(nn.Module):
    def __init__(self, config: LayerConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._cross_view = config.cross_view
        
        self.branches = nn.ModuleList()
        for branch_cfg in config.branches:
            attention = AttentionBranch(branch_cfg, skeleton)
            self.branches.append(Branch(branch_cfg, attention))
            
        match self._cross_view:
            case CrossViewType.LARGER:
                self.cross_proj = nn.ModuleList()
                for current_branch, before_branch in zip(config.branches[1:], config.branches):
                    if before_branch.out_channels != current_branch.channels:
                        proj = nn.Conv2d(before_branch.out_channels, current_branch.channels, kernel_size=1)
                    else:
                        proj = nn.Identity()
                    self.cross_proj.append(proj)
            case CrossViewType.SMALLER:
                self.cross_proj = nn.ModuleList()
                for current_branch, before_branch in zip(config.branches, config.branches[1:]):
                    if before_branch.out_channels != current_branch.channels:
                        proj = nn.Conv2d(before_branch.out_channels, current_branch.channels, kernel_size=1)
                    else:
                        proj = nn.Identity()
                    self.cross_proj.append(proj)
            case _:
                self.cross_proj = None
        
        if config.in_channels == config.out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(config.in_channels, config.out_channels, kernel_size=1)
    
    def _larger(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [None] * len(self.branches)
        cross_x = None 
        last_idx = len(self.branches) - 1
        for idx, branch in enumerate(self.branches):
            out = branch(x, cross_x)
            outputs[idx] = out
            
            if self.cross_proj is not None and idx < last_idx:
                cross_x = self.cross_proj[idx](out)

        output = torch.cat(outputs, dim=1)
        
        return output
    
    def _smaller(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [None] * len(self.branches)
        cross_x = None 
        for idx, branch in reversed(list(enumerate(self.branches))):
            out = branch(x, cross_x)
            outputs[idx] = out
            
            if self.cross_proj is not None and idx > 0:
                cross_x = self.cross_proj[idx-1](out)

        output = torch.cat(outputs, dim=1)
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        match self._cross_view:
            case CrossViewType.NONE:
                out = self._larger(x)
            case CrossViewType.LARGER:
                out = self._larger(x)
            case CrossViewType.SMALLER:
                out = self._smaller(x)
            case _:
                raise ValueError('Unknown cross-view type')
                
        out += res
        
        return out

class Branch(nn.Module):
    def __init__(self, config: BranchConfig, attention: AttentionBranch) -> None:
        super().__init__()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        # Attention
        self.att_branch = attention
        
        # Temporal convolution
        self.temp_norm = nn.GroupNorm(1, config.channels)
        self.temp = nn.Sequential(
            nn.Conv2d(config.channels, config.out_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Conv2d(
                config.out_channels,
                config.out_channels, 
                kernel_size=(config.window, 1),
                stride=(config.window, 1))
        )
        
    def _group_window_joints(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        #(N, C * window, T, V)
        x = self.unfold(x)
        # (N, C, window, T, V)
        x = x.view(N, C, self.window, T, V)
        # (N, C, T, window, V)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # (N, C, T, window * V)
        x = x.view(N, C, T, self.window * V)
        
        return x
    
    def _group_window_frames(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        # (N, C, T, window * V)
        N, C, T, _ = x.shape
        # (N, C, T, window, V)
        x = x.view(N, C, T, self.window, -1)
        # (N, C, T * window, V)
        x = x.view(N, C, T * self.window, -1)
        
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor]) -> torch.Tensor:
        N, _, _, _ = x.shape
        
        # Attention
        x = self._group_window_joints(x)
        if cross_x is not None:
            cross_x = self._group_window_joints(cross_x)
        
        att_out = self.att_branch(x, cross_x)
        
        # Temporal convolution
        temp_out = self.temp_norm(att_out)
        temp_out = self._group_window_frames(temp_out, batch=N)
        temp_out = self.temp(temp_out)
        
        return temp_out

class AttentionBranch(nn.Module):
    def __init__(self, config: BranchConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        # Attention
        self.first_norm = nn.GroupNorm(1, config.channels)
        self.attention = SpatioTemporalAttention(skeleton, config)
        
        if config.cross_view:
            self.second_norm = nn.GroupNorm(1, config.channels)
            self.cross_norm = nn.GroupNorm(1, config.channels)
            self.cross_attention = SpatioTemporalAttention(skeleton, config)
            
        self.dropout = nn.Dropout(config.sublayer_dropout)
        
        # FFN
        self.ffn_norm = nn.GroupNorm(1, config.channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, kernel_size=1),
            nn.GELU(), 
            nn.Dropout(config.feature_dropout),
            nn.Conv2d(config.channels, config.channels, kernel_size=1)
        )
    
    @staticmethod
    def generate_dict(cross_view: bool, config: BranchConfig) -> dict:
        d = {}
        
        d['attention'] = SpatioTemporalAttention(
            config.channels, config.num_heads, config.feature_dropout, config.structure_dropout)
        
        if cross_view:
            d['cross_attention'] = SpatioTemporalAttention(
            config.channels, config.num_heads, config.feature_dropout, config.structure_dropout)
        
        d['ffn'] = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, kernel_size=1),
            nn.GELU(), 
            nn.Dropout(config.feature_dropout),
            nn.Conv2d(config.channels, config.channels, kernel_size=1)
        )
        
        return d
        
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor]):
        norm_x = self.first_norm(x)
        att_out = self.attention(norm_x, norm_x)
        att_out = self.dropout(att_out)
        att_out += x
        
        if cross_x is not None:
            x_q = self.second_norm(att_out)
            x_kv = self.cross_norm(cross_x)
            cross_att_out = self.cross_attention(x_q, x_kv)
            cross_att_out = self.dropout(cross_att_out)
            cross_att_out += att_out
            att_out = cross_att_out
        
        ffn_out = self.ffn_norm(att_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out += att_out
        
        return ffn_out
    
class SpatioTemporalAttention(nn.Module):
    def __init__(self,
                 skeleton: SkeletonGraph,
                 config: BranchConfig) -> None:
        super().__init__()
        
        self._num_heads = config.num_heads
        self._channels = config.channels
        self._head_channels = config.channels // config.num_heads
        self._structure_dropout = config.structure_dropout
        self.feature_dropout = nn.Dropout(config.feature_dropout)
        
        # (V, V)
        global_attn = self._build_spatio_temporal_graph(
            skeleton.joints_bones_adjacency_matrix, 
            config.window)
        # (1, 1, V, V)
        global_attn = torch.from_numpy(global_attn).unsqueeze(0).unsqueeze(0)
        self.global_attn = nn.Parameter(global_attn)
        # (1, num_heads, 1, 1)
        self.alphas = nn.Parameter(torch.ones(1, config.num_heads, 1, 1))
        
        # Layers
        self.q_proj = nn.Conv2d(config.channels, config.channels, kernel_size=1)
        self.k_proj = nn.Conv2d(config.channels, config.channels, kernel_size=1)
        self.proj = nn.Conv2d(config.channels * config.num_heads, config.channels, kernel_size=1)
    
    def _build_spatio_temporal_graph(self, adj: np.ndarray, window: int) -> np.ndarray:
        window_adj = np.tile(adj, (window, window)).copy()
        window_adj = normalize_adjacency_matrix(window_adj)
        return window_adj
        
    @property
    def structure_dropout(self) -> float:
        return self._structure_dropout
    
    @structure_dropout.setter
    def structure_dropout(self, drop: float):
        self._structure_dropout = drop
        
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x_q.shape    
        # (N, C, T, V)
        q = self.q_proj(x_q)
        # (N, num_heads, C_head, T, V)
        q = q.view(N, self._num_heads, self._head_channels, T, V)
        q = self.feature_dropout(q)
        
        # (N, C, T, V)
        k = self.k_proj(x_kv)
        # (N, num_heads, C_head, T, V)
        k = k.view(N, self._num_heads, self._head_channels, T, V)
        k = self.feature_dropout(k)
        
        attn = torch.tanh(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self._head_channels * T)) * self.alphas
        attn += self.global_attn
        attn = self.feature_dropout(attn)
        # (N, num_heads, C, T, V)
        values = torch.einsum('nctu,nsuv->nsctv', [x_kv, attn]).contiguous()
        
        if self.training:
            # (N, num_heads)
            prob = torch.tensor([[1 - self._structure_dropout]], device=values.device).expand(values.shape[:2])
            # (N, num_heads)
            epsilon = torch.bernoulli(prob)
            # (N, num_heads, 1, 1, 1)
            binary_mask = epsilon.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # (N, num_heads, C_head, T, V)
            values = values * binary_mask
        
        # (N, C, T, V)
        values = values.view(N, -1, T, V)
        values = self.proj(values)
        
        if self.training:
            # (N)
            factor = torch.sum(epsilon, dim=-1)
            # (N, 1, 1. 1)
            factor = factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            factor = factor / self._num_heads
            factor[factor == 0] = 1
            # # (N, C, T, V)
            values = values / factor
        
        return values

'''
class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, 
                 channels: int, 
                 num_heads: int, 
                 feature_dropout: float, 
                 structure_dropout: float) -> None:
        super().__init__()

        self._num_heads = num_heads
        self._out_channels = channels
        self._head_channels = channels // num_heads
        self._structure_dropout = structure_dropout
        self.feature_dropout = nn.Dropout(feature_dropout)
        
        # Layers
        self.query = nn.Linear(channels, channels)
        self.query_att = nn.Linear(self._out_channels, self._num_heads)
        
        self.key = nn.Linear(channels, channels)
        self.key_att = nn.Linear(self._out_channels, self._num_heads)
        
        self.value = nn.Linear(channels, channels)
        self.transform = nn.Linear(self._out_channels, self._out_channels)
        
        self.proj = nn.Linear(channels, channels)
    
    @property
    def structure_dropout(self) -> float:
        return self._structure_dropout
    
    @structure_dropout.setter
    def structure_dropout(self, drop: float):
        self._structure_dropout = drop
    
    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self._num_heads, self._head_channels)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        return x
    
    def _group_heads(self, x: torch.Tensor) -> torch.Tensor:
        N, _, L, _ = x.shape
        # (N, L, num_heads, C_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (N, L, C_out)
        x = x.view(N, L, -1)
        
        return x
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        
        # (N, L, C_out)
        queries: torch.Tensor = self.query(x_q)
        queries = self.feature_dropout(queries)
        # (N, L, num_heads)
        query_score = self.query_att(queries) / (math.sqrt(self._head_channels))
        # (N, num_heads, L)
        query_score = query_score.transpose(-2, -1)
        # (N, num_heads, 1, L)
        query_weight = F.softmax(query_score, dim=-1).unsqueeze(2)
        # (N, num_heads, L, C_head)
        queries = self._separate_heads(queries)
        # (N, num_heads, 1, C_head)
        pooled_query = torch.matmul(query_weight, queries)
        # (N, 1, C_out)
        pooled_query = self._group_heads(pooled_query)
        
        # (N, L, C_out)
        keys = self.key(x_kv)
        keys = self.feature_dropout(keys)
        # (N, L, C_out)
        keys_queries = keys * pooled_query
        # (N, L, num_heads)
        keys_score = self.key_att(keys_queries) / math.sqrt(self._head_channels)
        # (N, num_heads, L)
        keys_score = keys_score.transpose(-2, -1)
        # (N, num_head, 1, L)
        keys_weight = F.softmax(keys_score, dim=-1).unsqueeze(2)
        # (N, num_heads, L, C_head)
        keys = self._separate_heads(keys)  
        # (N, num_head, 1, C_head)
        pooled_key = torch.matmul(keys_weight, keys)
        
        # (N, L, C_out)
        values = self.value(x_kv)
        values = self.feature_dropout(values)
        # (N, num_heads, L, C_head)
        values = self._separate_heads(values)
        # (N, num_heads, L, C_head)
        keys_values = values * pooled_key
        # (N, L, C_out)
        keys_values = self._group_heads(keys_values)
        # (N, L, C_out)
        keys_values = self.transform(keys_values)
        # (N, num_heads, L, C_head)
        keys_values = self._separate_heads(keys_values)
        # (N, num_heads, L, C_head)
        output = keys_values + queries
        
        if self.training:
            # (N, num_heads)
            prob = torch.tensor([[1 - self._structure_dropout]], device=output.device).expand(output.shape[:2])
            # (N, num_heads)
            epsilon = torch.bernoulli(prob)
            # (N, num_heads, 1, 1)
            binary_mask = epsilon.unsqueeze(-1).unsqueeze(-1)
            # (N, num_heads, L, C_heads)
            mask = binary_mask.expand(output.shape)
            # (N, num_heads, L, C_heads)
            output = output * mask
        
        # (N, L, C_out)
        output = self._group_heads(output)
        # (N, L, C_out)
        output = self.proj(output)
        
        if self.training:
            # (N)
            factor = torch.sum(epsilon, dim=-1)
            # (N, 1, 1)
            factor = factor.unsqueeze(-1).unsqueeze(-1)
            factor = factor / self._num_heads
            factor[factor == 0] = 1
            # (N, L, C_out)
            output = output / factor
        
        return output
'''