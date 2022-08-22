##
##
##

from __future__ import annotations
from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import LayerNorm2d
from .config import BlockConfig, CrossViewType, LayerConfig, BranchConfig, SampleType, TemporalConfig
from ..dataset.skeleton import SkeletonGraph, symmetric_normalized_adjacency

class Block(nn.Module):
    
    def __init__(self, config: BlockConfig,skeleton: SkeletonGraph) -> None:
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
            self.branches.append(Branch(branch_cfg, skeleton))
            
        self.temporal = TemporalConvolution(config.temporal)
    
    def _larger(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        cross_x = None 
        for branch in self.branches:
            out = branch(x, cross_x)
            output.append(out)
            
            if self._cross_view == CrossViewType.LARGER:
                cross_x = out
        
        output = torch.cat(output, dim=1)
        return output
    
    def _smaller(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        cross_x = None 
        for branch in reversed(self.branches):
            out = branch(x, cross_x)
            output.append(out)
            
            if self._cross_view == CrossViewType.SMALLER:
                cross_x = out
        
        output = torch.cat(output, dim=1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        match self._cross_view:
            case CrossViewType.NONE:
                out = self._larger(x)
            case CrossViewType.LARGER:
                out = self._larger(x)
            case CrossViewType.SMALLER:
                out = self._smaller(x)
            case _:
                raise ValueError('Unknown cross-view type')
            
        out = self.temporal(out)
        
        return out

class Branch(nn.Module):
    def __init__(self, config: BranchConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        #if config.in_channels != config.channels:
        #    self.reduce = nn.Conv2d(config.in_channels, config.channels, kernel_size=1)
        #else:
        #    self.reduce = nn.Identity()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        # Attention
        self.first_norm = nn.LayerNorm(config.channels)
        self.attention = SpatioTemporalAttention(config, skeleton)
        self.aggregate = nn.Conv2d(config.window * config.channels, config.channels, kernel_size=1)
        
        if config.cross_view:
            self.second_norm = nn.LayerNorm(config.channels)
            self.cross_norm = nn.LayerNorm(config.channels)
            self.cross_attention = SpatioTemporalAttention(config, skeleton)
            self.cross_aggregate = nn.Conv2d(config.window * config.channels, config.channels, kernel_size=1)
        
        # FFN
        self.ffn_norm = LayerNorm2d(config.channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, kernel_size=1),
            nn.GELU(), 
            nn.Dropout(config.dropout.feature),
            nn.Conv2d(config.channels, config.channels, kernel_size=1)
        )
        
        self.sublayer_dropout = nn.Dropout(config.dropout.sublayer)
    
    def _group_window(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        #(N, C * window, T, V)
        x = self.unfold(x)
        # (N, C, window, T, V)
        x = x.view(N, C, self.window, T, V)
        # (N, T, window, V, C)
        x = x.permute(0, 3, 2, 4, 1).contiguous()
        # (N * T, window * V, C)
        x = x.view(N * T, self.window * V, C)
        
        return x
    
    def _aggregate_window(self, x: torch.Tensor, cross: bool, batch: int):
        M, V, C = x.shape
        T = M // batch
        V = V // self.window
        # (N, T, window, V, C)
        x = x.view(batch, T, self.window, V, C)
        # (N, C, window, T, V)
        x = x.permute(0, 4, 2, 1, 3).contiguous()
        # (N, C * window, T, V)
        x = x.view(batch, C * self.window, T, V)
        
        # (N, C, T, V)
        if cross:
            x = self.cross_aggregate(x)
        else:
            x = self.aggregate(x)
            
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        #x = self.reduce(x)
        N, _, _, _ = x.shape
              
        # Attention
        window_x = self._group_window(x)
        norm_x = self.first_norm(window_x)
        att_out = self.attention(norm_x, norm_x)
        att_out = self._aggregate_window(att_out, False, N)
        att_out = self.sublayer_dropout(att_out)
        att_out += x
        
        # Cross Attention
        if cross_x is not None:
            x_q = self.second_norm(self._group_window(att_out))
            x_kv = self.cross_norm(self._group_window(cross_x))
            cross_att_out = self.cross_attention(x_q, x_kv)
            cross_att_out = self._aggregate_window(cross_att_out, True, N)
            cross_att_out = self.sublayer_dropout(cross_att_out)
            cross_att_out += att_out
        else:
            cross_att_out = att_out
        
        # FFN
        ffn_out = self.ffn_norm(cross_att_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.sublayer_dropout(ffn_out)
        ffn_out += cross_att_out

        return ffn_out
    
class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, config: BranchConfig, _: SkeletonGraph) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._channels = config.channels
        self._head_channels = config.channels // config.num_heads
        self._drop_head = config.dropout.head.start
        self.feature_dropout = nn.Dropout(config.dropout.feature)
        
        # Layers
        self.q_proj = nn.Linear(config.channels, config.channels)
        self.query_att = nn.Linear(self._channels, self._num_heads)
        
        self.k_proj = nn.Linear(config.channels, config.channels)
        self.key_att = nn.Linear(self._channels, self._num_heads)
        
        self.v_proj = nn.Linear(config.channels, config.channels)
        self.transform = nn.Linear(self._channels, self._channels)
        
        self.o_proj = nn.Linear(config.channels, config.channels)
    
    @property
    def drop_head(self) -> float:
        return self._drop_head
    
    @drop_head.setter
    def drop_head(self, drop: float):
        self._drop_head = drop
    
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
        queries: torch.Tensor = self.q_proj(x_q)
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
        keys = self.k_proj(x_kv)
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
        values = self.v_proj(x_kv)
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
            prob = torch.tensor([[1 - self._drop_head]], device=output.device).expand(output.shape[:2])
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
        output = self.o_proj(output)
        
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

class TemporalConvolution(nn.Module):
    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        
        self._layer_drop = config.dropout.layer
        
        if config.sample == SampleType.UP:
            raise NotImplemented
        
        stride = 1 if config.sample == SampleType.NONE else 2
        
        self.branches = nn.ModuleList()
        branch_channels = config.out_channels // (len(config.branches) + 2)
        for (window, dilation) in config.branches:
            pad = (window + (window - 1) * (dilation - 1) - 1) // 2
            self.branches.append(nn.Sequential(
                LayerNorm2d(config.in_channels),
                nn.Conv2d(config.in_channels, branch_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout(config.dropout.feature),
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=(window, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                    dilation=(dilation, 1)
                ),
            ))
        
        # Add MaxPool
        self.branches.append(nn.Sequential(
            LayerNorm2d(config.in_channels),
            nn.Conv2d(config.in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(config.dropout.feature),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
        ))
        
        # Add conv1x1
        self.branches.append(nn.Sequential(
            LayerNorm2d(config.in_channels),
            nn.Conv2d(config.in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
        ))
        
        if not config.residual:
            self.residual = lambda _: 0
        elif (config.in_channels == config.out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                config.in_channels, 
                config.out_channels, 
                kernel_size=1, 
                stride=(stride, 1))
            
        self.sublayer_dropout = nn.Dropout(config.dropout.sublayer)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        outputs = []
        for branch in self.branches:
            out = branch(x)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.sublayer_dropout(out)
        
        out += res
        
        return out