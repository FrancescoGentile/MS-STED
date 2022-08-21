##
##
##

from __future__ import annotations
from typing import Optional
import numpy as np

import torch
import torch.nn as nn

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
        self._layer_drop = config.dropout.layer

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
        prob = torch.tensor(1 - self._layer_drop)
        keep = torch.bernoulli(prob) == 1.0
        
        if keep:
            match self._cross_view:
                case CrossViewType.NONE:
                    out = self._larger(x)
                case CrossViewType.LARGER:
                    out = self._larger(x)
                case CrossViewType.SMALLER:
                    out = self._smaller(x)
                case _:
                    raise ValueError('Unknown cross-view type')
        
            out += x
        else:
            out = x
            
        out = self.temporal(out)
        
        return out

class Branch(nn.Module):
    def __init__(self, config: BranchConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        if config.in_channels != config.channels:
            self.reduce = nn.Conv2d(config.in_channels, config.channels, kernel_size=1)
        else:
            self.reduce = nn.Identity()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        # Attention
        self.first_norm = LayerNorm2d(config.channels)
        self.attention = SpatioTemporalAttention(config, skeleton)
        self.aggregate = nn.Conv3d(
            config.channels, 
            config.channels, 
            kernel_size=(1, config.window, 1))
        
        if config.cross_view:
            self.second_norm = LayerNorm2d(config.channels)
            self.cross_norm = LayerNorm2d(config.channels)
            self.cross_attention = SpatioTemporalAttention(config, skeleton)
            self.cross_aggregate = nn.Conv3d(
                config.channels, 
                config.channels, 
                kernel_size=(1, config.window, 1))
        
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
        # (N, C, T, window, V)
        x = x.transpose(2, 3).contiguous()
        # (N, C, T, window * V)
        x = x.view(N, C, T, self.window * V)
        
        return x
    
    def _aggregate_window(self, x: torch.Tensor, cross: bool) -> torch.Tensor:
        N, C, T, _ = x.shape
        # (N, C, T, window, V)
        x = x.view(N, C, T, self.window, -1)
        # (N, C, T, V)
        #x = torch.mean(x, dim=3)
        if not cross:
            x = self.aggregate(x)
        else:
            x = self.cross_aggregate(x)
        
        x = x.squeeze(3)
        
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.reduce(x)
              
        # Attention
        window_x = self._group_window(x)
        norm_x = self.first_norm(window_x)
        att_out = self.attention(norm_x, norm_x)
        att_out = self._aggregate_window(att_out, False)
        att_out = self.sublayer_dropout(att_out)
        att_out += x
        
        # Cross Attention
        if cross_x is not None:
            x_q = self.second_norm(self._group_window(att_out))
            x_kv = self.cross_norm(self._group_window(cross_x))
            cross_att_out = self.cross_attention(x_q, x_kv)
            cross_att_out = self._aggregate_window(cross_att_out, True)
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
    def __init__(self, config: BranchConfig, skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._channels = config.channels
        self._head_channels = config.channels // config.num_heads
        self._num_heads = config.num_heads
        
        # Dropout
        self.feature_dropout = nn.Dropout(p=config.dropout.feature)
        self._drop_head = config.dropout.head.start
        
        # Learnable attention map
        global_attn = self._build_spatio_temporal_graph(
            skeleton.joints_bones_adjacency(True),
            config.window)
        global_attn = torch.from_numpy(global_attn).unsqueeze(0).unsqueeze(0)
        self.global_attn = nn.Parameter(global_attn)
        
        self.alphas = nn.Parameter(torch.ones(1, config.num_heads, 1, 1))
        
        # Layers
        self.q_proj = nn.Conv2d(config.channels, config.channels, kernel_size=1)
        self.k_proj = nn.Conv2d(config.channels, config.channels, kernel_size=1)
        self.v_proj = nn.Conv2d(config.channels, config.channels, kernel_size=1)
        self.o_proj = nn.Conv2d(config.channels * config.num_heads, config.channels, kernel_size=1)
    
    def _build_spatio_temporal_graph(self, adj: np.ndarray, window: int) -> np.ndarray:
        window_adj = np.tile(adj, (window, window)).copy()
        window_adj = symmetric_normalized_adjacency(window_adj)
        return window_adj
    
    @property
    def drop_head(self) -> float:
        return self._drop_head
    
    @drop_head.setter
    def drop_head(self, drop: float):
        self._drop_head = drop
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        N, _, T, V = x_q.shape
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
        
        # (N, C, T, V)
        v = self.v_proj(x_kv)
        v = self.feature_dropout(v)
        
        attn = torch.tanh(
            torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self._head_channels * T)) * self.alphas
        attn += self.global_attn
        attn = self.feature_dropout(attn)
        
        # (N, num_heads, C, T, V)
        values = torch.einsum('nctu,nsuv->nsctv', [v, attn]).contiguous()
        
        if self.training:
            # (N, num_heads)
            prob = torch.tensor([[1 - self._drop_head]], device=values.device).expand(values.shape[:2])
            # (N, num_heads)
            epsilon = torch.bernoulli(prob)
            # (N, num_heads, 1, 1, 1)
            binary_mask = epsilon.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # (N, num_heads, C_head, T, V)
            values = values * binary_mask
        
        # (N, C * num_head, T, V)
        values = values.view(N, -1, T, V)
        out = self.o_proj(values)
        
        if self.training:
            # (N)
            factor = torch.sum(epsilon, dim=-1)
            # (N, 1, 1. 1)
            factor = factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            factor = factor / self._num_heads
            factor[factor == 0] = 1
            # (N, C, T, V)
            out = out / factor
        
        return out

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
        prob = torch.tensor(1 - self._layer_drop)
        keep = torch.bernoulli(prob) == 1.0
        
        res = self.residual(x)
        if keep:
            outputs = []
            for branch in self.branches:
                out = branch(x)
                outputs.append(out)

            out = torch.cat(outputs, dim=1)
            out = self.sublayer_dropout(out)
        
            out += res
        else:
            out = res
        
        return out