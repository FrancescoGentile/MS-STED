##
##
##

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import LayerNorm2d
from .config import BlockConfig, CrossViewType, LayerConfig, BranchConfig, SampleType, TemporalConfig


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
    
    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for layer_cfg in config.layers:
            self.layers.append(Layer(layer_cfg))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            out = layer(x)
            x = out 
        
        return out

class Layer(nn.Module):
    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        
        self._cross_view = config.cross_view

        self.branches = nn.ModuleList()
        for branch_cfg in config.branches:
            self.branches.append(Branch(branch_cfg))
            
        self.temporal = TemporalConvolution(config.temporal)
        
        #if config.in_channels == config.out_channels:
        #    self.residual = nn.Identity()
        #else:
        #    self.residual = nn.Conv2d(config.in_channels, config.out_channels, kernel_size=1)
    
    def _larger(self, x: torch.Tensor) -> torch.Tensor:
        output = 0
        cross_x = None 
        for branch in self.branches:
            out = branch(x, cross_x)
            output += out
            
            if self._cross_view == CrossViewType.LARGER:
                cross_x = out
        
        output /= len(self.branches)
        return output
    
    def _smaller(self, x: torch.Tensor) -> torch.Tensor:
        output = 0
        cross_x = None 
        for branch in reversed(self.branches):
            out = branch(x, cross_x)
            output += out
            
            if self._cross_view == CrossViewType.SMALLER:
                cross_x = out
        
        output /= len(self.branches)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #res = self.residual(x)
        
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
    def __init__(self, config: BranchConfig) -> None:
        super().__init__()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        # Attention
        self.first_norm = nn.LayerNorm(config.channels)
        self.attention = SpatioTemporalAttention(
            config.channels, 
            config.num_heads, 
            config.feature_dropout, 
            config.structure_dropout)
        
        if config.cross_view:
            self.second_norm = nn.LayerNorm(config.channels)
            self.cross_norm = nn.LayerNorm(config.channels)
            self.cross_attention = SpatioTemporalAttention(
                config.channels, 
                config.num_heads, 
                config.feature_dropout, 
                config.structure_dropout)
            
        self.dropout = nn.Dropout(config.sublayer_dropout)
        
        # FFN
        self.ffn_norm = nn.LayerNorm(config.channels)
        self.ffn = nn.Sequential(
            nn.Linear(config.channels, config.channels),
            nn.GELU(), 
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.channels, config.channels)
        )
        
    def _group_window_joints(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def _average_window_joints(self, x: torch.Tensor) -> torch.Tensor:
        N, _, C = x.shape
        # (N * T, window, V, C)
        x = x.view(N, self.window, -1, C)
        # (N * T, V, C)
        x = torch.mean(x, dim=1)
        
        return x
    
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        # (N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # (N * T, V, C)
        x = x.view(N * T, V, C)
        
        return x
    
    def _unflatten(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        _, V, C = x.shape
        # (N, T, V, C)
        x = x.view(batch, -1, V, C)
        # (N, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, _, _, _ = x.shape
        
        # Attention
        window_x = self._group_window_joints(x)
        norm_x = self.first_norm(window_x)
        att_out = self.attention(norm_x, norm_x)
        att_out = self._average_window_joints(att_out)
        att_out = self.dropout(att_out)
        att_out += self._flatten(x)
        
        if cross_x is not None:
            x_q = self.second_norm(self._group_window_joints(self._unflatten(att_out, N)))
            x_kv = self.cross_norm(self._group_window_joints(cross_x))
            cross_att_out = self.cross_attention(x_q, x_kv)
            cross_att_out = self._average_window_joints(cross_att_out)
            cross_att_out = self.dropout(cross_att_out)
            cross_att_out += att_out
            att_out = cross_att_out
        
        # FFN
        ffn_out = self.ffn_norm(att_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out += att_out
        
        out = self._unflatten(ffn_out, batch=N)
        
        return out

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


class TemporalConvolution(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        
        if cfg.sample == SampleType.UP:
            raise NotImplemented
        
        stride = 1 if cfg.sample == SampleType.NONE else 2
        
        self.branches = nn.ModuleList()
        branch_channels = cfg.out_channels // (len(cfg.branches) + 2)
        for (window, dilation) in cfg.branches:
            pad = (window + (window - 1) * (dilation - 1) - 1) // 2
            self.branches.append(nn.Sequential(
                LayerNorm2d(cfg.in_channels),
                nn.Conv2d(cfg.in_channels, branch_channels, kernel_size=1),
                #LayerNorm2d(branch_channels),
                nn.GELU(),
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=(window, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                    dilation=(dilation, 1)
                ),
                #LayerNorm2d(branch_channels),
            ))
        
        # Add MaxPool
        self.branches.append(nn.Sequential(
            LayerNorm2d(cfg.in_channels),
            nn.Conv2d(cfg.in_channels, branch_channels, kernel_size=1),
            #LayerNorm2d(branch_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            #LayerNorm2d(branch_channels),
        ))
        
        # Add conv1x1
        self.branches.append(nn.Sequential(
            LayerNorm2d(cfg.in_channels),
            nn.Conv2d(cfg.in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
            #LayerNorm2d(branch_channels),
        ))
        
        if cfg.residual:
            if cfg.in_channels == cfg.out_channels:
                if stride == 1:
                    self.residual = nn.Identity()
                else:
                    self.residual = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=(2, 1), stride=(stride, 1))
            else:
                if stride == 1:
                    self.residual = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=1, stride=1)
                else:
                    self.residual = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=(2, 1), stride=(stride, 1))
        else:
            self.residual = lambda _: 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        outputs = []
        for branch in self.branches:
            out = branch(x)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out += res
        
        return out