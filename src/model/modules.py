##
##
##

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

from .config import BlockConfig, LayerConfig, BranchConfig

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
        attention = SpatioTemporalAttention(config.branches[0])
        for branch_cfg in config.branches:
            self.branches.append(Branch(branch_cfg, attention))
            
        self.proj = nn.Conv2d(
            in_channels=config.in_channels, 
            out_channels=config.out_channels // len(self.branches), 
            kernel_size=1)
    
    def _ascending(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [None] * len(self.branches)
        cross_x = None 
        for idx, branch in enumerate(self.branches):
            out = branch(x, cross_x)
            outputs[idx] = self.proj(out)
            
            if self._cross_view:
                cross_x = out

        output = torch.cat(outputs, dim=1)
        
        return output
    
    def _descending(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [None] * len(self.branches)
        cross_x = None 
        for idx, branch in reversed(list(enumerate(self.branches))):
            out = branch(x, cross_x)
            outputs[idx] = self.proj(out)
            
            if self._cross_view:
                cross_x = out

        output = torch.cat(outputs, dim=1)
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cross_view:
            out = self._ascending(x)
        else:
            asc = self._ascending(x)
            dsc = self._descending(x)
            out = (asc + dsc) / 2
            
        return out

class Branch(nn.Module):
    def __init__(self, config: BranchConfig, attention: SpatioTemporalAttention) -> None:
        super().__init__()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        self.q_norm = nn.LayerNorm(config.channels)
        self.kv_norm = nn.LayerNorm(config.channels)
        self.attention = attention
        self.dropout = nn.Dropout(p=config.sublayer_dropout)
        
        self.ffn_norm = nn.LayerNorm(config.channels)
        dilation = (config.window - 1) + (config.dilation - 1) * config.window
        padding = ((config.window + (config.window - 1) * (dilation - 1) - 1) // 2) - (config.window // 2)
        self.fnn = nn.Sequential(
            nn.Conv2d(
                config.channels, 
                config.channels, 
                kernel_size=(config.window, 1),
                stride=(config.window, 1),
                dilation=(dilation, 1), 
                padding=(padding, 0)),
            nn.GELU(), 
            nn.Dropout(config.feature_dropout),
            nn.Conv2d(config.channels, config.channels, kernel_size=1),
            nn.Dropout(p=config.sublayer_dropout)
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
    
    def _group_window(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        N, _, C = x.shape
        # (N, window, V, C)
        x = x.view(N, self.window, -1, C)
        # (N, T * window, V, C)
        x = x.view(batch, -1, *(x.shape[2:]))
        # (N, C, T * window, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, _, _, _ = x.shape
        
        x_q = self._group_window_joints(x)
        x_kv = self._group_window_joints(cross_x) if cross_x is not None else x_q
        
        # Attention
        norm_x_q = self.q_norm(x_q)
        norm_x_kv = self.kv_norm(x_kv)
        tmp: torch.Tensor = self.attention(norm_x_q, norm_x_kv)
        tmp = self.dropout(tmp)
        tmp += x_q
        
        # Position-wise FFN
        out = self.ffn_norm(tmp)
        out = self._group_window(out, batch)
        out: torch.Tensor = self.fnn(out)
        out += x
        
        return out

class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, config: BranchConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._out_channels = config.channels
        self._head_channels = config.channels // config.num_heads
        self.fdrop = nn.Dropout(config.feature_dropout)
        self._sdrop = None
        
        # Layers
        self.query = nn.Linear(config.channels, config.channels)
        self.query_att = nn.Linear(self._head_channels, 1)
        
        self.key = nn.Linear(config.channels, config.channels)
        self.key_att = nn.Linear(self._head_channels, 1)
        
        self.value = nn.Linear(config.channels, config.channels)
        self.transform = nn.Linear(self._head_channels, self._head_channels)
        
        self.proj = nn.Linear(config.channels, config.channels)
        
        self.softmax = nn.Softmax(-1)
    
    @property
    def structure_dropout(self) -> float:
        return self._sdrop
    
    @structure_dropout.setter
    def structure_dropout(self, drop: float):
        self._sdrop = drop
    
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
        queries = self.fdrop(queries)
        # (N, num_heads, L, C_head)
        queries = self._separate_heads(queries)
        # (N, num_heads, L, 1)
        query_score = self.query_att(queries) / (math.sqrt(self._head_channels))
        # (N, num_heads, 1, L)
        query_score = query_score.transpose(-2, -1)
        # (N, num_heads, 1, L)
        query_weight = self.softmax(query_score)
        # (N, num_heads, 1, C_head)
        pooled_query = torch.matmul(query_weight, queries)
        
        # (N, L, C_out)
        keys = self.key(x_kv)
        keys = self.fdrop(keys)
        # (N, num_heads, L, C_head)
        keys = self._separate_heads(keys)  
        # (N, num_heads, L, C_head)
        keys_queries = keys * pooled_query
        # (N, num_heads, L, 1)
        keys_score = self.key_att(keys_queries) / math.sqrt(self._head_channels)
        # (N, num_heads, 1, L)
        keys_score = keys_score.transpose(-2, -1)
        # (N, num_head, 1, L)
        keys_weight = self.softmax(keys_score)
        # (N, num_head, 1, C_head)
        pooled_key = torch.matmul(keys_weight, keys)
        
        # (N, L, C_out)
        values = self.value(x_kv)
        values = self.fdrop(values)
        # (N, num_heads, L, C_head)
        values = self._separate_heads(values)
        # (N, num_heads, L, C_head)
        keys_values = values * pooled_key
        # (N, num_heads, L, C_head)
        keys_values = self.transform(keys_values)
        # (N, num_heads, L, C_head)
        output = keys_values + queries
        
        if self.training:
            # (N, num_heads)
            prob = torch.tensor([[1 - self._sdrop]], device=output.device).expand(output.shape[:2])
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
#''' 