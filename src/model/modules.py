##
##
##

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
        total_channels = 0
        for branch_cfg in config.branches:
            self.branches.append(Branch(branch_cfg))
            total_channels += branch_cfg.out_channels
            
        self.proj = nn.Conv2d(
            in_channels=total_channels, 
            out_channels=config.out_channels, 
            kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        outputs = []
        cross_x = None 
        for branch in self.branches[::-1]:
            out = branch(x, cross_x)
            outputs.append(out)
            
            if self._cross_view:
                cross_x = out

        output = torch.cat(outputs, dim=1)
        output = self.proj(output)
        
        return output

class Branch(nn.Module):
    def __init__(self, config: BranchConfig) -> None:
        super().__init__()
        
        self.window = config.window
        padding = (config.window + (config.window - 1) * (config.dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(config.window, 1),
            dilation=(config.dilation, 1),
            padding=(padding, 0))
        
        self.q_norm = nn.LayerNorm(config.in_channels)
        self.kv_norm = nn.LayerNorm(config.in_channels)
        self.attention = SpatioTemporalAttention(config)
        self.dropout = nn.Dropout(p=0.1)
        
        self.ffn_norm = nn.LayerNorm(config.out_channels)
        self.fnn = nn.Sequential(
            nn.Linear(config.out_channels, config.out_channels),
            nn.ReLU(inplace=True), 
            nn.Linear(config.out_channels, config.out_channels),
            nn.Dropout(p=0.1)
        )
        
    def _group_window(self, x: torch.Tensor) -> torch.Tensor:
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

    def _ungroup_window(self, x: torch.Tensor) -> torch.Tensor:
        N, V, C = x.shape
        # (N, window, V, C)
        x = x.view(N, self.window, -1, C)
        # (N, V, C)
        x = x.mean(dim=1)
        
        return x
    
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        #(N, T, V, C)
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
        batch, _, _, _ = x.shape
        
        x_q = self._flatten(self._group_window(x))
        x_kv = self._flatten(self._group_window(cross_x)) if cross_x is not None else x_q
        xf = self._flatten(x)
        
        # Attention
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        tmp: torch.Tensor = self.attention(x_q, x_kv)
        tmp = self._ungroup_window(tmp)
        tmp = self.dropout(tmp)
        tmp += xf
        
        # Position-wise FFN
        out = self.ffn_norm(tmp)
        out: torch.Tensor = self.fnn(out)
        out += tmp
        out = self._unflatten(out, batch)
        
        return out

'''
class SpatioTemporalAttention(nn.Module):
    def __init__(self, config: BranchConfig):
        super().__init__()
        *
        self.config = config
        self.attention_head_size = config.out_channels // config.num_heads
        self.num_attention_heads = config.num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.in_channels
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.value = nn.Linear(self.input_dim, self.all_head_size)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
                
    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x: torch.Tensor, cross_x: Optional[torch.Tensor] = None):
        
        print('-------------------------------')
        print(f'(before) mean: {x.mean().detach().item():.5e} - std: {x.std().detach().item():.5e}')
        
        x_q = x
        x_kv = cross_x if cross_x is not None else x
        
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        _, seq_len, _ = x.shape
        mixed_query_layer = self.query(x_q)
        mixed_key_layer = self.key(x_kv)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        #query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        #query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        #mixed_value_layer = self.value(x_kv)
        #value_layer = self.transpose_for_scores(mixed_value_layer)   
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
        
        print(f'(after) mean: {weighted_value.mean().detach().item():.5e} - std: {weighted_value.std().detach().item():.5e}')
        print('-------------------------------')
      
        return weighted_value

''' 
  
class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, config: BranchConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._out_channels = config.out_channels
        self._head_channels = config.out_channels // config.num_heads
        
        # Layers
        self.query = nn.Linear(config.in_channels, config.out_channels)
        self.query_att = nn.Linear(config.out_channels, config.num_heads)
        
        self.key = nn.Linear(config.in_channels, config.out_channels)
        self.key_att = nn.Linear(config.out_channels, config.num_heads)
        
        self.value = nn.Linear(config.in_channels, config.out_channels)
        self.transform = nn.Linear(config.out_channels, config.out_channels)
        
        self.linear = nn.Linear(config.out_channels, config.out_channels)
        
        self.softmax = nn.Softmax(-1)
    
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self._num_heads, self._head_channels)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        return x
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        
        #print('-------------------------------')
        #print(f'(before) mean: {x.mean().detach().item():.5e} - std: {x.std().detach().item():.5e}')
        
        # (N, L, C_out)
        queries: torch.Tensor = self.query(x_q)
        # (N, num_heads, L)
        query_score = self.query_att(queries).transpose(1, 2) / (math.sqrt(self._head_channels))
        # (N, num_heads, 1, L)
        query_weight = self.softmax(query_score).unsqueeze(2)
        # (N, num_head, L, C_head)
        query_transposed = self._transpose_for_scores(queries)
        # N, num_head, 1, C_head
        pooled_query = torch.matmul(query_weight, query_transposed)
        # (N, 1, num_head, C_head)
        pooled_query = pooled_query.transpose(1, 2)
        # (N, 1, C_out)
        pooled_query = pooled_query.view(-1, 1, self._out_channels)
        
        # (N, L, C_out)
        keys = self.key(x_kv)        
        # (N, L, C_out)
        keys_queries = keys * pooled_query
        # (N, num_head, L)
        keys_score = (self.key_att(keys_queries) / math.sqrt(self._head_channels)).transpose(1, 2)
        # (N, num_head, 1, L)
        keys_weight = self.softmax(keys_score).unsqueeze(2)
        # (N, num_head, L, C_head)
        keys_transposed = self._transpose_for_scores(keys_queries)
        # (N, num_head, 1, C_head)
        pooled_key = torch.matmul(keys_weight, keys_transposed)
        # (N, 1, num_head, C_head)
        pooled_key = pooled_key.transpose(1, 2)
        # (N, 1, C_out)
        pooled_key = pooled_key.view(-1, 1, self._out_channels)
        
        # (N, L, C_out)
        values = self.value(x_kv)
        # (N, L, C_out)
        keys_values = values * pooled_key
        # (N, L, C_out)
        keys_values = self.transform(keys_values)
        # (N, L, C_out)
        output = keys_values + queries
        # (N, L, C_out)
        output = self.linear(output)
        
        return output
#''' 