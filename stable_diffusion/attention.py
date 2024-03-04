import torch 
from torch import nn
from torch.nn import functional as F
import math 


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True) -> None:
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, heads, dim/heads) -> (batch_size, heads, seq_len, dim/heads)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)


        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu()
            weight.masked_fill_(mask, float('-inf'))

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, heads, seq_len, seq_len) @ (batch_size, heads, seq_len, dim/heads) -> (batch_size, heads, seq_len, dim/heads)
        output = weight @ v

        # (batch_size, heads, seq_len, dim/heads) -> (batch_size, seq_len, heads, dim/heads)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (batch_size, seq_len, dim)
        return output



