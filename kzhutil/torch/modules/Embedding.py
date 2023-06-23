import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.proj(emb)
        return emb


class CompressedEmb(nn.Module):
    def __init__(self, num_embeddings, embed_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.output_proj(self.embedding(x))
