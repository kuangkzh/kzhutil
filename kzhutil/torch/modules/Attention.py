import math
import torch
from torch import nn
import torch.nn.functional as F
from .Embedding import SinusoidalPosEmb


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, q_dim=None, k_dim=None, v_dim=None, out_dim=None, dropout=0., bias=True, kv_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_dim = embed_dim if q_dim is None else q_dim
        self.k_dim = embed_dim if k_dim is None else k_dim
        self.v_dim = embed_dim if v_dim is None else v_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.proj_q = nn.Linear(self.q_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(self.k_dim, embed_dim, bias=kv_bias)
        self.proj_v = nn.Linear(self.v_dim, embed_dim, bias=kv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, self.out_dim, bias=bias)

    def forward(self, q, k, v, mask=None, output_attn_score=False):
        """
        :param q: query of [Batch, Length, Embedding]
        :param k: key of [Batch, S, Embedding]
        :param v: key of [Batch, S, Embedding]
        :param mask: [Length, S], 0 means masked
        :param output_attn_score:
        :return: [Batch, Length, Embedding]
        """
        b, l, _ = q.shape
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # [Batch, h, L, dim//h]
        q, k, v = [x.view(b, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) for x in (q, k, v)]
        w = q@k.transpose(-1, -2) / math.sqrt(self.embed_dim // self.num_heads)
        w = w.masked_fill(mask == 0, -1e9) if mask is not None else w
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        out = (w@v).transpose(1, 2).contiguous().view(b, l, self.embed_dim)
        out = self.out_proj(out)

        if output_attn_score:
            return out, w

        return out


class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads, cross_attn=False, add_pos_embedding=True):
        super(AttnBlock, self).__init__()
        self.ln_input = nn.LayerNorm(dim)
        self.positional_embedding = SinusoidalPosEmb(dim) if add_pos_embedding else None
        self.self_attn = nn.MultiheadAttention(dim, num_heads)
        self.ln_self = nn.LayerNorm(dim)
        self.cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(dim, num_heads)
            self.ln_cross = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
        self.ln_feed = nn.LayerNorm(dim)

    def forward(self, q, k=None, v=None):
        """
        :param q: [B, T, E]
        :param k: [B, L, E]
        :param v: [B, L, E]
        :return:
        """
        x = q = self.ln_input(q)
        if self.positional_embedding is not None:
            x = x + self.positional_embedding(torch.arange(x.shape[0]).to(x.device)).unsqueeze(1)
        q = q + self.ln_self(self.self_attn(x, x, x)[0])
        if self.cross_attn:
            q = q + self.ln_cross(self.cross_attn(q, k, v)[0])
        out = (self.ln_feed(self.feed_forward(q)) + q)
        return out


class FactorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor_num, dropout=0., query_factors=True):
        """
        Suppose there are factors in attentions that can be decomposed.
        If query_factors is True: softmax(Q @ E^T) @ softmax(E @ K^T) @ V
        else: EKV := softmax(E @ K^T) @ V
              softmax(E @ EKV^T) @ EKV
        """
        super(FactorAttention, self).__init__()
        self.factor_num = factor_num
        self.factor_embeds = nn.Embedding(factor_num, embed_dim)
        self.factor_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.output_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.query_factors = query_factors

    def forward(self, q, k, v, output_attn_score=False):
        e = self.factor_embeds(torch.arange(self.factor_num).repeat(q.shape[0], 1))
        if output_attn_score:
            ekv, w1 = self.factor_attn(e, k, v, output_attn_score=True)
            out, w2 = self.output_attn(q, e if self.query_factors else ekv, ekv, output_attn_score=True)
            return out, w2@w1
        else:
            ekv = self.factor_attn(e, k, v)
            out = self.output_attn(q, e if self.query_factors else ekv, ekv)
            return out


class ConvFactorSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, factor_num, dropout=0., query_factors=True):
        """
        Suppose there are factors in attentions that can be decomposed.
        If query_factors is True: softmax(Q @ E^T) @ softmax(E @ K^T) @ V
        else: EKV := softmax(E @ K^T) @ V
              softmax(E @ EKV^T) @ EKV
        """
        super(ConvFactorSelfAttention, self).__init__()
        d_k = embed_dim // num_heads
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (1, kernel_size)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "kernel_size must be odd numbers"
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.num_heads = num_heads

        self.conv = nn.Conv2d(d_k, d_k, kernel_size=kernel_size, padding=padding)
        self.attn = FactorAttention(embed_dim, num_heads, factor_num, dropout, query_factors)

    def forward(self, x):
        """
        Args:
            x: [Batch, Length, Embedding]
        """
        b, l, d = x.shape
        x = x.view(b, l, self.num_heads, d // self.num_heads).transpose(1, 3)
        x = self.conv(x).transpose(1, 3).reshape(b, l, d)
        x = self.attn(x, x, x)
        return x


class SlidingSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, dropout=0.):
        super(SlidingSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        b, l, d = x.shape
        h, w = self.num_heads, self.window_size

        padding = torch.zeros_like(x[:, :1]).repeat(1, w//2, 1)
        pl = l + w // 2 * 2

        # [Batch, length, h, 1, d_h]
        q = self.proj_q(x).view(b, l, h, 1, d//h)
        # [Batch, length, h, window_size, d_h]
        k = self.proj_k(torch.cat((padding, x, padding), dim=1)).as_strided((b, l, h, w, d//h), (pl*d, d, d//h, d, 1))
        v = self.proj_v(torch.cat((padding, x, padding), dim=1)).as_strided((b, l, h, w, d//h), (pl*d, d, d//h, d, 1))

        # [Batch, length, h, 1, window_size]
        w = q@k.transpose(-1, -2) / math.sqrt(d//h)
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        out = (w@v).contiguous().view(b, l, d)
        out = self.out_proj(out)

        return out
