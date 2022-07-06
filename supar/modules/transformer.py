# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        pre_norm: bool = False,
        embed_scale: Optional[int] = None,
        dropout: float = 0.1
    ) -> TransformerEncoder:
        super(TransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner
        self.embed_scale = embed_scale
        self.pre_norm = pre_norm
        self.embed_scale = embed_scale

        self.pos_embed = SinusoidPositionalEmbedding()
        self.layers = nn.ModuleList([TransformerEncoderLayer(n_heads=n_heads,
                                                             n_model=n_model,
                                                             n_inner=n_inner,
                                                             pre_norm=pre_norm,
                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.pre_norm:
            s += f", pre_norm={self.pre_norm}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.embed_scale:
            x = x * self.embed_scale
        x = self.dropout(x + self.pos_embed(x)).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, mask)
        if self.pre_norm:
            x = self.norm(x)
        return x.transpose(0, 1)


class RelativePositionTransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers: int,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        pre_norm: bool = False,
        embed_scale: Optional[int] = None,
        dropout: float = 0.1
    ) -> RelativePositionTransformerEncoder:
        super(RelativePositionTransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner
        self.pre_norm = pre_norm
        self.embed_scale = embed_scale

        self.layers = nn.ModuleList([RelativePositionTransformerEncoderLayer(n_heads=n_heads,
                                                                             n_model=n_model,
                                                                             n_inner=n_inner,
                                                                             pre_norm=pre_norm,
                                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.pre_norm:
            s += f", pre_norm={self.pre_norm}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.embed_scale:
            x = x * self.embed_scale
        x = self.dropout(x).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, mask)
        if self.pre_norm:
            x = self.norm(x)
        return x.transpose(0, 1)


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        pre_norm: bool = False,
        embed_scale: Optional[int] = None,
        dropout: float = 0.1
    ) -> TransformerDecoder:
        super(TransformerDecoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner
        self.pre_norm = pre_norm
        self.embed_scale = embed_scale

        self.pos_embed = SinusoidPositionalEmbedding()
        self.layers = nn.ModuleList([TransformerDecoderLayer(n_heads=n_heads,
                                                             n_model=n_model,
                                                             n_inner=n_inner,
                                                             pre_norm=pre_norm,
                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.pre_norm:
            s += f", pre_norm={self.pre_norm}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if self.embed_scale:
            x_tgt = x_tgt * self.embed_scale
        x_tgt = self.dropout(x_tgt + self.pos_embed(x_tgt))
        x_tgt, x_src = x_tgt.transpose(0, 1), x_src.transpose(0, 1)
        for layer in self.layers:
            x_tgt = layer(x_tgt=x_tgt,
                          x_src=x_src,
                          tgt_mask=tgt_mask,
                          src_mask=src_mask,
                          attn_mask=attn_mask)
        if self.pre_norm:
            x_tgt = self.norm(x_tgt)
        return x_tgt.transpose(0, 1)


class RelativePositionTransformerDecoder(nn.Module):

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 8,
        n_model: int = 1024,
        n_inner: int = 2048,
        pre_norm: bool = False,
        embed_scale: Optional[int] = None,
        dropout: float = 0.1
    ) -> RelativePositionTransformerDecoder:
        super(RelativePositionTransformerDecoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner
        self.pre_norm = pre_norm
        self.embed_scale = embed_scale

        self.layers = nn.ModuleList([RelativePositionTransformerDecoderLayer(n_heads=n_heads,
                                                                             n_model=n_model,
                                                                             n_inner=n_inner,
                                                                             pre_norm=pre_norm,
                                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.pre_norm:
            s += f", pre_norm={self.pre_norm}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if self.embed_scale:
            x_tgt = x_tgt * self.embed_scale
        x_tgt = self.dropout(x_tgt)
        x_tgt, x_src = x_tgt.transpose(0, 1), x_src.transpose(0, 1)
        for layer in self.layers:
            x_tgt = layer(x_tgt=x_tgt,
                          x_src=x_src,
                          tgt_mask=tgt_mask,
                          src_mask=src_mask,
                          attn_mask=attn_mask)
        if self.pre_norm:
            x_tgt = self.norm(x_tgt)
        return x_tgt.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_inner: int,
        activation: str = 'relu',
        pre_norm: bool = False,
        dropout: float = 0.1
    ) -> TransformerEncoderLayer:
        super(TransformerEncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.attn = MultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.attn_norm = nn.LayerNorm(n_model)
        self.ffn = nn.Sequential(
            nn.Linear(n_model, n_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_model)
        )
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.pre_norm:
            n = self.attn_norm(x)
            x = x + self.dropout(self.attn(n, n, n, mask))
            n = self.ffn_norm(x)
            x = x + self.dropout(self.ffn(n))
        else:
            x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask)))
            x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class RelativePositionTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_inner: int,
        activation: str = 'relu',
        pre_norm: bool = False,
        dropout: float = 0.1
    ) -> RelativePositionTransformerEncoderLayer:
        super(RelativePositionTransformerEncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.attn = RelativePositionMultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.attn_norm = nn.LayerNorm(n_model)
        self.ffn = nn.Sequential(
            nn.Linear(n_model, n_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_model)
        )
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.pre_norm:
            n = self.attn_norm(x)
            x = x + self.dropout(self.attn(n, n, n, mask))
            n = self.ffn_norm(x)
            x = x + self.dropout(self.ffn(n))
        else:
            x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask)))
            x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_inner: int,
        activation: str = 'relu',
        pre_norm: bool = False,
        dropout: float = 0.1
    ) -> TransformerDecoderLayer:
        super(TransformerDecoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.self_attn_norm = nn.LayerNorm(n_model)
        self.mha_attn = MultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.mha_attn_norm = nn.LayerNorm(n_model)
        self.ffn = nn.Sequential(
            nn.Linear(n_model, n_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_model)
        )
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            n_tgt = self.self_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.self_attn(n_tgt, n_tgt, n_tgt, tgt_mask, attn_mask))
            n_tgt = self.mha_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.mha_attn(n_tgt, x_src, x_src, src_mask))
            n_tgt = self.ffn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.ffn(x_tgt))
        else:
            x_tgt = self.self_attn_norm(x_tgt + self.dropout(self.self_attn(x_tgt, x_tgt, x_tgt, tgt_mask, attn_mask)))
            x_tgt = self.mha_attn_norm(x_tgt + self.dropout(self.mha_attn(x_tgt, x_src, x_src, src_mask)))
            x_tgt = self.ffn_norm(x_tgt + self.dropout(self.ffn(x_tgt)))
        return x_tgt


class RelativePositionTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_inner: int,
        activation: str = 'relu',
        pre_norm: bool = False,
        dropout: float = 0.1
    ) -> RelativePositionTransformerDecoderLayer:
        super(RelativePositionTransformerDecoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.self_attn = RelativePositionMultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.self_attn_norm = nn.LayerNorm(n_model)
        self.mha_attn = RelativePositionMultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.mha_attn_norm = nn.LayerNorm(n_model)
        self.ffn = nn.Sequential(
            nn.Linear(n_model, n_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_model)
        )
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if self.pre_norm:
            n_tgt = self.self_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.self_attn(n_tgt, n_tgt, n_tgt, tgt_mask, attn_mask))
            n_tgt = self.mha_attn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.mha_attn(n_tgt, x_src, x_src, src_mask))
            n_tgt = self.ffn_norm(x_tgt)
            x_tgt = x_tgt + self.dropout(self.ffn(x_tgt))
        else:
            x_tgt = self.self_attn_norm(x_tgt + self.dropout(self.self_attn(x_tgt, x_tgt, x_tgt, tgt_mask, attn_mask)))
            x_tgt = self.mha_attn_norm(x_tgt + self.dropout(self.mha_attn(x_tgt, x_src, x_src, src_mask)))
            x_tgt = self.ffn_norm(x_tgt + self.dropout(self.ffn(x_tgt)))
        return x_tgt


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_embed: int,
        dropout: float = 0.1
    ) -> MultiHeadAttention:
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed
        self.scale = n_embed**0.5

        self.wq = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wk = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wv = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wo = nn.Parameter(torch.zeros(n_heads * n_embed, n_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        batch_size, _ = mask.shape
        # [seq_len, batch_size * n_heads, n_embed]
        q = F.linear(q, self.wq).view(-1, batch_size * self.n_heads, self.n_embed)
        # [src_len, batch_size * n_heads, n_embed]
        k = F.linear(k, self.wk).view(-1, batch_size * self.n_heads, self.n_embed)
        v = F.linear(v, self.wv).view(-1, batch_size * self.n_heads, self.n_embed)

        mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
        # [batch_size * n_heads, seq_len, src_len]
        if attn_mask is not None:
            mask = mask & attn_mask
        # [batch_size * n_heads, seq_len, src_len]
        attn = torch.bmm(q.transpose(0, 1) / self.scale, k.movedim((0, 1), (2, 0)))
        attn = torch.softmax(attn + torch.where(mask, 0., float('-inf')), -1)
        # [seq_len, batch_size * n_heads, n_embed]
        x = torch.bmm(self.dropout(attn), v.transpose(0, 1)).transpose(0, 1)
        # [seq_len, batch_size, n_model]
        x = F.linear(x.reshape(-1, batch_size, self.n_heads * self.n_embed), self.wo)

        return x


class RelativePositionMultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads: int,
        n_model: int,
        n_embed: int,
        dropout: float = 0.1
    ) -> RelativePositionMultiHeadAttention:
        super(RelativePositionMultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed
        self.scale = n_embed**0.5

        self.pos_embed = RelativePositionalEmbedding(n_model=n_embed)
        self.wq = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wk = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wv = nn.Parameter(torch.zeros(n_model, n_heads * n_embed))
        self.wo = nn.Parameter(torch.zeros(n_heads * n_embed, n_model))
        self.bu = nn.Parameter(torch.zeros(n_heads, n_embed))
        self.bv = nn.Parameter(torch.zeros(n_heads, n_embed))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.BoolTensor,
        attn_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        batch_size, _ = mask.shape
        # [seq_len, batch_size, n_heads, n_embed]
        q = F.linear(q, self.wq).view(-1, batch_size, self.n_heads, self.n_embed)
        # [src_len, batch_size * n_heads, n_embed]
        k = F.linear(k, self.wk).view(-1, batch_size * self.n_heads, self.n_embed)
        v = F.linear(v, self.wv).view(-1, batch_size * self.n_heads, self.n_embed)
        # [seq_len, src_len, n_embed]
        p = self.pos_embed(q[:, 0, 0], k[:, 0])
        # [seq_len, batch_size * n_heads, n_embed]
        qu, qv = (q + self.bu).view(-1, *k.shape[1:]), (q + self.bv).view(-1, *k.shape[1:])

        mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
        if attn_mask is not None:
            mask = mask & attn_mask
        # [batch_size * n_heads, seq_len, src_len]
        attn = torch.bmm(qu.transpose(0, 1), k.movedim((0, 1), (2, 0)))
        attn = attn + torch.matmul(qv.transpose(0, 1).unsqueeze(2), p.transpose(1, 2)).squeeze(2)
        attn = torch.softmax(attn / self.scale + torch.where(mask, 0., float('-inf')), -1)
        # [seq_len, batch_size * n_heads, n_embed]
        x = torch.bmm(self.dropout(attn), v.transpose(0, 1)).transpose(0, 1)
        # [seq_len, batch_size, n_model]
        x = F.linear(x.reshape(-1, batch_size, self.n_heads * self.n_embed), self.wo)

        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, n_model: int, max_len: int = 1024) -> PositionalEmbedding:
        super().__init__()

        self.embed = nn.Embedding(max_len, n_model)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        w = self.embed.weight
        max_len, n_model = w.shape
        w = w.new_tensor(range(max_len)).unsqueeze(-1)
        w = w / 10000 ** (w.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        w[:, 0::2], w[:, 1::2] = w[:, 0::2].sin(), w[:, 1::2].cos()
        self.embed.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.new_tensor(range(x.shape[1])).long())


class RelativePositionalEmbedding(nn.Module):

    def __init__(self, n_model: int, max_len: int = 1024) -> RelativePositionalEmbedding:
        super().__init__()

        self.embed = nn.Embedding(max_len, n_model)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        w = self.embed.weight
        max_len, n_model = w.shape
        pos = torch.cat((w.new_tensor(range(-max_len//2, 0)), w.new_tensor(range(max_len//2))))
        w = pos.unsqueeze(-1) / 10000 ** (w.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        w[:, 0::2], w[:, 1::2] = w[:, 0::2].sin(), w[:, 1::2].cos()
        self.embed.weight.copy_(w)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        offset = sum(divmod(self.embed.weight.shape[0], 2))
        return self.embed((k.new_tensor(range(k.shape[0])) - q.new_tensor(range(q.shape[0])).unsqueeze(-1)).long() + offset)


class SinusoidPositionalEmbedding(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[:, 0::2], pos[:, 1::2] = pos[:, 0::2].sin(), pos[:, 1::2].cos()
        return pos


class SinusoidRelativePositionalEmbedding(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len))
        pos = (pos - pos.unsqueeze(-1)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[..., 0::2], pos[..., 1::2] = pos[..., 0::2].sin(), pos[..., 1::2].cos()
        return pos


class InverseSquareRootLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ) -> InverseSquareRootLR:
        self.warmup_steps = warmup_steps
        self.factor = warmup_steps ** 0.5
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        scale = min(epoch ** -0.5, epoch * self.warmup_steps ** -1.5) * self.factor
        return [scale * lr for lr in self.base_lrs]
