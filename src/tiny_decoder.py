import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    keys: List[torch.Tensor]
    values: List[torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Minimal multi-head causal self-attention with optional KV cache."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if layer_cache is not None:
            prev_k, prev_v = layer_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        query_len = q.size(2)
        key_len = k.size(2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if layer_cache is None:
            mask = torch.triu(
                torch.ones(query_len, key_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, v)
        y = self.out(self._merge_heads(y))

        new_cache = (k, v) if use_cache else None
        return y, new_cache


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), layer_cache=layer_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyDecoderLM(nn.Module):
    """Small decoder-only LM for systems experiments, not for model quality."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_position_embeddings, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        bsz, seq_len = input_ids.shape
        past_len = 0 if cache is None else cache.keys[0].size(2)

        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        new_keys, new_values = [], []
        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else (cache.keys[i], cache.values[i])
            x, new_cache = block(x, layer_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                k, v = new_cache
                new_keys.append(k)
                new_values.append(v)

        logits = self.lm_head(self.ln_f(x))
        new_cache_obj = KVCache(new_keys, new_values) if use_cache else None
        return logits, new_cache_obj

    @torch.no_grad()
    def generate_naive(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        output = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(output, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)
        return output

    @torch.no_grad()
    def generate_with_cache(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        logits, cache = self(input_ids, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output = torch.cat([input_ids, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            logits, cache = self(next_token, cache=cache, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

        return output
