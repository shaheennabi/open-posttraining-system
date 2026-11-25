"""Memory-efficient causal grouped-query attention implemented with Pallas.

The kernel keeps only a query tile, a key/value tile, and online-softmax
accumulators live at once. It is intentionally not wired into ``qwen3.py`` yet.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


def _flash_attention_kernel(
    query_ref,
    key_ref,
    value_ref,
    cache_position_ref,
    query_length_ref,
    key_length_ref,
    output_ref,
    *,
    block_q: int,
    block_k: int,
    head_dim: int,
    scale: float,
) -> None:
    query_block_index = pl.program_id(0)
    query_start = query_block_index * block_q
    query_indices = query_start + jnp.arange(block_q, dtype=jnp.int32)

    query = query_ref[pl.ds(query_start, block_q), :]
    cache_position = cache_position_ref[()]
    query_length = query_length_ref[()]
    key_length = key_length_ref[()]

    row_max = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    row_sum = jnp.zeros((block_q,), dtype=jnp.float32)
    output_accumulator = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

    query_end = jnp.minimum(query_start + block_q, query_length)
    visible_key_end = jnp.minimum(cache_position + query_end, key_length)
    num_key_blocks = pl.cdiv(visible_key_end, block_k)

    def attend_to_key_block(block_index, carry):
        previous_max, previous_sum, previous_output = carry
        key_start = block_index * block_k

        key = key_ref[pl.ds(key_start, block_k), :]
        value = value_ref[pl.ds(key_start, block_k), :]
        scores = jnp.matmul(
            query, key.T, preferred_element_type=jnp.float32
        ) * jnp.float32(scale)

        key_indices = key_start + jnp.arange(block_k, dtype=jnp.int32)
        causal_mask = key_indices[None, :] <= (
            cache_position + query_indices[:, None]
        )
        valid_key_mask = key_indices < key_length
        scores = jnp.where(
            causal_mask & valid_key_mask[None, :], scores, -jnp.inf
        )

        block_max = jnp.max(scores, axis=-1)
        next_max = jnp.maximum(previous_max, block_max)
        previous_scale = jnp.where(
            jnp.isfinite(previous_max),
            jnp.exp(previous_max - next_max),
            0.0,
        )
        probabilities = jnp.exp(scores - next_max[:, None])
        next_sum = previous_sum * previous_scale + jnp.sum(
            probabilities, axis=-1
        )
        next_output = (
            previous_output * previous_scale[:, None]
            + jnp.matmul(
                probabilities.astype(value.dtype),
                value,
                preferred_element_type=jnp.float32,
            )
        )
        return next_max, next_sum, next_output

    _, row_sum, output_accumulator = lax.fori_loop(
        0,
        num_key_blocks,
        attend_to_key_block,
        (row_max, row_sum, output_accumulator),
    )

    output = output_accumulator / row_sum[:, None]
    query_is_valid = query_indices < query_length
    output = jnp.where(query_is_valid[:, None], output, 0.0)
    output_ref[pl.ds(query_start, block_q), :] = output.astype(output_ref.dtype)


def flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cache_position: int | jax.Array = 0,
    *,
    block_q: int = 16,
    block_k: int = 64,
    interpret: bool | None = None,
) -> jax.Array:
    """Compute causal GQA attention with online softmax using a Pallas kernel.

    Args:
        query: ``[batch, query_length, query_heads, head_dim]``.
        key: ``[batch, cache_length, kv_heads, head_dim]``.
        value: Same shape and dtype as ``key``.
        cache_position: Absolute position of the first query in the KV cache.
            Use zero for prompt prefill and the current cache size for decoding.
        block_q: Number of query rows handled by one Pallas program.
        block_k: Number of cached keys consumed per online-softmax iteration.
        interpret: Run Pallas through its reference interpreter. Defaults to
            ``True`` on CPU (for correctness checks) and ``False`` elsewhere.

    Returns:
        Attention output with shape ``[batch, query_length, query_heads,
        head_dim]`` and the same dtype as ``query``.

    This forward-only kernel targets Qwen3's causal grouped-query attention.
    It does not compute gradients and is not yet connected to ``qwen3.py``.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 arrays")
    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")

    batch_size, query_length, query_heads, head_dim = query.shape
    key_batch_size, key_length, kv_heads, key_head_dim = key.shape
    if batch_size != key_batch_size:
        raise ValueError("query, key, and value batch sizes must match")
    if head_dim != key_head_dim:
        raise ValueError("query, key, and value head dimensions must match")
    if query_heads == 0 or kv_heads == 0:
        raise ValueError("query_heads and kv_heads must be positive")
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    if query_length == 0 or key_length == 0:
        raise ValueError("query and key sequences must be non-empty")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if isinstance(cache_position, int):
        if cache_position < 0:
            raise ValueError("cache_position must be non-negative")
        if cache_position + query_length > key_length:
            raise ValueError(
                "key must contain entries through cache_position + query_length"
            )
    if block_q <= 0 or block_q & (block_q - 1):
        raise ValueError("block_q must be a positive power of two")
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError("block_k must be a positive power of two")
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two")

    if interpret is None:
        interpret = jax.default_backend() == "cpu"

    query_heads_per_kv_head = query_heads // kv_heads
    padded_query_length = pl.cdiv(query_length, block_q) * block_q
    padded_key_length = pl.cdiv(key_length, block_k) * block_k
    padded_query = jnp.pad(
        query,
        ((0, 0), (0, padded_query_length - query_length), (0, 0), (0, 0)),
    )
    padded_key = jnp.pad(
        key, ((0, 0), (0, padded_key_length - key_length), (0, 0), (0, 0))
    )
    padded_value = jnp.pad(
        value, ((0, 0), (0, padded_key_length - key_length), (0, 0), (0, 0))
    )

    def query_block_map(query_block_index, batch_index, head_index):
        return batch_index, query_block_index, head_index, 0

    def key_head_map(_query_block_index, batch_index, head_index):
        del _query_block_index
        return batch_index, 0, head_index // query_heads_per_kv_head, 0

    scalar_spec = pl.BlockSpec((), lambda *_: ())
    kernel = partial(
        _flash_attention_kernel,
        block_q=block_q,
        block_k=block_k,
        head_dim=head_dim,
        scale=1.0 / math.sqrt(head_dim),
    )
    compiler_params = None
    if not interpret and jax.default_backend() == "gpu":
        from jax.experimental.pallas import triton as plgpu

        compiler_params = plgpu.CompilerParams(num_warps=4, num_stages=2)

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, padded_query_length, query_heads, head_dim),
            query.dtype,
        ),
        grid=(
            pl.cdiv(padded_query_length, block_q),
            batch_size,
            query_heads,
        ),
        in_specs=[
            pl.BlockSpec(
                (None, block_q, None, head_dim), query_block_map
            ),
            pl.BlockSpec(
                (None, padded_key_length, None, head_dim), key_head_map
            ),
            pl.BlockSpec(
                (None, padded_key_length, None, head_dim), key_head_map
            ),
            scalar_spec,
            scalar_spec,
            scalar_spec,
        ],
        out_specs=pl.BlockSpec(
            (None, block_q, None, head_dim), query_block_map
        ),
        interpret=interpret,
        compiler_params=compiler_params,
        name="qwen3_causal_gqa_flash_attention",
    )(
        padded_query,
        padded_key,
        padded_value,
        jnp.asarray(cache_position, dtype=jnp.int32),
        jnp.asarray(query_length, dtype=jnp.int32),
        jnp.asarray(key_length, dtype=jnp.int32),
    )
    return output[:, :query_length, :, :]