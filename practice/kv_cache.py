
import jax.numpy as jnp


def init_kv_cache(batch_size, max_seq_len):
    # [28, B, max_seq_len, 8, 128]
    cache_shape = (28, batch_size, max_seq_len, 8, 128)

    key_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
    value_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

    return key_cache, value_cache


def update_kv_cache(key_cache, value_cache, new_key, new_value, cache_position):
    # key_cache:   [28, B, max_seq_len, 8, 128]
    # value_cache: [28, B, max_seq_len, 8, 128]
    # new_key:     [28, B, T, 1024]
    # new_value:   [28, B, T, 1024]

    B, T, _ = new_key.shape[1:]

    new_key = new_key.reshape(28, B, T, 8, 128)
    # new_key: [28, B, T, 8, 128]

    new_value = new_value.reshape(28, B, T, 8, 128)
    # new_value: [28, B, T, 8, 128]

    cache_indices = cache_position + jnp.arange(T)
    # cache_indices: [T]

    key_cache = key_cache.at[:, :, cache_indices, :, :].set(new_key)
    value_cache = value_cache.at[:, :, cache_indices, :, :].set(new_value)

    return key_cache, value_cache
