import jax
import jax.numpy as jnp



class KVCache:
    def __init__(self, n_layers=28, batch_size=1, num_kv_heads=8, head_dim=128, initial_capacity=128, dtype=jnp.bfloat16):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.capacity = initial_capacity
        self.cache_size = 0
        self.key_cache, self.value_cache = self.init_cache()


    def init_cache(self, capacity=None):
        if capacity is None:
            capacity = self.capacity

        cache_shape = (self.n_layers, self.batch_size, capacity, self.num_kv_heads, self.head_dim) #[28, 1, 128, 8, 128]

        key_cache = jnp.zeros(cache_shape, dtype=self.dtype)
        value_cache = jnp.zeros(cache_shape, dtype=self.dtype)

        return key_cache, value_cache

    def grow_cache(self):
        new_capacity = self.capacity * 2
        new_key_cache, new_value_cache = self.init_cache(new_capacity)

        new_key_cache = new_key_cache.at[:, :, :self.cache_size, :, :].set(self.key_cache[:, :, :self.cache_size, :, :])
        new_value_cache = new_value_cache.at[:, :, :self.cache_size, :, :].set(self.value_cache[:, :, :self.cache_size, :, :])

        self.key_cache = new_key_cache
        self.value_cache = new_value_cache
        self.capacity = new_capacity

    def ensure_capacity(self, T):
        # call BEFORE the model forward, so the cache has room for T new tokens
        while self.cache_size + T > self.capacity:
            self.grow_cache()

    def set_cache(self, key_cache, value_cache, T):
        # call AFTER the model forward, with the caches returned by qwen3_model
        # key_cache / value_cache: [n_layers, B, capacity, num_kv_heads, head_dim]
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.cache_size += T


    def get_kv_cache(self):
        key_cache = self.key_cache[:, :, :self.cache_size, :, :]
        value_cache = self.value_cache[:, :, :self.cache_size, :, :]
        return key_cache, value_cache
