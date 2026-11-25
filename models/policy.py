from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.flax import load_file

from .kv_cache import KVCache
from .qwen3 import qwen3_model

_qwen3_forward = jax.jit(qwen3_model, static_argnames=("rope_theta",))

NUM_LAYERS = 28

LAYER_WEIGHT_NAMES = {
    "input_layernorm": "input_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "q_norm": "self_attn.q_norm",
    "k_norm": "self_attn.k_norm",
    "o_proj": "self_attn.o_proj",
    "post_attention_layernorm": "post_attention_layernorm",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def load_model(model_path, rope_theta=1_000_000):
    weights = load_file(str(model_path))
    layer_params = {
        param_name: jnp.stack(
            [
                weights[f"model.layers.{i}.{checkpoint_name}.weight"]
                for i in range(NUM_LAYERS)
            ],
            axis=0,
        )
        for param_name, checkpoint_name in LAYER_WEIGHT_NAMES.items()
    }

    params = {
        "embed_tokens": weights["model.embed_tokens.weight"],
        "layers": layer_params,
        "final_norm": weights["model.norm.weight"],
    }
    del weights

    def model(input_ids, key_cache, value_cache, cache_position):
        return _qwen3_forward(
            input_ids,
            params,
            key_cache,
            value_cache,
            cache_position,
            rope_theta=rope_theta,
        )

    return model


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "qwen" / "model.safetensors"
    model = load_model(model_path)

    cache = KVCache()
    input_ids = jnp.arange(4).reshape(1, 4)
    cache.ensure_capacity(4)
    logits, key_cache, value_cache = model(
        input_ids, cache.key_cache, cache.value_cache, cache.cache_size
    )
    cache.set_cache(key_cache, value_cache, 4)

    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(
        f"Cache: size={cache.cache_size}, capacity={cache.capacity}, "
        f"key_cache={cache.key_cache.shape}"
    )
    assert logits.shape == (1, 4, 151936)
    assert jnp.all(jnp.isfinite(logits))
    print("load_model OK")
