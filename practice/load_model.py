import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.flax import load_file

if __package__:
    from practice.qwen3 import qwen3_model
else:
    from qwen3 import qwen3_model

# jit once at import. params is passed as a (traced) argument, NOT closed over, so the ~1.2 GB of
# weights are not baked into the compiled program as constants. rope_theta must be static.
_qwen3_forward = jax.jit(qwen3_model, static_argnames=("rope_theta",))

# Qwen3-0.6B architecture (from config.json)
EXPECTED = {
    "hidden": 1024,
    "q_out": 16 * 128,
    "kv_out": 8 * 128,
    "head_dim": 128,
    "ffn": 3072,
    "vocab": 151936,
}


def load_model(model_path, rope_theta=1_000_000):
    model = load_file(str(model_path))  # reads model.safetensors -> dict[str, jax array] (bf16 stays bf16)

    # Infer the number of layers from the checkpoint instead of hard-coding 28
    layer_ids = {
        int(m.group(1))
        for k in model
        if (m := re.match(r"model\.layers\.(\d+)\.", k))
    }
    num_layers = len(layer_ids)
    assert layer_ids == set(range(num_layers)), f"non-contiguous layer ids: {sorted(layer_ids)}"

    # Map our parameter names to checkpoint tensor names
    layer_weight_names = {
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

    # Stack the layers: every leaf becomes [num_layers, ...]
    layer_params = {}
    for param_name, checkpoint_name in layer_weight_names.items():
        weights = [model[f"model.layers.{i}.{checkpoint_name}.weight"] for i in range(num_layers)]
        layer_params[param_name] = jnp.stack(weights, axis=0)

    params = {
        "embed_tokens": model["model.embed_tokens.weight"],
        "layers": layer_params,
        "final_norm": model["model.norm.weight"],
    }
    del model  # free the un-stacked copies

    # Qwen3-0.6B ties lm_head to the embedding. We rely on that, so fail loudly if a checkpoint
    # ships a separate lm_head that differs.
    # (the 0.6B safetensors has no "lm_head.weight" key at all)

    # Shape sanity checks: catch a wrong checkpoint / wrong transpose early
    L = params["layers"]
    e = EXPECTED
    expected_shapes = {
        "embed_tokens": (e["vocab"], e["hidden"]),
        "final_norm": (e["hidden"],),
    }
    
    for name, shape in expected_shapes.items():
        assert params[name].shape == shape, f"{name}: {params[name].shape} != {shape}"
    expected_layer_shapes = {
        "input_layernorm": (num_layers, e["hidden"]),
        "post_attention_layernorm": (num_layers, e["hidden"]),
        "q_proj": (num_layers, e["q_out"], e["hidden"]),
        "k_proj": (num_layers, e["kv_out"], e["hidden"]),
        "v_proj": (num_layers, e["kv_out"], e["hidden"]),
        "o_proj": (num_layers, e["hidden"], e["q_out"]),
        "q_norm": (num_layers, e["head_dim"]),
        "k_norm": (num_layers, e["head_dim"]),
        "gate_proj": (num_layers, e["ffn"], e["hidden"]),
        "up_proj": (num_layers, e["ffn"], e["hidden"]),
        "down_proj": (num_layers, e["hidden"], e["ffn"]),
    }
    for name, shape in expected_layer_shapes.items():
        assert L[name].shape == shape, f"layers.{name}: {L[name].shape} != {shape}"

    return partial(_qwen3_forward, params=params, rope_theta=rope_theta)


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "qwen" / "model.safetensors"
    model = load_model(model_path)
    input_ids = jnp.arange(4).reshape(1, 4)
    logits = model(input_ids)

    expected_shape = (1, 4, 151936)
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    assert jnp.all(jnp.isfinite(logits))
    print("load_model OK")
