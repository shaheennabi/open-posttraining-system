from pathlib import Path
import jax.numpy as jnp
from safetensors.flax import load_file


def load_model(model_path):
    model = load_file(str(model_path)) # this reads the model.safetensors file


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

    # Stack the 28 layers
    layer_params = {}

    for param_name, checkpoint_name in layer_weight_names.items():

        weights = [
            model[f"model.layers.{i}.{checkpoint_name}.weight"]
            for i in range(28)
        ]

        layer_params[param_name] = jnp.stack(weights, axis=0) #stacking weights


    # Construct the complete model parameter PyTree
    params = {
        "embed_tokens": model["model.embed_tokens.weight"],
        "layers": layer_params,
        "final_norm": model["model.norm.weight"],
    }


    return params


if __name__ == "__main__":
    if __package__:
        from .qwen3 import qwen3_model
    else:
        from qwen3 import qwen3_model

    model_path = Path(__file__).resolve().parent / "qwen" / "model.safetensors"
    params = load_model(model_path)
    input_ids = jnp.arange(4).reshape(1, 4)
    logits = qwen3_model(input_ids, params, 1000000)

    expected_shape = (1, 4, 151936)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    assert jnp.all(jnp.isfinite(logits))