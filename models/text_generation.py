from pathlib import Path

import jax.numpy as jnp

from .kv_cache import KVCache
from .policy import load_model
from .tokenizer import Qwen3Tokenizer

BUCKET = 64  # the prompt is right-padded to a multiple of this so the prefill step recompiles rarely


def generate_text(prompt: str, model, tokenizer: Qwen3Tokenizer, max_new_tokens: int = 128, use_chat_template: bool = True, enable_thinking: bool = False) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return ""

    # Qwen3-0.6B is a chat model: it must see <|im_start|>user ... <|im_end|><|im_start|>assistant
    if use_chat_template:
        input_ids = tokenizer.encode_chat(prompt, add_generation_prompt=True, enable_thinking=enable_thinking)
    else:
        input_ids = tokenizer.encode(prompt, chat_wrapped=False)

    n = len(input_ids)
    kv_cache = KVCache()

    # ---- prefill: the whole prompt goes through the model once and fills the cache ----
    # Right-padding is safe: attention is causal, so real tokens never see the pads. The pads also write
    # K/V into slots n..padded_len-1, but cache_size stays n, so those slots are masked and later
    # overwritten by real decoded tokens.
    padded_len = ((n + BUCKET - 1) // BUCKET) * BUCKET
    padded = input_ids + [tokenizer.pad_token_id] * (padded_len - n)

    kv_cache.ensure_capacity(padded_len)
    logits, key_cache, value_cache = model(
        jnp.asarray([padded], dtype=jnp.int32), kv_cache.key_cache, kv_cache.value_cache, kv_cache.cache_size
    )  # logits: [1, padded_len, vocab]
    kv_cache.set_cache(key_cache, value_cache, n)  # only the n real tokens count
    next_token_id = int(jnp.argmax(logits[0, n - 1]))  # logits of the LAST REAL token

    # ---- decode: feed ONE new token per step, K/V of the history comes from the cache ----
    generated_ids = []
    streamed_text = ""
    for step in range(max_new_tokens):
        if next_token_id in tokenizer.stop_token_ids:
            break

        generated_ids.append(next_token_id)

        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        stable_text = decoded_text
        # hold back a trailing "\ufffd" (half of a multi-byte character) until the rest arrives
        if tokenizer.errors == "replace" and stable_text.endswith("\ufffd"):
            stable_text = stable_text[:-1]
        if stable_text.startswith(streamed_text):
            new_text = stable_text[len(streamed_text):]
            if new_text:
                print(new_text, end="", flush=True)
                streamed_text = stable_text

        if step == max_new_tokens - 1:
            break  # no need to run the model for a token we will not use

        kv_cache.ensure_capacity(1)
        logits, key_cache, value_cache = model(
            jnp.asarray([[next_token_id]], dtype=jnp.int32), kv_cache.key_cache, kv_cache.value_cache, kv_cache.cache_size
        )  # logits: [1, 1, vocab], the new token is written at position cache_size
        kv_cache.set_cache(key_cache, value_cache, 1)

        next_token_id = int(jnp.argmax(logits[0, 0]))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if generated_text.startswith(streamed_text):
        print(generated_text[len(streamed_text):], end="", flush=True)
    return generated_text


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent / "qwen"
    # load only when run as a script (it used to load at import time)
    model = load_model(model_dir / "model.safetensors")
    tokenizer = Qwen3Tokenizer(model_dir / "tokenizer.json")
    generate_text(
        "Explain RL Environment.",
        model,
        tokenizer,
        max_new_tokens=128,
    )
    print(flush=True)
