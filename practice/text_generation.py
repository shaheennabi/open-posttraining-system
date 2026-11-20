from pathlib import Path

import jax.numpy as jnp

if __package__:
    from practice.load_model import load_model
    from practice.Qwen3_tokenizer import Qwen3Tokenizer
else:
    from load_model import load_model
    from Qwen3_tokenizer import Qwen3Tokenizer

BUCKET = 64  # sequences are right-padded to a multiple of this so jit recompiles rarely


def generate_text(
    prompt: str,
    model,
    tokenizer: Qwen3Tokenizer,
    max_new_tokens: int = 128,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Greedy generation with a loaded Qwen3 model (no KV cache: recomputes the sequence each step)."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    # Qwen3-0.6B is a chat/instruct model: it must see <|im_start|>user ... <|im_end|><|im_start|>assistant
    # Feeding the raw string makes it behave like a confused base model.
    if use_chat_template:
        input_ids = tokenizer.encode_chat(prompt, add_generation_prompt=True, enable_thinking=enable_thinking)
    else:
        input_ids = tokenizer.encode(prompt, chat_wrapped=False)
    if not input_ids:
        raise ValueError("prompt must encode to at least one token")

    generated_ids = []
    if max_new_tokens == 0:
        return ""

    stop_token_ids = set(getattr(tokenizer, "stop_token_ids", ()))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not stop_token_ids and eos_token_id is not None:
        stop_token_ids.add(eos_token_id)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    streamed_text = ""
    for _ in range(max_new_tokens):
        n = len(input_ids)
        padded_len = ((n + BUCKET - 1) // BUCKET) * BUCKET
        # Right-padding is safe: attention is causal, so real positions never see the pad tokens.
        padded = input_ids + [pad_id] * (padded_len - n)
        model_input = jnp.asarray([padded], dtype=jnp.int32)

        logits = model(model_input)                         # [1, padded_len, vocab]
        next_token_id = int(jnp.argmax(logits[0, n - 1]))   # logits of the LAST REAL token

        if next_token_id in stop_token_ids:
            break

        input_ids.append(next_token_id)
        generated_ids.append(next_token_id)

        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        stable_text = decoded_text
        # hold back a trailing "\ufffd" (half of a multi-byte character) until the rest arrives
        if getattr(tokenizer, "errors", "replace") == "replace" and stable_text.endswith("\ufffd"):
            stable_text = stable_text[:-1]
        if stable_text.startswith(streamed_text):
            new_text = stable_text[len(streamed_text):]
            if new_text:
                print(new_text, end="", flush=True)
                streamed_text = stable_text

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
        "Explain reinforcement learning.",
        model,
        tokenizer,
        max_new_tokens=128,
    )
    print(flush=True)
