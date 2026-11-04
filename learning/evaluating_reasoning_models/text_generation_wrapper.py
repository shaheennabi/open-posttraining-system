import torch
from base_model.qwen import KVCache
from evaluating_reasoning_models.complete_boxed import has_complete_boxed_answer


@torch.inference_mode()
def kvcache_based_token_generation(
    input_ids,
    model,
    max_new_tokens,
    stop_token_ids=None,
):
    # initializing model in eval_model()
    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    stop_token_ids = set(stop_token_ids or [])

    logits = model(input_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        next_token_id = next_token.squeeze().item()

        if next_token_id in stop_token_ids:
            break

        yield next_token

        logits = model(next_token, cache=cache)[:, -1]


# this function will use the existing kv_cache stream function
def text_generation_wrapper(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens,
    verbose=False,
    stop_after_boxed=True,
    stop_texts=("\nQuestion:", "\nAnswer:"),
):
    # encode
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
    ).unsqueeze(0)

    stop_token_ids = getattr(tokenizer, "stop_token_ids", None)

    if not stop_token_ids:
        stop_token_ids = {tokenizer.eos_token_id}

    stop_token_ids = {
        token_id
        for token_id in stop_token_ids
        if token_id is not None
    }

    generated_ids = []
    generated_text = ""

    for token in kvcache_based_token_generation(
        input_ids,
        model,
        max_new_tokens,
        stop_token_ids,
    ):
        output_tokens = token.squeeze(0).tolist()

        # append ids -- scalar
        generated_ids.append(token.squeeze(0).item())

        token_text = tokenizer.decode(output_tokens)
        generated_text += token_text

        if verbose:
            print(token_text, end="", flush=True)

        if stop_after_boxed and has_complete_boxed_answer(generated_text):
            break

        """
        if stop_texts and any(
            stop_text in generated_text
            for stop_text in stop_texts
        ):
            break
        """

    return tokenizer.decode(generated_ids)