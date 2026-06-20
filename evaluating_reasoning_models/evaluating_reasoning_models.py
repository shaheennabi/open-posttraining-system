from pathlib import Path
import json
import re
import time

import torch
from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser as spp
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError

from base_model.qwen import KVCache


RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

LATEX_FIXES = [  # LaTeX formatting to be replaced
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # Strip chat special tokens

SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}


@torch.inference_mode()
def generate_text_stream_with_kv_cache(
    prompt, model, tokenizer, device, max_new_tokens, eos_token_id
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Initial forward pass
    logits = model(input_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token

        logits = model(next_token, cache=cache)[:, -1]


def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens, verbose=False
):
    generated_ids = []

    for token in generate_text_stream_with_kv_cache(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0).item()
        generated_ids.append(next_token_id)

        if verbose:
            print(tokenizer.decode([next_token_id]), end="", flush=True)

    return tokenizer.decode(generated_ids)


def get_last_boxed(text: str):
    """Extract content from the last \boxed{...} expression."""
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip whitespace
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        if text[current_idx] == "{":
            brace_depth += 1
        elif text[current_idx] == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx : current_idx - 1]


def extract_final_candidate(text: str, fallback: str = "number_then_full") -> str:
    """Extract the most likely final answer from model output."""
    if not text:
        return ""

    # Prefer \boxed{} answer
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")

    # Fallback: extract number or full text
    if fallback in ("number_then_full", "number_only"):
        matches = RE_NUMBER.findall(text)
        if matches:
            return matches[-1]
        if fallback == "number_then_full":
            return text.strip()

    return ""


def has_complete_boxed_answer(text):
    start = text.rfind(r"\boxed")
    if start == -1:
        return False

    brace_start = text.find("{", start)
    if brace_start == -1:
        return False

    depth = 0
    for ch in text[brace_start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True

    return False


def normalize_text(text: str) -> str:
    """Clean and normalize mathematical text for comparison."""
    if not text:
        return ""

    text = RE_SPECIAL.sub("", text).strip()

    # Remove multiple-choice labels (e.g., "A. 42" → "42")
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    # Remove degree symbols and LaTeX wrappers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Unwrap \text{...}
    match = re.match(r"^\\text\{(.+?)\}$", text)
    if match:
        text = match.group(1)

    # LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # Convert superscripts
    def convert_superscripts(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        return f"{base}**{converted}" if base else converted

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    # Mathematical transformations
    text = text.replace("^", "**")
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")

    text = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s+([^\s{}]+)", r"sqrt(\1)", text)

    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)", r"(\1)/(\2)", text)

    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", r"+\1", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)  # Remove thousand separators

    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr: str):
    if not expr or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None


def equality_check(expr_gtruth: str, expr_pred: str) -> bool:
    if expr_gtruth == expr_pred:
        return True

    gtruth = sympy_parser(expr_gtruth)
    pred = sympy_parser(expr_pred)

    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False


def split_into_parts(text: str):
    """Split tuple/list-like answers into parts for comparison."""
    if not text:
        return []

    if len(text) >= 2 and text[0] in "([ " and text[-1] in ")]" and "," in text[1:-1]:
        items = [p.strip() for p in text[1:-1].split(",")]
        if all(items):
            return items

    return [text]


def grade_answer(pred_text: str, gt_text: str) -> bool:
    if not pred_text or not gt_text:
        return False

    gt_parts = split_into_parts(normalize_text(gt_text))
    pred_parts = split_into_parts(normalize_text(pred_text))

    if len(gt_parts) != len(pred_parts) or not gt_parts:
        return False

    return all(equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts))


# ====================== Utility & Evaluation ======================


def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows
    col_widths = [max(len(row[i]) for row in data) for i in range(len(header))]

    for row in data:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))

    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")


def render_prompt(problem: str) -> str:
    return (
        "You are a helpful math assistant.\n\n"
        "Solve the problem step by step.\n"
        "The last line of your response should contain only the final answer "
        "inside \\boxed{}.\n\n"
        f"Question:\n{problem}\n\n"
        "Answer:\n"
    )





def eta_progress_message(processed, total, start_time, show_eta=True, label="Progress"):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")

    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    avg_time = elapsed / processed
    eta_seconds = int(round(avg_time * (total - processed)))

    minutes, seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    eta_str = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else \
              f"{minutes:02d}m {seconds:02d}s" if minutes else f"{seconds:02d}s"

    return f"{progress} | ETA: {eta_str}".ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat(
                model, tokenizer, prompt, device, max_new_tokens, verbose=verbose
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(gen_text)
            is_correct = grade_answer(extracted, row["answer"])
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(eta_progress_message(i, num_examples, start_time), end="\r", flush=True)

            if verbose:
                print(f"\n\n{'='*60}\nExtracted: {extracted}\n"
                      f"Expected : {row['answer']}\n"
                      f"Correct  : {num_correct}/{i}\n{'-'*60}")

    seconds = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0

    print(f"\n\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds/60:.1f} minutes")
    print(f"Average response length: {total_len/num_examples:.1f} tokens")
    print(f"Results saved to: {out_path}")

    return num_correct, num_examples, acc