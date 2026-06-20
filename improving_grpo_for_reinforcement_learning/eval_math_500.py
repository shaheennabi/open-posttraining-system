import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluating_reasoning_models.evaluating_reasoning_models import (
    extract_final_candidate,
    generate_text_stream_concat,
    grade_answer,
    render_prompt,
)
from evaluating_reasoning_models.load_math_500 import load_math500_test
from evaluating_reasoning_models.model_and_tokenizer import load_model_and_tokenizer


DEFAULT_DATASET_PATH = ROOT_DIR / "evaluating_reasoning_models" / "math500_test.json"


def resolve_existing_path(path_value, description):
    path = Path(path_value)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            SCRIPT_DIR / path,
            ROOT_DIR / path,
        ]

        if path.parts and path.parts[0] == "checkpoints":
            checkpoint_name = path.name
            candidates.extend(
                [
                    SCRIPT_DIR / "checkpoints" / "rlvr_grpo_training_with_no_kl" / checkpoint_name,
                    ROOT_DIR / "improving_grpo_for_reinforcement_learning" / "checkpoints" / checkpoint_name,
                    ROOT_DIR / "improving_grpo_for_reinforcement_learning" / "checkpoints" / "rlvr_grpo_training_with_no_kl" / checkpoint_name,
                ]
            )

    unique_candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    if not path.is_absolute() and path.name:
        checkpoint_dirs = [
            SCRIPT_DIR / "checkpoints",
            ROOT_DIR / "checkpoints",
            ROOT_DIR / "improving_grpo_for_reinforcement_learning" / "checkpoints",
        ]
        for checkpoint_dir in checkpoint_dirs:
            if not checkpoint_dir.exists():
                continue
            matches = sorted(checkpoint_dir.rglob(path.name))
            if matches:
                return matches[-1].resolve()

    attempted = "\n  - ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        f"{description} not found. Tried:\n  - {attempted}"
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

    eta_str = (
        f"{hours}h {minutes:02d}m {seconds:02d}s"
        if hours
        else f"{minutes:02d}m {seconds:02d}s"
        if minutes
        else f"{seconds:02d}s"
    )

    return f"{progress} | ETA: {eta_str}".ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    step=None,
    out_path=None,
    metrics_csv="eval_metrics.csv",
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        step_label = "unknown" if step is None else f"{step:05d}"
        out_path = Path(f"math500-step{step_label}-{dev_name}.jsonl")
    else:
        out_path = Path(out_path)

    metrics_path = Path(metrics_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])

            gen_text = generate_text_stream_concat(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

            response_len = len(tokenizer.encode(gen_text))
            total_len += response_len

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

            print(
                eta_progress_message(
                    i,
                    num_examples,
                    start_time,
                    label="MATH-500 Eval",
                ),
                end="\r",
                flush=True,
            )

            if verbose:
                print(
                    f"\n\n{'='*60}\n"
                    f"Extracted: {extracted}\n"
                    f"Expected : {row['answer']}\n"
                    f"Correct  : {num_correct}/{i}\n"
                    f"{'-'*60}"
                )

    seconds = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    avg_response_len = total_len / num_examples if num_examples else 0.0

    csv_exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                [
                    "step",
                    "eval_acc",
                    "num_correct",
                    "num_examples",
                    "avg_response_len",
                    "runtime_minutes",
                    "jsonl_path",
                ]
            )
        writer.writerow(
            [
                step,
                round(acc, 4),
                num_correct,
                num_examples,
                round(avg_response_len, 2),
                round(seconds / 60, 2),
                str(out_path),
            ]
        )

    print("\n")
    print(f"Step: {step}")
    print(f"Accuracy: {acc * 100:.2f}% ({num_correct}/{num_examples})")
    print(f"Average response length: {avg_response_len:.1f} tokens")
    print(f"Runtime: {seconds / 60:.2f} minutes")
    print(f"Results saved to: {out_path}")
    print(f"Evaluation metrics saved to: {metrics_path}")

    return {
        "step": step,
        "eval_acc": acc,
        "num_correct": num_correct,
        "num_examples": num_examples,
        "avg_response_len": avg_response_len,
        "runtime_minutes": seconds / 60,
        "jsonl_path": str(out_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to a safetensors checkpoint.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step for logging.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional JSONL output path.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="eval_metrics.csv",
        help="CSV file to create or append evaluation metrics to.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_size < 0:
        raise ValueError("--dataset_size must be non-negative")

    checkpoint_path = resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_path = DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else Path("math500_test.json")

    print(f"Repository root: {ROOT_DIR}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Evaluation step: {args.step}")

    model, tokenizer = load_model_and_tokenizer(
        which_model="base",
        use_compile=False,
        checkpoint_path=checkpoint_path,
    )

    device = next(model.parameters()).device

    math_data = load_math500_test(local_path=dataset_path)
    if args.dataset_size:
        math_data = math_data[: args.dataset_size]
    else:
        math_data = []

    print(f"Loaded {len(math_data)} MATH-500 examples")

    evaluate_math500_stream(
        model=model,
        tokenizer=tokenizer,
        device=device,
        math_data=math_data,
        step=args.step,
        out_path=args.out_path,
        metrics_csv=args.metrics_csv,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
