import csv
import json
import time
from pathlib import Path

from evaluating_reasoning_models.evaluating_reasoning_models import (
    render_prompt,
    generate_text_stream_concat,
    extract_final_candidate,
    grade_answer,
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
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
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

            is_correct = grade_answer(
                extracted,
                row["answer"],
            )

            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

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
    avg_response_len = (
        total_len / num_examples if num_examples else 0.0
    )

    metrics_path = Path(metrics_csv)
    csv_exists = metrics_path.exists()

    with open(metrics_path, "a", newline="") as f:
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
            ]
        )

    print("\n")
    print(
        f"Accuracy: {acc * 100:.2f}% "
        f"({num_correct}/{num_examples})"
    )
    print(
        f"Average response length: "
        f"{avg_response_len:.1f} tokens"
    )
    print(
        f"Runtime: {seconds / 60:.2f} minutes"
    )
    print(
        f"Results saved to: {out_path}"
    )
    print(
        f"Evaluation metrics saved to: {metrics_csv}"
    )

    return {
        "step": step,
        "eval_acc": acc,
        "num_correct": num_correct,
        "num_examples": num_examples,
        "avg_response_len": avg_response_len,
        "runtime_minutes": seconds / 60,
    }


if __name__ == "__main__":
    print(
        "Import and call evaluate_math500_stream() from your notebook."
    )