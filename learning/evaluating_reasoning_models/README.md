# Evaluating Reasoning Models

## Overview

This folder contains a comprehensive evaluation framework for benchmarking and assessing the reasoning capabilities of language models. Use this to:

- **Measure reasoning accuracy** - How well does the model solve math/logic problems?
- **Test different strategies** - Compare greedy decoding vs. self-consistency
- **Generate evaluation reports** - Detailed metrics and analysis
- **Compare model versions** - Track improvement across different models or fine-tuning runs

**Primary benchmark:** MATH500 dataset (500 mathematical reasoning problems)

## Contents

### Core Scripts

#### `evaluating_reasoning_models.py`
Main evaluation pipeline that runs your model on benchmark datasets and collects results.

**Key responsibilities:**
- Load model and tokenizer
- Load benchmark dataset (MATH500)
- Generate predictions using the model
- Compare predictions with ground truth
- Calculate accuracy metrics
- Generate evaluation reports

**Usage:**
```bash
python evaluating_reasoning_models.py --model "path/to/model" --output "results.jsonl"
```

#### `model_and_tokenizer.py`
Utilities for loading pre-trained models and tokenizers.

**Key functions:**
- `load_model()` - Load model from Hugging Face or local path
- `load_tokenizer()` - Load corresponding tokenizer
- Handle GPU/CPU device selection
- Manage model configuration

**Usage:**
```python
from model_and_tokenizer import load_model, load_tokenizer
model = load_model("Qwen/Qwen-7B")
tokenizer = load_tokenizer("Qwen/Qwen-7B")
```

#### `text_generation_wrapper.py`
Wrapper for generating text from models with various decoding strategies.

**Key functions:**
- `generate_greedy()` - Deterministic generation
- `generate_with_sampling()` - Temperature/top-p sampling
- `generate_chain_of_thought()` - CoT prompting
- Batch generation for efficiency

**Supports:**
- Temperature scaling
- Top-k and top-p sampling
- Maximum token limits
- Early stopping conditions

#### `load_math_500.py`
Utilities for loading and processing the MATH500 benchmark dataset.

**Key functions:**
- `load_dataset()` - Load all 500 problems
- `load_split()` - Load specific train/test splits
- `format_problem()` - Format problem for model input
- `extract_answer()` - Extract final answer from problem text

### Datasets

#### `math500_test.json`
Test set of 500 mathematical problems in JSON format.

**Structure:**
```json
{
  "problem_id": "1",
  "problem": "Find the value of...",
  "solution": "Step-by-step solution...",
  "answer": "42"
}
```

**Properties:**
- 500 diverse math problems
- Difficulty range: algebra to competition mathematics
- Ground truth answers included
- Solutions with explanations

#### `math500-base-cuda-cot-greedy.jsonl`
Evaluation results: Model predictions using Chain-of-Thought (CoT) with greedy decoding.

**Format:** One JSON object per line (JSONL)
```json
{"problem_id": "1", "prediction": "42", "gold_answer": "42", "correct": true, "reasoning": "..."}
```

**Key metrics in results:**
- `correct` - Whether prediction matches ground truth
- `prediction` - Model's final answer
- `reasoning` - Model's reasoning process (if CoT)

#### `math500-base-cuda-cot-self-consistency.jsonl`
Evaluation results: Model predictions using Chain-of-Thought + Self-Consistency (multiple reasoning paths).

**Comparison:**
- **Greedy:** Single reasoning path, deterministic
- **Self-Consistency:** Multiple reasoning paths, take majority vote
- Usually achieves higher accuracy due to diversity

## Workflow

### Step 1: Prepare Model

Ensure you have:
1. Downloaded model weights (via `../downloading_the_base_model/`)
2. Custom architecture initialized (via `../base_model/`)

```python
# Load model and tokenizer
from model_and_tokenizer import load_model, load_tokenizer
model = load_model("Qwen/Qwen-7B")
tokenizer = load_tokenizer("Qwen/Qwen-7B")
```

### Step 2: Load Benchmark

```python
from load_math_500 import load_dataset
problems = load_dataset("math500_test.json")
print(f"Loaded {len(problems)} problems")
```

### Step 3: Generate Predictions

```python
from text_generation_wrapper import generate_chain_of_thought
predictions = []
for problem in problems:
    output = generate_chain_of_thought(
        model=model,
        tokenizer=tokenizer,
        prompt=problem["problem"],
        max_tokens=500
    )
    predictions.append(output)
```

### Step 4: Evaluate Results

```python
from evaluating_reasoning_models import evaluate
results = evaluate(predictions, problems)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct']} / {results['total']}")
```

## Output

The evaluation generates JSONL files (one JSON object per line) with results including `problem_id`, `prediction`, `ground_truth`, and `correct` fields.
