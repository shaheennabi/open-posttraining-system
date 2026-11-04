# Improving Reasoning with Inference-Time Scaling

## Overview

This folder contains the **primary focus** of this research project: advanced inference-time techniques to significantly improve the reasoning capabilities of language models **without retraining**.

**Key insight:** Don't just train bigger models—use smarter decoding strategies at inference time to get better reasoning out of existing models.

**Use this folder to:**
- Apply inference-time scaling techniques (CoT, self-consistency, temperature scaling)
- Compare different optimization strategies
- Measure improvements over baseline generation
- Benchmark reasoning accuracy improvements
- Understand the trade-offs between quality and cost

## Contents

### `improving_reasoning_with_inference_time_scaling.py`
Main implementation of inference-time scaling techniques for reasoning enhancement.

**Key techniques implemented:**
- Chain-of-Thought (CoT) prompting
- Self-Consistency (majority voting over multiple paths)
- Temperature scaling and sampling strategies
- Token probability analysis
- Ensemble methods

**Usage:**
```bash
python improving_reasoning_with_inference_time_scaling.py \
    --model "Qwen/Qwen-7B" \
    --strategy "cot-self-consistency" \
    --num_samples 5 \
    --output "results.jsonl"
```

### Result Files (Example Outputs)

#### `math500-base-cuda-cot-greedy.jsonl`
Baseline results: CoT with greedy decoding on MATH500 dataset.

**Interpretation:**
- Single reasoning path
- Deterministic generation
- Typical accuracy: ~50-70% (varies by model size)
- Fast inference: ~0.5-1 second per problem

#### `math500-base-cuda-cot-self-consistency.jsonl`
Optimized results: CoT with self-consistency (K=5 samples) on MATH500 dataset.

**Interpretation:**
- Multiple diverse reasoning paths
- Take majority vote of final answers
- Typical accuracy: ~60-80% (improvement over greedy)
- Slower inference: ~5-10 seconds per problem (K=5)

**Expected improvement:**
- Small models (1.8B): +5-10% accuracy
- Medium models (7B): +8-15% accuracy
- Large models (13B+): +10-20% accuracy

## Actual Experimental Results

### Main Findings from MATH-500 Evaluation

**Experiment 1: Base Model - CoT + Greedy (Single Sample)**
- Dataset: MATH-500, first 20 problems
- Configuration: Temperature=0.2, Top-p=0.9, Max tokens=2048
- **Result: 3/20 correct (15.0%)**
- Runtime: ~5.8 minutes on Tesla T4 GPU

**Experiment 2: Base Model - CoT + Self-Consistency (K=7 Samples)**
- Dataset: MATH-500, first 20 problems
- Configuration: Temperature=0.2, Top-p=0.9, K=7 samples, Max tokens=2048
- **Result: 4/20 correct (20.0%)**
- Runtime: ~36.6 minutes on Tesla T4 GPU
- **Improvement: +1 additional correct answer (+5 percentage points)**
- **Cost increase: 6.3x more inference time**

**Experiment 3: Reasoning Model - CoT + Greedy (Single Sample)**
- Dataset: MATH-500, first 20 problems
- Configuration: Temperature=0.2, Top-p=0.9, Max tokens=2048
- **Result: 3/20 correct (15.0%)**
- Runtime: ~5.8 minutes on Tesla T4 GPU
- **Average reasoning trace length: ~493 tokens (vs. baseline)**
- **Key observation:** Longer reasoning did NOT improve single-sample accuracy

### Critical Insights

**1. Self-Consistency Effectiveness is Limited**
```
Baseline (Greedy):           15.0% accuracy,  5.8 minutes
With Self-Consistency (K=7): 20.0% accuracy, 36.6 minutes
Gain: +5.0 percentage points for 6.3x cost increase
```

This demonstrates that self-consistency can recover answers that already exist within the model's reasoning distribution, but cannot create new reasoning capabilities. The model simply couldn't solve problems it previously couldn't solve, even with 7 diverse reasoning paths.

**2. Reasoning Model Generates Longer Traces But No Accuracy Improvement**

The reasoning model produced substantially longer reasoning chains (~493 tokens) compared to the base model. However, this increase in reasoning verbosity **did not translate to higher accuracy** on a single-sample basis (both 15.0%).

This suggests:
- Extended reasoning traces don't guarantee better problem-solving
- The model may be generating plausible-sounding but incorrect reasoning
- The fundamental reasoning capability is limited, not the verbosity

**3. Inference-Time Scaling Cannot Compensate for Lack of Reasoning Ability**

Key finding from these experiments:
> **Inference-time scaling techniques (CoT + self-consistency) can amplify existing reasoning capabilities, but cannot create reasoning skills absent in the model.**

**Implications:**
- If a base model can't solve a problem, voting over 7 attempts to the same wrong reasoning won't help
- Meaningful improvements in mathematical reasoning require **post-training methods** such as:
  - Supervised fine-tuning on correct reasoning examples
  - Preference optimization (DPO/ORPO/SimPO)
  - Reinforcement learning with better reward signals

### Computational Cost Analysis

| Technique | Samples | Base Accuracy | Final Accuracy | Time per 20 problems | Cost Multiplier |
|-----------|---------|---------------|-----------------|---------------------|-----------------|
| Greedy CoT | 1 | - | 15.0% | 5.8 min | 1x |
| CoT + SC (K=7) | 7 | 15.0% | 20.0% | 36.6 min | 6.3x |

**Cost-benefit analysis:**
- To improve from 15% to 20% (3→4 correct on 20 problems), you pay **~31 additional minutes** of inference
- That's **~31 minutes per 1 additional correct answer**
- For production systems, this may not be worth the cost unless accuracy is extremely critical

## Core Concepts

### 1. Chain-of-Thought (CoT) Prompting

**What it does:** Encourage the model to show step-by-step reasoning before providing the final answer.

**Baseline prompt:**
```
Q: What is 2 + 2?
A:
```

**CoT prompt:**
```
Q: What is 2 + 2?
A: Let me think step by step.
First, I need to add 2 and 2.
2 + 2 = 4.
Therefore, the answer is 4.
```

**Key insight:** Models reason better when forced to "show their work"

**Expected improvement:** +10-20% accuracy on reasoning tasks

**Trade-off:** Slower generation (longer outputs)

### 2. Self-Consistency (Majority Voting)

**What it does:** Generate K diverse reasoning paths and select the most common final answer.

**How it works:**
```
Generate K samples → Extract final answer from each → Vote → Return majority answer

Sample 1: "2+2 = 4" → Answer: 4
Sample 2: "Two plus two equals 4" → Answer: 4
Sample 3: "2 and 2 sum to 4" → Answer: 4
---
Majority vote: 4 (all 3 agreed)
```

**Why it works:**
1. Different reasoning paths can reach the same correct answer
2. Errors in reasoning are often idiosyncratic
3. Majority vote averages out errors
4. Surprisingly effective and cheap compared to ensemble models

**Expected improvement:** +5-15% accuracy (additional gain on top of CoT)

**Trade-off:** K times slower (K=5 = 5x cost)

**Typical K values:**
- K=1: Standard generation (no self-consistency)
- K=3: Moderate improvement, reasonable cost
- K=5: Good balance of quality and cost
- K=10: High quality, expensive

### 3. Temperature Scaling

**What it does:** Control the randomness of token generation.

**Formula:** `logits_scaled = logits / temperature`

**Temperature effects:**
- T=0.1 (very cold): Deterministic, focused on high-probability tokens
- T=1.0 (default): Standard behavior
- T=2.0 (very hot): Very random, high diversity

**For self-consistency use case:**
- Use higher temperature (0.8-1.2) to generate diverse paths
- Diversity is key to self-consistency effectiveness
- Lower temperature makes all samples similar (defeats purpose)

**Best practice:** Temperature=0.8-0.9 for self-consistency sampling

### 4. Token Probability Analysis

**What it does:** Analyze confidence of model predictions to improve decisions.

**Key metrics:**
- Average token probability per sample
- Entropy of final answer distribution
- Confidence in majority vote

**Use case:** 
- Reject low-confidence predictions
- Trigger resampling if uncertain
- Cascade to higher-cost inference strategies

## Workflow

### Step 1: Choose Your Strategy

```python
strategies = {
    "baseline": "Greedy decoding only",
    "cot": "Chain-of-Thought prompting",
    "self-consistency": "CoT + majority voting (K=1-10)",
    "adaptive": "Choose strategy based on difficulty"
}
```

### Step 2: Configure Parameters

```python
config = {
    "model": "Qwen/Qwen-7B",
    "strategy": "cot-self-consistency",
    "num_samples": 5,  # K for self-consistency
    "temperature": 0.8,  # For diversity
    "max_tokens": 500,  # Per reasoning path
    "top_p": 0.95,  # Nucleus sampling
}
```

### Step 3: Load Data and Model

```python
from improving_reasoning_with_inference_time_scaling import load_model, load_dataset

model = load_model("Qwen/Qwen-7B")
dataset = load_dataset("math500_test.json")
```

### Step 4: Run Inference

```python
from improving_reasoning_with_inference_time_scaling import inference_with_scaling

results = []
for problem in dataset:
    result = inference_with_scaling(
        model=model,
        problem=problem,
        strategy="cot-self-consistency",
        num_samples=5,
        temperature=0.8
    )
    results.append(result)
```

### Step 5: Analyze Results

```python
from improving_reasoning_with_inference_time_scaling import analyze_results

analysis = analyze_results(results, dataset)
print(f"Accuracy: {analysis['accuracy']:.2%}")
print(f"Improvement over baseline: {analysis['improvement']:.2%}")
print(f"Average inference time: {analysis['avg_time']:.2f}s")
```

## Experimental Setup

### Baseline Comparison

**Typical experimental structure:**

```
Model: Qwen-7B
Task: MATH500

Strategy                    | Accuracy | Time/Sample | Cost Multiplier
────────────────────────────┼──────────┼─────────────┼─────────────────
1. Greedy (baseline)        | 58%      | 0.5s        | 1x (reference)
2. CoT (greedy)             | 68%      | 2.0s        | 4x
3. Temperature 0.9 (greedy) | 60%      | 0.5s        | 1x
4. Top-p 0.95 (greedy)      | 62%      | 0.6s        | 1.2x
5. CoT + Sampling (K=1)     | 66%      | 1.5s        | 3x
6. CoT + SC (K=3)           | 72%      | 6.0s        | 12x
7. CoT + SC (K=5)           | 75%      | 10.0s       | 20x
8. CoT + SC (K=10)          | 77%      | 20.0s       | 40x
```

**Key insights:**
- CoT alone provides +10% improvement
- Each additional self-consistency sample improves ~1-2%
- K=5 offers best balance (75% accuracy, 20x cost)
- Diminishing returns after K=5

### Running Experiments

```bash
# Baseline (greedy)
python improving_reasoning_with_inference_time_scaling.py \
    --model "Qwen/Qwen-7B" \
    --strategy "greedy" \
    --output "results_baseline.jsonl"

# CoT + Self-Consistency with K=5
python improving_reasoning_with_inference_time_scaling.py \
    --model "Qwen/Qwen-7B" \
    --strategy "cot-self-consistency" \
    --num_samples 5 \
    --temperature 0.8 \
    --output "results_optimized.jsonl"

# Compare results
python analyze_results.py \
    --baseline "results_baseline.jsonl" \
    --optimized "results_optimized.jsonl"
```

## Advanced Techniques

### Adaptive Strategies

```python
# Use greedy for easy problems, self-consistency for hard ones
def adaptive_inference(model, problem, easy_threshold=0.7):
    # First, try greedy
    result_greedy = cot_greedy(model, problem)
    confidence = extract_confidence(result_greedy)
    
    # If low confidence, retry with self-consistency
    if confidence < easy_threshold:
        result_sc = cot_self_consistency(model, problem, k=5)
        return result_sc
    else:
        return result_greedy
```

### Ensemble Methods

```python
# Combine multiple models or checkpoints
def ensemble_inference(models, problem, k=5):
    all_predictions = []
    
    for model in models:
        for _ in range(k):
            pred = generate_with_cot(model, problem)
            all_predictions.append(pred)
    
    # Majority vote across all model + sample combinations
    return majority_vote(all_predictions)
```

### Hybrid Approaches

```python
# Combine different techniques
def hybrid_inference(model, problem):
    # Use CoT + Top-p sampling + Self-consistency
    predictions = []
    for _ in range(5):
        # Each sample uses temperature sampling
        output = model.generate(
            input_ids,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=500
        )
        predictions.append(extract_answer(output))
    
    return majority_vote(predictions)
```

## Result Analysis

### Understanding JSONL Results

```json
{
    "problem_id": "42",
    "problem": "Solve: x^2 + 2x + 1 = 0",
    "ground_truth": "x = -1",
    "strategy": "cot-self-consistency",
    "num_samples": 5,
    "samples": [
        {"answer": "x = -1", "reasoning": "..."},
        {"answer": "x = -1", "reasoning": "..."},
        {"answer": "x = -1", "reasoning": "..."}
    ],
    "prediction": "x = -1",
    "confidence": 0.95,
    "correct": true,
    "inference_time": 10.2
}
```

### Key Metrics to Track

1. **Accuracy:** Percentage of correct predictions
2. **Confidence:** Model's confidence in predictions
3. **Inference time:** Time per problem (seconds)
4. **Total cost:** Time × number of samples
5. **Improvement:** Accuracy gain vs. baseline

## Comparison with Other Approaches

### vs. Training-time Scaling
| Aspect | Inference-time | Training-time |
|--------|----------------|---------------|
| **Speed** | Can apply instantly | Requires retraining |
| **Cost** | Compute at inference | Compute at training |
| **Data** | No new data needed | Requires training data |
| **Flexibility** | Easy to change strategy | Fixed after training |
| **Improvement** | 10-20% | 20-50%+ (if sufficient data) |

### vs. Model Ensembles
| Aspect | Inference-time Scaling | Ensembles |
|--------|------------------------|-----------|
| **Space** | Single model | K models |
| **Cost** | K × single model | K × single model |
| **Diversity** | Sampling-based | Model variety |
| **Simplicity** | Simple decoding | Complex orchestration |

## Best Practices

1. **Establish baseline first** - Measure greedy decoding performance
2. **Incrementally add techniques** - CoT → then self-consistency
3. **Monitor cost/benefit** - Is 15% improvement worth 20x cost?
4. **Use validation set** - Optimize on held-out data
5. **Document everything** - Track which strategy for which model

## Troubleshooting

### Issue: Self-Consistency not improving accuracy
**Possible causes:**
- Temperature too low → increase to 0.8-1.0
- K too small → try K=5-10
- Wrong prompt template → verify CoT prompt format
- Model architecture issues → test with baseline model

**Solution:**
```python
# Debug: visualize samples
for i, sample in enumerate(samples):
    print(f"Sample {i}: {sample['answer']}")
```

### Issue: Very high inference cost
**Problem:** Self-consistency with K=10 is too expensive
**Solution:**
- Reduce K to 3-5
- Use adaptive strategy (K=1 for easy, K=5 for hard)
- Consider smaller model variant
- Use quantization to speed up inference

### Issue: Majority vote ties
**Problem:** Multiple answers tied for most common
**Solution:**
- Increase K to break ties
- Use confidence weighting (weight by token probability)
- Manual tie-breaking rule (alphabetical, first seen, etc.)

## Next Steps

1. **Evaluate on benchmarks** → Use `../evaluating_reasoning_models/`
2. **Fine-tune model** → Create training pipeline
3. **Deploy production system** → Handle cost vs. quality trade-off
4. **Explore other domains** → Adapt to other reasoning tasks beyond math

## Recommended Reading

- [Chain-of-Thought Prompting (Wei et al., 2023)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2023)](https://arxiv.org/abs/2203.11171)
- [Inference-Time Scaling (Extended Thinking)](https://openai.com/blog/)
- [Temperature in LLMs](https://arxiv.org/abs/2102.07033)

## Questions?

- Refer to script docstrings for detailed parameter explanations
- Check result JSONL files for output format examples
- See the main README for project-wide context
- Review the evaluating_reasoning_models folder for benchmarking guidance
