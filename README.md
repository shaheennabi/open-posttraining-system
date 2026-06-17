# Open Post Training

A compact research codebase for post-training large language models with a focus on reasoning and reinforcement learning experiments.

## What this repo does
- Implements reusable components for post-training workflows
- Evaluates reasoning capabilities on benchmarks like MATH-500
- Explores inference-time scaling and RL reward optimization
- Tracks progress on RLVR-GRPO experiments

## Latest experiment summary

### Base Model
- Correct: **32 / 50**
- Accuracy: **64.0%**
- Evaluation time: **7.8 minutes**
- Avg response length: **303 tokens**

### RLVR-GRPO (5-Step checkpoint)
- GRPO steps: **5**, rollouts: **4**, max length: **300 tokens**
- Reward: sparse binary correctness
- Correct: **24 / 50**
- Accuracy: **48.0%**
- Evaluation time: **5.9 minutes**
- Avg response length: **223 tokens**

### RLVR-GRPO (50-Step checkpoint)
- GRPO steps: **50**, rollouts: **8**, max length: **300 tokens**
- Reward: sparse binary correctness
- Correct: **30 / 50**
- Accuracy: **60.0%**
- Evaluation time: **3.9 minutes**
- Avg response length: **219 tokens**

## Key observations
- 5-step RLVR-GRPO underperformed the base model (**64% → 48%**).
- 50-step RLVR-GRPO recovered performance to **60%**, still slightly below base.
- RL training shortened responses significantly (**303 → ~220 tokens**) and reduced latency.
- Sparse final-answer reward is a weak learning signal; more training and better rewards are needed.

## Core focus areas
- Supervised fine-tuning (SFT)
- Preference optimization (DPO / ORPO / SimPO)
- RL-based post-training
- Reasoning and inference-time scaling
- Evaluation and benchmark analysis

## Primary folders
- `base_model/` — custom model architecture and Qwen implementation
- `downloading_the_base_model/` — model download pipeline
- `evaluating_reasoning_models/` — benchmark evaluation and metrics
- `generating_text_with_pre_trained_llm/` — text generation utilities
- `improving_reasoning_with_inference_time_scaling/` — inference scaling experiments

## Quick start
1. Download a model:
   ```bash
   cd downloading_the_base_model
   python download_model.py
   ```
2. Load or inspect model architecture:
   ```bash
   cd ../base_model
   python qwen.py
   ```
3. Generate predictions:
   ```bash
   cd ../generating_text_with_pre_trained_llm
   python generate.py
   ```
4. Evaluate reasoning:
   ```bash
   cd ../evaluating_reasoning_models
   python evaluating_reasoning_models.py
   ```
5. Test inference scaling:
   ```bash
   cd ../improving_reasoning_with_inference_time_scaling
   python improving_reasoning_with_inference_time_scaling.py
   ```

## Status
Active research development. Contributions and experimentation are welcome.
