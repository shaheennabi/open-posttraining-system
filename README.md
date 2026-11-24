# Open Post Training

**Open Post-Training System** is a JAX/TPU-native agent-swarm reinforcement-learning infrastructure project for long-horizon cybersecurity and social-engineering research. It also explores how culture and technology access shape interactions with AI, including research involving low-literacy and underserved communities. The project is being built from model loading and inference utilities toward a complete training and evaluation system.

## Current status

The project is actively under development. The most complete part today is the learning and inference workflow around Qwen3 models. The larger post-training architecture is laid out, but it is not yet a production-ready end-to-end training platform.

## What is implemented

- **Model and tokenizer workflow:** utilities for downloading model files, loading Qwen3 configurations and weights, and working with tokenizers.
- **Text generation:** a working generation path for pretrained models, including sampling and generation helpers.
- **KV-cached inference:** a KV cache implementation and a cached text-generation path for faster autoregressive generation.
- **Reasoning evaluation:** scripts and notebooks for running reasoning-model experiments on MATH-500, including result files and metric analysis.
- **Inference-time scaling:** examples for chain-of-thought, self-consistency, temperature scaling, and comparing inference strategies.
- **Project foundations:** initial interfaces and modules for agents, trajectories, rollouts, rewards, environments, distributed actors, evaluation, and checkpointing.
- **Research experiments:** exploratory work on GRPO and representation analysis, including the latest Qwen3-0.6B study.

## What is still in progress

- Connecting the agent, rollout, reward, environment, and learner modules into one reliable end-to-end training loop.
- Completing the GRPO, PPO, and SFT algorithm implementations.
- Adding robust checkpointing, configuration management, distributed execution, and recovery workflows.
- Expanding coding-agent environments, verifiers, benchmarks, and automated tests.
- Improving reproducibility, documentation, examples, and performance across hardware setups.

## Repository layout

| Area | Purpose |
| --- | --- |
| `agents/` | Agent execution loops, context management, and tool registration |
| `algorithms/` | Losses and planned post-training algorithms |
| `core/` | Shared configuration, protocols, trajectories, and types |
| `distributed/` | Actor, learner, communication, and synchronization components |
| `evaluation/` | Benchmarks, metrics, and analysis utilities |
| `learning/` | Guided model-loading, generation, evaluation, and reasoning experiments |
| `models/`, `practice/` | Model interfaces and hands-on Qwen3 implementations |
| `rewards/`, `rollouts/`, `environments/` | Building blocks for reinforcement-learning experiments |

## Getting started

Install the project dependencies:

```bash
pip install -r requirements.txt
```

For the most complete examples, start with the guides under [`learning/`](learning/), especially:

- [`learning/downloading_the_base_model/`](learning/downloading_the_base_model/)
- [`learning/base_model/`](learning/base_model/)
- [`learning/generating_text_with_pre_trained_llm/`](learning/generating_text_with_pre_trained_llm/)
- [`learning/evaluating_reasoning_models/`](learning/evaluating_reasoning_models/)
- [`learning/improving_reasoning_with_inference_time_scaling/`](learning/improving_reasoning_with_inference_time_scaling/)

## Recent research

### Beyond Fixed Directions: Adaptive Representation Analysis of Reasoning and Memorization in LLMs

This work studies whether reasoning and memorization can be characterized by a single representation direction and whether that direction remains stable after reinforcement learning. Using Qwen3-0.6B and a controlled 400-example dataset, the study finds strong single-direction decodability while observing substantial geometric reorganization after RL.

**Preprint:** https://arxiv.org/pdf/2608.21919

Contributions and experiments are welcome as the system develops.
