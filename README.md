# Open Post-Training System

An open research codebase exploring how to build post-training infrastructure for long-horizon, multi-agent tasks. The project is being developed from the model layer upward, beginning with a hands-on JAX implementation of Qwen3-0.6B.

## Current status

The Qwen3 model-loading and inference path is the most developed part of the repository. The broader agent, rollout, environment, reward, and learner architecture is being scaffolded; it is not yet a complete end-to-end reinforcement-learning system.

### Implemented so far

- **Qwen3-0.6B forward model:** JAX implementation of the decoder model, including grouped-query attention, RoPE, RMSNorm/QK-Norm, and the gated MLP.
- **Checkpoint loading:** loads local Qwen3 safetensors and stacks transformer-layer weights for the JAX model.
- **Tokenizer and generation:** local Qwen3 tokenizer, autoregressive generation, and a growing KV cache.
- **Pallas experiment:** a standalone flash-attention prototype exists, but is not connected to the Qwen3 forward path.
- **System structure:** initial directories for agents, environments, tasks, rollouts, rewards, algorithms, distributed training, evaluation, and checkpointing.

### Still in progress

- Connecting agents, environments, rollout collection, rewards, and a learner into a training loop.
- Implementing the Countdown environment and supplying its train and held-out task data.
- Implementing training algorithms and validating their gradients and behavior.
- Building distributed execution, checkpointing, and task-specific evaluation.

Directories and files for these planned components are scaffolding, not claims that those components are already implemented.

## Repository layout

| Area | Purpose |
| --- | --- |
| `core/` | Shared configuration, protocols, trajectories, and types |
| `environments/` | Environment interfaces, sandbox, and planned task environments |
| `agents/` | Agent execution, context, tools, and action parsing |
| `tasks/` | Training and held-out task data |
| `rollouts/` | Sampling, workers, scheduling, and collection |
| `models/` | Qwen3 implementation, tokenizer, KV cache, policy loading, and adapters |
| `algorithms/` | Losses and planned post-training algorithms |
| `distributed/` | Actor, learner, communication, and synchronization components |
| `rewards/` | Reward functions and verifiers |
| `evaluation/` | Benchmarks, metrics, and analysis |
| `checkpointing/` | Planned checkpoint management |
| `configs/` | Model, rollout, environment, and training configuration |
| `tests/` | Planned focused tests |
| `notebooks/` | Exploratory notebooks |

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download Qwen3-0.6B assets into `models/qwen/`:

```bash
python -m models.download_model
```

Run the local generation example:

```bash
python -m models.text_generation
```

The downloaded model files are local assets and are not included in the repository. Model loading and generation require sufficient system memory.

## Project article

The first article in the series describes the project’s starting point: choosing Qwen3-0.6B for constrained hardware, understanding its architecture from checkpoint weights, accounting for parameters and memory, and beginning a JAX implementation.

[Building an Open Post-Training System: Starting with Qwen3-0.6B](https://shaheennabi.substack.com/p/open-post-training-system-a-jaxtpu)

It is a progress report for the model-understanding and implementation stage, not a claim that the planned distributed RL system is complete.
