# Attractor Models


## About

Attractor models augment a standard transformer backbone with a weight-tied fixed-point refinement head. A conventional transformer encodes the input into contextual features; a single weight-tied block is then iterated to a fixed point via Anderson acceleration, with the backbone context injected at every step. Gradients flow back through the implicit function theorem, keeping training memory constant in the number of solver iterations.

This codebase also includes experiments on Tiny recursive model experiments

## Acknowledgments

This codebase builds on:

- [Parcae](https://github.com/sandyresearch/parcae) — Parcae
- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) — TRM


## Installation

Requires Python 3.11+ and PyTorch 2.4+. Install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .
```

## Pretrained Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Attractor-140M | 140M | [jacobfa1/attractor-140m](https://huggingface.co/jacobfa1/attractor-140m) |
| Attractor-370M | 370M | [jacobfa1/attractor-370m](https://huggingface.co/jacobfa1/attractor-370m) |
| Attractor-770M | 770M | [jacobfa1/attractor-770m](https://huggingface.co/jacobfa1/attractor-770m) |

## Quick Start

```python
from attractor.models.attractor import Attractor, AttractorConfig

config = AttractorConfig.from_name("attractor-small-140m")
model = config.construct_model()
```

## Training

### Language Modeling

Training is configured via YAML files in `launch_configs/`.

| Config | Architecture | Parameters |
|--------|-------------|------------|
| `attractor-small-140m.yaml` | Attractor | 140M |
| `attractor-medium-370m.yaml` | Attractor | 370M |
| `attractor-large-770m.yaml` | Attractor | 770M |
| `attractor-xlarge-1_3b.yaml` | Attractor | 1.3B |
| `parcae-small-140m.yaml` | Parcae | 140M |
| `parcae-medium-370m.yaml` | Parcae | 370M |
| `parcae-large-770m.yaml` | Parcae | 770M |
| `parcae-xlarge-1_3bm.yaml` | Parcae | 1.3B |
| `gpt-small-140m.yaml` | GPT | 140M |
| `gpt-medium-370m.yaml` | GPT | 370M |
| `gpt-medium-770m.yaml` | GPT | 770M |
| `gpt-medium-1_3b.yaml` | GPT | 1.3B |

Launch with:

```bash
bash runs/run_training.sh launch_configs/attractor-small-140m.yaml attractor-small 2
```

### ARC-AGI Puzzles

The TRM-DEQ experiment in `experiments/attractor_puzzles/` 

```bash
bash experiments/attractor_puzzles/launch_arc_deq.sh
```

### Sudoku

```bash
torchrun --standalone --nproc_per_node=2 \
    -m experiments.attractor_puzzles.trm.train_trm_deq \
    --data_dir /path/to/sudoku-data \
    --out_dir /path/to/output
```

## Eval

```bash
python scripts/eval.py --out_dir /path/to/checkpoint --eval_tasks core
```

## Project Structure

```
attractor/
  models/
    attractor/    # Attractor model (fixed-point head + IFT backward)
    parcae/       # Parcae looped model (baseline)
    gpt/          # Standard transformer (baseline)
  configs/
    attractor/    # Attractor model configs (140M - 1.3B)
    parcae/       # Parcae model configs
    gpt/          # GPT model configs
experiments/
  attractor_puzzles/   # ARC-AGI and Sudoku with TRM-DEQ
recpre/               # Training infrastructure (dataloading, optimizer, monitoring)
receval/              # Evaluation infrastructure
scripts/              # Training, eval, generation entry points
```
