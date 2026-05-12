<div align="center">

# Solve the Loop: Attractor Models for Language and Reasoning

**Jacob Fein-Ashley &nbsp;&middot;&nbsp; Paria Rashidinejad**

University of Southern California

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://attractor-models.github.io)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey?style=for-the-badge&logo=arxiv)](https://attractor-models.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Attractor-140M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--140M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-140m)
[![Attractor-370M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--370M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-370m)
[![Attractor-770M](https://img.shields.io/badge/%F0%9F%A4%97-Attractor--770M-yellow?style=flat-square)](https://huggingface.co/jacobfa1/attractor-770m)

</div>

## Installation

Requires Python 3.11+ and PyTorch 2.4+. Install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .
```

## Quick Start

```python
from attractor.models.attractor import Attractor, AttractorConfig

config = AttractorConfig.from_name("attractor-small-140m")
model = config.construct_model()
```

## Pretrained Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Attractor-140M | 140M | [jacobfa1/attractor-140m](https://huggingface.co/jacobfa1/attractor-140m) |
| Attractor-370M | 370M | [jacobfa1/attractor-370m](https://huggingface.co/jacobfa1/attractor-370m) |
| Attractor-770M | 770M | [jacobfa1/attractor-770m](https://huggingface.co/jacobfa1/attractor-770m) |

## Training

### Language Modeling

Training is configured via YAML files in `launch_configs/`.

| Config | Architecture | Parameters |
|--------|-------------|------------|
| `attractor-small-140m.yaml` | Attractor | 140M |
| `attractor-medium-370m.yaml` | Attractor | 370M |
| `attractor-large-770m.yaml` | Attractor | 770M |
| `attractor-xlarge-1_3b.yaml` | Attractor | 1.3B |
| `parcae-small-140m.yaml` | Parcae (baseline) | 140M |
| `parcae-medium-370m.yaml` | Parcae (baseline) | 370M |
| `gpt-small-140m.yaml` | GPT (baseline) | 140M |
| `gpt-medium-370m.yaml` | GPT (baseline) | 370M |

Launch with:

```bash
bash runs/run_training.sh launch_configs/attractor-small-140m.yaml attractor-small 2
```

### Sudoku & Maze Reasoning

```bash
torchrun --standalone --nproc_per_node=2 \
    -m experiments.attractor_puzzles.trm.train_trm_deq \
    --data_dir /path/to/sudoku-data \
    --out_dir /path/to/output
```

### ARC-AGI Puzzles

```bash
bash experiments/attractor_puzzles/launch_arc_deq.sh
```

## Evaluation

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
  configs/        # Model configs (140M – 1.3B)
experiments/
  attractor_puzzles/   # Sudoku, Maze, ARC-AGI with TRM-DEQ
recpre/               # Training infrastructure
receval/              # Evaluation infrastructure
scripts/              # Training, eval, generation entry points
```

## Citation

```bibtex
@article{feinashley2026attractor,
  title={Solve the Loop: Attractor Models for Language and Reasoning},
  author={Fein-Ashley, Jacob and Rashidinejad, Paria},
  year={2026},
  url={https://attractor-models.github.io}
}
```

## Acknowledgments

This codebase builds on [Parcae](https://github.com/sandyresearch/parcae) and [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). This work was supported in part by a grant from Coefficient Giving.
