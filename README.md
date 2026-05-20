<div align="center">

# Solve the Loop

### Attractor Models for Language and Reasoning

<br>

[Jacob Fein-Ashley](jacobfa.github.io) &nbsp;&middot;&nbsp; [Paria Rashidinejad](https://pariard.github.io/)

University of Southern California

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2605.12466-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.12466)
[![Project Page](https://img.shields.io/badge/Project-Page-1D4ED8?style=for-the-badge&logo=github&logoColor=white)](https://attractor-models.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

[![Attractor-140M on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Attractor--140M-FFD21E?style=for-the-badge)](https://huggingface.co/jacobfa1/attractor-140m)
[![Attractor-370M on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Attractor--370M-FFD21E?style=for-the-badge)](https://huggingface.co/jacobfa1/attractor-370m)
[![Attractor-770M on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Attractor--770M-FFD21E?style=for-the-badge)](https://huggingface.co/jacobfa1/attractor-770m)

<br>

</div>

> **Attractor models** are looped language models that replace the recurrent block with a fixed-point iteration head.

## Installation

Requires Python 3.11+ and PyTorch 2.4+. Install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .
```

## Quick Start

```python
import attractor

model = attractor.create_model("attractor-small-140m")
```

Or, more explicitly:

```python
from attractor.models.attractor import Attractor, AttractorConfig

config = AttractorConfig.from_name("attractor-small-140m")
model = config.construct_model()
```

Other valid names include `parcae-*`, `gpt-*`, `eqlm-*`, and the `*-sudoku-{7m,27m}` reasoning configs (see [Project Structure](#project-structure)).

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
| `parcae-small-140m.yaml` | Parcae (baseline) | 140M |
| `parcae-medium-370m.yaml` | Parcae (baseline) | 370M |
| `parcae-large-770m.yaml` | Parcae (baseline) | 770M |
| `parcae-xlarge-1_3b.yaml` | Parcae (baseline) | 1.3B |
| `gpt-small-140m.yaml` | GPT (baseline) | 140M |
| `gpt-medium-370m.yaml` | GPT (baseline) | 370M |
| `gpt-large-770m.yaml` | GPT (baseline) | 770M |
| `gpt-xlarge-1_3b.yaml` | GPT (baseline) | 1.3B |

`runs/run_training.sh` takes positional arguments `CONFIG RUN_NAME RUN_GROUP NUM_GPUS` (defaults: `parcae-small-140m`, group `parcae`, 8 GPUs). For example, to train Attractor-140M on 2 GPUs:

```bash
bash runs/run_training.sh \
    launch_configs/attractor-small-140m.yaml \
    attractor-small-140m attractor 2
```

Tiny reasoning models:

| Config | Architecture | Parameters |
|--------|-------------|------------|
| `attractor-sudoku-7m` | Attractor (TRM-DEQ) | 7M |
| `attractor-sudoku-27m` | Attractor (TRM-DEQ) | 27M |

### Sudoku & Maze Reasoning

```bash
torchrun --standalone --nproc_per_node=2 \
    -m experiments.attractor_puzzles.trm.train_trm_deq \
    --data_dir /path/to/sudoku-data \
    --out_dir  /path/to/output \
    --L_layers 8 \
    --deq_max_iter 8 --deq_min_iter 4 --deq_tol 1e-3 \
    --bptt_through 2 --jacobian_reg_lambda 1e-3
```

The sibling pipeline under `experiments/eqlm_sudoku/trm/train_trm_deq.py` runs the same trainer against the EQLM head; both share `experiments/*/data/` and `experiments/*/configs/`.

### ARC-AGI Puzzles

```bash
bash experiments/attractor_puzzles/launch_arc_deq.sh
```

This launches `experiments/eqlm_sudoku/trm/train_trm_deq.py` on 3 GPUs under SLURM with the ARC-AGI dataset.

## Evaluation

Eval is configured via YAML files in `eval_configs/` (`eval-core.yaml`, `eval-core-extended.yaml`, `eval-attractor.yaml`, `eval-eqlm.yaml`, `eval-val-loss.yaml`, `eval-lambada.yaml`). Launch with the wrapper:

```bash
bash runs/run_eval.sh /path/to/out_dir eval_configs/eval-core.yaml 8
```

or directly:

```bash
torchrun --nproc_per_node=8 scripts/eval.py \
    --out_dir /path/to/out_dir \
    --config  eval_configs/eval-core.yaml
```

Supported `eval_tasks` values: `lm_eval`, `sample`, `bpb`, `core`, `core_extended`.

## Project Structure

```
attractor/
  models/
    attractor/    # Attractor model (fixed-point head + IFT backward)
    eqlm/         # EQLM (alias of Attractor; used for scaling-law runs)
    parcae/       # Parcae looped model (baseline)
    gpt/          # Standard transformer (baseline)
  configs/        # Model configs:
    attractor/    #   attractor-{small-140m, medium-370m, large-770m, xlarge-1_3b,
                  #              sudoku-7m, sudoku-27m}
    parcae/       #   parcae-{small-140m, medium-370m, large-770m, xlarge-1_3b}
    gpt/          #   gpt-{small-140m, medium-370m, large-770m, xlarge-1_3b}
experiments/
  attractor_puzzles/   # Sudoku, Maze, ARC-AGI with TRM-DEQ
  eqlm_sudoku/         # Sibling pipeline used by the ARC launcher
launch_configs/        # YAML training configs (parcae / attractor / eqlm / gpt × 4 sizes)
eval_configs/          # YAML evaluation configs (core / core_extended / val-loss / lambada)
recpre/                # Training infrastructure
receval/               # Evaluation infrastructure
runs/                  # Shell wrappers: run_training.sh, run_eval.sh,
                       #                 sweep_recurrence.sh, run_batch_eval.sh
scripts/               # Training, eval, generation, FLOPs estimation entry points
```

## Citation

```bibtex
@article{feinashley2026attractor,
  title={Solve the Loop: Attractor Models for Language and Reasoning},
  author={Fein-Ashley, Jacob and Rashidinejad, Paria},
  year={2026},
  url={https://arxiv.org/abs/2605.12466}
}
```

## Acknowledgments

This codebase builds on [Parcae](https://github.com/sandyresearch/parcae) and [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). This work was supported in part by a grant from Coefficient Giving.
