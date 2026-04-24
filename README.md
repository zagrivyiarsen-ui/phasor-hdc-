```markdown
# Phasor-HDC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Official implementation of the paper **"Phasor-HDC: An Algebraic Substrate for O(1) Causal Interventions and Safe HALT in Hyperdimensional Complex Space"** (preprint, 2026).

## Overview

Phasor-HDC is a neuro-symbolic architecture operating in a complex hyperdimensional space $\mathbb{C}^D$.  
It provides three core contributions:

- **Algebraic `do`-operator** – performs causal interventions in $O(1)$ time without retraining.
- **Safe HALT** – detects logical contradictions via destructive interference and refuses to answer.
- **Phasor-RAG** – scalable memory retaining **100% retrieval accuracy** up to $10^5$ stored rules.

All experiments from the paper are fully reproducible using the code in this repository.

## Installation

```bash
git clone https://github.com/ArseniyZagr/phasor-hdc.git
cd phasor-hdc
```

It is recommended to use a virtual environment with Python 3.9+:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Running the Experiments

Each experiment is self-contained and saves its results under results/expN/.

```bash
# Experiment 1 – Colored MNIST (OOD generalization & do-intervention)
python exp1_colored_mnist.py

# Experiment 2 – CLEVR-Causal (front-door adjustment & causal surgery)
python exp2_clevr_causal.py

# Experiment 3 – Safe HALT (destructive interference)
python exp3_safe_halt.py

# Experiment 4 – Memory Scaling (HRR vs Phasor-RAG)
python exp4_memory_scaling.py

# Experiment 5 – Multi-hop Reasoning (bAbI-style chains)
python exp5_multihop.py
```

After execution, the corresponding results/expN/ folder will contain:

· summary.json – aggregated numerical metrics.
· plots/ – figures in both PDF and PNG formats.
· models/ – saved model checkpoints (Experiment 1 only).
· raw.csv – per-seed raw data (where applicable).

Reproducing Paper Results

All hyperparameters (dimensionality $D$, number of seeds, learning rates, etc.) are defined in the CONFIG dictionary at the top of each script.
Run the scripts without modifications to reproduce the exact numbers reported in the paper.

Experiment Expected Outcome
Exp 1 Supervised OOD = 100%, Unsupervised OOD ≈ 64%
Exp 2 Front‑door accuracy = 100%, intervention time < 0.1 ms
Exp 3 Energy collapses to 0 at $\alpha = 0.5$
Exp 4 HRR collapses after ≈540 rules; Phasor‑RAG stays at 100% up to $10^5$
Exp 5 100% accuracy on both 2‑hop and 4‑hop reasoning chains

Hardware Requirements

· A CUDA‑capable GPU is recommended for Experiments 1 and 4, but all scripts can run on CPU (with longer runtime).
· Approximately 8 GB of RAM is sufficient for $D = 10\,000$ and up to $N = 100\,000$ stored rules.

Repository Structure

```
.
├── exp1_colored_mnist.py      # Colored MNIST: slot learning, knockoff, do-intervention
├── exp2_clevr_causal.py       # CLEVR-like causal graph: front-door, causal surgery
├── exp3_safe_halt.py          # Safe HALT: energy collapse under contradiction
├── exp4_memory_scaling.py     # HRR superposition vs Phasor-RAG retrieval accuracy
├── exp5_multihop.py           # Multi-step logical deduction
├── requirements.txt           # Python dependencies
├── LICENSE
├── CITATION.cff
├── README.md                  # This file
└── results/                   # Automatically generated output directory
    ├── exp1/
    ├── exp2/
    ├── exp3/
    ├── exp4/
    └── exp5/
```

Citation

If you use this code in your research, please cite the corresponding paper:

```bibtex
@article{zagr2026phasor,
  title   = {Phasor-HDC: An Algebraic Substrate for O(1) Causal Interventions and Safe HALT in Hyperdimensional Complex Space},
  author  = {Arseniy Zagr},
  journal = {arXiv preprint},
  year    = {2026}
}
```

Contact

Arseniy Zagr – arseniy.zagr@uni-wuppertal.de
University of Wuppertal

License

This project is licensed under the MIT License – see the LICENSE file for details.

```
