# BayesPCN: A Continually Learnable Predictive Coding Associative Memory

## Installation

1. Install python 3.9
2. Run `pip install -r requirements.txt`

## Instructions

### Training

- Train MHN: `python -m bayes_pcn --run-name=sample --layer-update-strat=mhn --n-layers=1`
- Train Generative PCN: `python -m bayes_pcn --run-name=sample --layer-update-strat=ml --n-layers=2`
- Train Bayes PCN: `python -m bayes_pcn --run-name=sample --layer-update-strat=bayes --n-layers=2`

### Scoring

- `python score.py --model-path=runs/default/sample/latest.pt --dataset-mode=all --wandb-mode=online`

### Examining Model Performance

- `python examine.py --model-path=runs/default/sample/latest.pt --index=0`
