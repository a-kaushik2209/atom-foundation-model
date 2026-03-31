# ALFM: Atom-Level Foundation Model for Materials Property Prediction

Self-supervised pretraining for crystal property prediction. Pretrain a GNN encoder on 238k unlabeled crystal structures, then fine-tune on formation energy and band gap prediction.

## Results

| Task | Pretrained | Scratch | Improvement |
|------|:---:|:---:|:---:|
| Formation Energy (MAE, eV/atom) | **0.0494** | 0.0757 | 34.8% |
| Formation Energy (R²) | **0.995** | — | — |
| Band Gap (MAE, eV) | **0.267** | 0.285 | 6.4% |
| Band Gap (R²) | **0.900** | — | — |

The pretrained model trained on 50% of labeled data outperforms the scratch model trained on 100% of labeled data.

### Ablation

| Pretraining | e_form MAE | vs Scratch |
|---|:---:|:---:|
| None (scratch) | 0.0757 | — |
| MAR only | 0.0577 | -23.8% |
| Contrastive only | 0.0579 | -23.5% |
| MAR + Contrastive | **0.0494** | **-34.8%** |

## How it works

1. **Graph construction**: Each crystal becomes a graph. Atoms are nodes, bonds within 6A cutoff are edges. Node features: atomic number, electronegativity, radius, etc. Edge features: 32 Bessel radial basis functions encoding distances.

2. **Pretraining** (15 epochs on 238k structures):
   - Masked Atom Reconstruction (MAR): mask 20% of atoms, predict their features from context
   - Contrastive learning (NT-Xent): two augmented views of same crystal should have similar embeddings

3. **Fine-tuning** (30 epochs): attach a regression head, freeze encoder for 5 epochs then unfreeze. Huber loss, OneCycleLR, early stopping.

## Architecture

- 4-layer gated message passing network
- 128-dim node embeddings, 64-dim edge embeddings
- Mean + std pooling for graph-level representation
- EMA (decay 0.999) on encoder weights during pretraining
- Mixed precision training (AMP)

## Dataset

MatBench formation energy and band gap datasets from Materials Project. 238,865 total crystal structures.

- Formation energy: ~132k structures, range [-4.5, 2.5] eV/atom
- Band gap: ~106k structures, range [0, 9] eV

70/15/15 train/val/test split.

## Running

The full pipeline runs in a single Kaggle notebook on a T4 GPU (16GB VRAM). Total runtime is around 6-8 hours.

```
# Dependencies (install in Kaggle)
pip install torch-geometric pymatgen mp-api

# The notebook runs sequentially:
# Cell 1-3: imports, config, graph construction
# Cell 4: load data + build/cache graphs
# Cell 5-8: model, pretraining objectives, pretrain loop
# Cell 9-13: fine-tuning (pretrained + scratch for both tasks)
# Cell 14-16: data efficiency experiments
# Cell 17-22: evaluation plots (parity, t-SNE, ablation, etc.)
```

### Key config

```python
cfg = dict(
    cutoff      = 6.0,       # angstrom cutoff for neighbor search
    max_nbrs    = 12,        # max neighbors per atom
    n_rbf       = 32,        # bessel basis functions
    node_dim    = 128,       # node embedding dimension
    edge_dim    = 64,        # edge embedding dimension
    n_mp_layers = 4,         # message passing layers
    pt_epochs   = 15,        # pretraining epochs
    pt_batch    = 128,       # pretraining batch size
    ft_epochs   = 30,        # fine-tuning epochs
    ft_batch    = 128,       # fine-tuning batch size
)
```

## File structure

```
cammp/
  alfm-n.ipynb          # full training notebook
  ALFM_CAMMP_Report.docx  # project report with all figures
  README.md             # this file
```

## Output plots

The notebook generates 15 plots covering pretraining loss curves, t-SNE embeddings, parity plots, fine-tuning curves, data efficiency, benchmark comparisons, per-element error analysis, uncertainty quantification, CKA similarity, and ablation results.

## Author

Aryan Kaushik — CAMMP Project
