# ALFM

Atom-Level Foundation Model for materials property prediction. This is my CAMMP internship project.

## What this is

I pretrained a graph neural network on 238k crystal structures using two self-supervised objectives (masked atom reconstruction and contrastive learning), then fine-tuned it to predict formation energy and band gap. The idea is simple: if the model first learns what crystals "look like" without any labels, it gets better at predicting properties with labels afterwards.

## Results

Formation energy prediction:
- Pretrained: **0.0494 eV/atom** (R² = 0.995)
- From scratch: 0.0757 eV/atom
- Pretraining cuts the error by about 35%

Band gap prediction:
- Pretrained: **0.267 eV** (R² = 0.900)
- From scratch: 0.285 eV
- About 6% improvement here

The pretrained model trained on half the labeled data beats the scratch model trained on all of it. That's probably the most interesting finding.

## How it works

Each crystal gets turned into a graph where atoms are nodes and nearby atoms (within 6 angstroms) are connected by edges. Node features are things like atomic number, electronegativity, radius. Edge features use Bessel radial basis functions to encode distances.

The encoder is a 4-layer gated message passing network (128-dim embeddings). During pretraining, we mask 20% of atoms and make the model predict what was there, and also train it to produce similar embeddings for augmented versions of the same crystal. After pretraining, we attach a regression head and fine-tune on the actual property prediction task.

Training details: EMA on encoder weights, Huber loss for fine-tuning, OneCycleLR scheduler, mixed precision, gradient accumulation. Everything runs on a single T4 GPU on Kaggle.

## Dataset

MatBench formation energy and band gap from the Materials Project. 238,865 structures total. 70/15/15 split.

## Running

The notebook runs end-to-end on Kaggle with a T4 GPU. Takes about 6 to 8 hours total. You need torch-geometric and pymatgen installed.

## Author

Aryan Kaushik
