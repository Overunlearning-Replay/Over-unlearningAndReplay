# Over-Unlearning and Replay

Welcome to the official repository for the paper "Over-Unlearning and Replay Is All You Need." This research explores a critical vulnerability in class-wise approximate unlearning algorithms and introduces the Reminiscence Attack, targeting the membership privacy of unlearned data.

## Overview

The provided codebase enables:

1. Estimation of distribution attributes across various approximate unlearning models.
2. Implementation of the Reminiscence Attack on models processed by approximate unlearning algorithms.
3. Execution of the Over-Unlearning and Replay framework for these algorithms.

## Usage Guide

### 1. Pretraining

Begin with pretraining a model using `pretrain_model_{dataset}.py`. Modify the training loss function to incorporate an L1 regularization for sparsity. Exclude out-of-distribution (OOD) data by setting the `ood_classes` in `config.py`.

### 2. Running Unlearning Algorithms

Post-pretraining, execute unlearning algorithms by running `run.py` with the following specifications:

- `dataset`: Options include `CIFAR20`, `CIFAR100`, `Imagenet64`.
- `model`: Specify the model to use.
- `total_number_of_classes`: Total classes in the dataset.
- `forget_class_index`: Index of the class to be forgotten.
- `pretrained_model_path`: Path to the pretrained model.
- `mu_method_list`: the configuration for recommended unlearning algorithms and parameters.

Use `forget_full_class_main_{dataset}.py` for a complete class unlearning session.

- **Standard Algorithms**: Use the standard method names (e.g., `Wfisher`).
- **Over-Unlearning & Replay**: Append `_our` to the method name (e.g., `Wfisher_our`).

Results are saved in the `log_files` directory.

### 3. Distribution Metrics Analysis

Analyze the distribution of unlearned models by running `run-distribution-metric.py`. This script outputs various metrics such as Intra-class Variance, Silhouette Score, Overlap Score, KDE-estimated Overlap, and t-SNE visualizations.

### 4. Reminiscence Attack

To initiate the Reminiscence Attack, execute `run.py` with the same parameters as in the unlearning phase but change the `python_file` to `mia_on_mu_main_{dataset}_fullclass.py`.

## Contributing

We welcome contributions to enhance the functionality and breadth of our research. Please feel free to fork the repository, make your changes, and submit a pull request.






