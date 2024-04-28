# Over-Unlearning and Replay

Welcome to the official repository for the paper **"How to Forget Data without A Trace: Over-Unlearning and Replay Are All You Need"**. This research explores a critical vulnerability in class-wise approximate unlearning algorithms and introduces the **Reminiscence Attack**, targeting the membership privacy of unlearned data.

## Overview

The provided codebase enables:
1. **Estimation of distribution attributes** across various approximate unlearning models.
2. **Implementation of the Reminiscence Attack** on models processed by approximate unlearning algorithms.
3. **Execution of the Over-Unlearning and Replay framework** for these algorithms.

## Usage Guide

### 1. Pretraining

**To train a normal model:**  
Begin with pretraining a model using `pretrain_model_{dataset}.py`. 

**To train a sparse model:**  
Modify the training loss function to incorporate an L1 regularization for sparsity. Exclude out-of-distribution (OOD) data by setting the `ood_classes` in `config.py`.

### 2. Running Unlearning Algorithms

Post-pretraining, execute unlearning algorithms by running `run.py` with the following specifications:
- `dataset`: Options include `CIFAR20`, `CIFAR100`, `Imagenet64`.
- `model`: Specify the model to use.
- `total_number_of_classes`: Total classes in the dataset.
- `forget_class_index`: Index of the class to be forgotten.
- `pretrained_model_path`: Path to the pretrained model.
- `mu_method_list`: the configuration for recommended unlearning algorithms and parameters.

**Use `forget_full_class_main_{dataset}.py` for a complete class unlearning session.**

- **Standard Algorithms**: Use the standard method names (e.g., `Wfisher`).
- **Over-Unlearning & Replay**: Append `_our` to the method name (e.g., `Wfisher_our`).

Results are saved in the `./log_files` directory.

### 3. Distribution Metrics Analysis

Analyze the distribution of unlearned models by running `run-distribution-metric.py`. This script outputs various metrics such as Intra-class Variance, Silhouette Score, Overlap Score, KDE-estimated Overlap, and t-SNE visualizations.

### 4. Reminiscence Attack

To initiate the Reminiscence Attack, execute `run.py` with the same parameters as in the unlearning phase but change the `python_file` to `mia_on_mu_main_{dataset}_fullclass.py`.

### 5. Optimization Strategies

- **L1 regularization**: Add `l1_regularization` to the loss term. This parameter is defined in `forget_full_class_strategies.py`.
- **L2 regularization**: Add `l2_penalty` to the loss term. This parameter is also defined in `forget_full_class_strategies.py`.
- **SalUn**: This requires three steps:
  1. First, generate masks by running `python_file = "saliency_mu\\generate_mask.py"` in `run.py`, specifying `masked_save_dir`.
  2. Then, pass `masked_save_dir` into the `--mask_path` parameter of the `forget_full_class_main_{dataset}.py` file.
  3. If `mask_path` is not None, SalUn strategies will be enabled, as already integrated in the code.




