# Over-Unlearning and Replay

This repository contains the official code for the paper titled "Over-Unlearning and Replay Is All You Need". The paper identifies a significant vulnerability in class-wise approximate unlearning algorithms concerning residual knowledge at the distribution level. To address this vulnerability and explore associated privacy risks, the paper introduces the Reminiscence Attack, targeting the membership privacy of unlearned data.

## Overview

The codebase provides functionality for three main aspects:

1. Estimating distribution attributes of various approximate unlearning models.
2. Conducting the Reminiscence Attack on models obtained from various approximate unlearning algorithms.
3. Executing the Over-Unlearning and Replay framework for various approximate unlearning algorithms.

## Usage

### 1. Pretraining

Before utilizing the unlearning algorithms, it is necessary to obtain a pretrained model. This can be achieved by running `pretrain_model_{dataset}.py`. To obtain a sparse pretrained model, modify the training loss to include an L1 regularization penalty term. Note that the training excludes out-of-distribution (OOD) data, as specified by the `ood_classes` parameter in `config.py`.

### 2. Running Unlearning Algorithms

After obtaining the pretrained model, various unlearning algorithms can be executed. Run `run.py` and specify the dataset (`CIFAR20`, `CIFAR100`, `Imagenet64`), the model, the total number of classes, the index of the forget class, and the path of the pretrained model. The file recommends different unlearning algorithms and their parameters in the `mu_method_list` parameter. If the executed Python file is named `forget_full_class_main_{dataset}.py`, a series of machine unlearning operations will be performed. 
- For original unlearning algorithms, the method name remains unchanged (e.g., `Wfisher`).
- For unlearning algorithms under Over-Unlearning & Replay, append `_our` to the method name (e.g., `Wfisher_our`).

All generated results will be stored in the `log_files` directory.

### 3. Distribution Metrics Analysis

For analyzing the unlearned models obtained from various unlearning algorithms, execute `run-distribution-metric.py`. This will generate Intra-class Variance, Silhouette Score, Overlap Score, KDE-estimated Overlap visualization results, as well as t-SNE visualization results.

### 4. Reminiscence Attack

To conduct the Reminiscence Attack, run `run.py` with the same configuration as before, but set the `python_file` parameter to `mia_on_mu_main_{dataset}_fullclass.py`.






