#!/bin/python3.8

"""
This file is used to collect all arguments for the experiment, prepare the dataloaders, call the method for forgetting, and gather/log the metrics.
Methods are executed in the strategies file.
"""

import random
import os

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import models
from pretrain_model_cifar15 import get_classwise_ds
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import models
import config
from training_utils import *
import os.path as osp

@torch.no_grad()
def eval_training(net,testloader):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(testloader.dataset)

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    default="./log_files/model/pretrain/ResNet18-Cifar20-19/39-best.pth",
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    default="Cifar20",
    nargs="?",
    choices=["Cifar10", "Cifar19", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, default=15, help="number of classes")
parser.add_argument("-gpu", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    default="baseline",
    nargs="?",
    # choices=[
    #     "baseline",
    #     "retrain",
    #     "finetune",
    #     "blindspot",
    #     "amnesiac",
    #     "UNSIR",
    #     "NTK",
    #     "ssd_tuning",
    #     "FisherForgetting",
    #     'Wfisher',
    #     'FT_prune',
    #     'negative_grad',
    #     'negative_grad_relabel',
    #     'negative_grad_with_prune',
    #     "blindspot_with_prune"
    # ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_class",
    type=str,
    default="4",  # 4
    nargs="?",
    help="class to forget"
)

parser.add_argument(
    "-mia_mu_method",
    type=str,
    default="mia_mu_adversarial",
    nargs="?",
    help="select unlearning method from choice set",
) #not to use: "UNSIR", "ssd_tuning"

parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")

parser.add_argument(
    "--para1", type=str, default=0.01, help="the first parameters, lr, etc."
)
parser.add_argument(
    "--para2", type=str, default=0, help="number of epochs"
)

#############masked related##########################
parser.add_argument("--masked_path", default=None, help="the path where masks are saved")

args = parser.parse_args()

def extract_features(dataloader, model_temp, device):
    features = []
    labels = []
    model_temp.eval()  # 设置为评估模式
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, _, label = data
            inputs = inputs.to(device)
            output = model_temp(inputs)
            features.extend(output.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

# executes the method passed via args
def tsne_visualization(train_loader, model, forget_class, device, name):
    features, labels = extract_features(train_loader, model, device)  # full_trainloader #train_dl_w_ood

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features) #[class]
    retain_classes = list(set(range(20)) - set(config.ood_classes))
    # print("retain_classes", retain_classes)
    index_forget_class = retain_classes.index(forget_class)
    plt.figure(figsize=(5.1, 4.5))
    # colors = ['lightcoral',
    #           'sandybrown',
    #           'greenyellow',
    #           'gold',
    #           'darkcyan',
    #           'darkgreen',
    #           'aquamarine',
    #           'skyblue',
    #           'cornflowerblue',
    #           'slateblue',
    #           'darkviolet',
    #           'lightpink',
    #           'coral',
    #           'slategray',
    #           'navy']
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
              "#d62728", "#8c564b", "#e377c2", "#7f7f7f",
              "#bcbd22", "#17becf", "#65c2a4", "#89a6af",
              "#df7976", "#e4d1dd", "#d1e0ab"]
    for i in list(set(range(15)) - set([index_forget_class])):  # CIFAR-10有10个类
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], color=colors[i],
                    label=str(retain_classes[i]), alpha=0.5)

    for i in set([index_forget_class]):  # CIFAR-10有10个类
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], color=colors[i],
                    label=str(retain_classes[i]), alpha=0.5)

    plt.legend(loc=4)
    plt.savefig(visual_path + f"/tsne_{name}.pdf", bbox_inches='tight', format='pdf')

def feature_visualization(model, unlearned_model, device):
    # 特征提取
    # features, labels = extract_features(full_train_dl, model)
    # # 使用t-SNE进行降维
    # tsne = TSNE(n_components=2, random_state=42)
    # reduced_features = tsne.fit_transform(features)
    #
    # plt.figure(figsize=(10, 10))
    # for i in range(19):
    #     indices = labels == i
    #     plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=str(i), alpha=0.5)
    # plt.legend()
    # plt.savefig(visual_path + f"/{args.method}_feature_visualization_pretrain.png")

    # tsne_visualization(full_train_dl, model, int(args.forget_class), device, name='pretrained_model')

    tsne_visualization(full_train_dl, unlearned_model,  int(args.forget_class), device, name='unlearned_model')
    # 使用t-SNE进行降维
    # tsne_visualization(train_dl_w_ood, unlearned_model, int(args.forget_class), device, name='unlearned_model_w_ood')

if __name__ == '__main__':

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check that the correct things were loaded
    # if args.dataset == "Cifar20":
    #     assert args.forget_class in config.cifar20_classes
    # elif args.dataset == "Cifar100":
    #     assert args.forget_class in config.cifar100_classes

    # forget_class = config.class_dict[args.forget_class]

    batch_size = args.b

    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.weight_path))

    checkpoint_path = os.path.join(config.CHECKPOINT_PATH,
                                          "{unlearning_scenarios}".format(
                                              unlearning_scenarios="forget_full_class_main"),
                                          "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                             classes=args.classes),
                                          "{task}".format(task="unlearning"),
                                          "{unlearning_method}".format(unlearning_method=args.method))
                                          # "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method, para1=args.para1, para2=args.para2))

    print("#####", checkpoint_path)
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    if args.masked_path:
        weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.epochs, type="last-masked")
    else:
        weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.epochs, type="last")

    if args.gpu:
        net = net.cuda()

    # For celebritiy faces
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    # Scale for ViT (faster training, better performance)
    # Scale for ViT (faster training, better performance)
    img_size = 224 if args.net == "ViT" else 32

    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)

    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)

    classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=20), \
        get_classwise_ds(validset, num_classes=20)

    (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, 20,
                                                                   int(args.forget_class), config.ood_classes)

    forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]

    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)
    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
                               batch_size=batch_size, )
    full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)),
                               batch_size=batch_size, )

    ood_valid_ds = {}
    ood_train_ds = {}
    ood_valid_dl = []
    ood_train_dl = []  # 存不同类别的分布外数据的dataloader
    for cls in config.ood_classes:
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(args.forget_class)))  # 分布外的数据默认为遗忘类

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(args.forget_class)))  # 分布外的数据默认为遗忘类

        ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size))
        ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size))

    train_dl_w_ood = DataLoader(ConcatDataset((retain_train_dl.dataset, ood_train_dl[0].dataset)),
                                batch_size=batch_size, shuffle=True)
    # Change alpha here as described in the paper
    # For PinsFaceRe-cognition, we use α=50 and λ=0.1
    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 0.5
    else:
        model_size_scaler = 1

    unlearned_net = getattr(models, args.net)(num_classes=args.classes)
    unlearned_net.load_state_dict(torch.load(weights_path))
    if args.gpu:
        unlearned_net = unlearned_net.cuda()

    kwargs = {
        "model": net,
        "unlearned_model": unlearned_net,
        # "retain_train_dl": retain_train_dl,
        # "forget_train_dl": forget_train_dl,
        # "full_train_dl": full_train_dl,
        # "valid_dl": full_valid_dl,#validloader,
        # "forget_class": forget_class,
        # "num_classes": args.classes,
        # "dataset_name": args.dataset,
        "device": "cuda" if args.gpu else "cpu",
        # "model_name": args.net,
    }

    visual_path = os.path.join(config.CHECKPOINT_PATH,
                               "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                               "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                  classes=args.classes),
                               "{task}".format(task="visualization"),
                               "{unlearning_method}".format(unlearning_method=args.method))
                               # "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                               #                                                         para1=args.para1,
                               #                                                         para2=args.para2))

    print("#####", visual_path)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    feature_visualization(**kwargs)