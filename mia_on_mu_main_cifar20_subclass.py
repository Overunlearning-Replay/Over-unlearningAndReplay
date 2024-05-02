import random
import os
# import wandb
# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset, Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import models
import mia_mu_strategies
from unlearn import *
from utils import *
import datasets
import models
import config
from training_utils import *
import os.path as osp
import forget_subclass_strategies
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    required=True,
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    default="Cifar20",
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)

# ## forget random/fullclass
# parser.add_argument("-classes", type=int, required=True, help="number of classes")
## forget subclass
parser.add_argument("-superclasses", type=int, default=20, help="number of superclasses")
parser.add_argument("-subclasses", type=int, default=100, help="number of subclasses")

parser.add_argument("-gpu", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=32, help="batch size for dataloader")#128
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    default="FT_prune",
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
    #     'FT_prune'
    # ],
    help="select unlearning method from choice set",
) #not to use: "UNSIR", "ssd_tuning"

parser.add_argument(
    "-mia_mu_method",
    type=str,
    default="mia_mu_relearning",
    nargs="?",
    choices=[
        "mia_mu_relearning",
        "mia_mu_adversarial"
    ],
    help="select unlearning method from choice set",
) #not to use: "UNSIR", "ssd_tuning"

# ## forget random
# parser.add_argument(
#     "-forget_perc", type=float, required=True, default=0.1, help="Percentage of trainset to forget"
# )

## forget fullclass "vehicle2"/subclass 'rocket'
parser.add_argument(
    "-forget_class",
    type=str,
    default='rocket',
    nargs="?",
    help="class to forget",
    choices=list(config.class_dict),
)

parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")

parser.add_argument(
        "--forget_class",
        type=int,
        default=0,  # 4
        nargs="?",
        help="class to forget",
)
parser.add_argument("--num_ood_dataset", type=str, default='1')
parser.add_argument("--unlearn_data_percent", type=str, default='0.1')

parser.add_argument(
    "--para1", type=str, default=0.01, help="the first parameters, lr, etc."
)
parser.add_argument(
    "--para2", type=str, default=0, help="number of epochs"
)

parser.add_argument(
    "--relearning_lr", type=str, default=0.01, help="number of epochs"
)

parser.add_argument(
    "--sub_ood_class", type=int, default=30
)

args = parser.parse_args()

# Set seeds
def set_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    set_seeds(args)

    batch_size = args.b

    # get network
    net = getattr(models, args.net)(num_classes=args.superclasses)
    # net.load_state_dict(torch.load(args.weight_path))

    unlearning_teacher = getattr(models, args.net)(num_classes=args.superclasses)

    if args.gpu and torch.cuda.is_available():
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = 224 if args.net == "ViT" else 32

    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True, img_size=img_size
    )
    validset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=True, img_size=img_size
    )

    classwise_train, classwise_test = forget_subclass_strategies.get_classwise_ds(trainset, 100), \
                                      forget_subclass_strategies.get_classwise_ds(validset, 100)
    (retain_train, retain_valid) = build_retain_sets_in_sublabel_unlearning(classwise_train, classwise_test,
                                                                            20,
                                                                            100, args.forget_class,
                                                                            [int(args.sub_ood_class)])  # forget_subclass_strategies.build_retain_sets

    forget_train, forget_valid = classwise_train[args.forget_class], classwise_test[args.forget_class]
    len_infer_data = int(float(args.unlearn_data_percent) * len(forget_train))
    print("len_infer_data:", len_infer_data)
    forget_train = Subset(forget_train, np.random.choice(range(len(forget_train)),
                                                         size=len_infer_data, replace=False))

    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)

    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
                               batch_size=batch_size, )
    full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)),
                               batch_size=batch_size, )

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
    for cls in [int(args.sub_ood_class)]: #30
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(0)))  #args.forget_class 分布外的数据默认为遗忘类

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(0)))  #args.forget_class 分布外的数据默认为遗忘类

        ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size))
        ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size))

    for idx, li in validset.coarse_map.items():
        if args.forget_class in li:
            forget_superclass = idx
            break

    data_length = len(forget_train)
    retain_index_1 = np.random.choice(range(len(retain_train)), size=data_length, replace=False)
    retain_index_2 = list(set(range(len(retain_train))) - set(retain_index_1))
    retrain_train_1 = Subset(retain_train, retain_index_1)
    retrain_train_2 = Subset(retain_train, retain_index_2)
    retain_train_dl = DataLoader(retrain_train_1, num_workers=0, batch_size=batch_size, shuffle=True)
    rest_retain_train_dl = DataLoader(retrain_train_2, num_workers=0, batch_size=14, shuffle=True)

    full_train_dataloader = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
                                       batch_size=batch_size)

    mia_path = os.path.join(config.CHECKPOINT_PATH,
                            "{unlearning_scenarios}-{forget_class}-ood{ood_class}".format(unlearning_scenarios="forget_sub_class_main",
                                                                           forget_class=args.forget_class,
                                                                            ood_class=args.sub_ood_class),
                            "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                               classes=args.superclasses),
                            "{task}".format(task="mia_mu_attack"),
                            "{mia_method}-lr{relearning_lr}".format(mia_method=args.mia_mu_method,
                                                                    relearning_lr=args.relearning_lr),
                            "{unlearning_method}-{para1}-{para2}-{percent}".format(unlearning_method=args.method,
                                                                                   para1=args.para1,
                                                                                   para2=args.para2,
                                                                                   percent=args.unlearn_data_percent))

    # where the unlearned model is
    checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}-{forget_class}-ood{ood_class}".format(unlearning_scenarios="forget_sub_class_main",
                                                                           forget_class=args.forget_class,
                                                                            ood_class=args.sub_ood_class),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.superclasses),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                                                                                para1=args.para1,
                                                                                para2=args.para2))


    print("#####", mia_path)
    if not os.path.exists(checkpoint_path_folder):
        os.makedirs(checkpoint_path_folder)
    checkpoint_path = os.path.join(checkpoint_path_folder, "{epoch}-{type}.pth")
    weights_path = checkpoint_path.format(epoch=args.epochs, type="last")

    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 1
    else:
        model_size_scaler = 1

    kwargs = {
        "unlearned_model": net,
        "retain_train_dataloader": retain_train_dl,  # 保留集
        "forget_train_dataloader": forget_train_dl,  # 遗忘集
        # "full_train_dataloader": full_train_dataloader,  # 保留集+遗忘集
        "ood_dataloader": ood_train_dl,  # 分布外的数据，列表，可能有多个大类别
        # "valid_poisonedloader": valid_poisonedloader,
        "rest_retain_dataloader": rest_retain_train_dl,
        "dampening_constant": 1,
        "selection_weighting": 10 * model_size_scaler,  # used in ssd_tuning
        "num_classes": args.superclasses,
        "dataset_name": args.dataset,
        "device": "cuda" if args.gpu else "cpu",
        "model_name": args.net,
        "weight_path": weights_path,
        "mia_path": mia_path,
        "relearning_lr": float(args.relearning_lr)
    }
    # mia attack
    getattr(mia_mu_strategies, args.mia_mu_method)(**kwargs)

    logname = osp.join(mia_path, 'log_{}-{}-{}.tsv'.format(args.net, args.dataset, args.superclasses))