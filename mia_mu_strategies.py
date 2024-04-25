from tqdm import tqdm

from process_mia_data import view_mia_results
from utils import *
from metrics import UnLearningScore, get_membership_attack_prob
import argparse
import models
import torch.optim as optim
import csv
import os.path as osp

def mia_mu_relearning(unlearned_model, forget_train_dataloader,
                      retain_train_dataloader, ood_dataloader, rest_retain_dataloader,
                      num_classes, device, weight_path, mia_path, relearning_lr=0.01,
                      **kwargs):
    unlearned_model.load_state_dict(torch.load(weight_path))

    if isinstance(ood_dataloader, list):
        lognames = ['log_forget_train_dataloader.tsv']
        for i in range(len(ood_dataloader)):
            lognames.append('log_ood_dataloader_'+str(i)+'.tsv')
    else:
        lognames = ['log_forget_train_dataloader.tsv', 'log_ood_dataloader.tsv']
    if not os.path.exists(mia_path):
        os.makedirs(mia_path)

    for logname in lognames:
        logname = osp.join(mia_path, logname)
        with open(logname, 'w+') as f:
            columns = ['epoch',
                       'loss(Train)',
                       'acc(Train)'
                       ]
            f.write('\t'.join(columns) + '\n')

    unlearned_model = unlearned_model.to(device)

    # get_metric_scores(unlearned_model, unlearning_teacher, retain_train_dataloader,
    #                   forget_train_dataloader, valid_dataloader,
    #                   valid_poisonedloader, device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=float(relearning_lr), momentum=0.9)#

    def relearn(model, dataloader, epoch, logfile):

        total_loss = 0.0
        correct = 0
        total = 0
        total_batch = len(dataloader)
        iter_rest_retaining_dataloader = iter(rest_retain_dataloader)
        for batch_idx, (inputs, _, targets) in enumerate(tqdm(dataloader)):
            model.train()
            try:
                rest_inputs, _, rest_targets = iter_rest_retaining_dataloader.next()
            except:
                iter_rest_retaining_dataloader = iter(rest_retain_dataloader)
                rest_inputs, _, rest_targets = iter_rest_retaining_dataloader.next()

            inputs = torch.cat((inputs, rest_inputs), dim=0)
            targets = torch.cat((targets, rest_targets), dim=0)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # total_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets[:args.b].size(0)
            # correct += predicted[:args.b].eq(targets[:args.b]).sum().item()
            #
            # if batch_idx % 10 == 0:
            #     avg_loss = total_loss / 10
            #     accuracy = correct / total
            #     print("[epoch]:", "{:.2f}".format(epoch + (batch_idx + 1) / total_batch), ", avg_loss:",
            #           "{:.6f}".format(avg_loss), ", training_acc:", "{:.4f}".format(accuracy))
            #     with open(logfile, 'a+') as f:
            #         cols = ["{:.2f}".format(epoch + (batch_idx + 1) / total_batch),
            #                 "{:.6f}".format(avg_loss),
            #                 "{:.4f}".format(accuracy)]
            #         f.write('\t'.join([str(c) for c in cols]) + '\n')
            #     total_loss = 0.0
            #     correct = 0
            #     total = 0

            model.eval()
            for inputs, _, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / len(dataloader.dataset)
            accuracy = correct / len(dataloader.dataset)
            print("MIA Attack: [epoch]:", "{:.2f}".format(epoch + (batch_idx + 1) / total_batch), ", avg_loss:",
                  "{:.6f}".format(avg_loss), ", training_acc:", "{:.4f}".format(accuracy))
            with open(logfile, 'a+') as f:
                cols = ["{:.2f}".format(epoch + (batch_idx + 1) / total_batch),
                        "{:.6f}".format(avg_loss),
                        "{:.4f}".format(accuracy)]
                f.write('\t'.join([str(c) for c in cols]) + '\n')
            total_loss = 0.0
            correct = 0

            if batch_idx > 50:
                break

    epochs = 1
    unlearned_model.load_state_dict(torch.load(weight_path))
    for epoch in range(epochs):
        # 训练第一个数据加载器
        relearn(unlearned_model, forget_train_dataloader, epoch, osp.join(mia_path, 'log_forget_train_dataloader.tsv'))

    unlearned_model.load_state_dict(torch.load(weight_path))
    for epoch in range(epochs):
        # 训练第三个数据加载器
        if isinstance(ood_dataloader, list):
            for i in range(len(ood_dataloader)):
                relearn(unlearned_model, ood_dataloader[i], epoch, osp.join(mia_path, 'log_ood_dataloader_'+str(i)+'.tsv'))
        else:
            relearn(unlearned_model, ood_dataloader, epoch, osp.join(mia_path, 'log_ood_dataloader.tsv'))
    #view the mia results
    view_mia_results(lognames, mia_path)
