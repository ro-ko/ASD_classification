import os, sys
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import csv
import scipy.io as sio
from scipy import stats
# from nilearn import connectome
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split

from data_preprocessing import data_loader, cls_data_loader, augment_data_loader
from metric import accuracy, sensitivity, specificity, get_clf_eval
from model import GATR
from seed import set_seed
from tqdm import tqdm
import wandb 


def train(model, optimizer, criterion, train_loader, epoch, num_epochs):
    model.train()
    train_epoch_cost = 0
    train_out_stack = []; train_label_stack = []


    for batch_idx, loader in enumerate(train_loader):
        train_label = loader.y
        group_label = loader.g_y
        
        optimizer.zero_grad()
        outputs, cls_outputs = model(loader)
        _, predicted = torch.max(outputs.data, 1)
        _, cls_predicted = torch.max(cls_outputs.data, 1)

        # binary loss
        loss = criterion(outputs, train_label)

        # group loss
        g_loss = criterion(cls_outputs, group_label)

        # total_loss = loss + lambda_ * g_loss
        total_loss = loss + 0.1* g_loss
        train_epoch_cost += total_loss.item()

        total_loss.backward()
        optimizer.step()

        # final
        train_label_stack.extend(train_label.cpu().detach().tolist())
        train_out_stack.extend(np.argmax((outputs).cpu().detach().tolist(), axis=1))

    return


def test(model, criterion, test_loader):
    model.eval()
    # total final evaluation
    test_epoch_cost = 0
    test_out_stack = []; test_label_stack = []

    with torch.no_grad():
        for batch_idx, loader in enumerate(test_loader):
            test_label = loader.y
            outputs, _ = model(loader)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, test_label)
            test_epoch_cost += loss.item()
            test_label_stack.extend(test_label.cpu().detach().tolist())
            test_out_stack.extend(np.argmax((outputs).cpu().detach().tolist(), axis=1))

        acc = accuracy(test_out_stack, test_label_stack),
        loss = test_epoch_cost / len(test_loader),
        sen = sensitivity(test_out_stack, test_label_stack),
        spe = specificity(test_out_stack, test_label_stack),
        _, _, _, f1, _ = get_clf_eval(test_label_stack,test_out_stack)

    return loss[0], acc[0], sen[0], spe[0], f1



def trainer(args, seed, train_x, test_x, train_y, train_g_y, test_y):

    # define model
    # model = GIN(args, 128).to(args.device)  #  GAT, GCN, ChebNet
    model = GATR(args, 128, 2, 1).to(args.device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    train_loader, test_loader = augment_data_loader(args, train_x, train_y, train_g_y, test_x, test_y)

    # training
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        train(model, optimizer, criterion, train_loader, epoch, args.num_epochs)
    
    torch.save(model.state_dict(), f"./weight/model_{args.detail}.pt")

    return



def main(seed):

    ## config
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a GNN')

    # pytorch base
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda 0, 1
    parser.add_argument("--device", default=device)

    # training hyperparameter
    # parser.add_argument('--seed', type=int, default=500, help='random seed')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') # 1e-5, 5e-6
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=1000, type=int, help='num_epochs')
    parser.add_argument('--optim', default="Adam", type=str, help='optimizer')
    parser.add_argument('--betas', default=(0.5, 0.9), type=tuple, help='adam betas')
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum - SGD, MADGRAD")
    parser.add_argument("--gamma", default=0.995, type=float, help="gamma for lr learning")
    parser.add_argument("--info_gain_weight", default=[0.522, 0.478], type=list, help="info gain weight")

    # GNN parameter
    parser.add_argument("--embCh", default="[200, 128, 64]", type=str, help="")  # gnn channel
    parser.add_argument("--numROI", default=200, type=int, help="cc200")  # number of RoI
    parser.add_argument("--p_value", default=0.1, type=float, help="topology ratio")  # p_value
    parser.add_argument("--dropout_ratio", default=0.5, type=float)
    # directory path
    parser.add_argument('--root_dir', default="Data/", type=str, help='Download dir')
    parser.add_argument('--grl', default=1.0, type=float, help='grl ratio')
    parser.add_argument('--detail', default="multi_site + grl + augmentation except original + weighted loss", type=str, help='experiment detail')

    args = parser.parse_args('')
    print('Arguments: \n', args)

    args.embCh = list(map(int, re.findall("\d+", str(args.embCh))))

    print(f"seed: {seed}")
    set_seed(seed)

    ## Load dataset
    NYU_site_data = np.load(f'{args.root_dir}NYU_site_data.npy')
    NYU_site_label = np.load(f'{args.root_dir}NYU_site_label.npy')

    PITT_site_data = np.load(f'{args.root_dir}PITT_site_data.npy')
    PITT_site_label = np.load(f'{args.root_dir}PITT_site_label.npy')
    PITT_site_group_label = np.ones((PITT_site_label.shape[0], 1))

    UM_1_site_data = np.load(f'{args.root_dir}UM_1_site_data.npy')
    UM_1_site_label = np.load(f'{args.root_dir}UM_1_site_label.npy')
    UM_1_site_group_label = np.full((UM_1_site_label.shape[0], 1), 2)

    USM_site_data = np.load(f'{args.root_dir}USM_site_data.npy')
    USM_site_label = np.load(f'{args.root_dir}USM_site_label.npy')
    USM_site_group_label = np.full((USM_site_label.shape[0], 1), 3)


    YALE_site_data = np.load(f'{args.root_dir}YALE_site_data.npy')
    YALE_site_label = np.load(f'{args.root_dir}YALE_site_label.npy')
    YALE_site_group_label = np.full((YALE_site_label.shape[0], 1), 4)

    UCLA_1_site_data = np.load(f'{args.root_dir}UCLA_1_site_data.npy')
    UCLA_1_site_label = np.load(f'{args.root_dir}UCLA_1_site_label.npy')
    UCLA_1_site_group_label = np.full((UCLA_1_site_label.shape[0], 1), 5)

    # To seperate train & test: only use NYU test
    NYU_train, NYU_test, NYU_train_label, NYU_test_label = train_test_split(NYU_site_data, NYU_site_label, test_size=0.3, random_state=seed)  # 7:3
    NYU_site_group_label = np.zeros((NYU_train_label.shape[0], 1))


    data_list = np.concatenate((PITT_site_data, UM_1_site_data, USM_site_data, YALE_site_data, UCLA_1_site_data, NYU_train), axis=0)
    label_list = np.concatenate((PITT_site_label, UM_1_site_label, USM_site_label, YALE_site_label, UCLA_1_site_label, NYU_train_label), axis=0)
    group_list = np.concatenate((PITT_site_group_label, UM_1_site_group_label, USM_site_group_label, YALE_site_group_label, UCLA_1_site_group_label, NYU_site_group_label), axis=0)

    # 데이터를 섞기 위한 랜덤 인덱스 생성
    indices = np.random.permutation(data_list.shape[0])

    # 생성된 인덱스를 이용하여 train_x와 train_label 데이터를 섞어줌
    train_x = data_list[indices]
    train_label = label_list[indices]
    train_label = np.squeeze(train_label)

    train_group = group_list[indices]
    train_group = np.squeeze(train_group)

    trainer(args, seed, train_x, NYU_test, train_label, train_group, NYU_test_label)
    
    return

if __name__ == "__main__":
    total_result = [[0 for j in range(5)] for i in range(4)]


    seed = 96
    
    main(seed)


