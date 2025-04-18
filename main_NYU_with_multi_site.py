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

from data_preprocessing import data_loader
from metric import accuracy, sensitivity, specificity, get_clf_eval
from model import GAT, GCN, ChebyNet, GIN
from seed import set_seed
from tqdm import tqdm
import wandb 


def train(model, optimizer, criterion, train_loader):
    model.train()
    train_epoch_cost = 0
    train_out_stack = []; train_label_stack = []

    for batch_idx, loader in enumerate(train_loader):
        train_label = loader.y
        optimizer.zero_grad()
        outputs = model(loader)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, train_label)
        train_epoch_cost += loss.item()
        loss.backward()
        optimizer.step()

        # final
        train_label_stack.extend(train_label.cpu().detach().tolist())
        train_out_stack.extend(np.argmax((outputs).cpu().detach().tolist(), axis=1))

    acc = accuracy(train_out_stack, train_label_stack),
    loss = train_epoch_cost / len(train_loader),
    sen = sensitivity(train_out_stack, train_label_stack),
    spe = specificity(train_out_stack, train_label_stack),
    _, _, _, f1, _ = get_clf_eval(train_label_stack, train_out_stack)

    return loss[0], acc[0], sen[0], spe[0], f1


def test(model, criterion, test_loader):
    model.eval()
    # total final evaluation
    test_epoch_cost = 0
    test_out_stack = []; test_label_stack = []

    with torch.no_grad():
        for batch_idx, loader in enumerate(test_loader):
            test_label = loader.y
            outputs = model(loader)
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



def trainer(args, seed, train_x, test_x, train_y, test_y):

    # define model
    model = GAT(args, 128, 2, 1).to(args.device)  #  GAT, GCN, ChebNet
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    train_loader, test_loader = data_loader(args, train_x, train_y, test_x, test_y)

    # save log
    if not (os.path.isdir(f'{args.root_dir}results/model{args.timestamp}')):
        os.makedirs(os.path.join(f'{args.root_dir}results/model{args.timestamp}'))
    path_save_info = f'{args.root_dir}results/model{args.timestamp}' + os.path.sep + f'train_info{args.timestamp}_{seed}.csv'
    # train
    with open(path_save_info.replace(".csv", "_train.csv"), "w") as f:
        f.write("epoch, loss, acc, sen, spec, F1\n")
    # test
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("epoch, loss, acc, sen, spec, F1\n")

    # save parameter log
    with open(path_save_info.replace(".csv", "info.txt"), "w") as f:
        f.write(f'model: {str(model)}\n\n Parameters : {args}\n\n Optimizer C : {optimizer}\n\n ')

    # training
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        train_loss, train_acc, train_sen, train_spe, train_f1 = train(model, optimizer, criterion, train_loader)
        test_loss, test_acc, test_sen, test_spe, test_f1 = test(model, criterion, test_loader)
       
        wandb.log({f"train_loss_{seed}": train_loss, f"train_acc_{seed}": train_acc, f"train_sen_{seed}": train_sen, f"train_spe_{seed}": train_spe, f"train_f1_{seed}": train_f1,
                    f"test_loss_{seed}": test_loss, f"test_acc_{seed}": test_acc, f"test_sen_{seed}": test_sen, f"test_spe_{seed}": test_spe, f"test_f1_{seed}": test_f1})

        # save results
        with open(path_save_info.replace(".csv", "_train.csv"), "a") as f:
            f.write('{},{},{},{},{},{}\n'.format(epoch, train_loss, train_acc, train_sen, train_spe, train_f1))

        with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
            f.write('{},{},{},{},{},{}\n'.format(epoch, test_loss, test_acc, test_sen, test_spe, test_f1))


    return train_loss, train_acc, train_sen, train_spe, train_f1, \
           test_loss, test_acc, test_sen, test_spe, test_f1


def main(seed, timestamp):

    ## config
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a GNN')

    # pytorch base
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # cuda 0, 1
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
    parser.add_argument("--timestamp", default=timestamp, type=str, help="")
    parser.add_argument('--root_dir', default="Data/", type=str, help='Download dir')

    args = parser.parse_args('')
    print('Arguments: \n', args)

    wandb.init(project="ASD_classification", name=f"NYU_multi_site_{args.timestamp}")
    wandb.config.update(args)

    args.embCh = list(map(int, re.findall("\d+", str(args.embCh))))

    print(f"seed: {seed}")
    set_seed(seed)

    ## Load dataset
    NYU_site_data = np.load(f'{args.root_dir}NYU_site_data.npy')
    NYU_site_label = np.load(f'{args.root_dir}NYU_site_label.npy')
    PITT_site_data = np.load(f'{args.root_dir}PITT_site_data.npy')
    PITT_site_label = np.load(f'{args.root_dir}PITT_site_label.npy')
    UM_1_site_data = np.load(f'{args.root_dir}UM_1_site_data.npy')
    UM_1_site_label = np.load(f'{args.root_dir}UM_1_site_label.npy')
    USM_site_data = np.load(f'{args.root_dir}USM_site_data.npy')
    USM_site_label = np.load(f'{args.root_dir}USM_site_label.npy')
    YALE_site_data = np.load(f'{args.root_dir}YALE_site_data.npy')
    YALE_site_label = np.load(f'{args.root_dir}YALE_site_label.npy')
    UCLA_1_site_data = np.load(f'{args.root_dir}UCLA_1_site_data.npy')
    UCLA_1_site_label = np.load(f'{args.root_dir}UCLA_1_site_label.npy')

    # To seperate train & test: only use NYU test
    NYU_train, NYU_test, NYU_train_label, NYU_test_label = train_test_split(NYU_site_data, NYU_site_label, test_size=0.3, random_state=seed)  # 7:3

    data_list = np.concatenate((PITT_site_data, UM_1_site_data, USM_site_data, YALE_site_data, UCLA_1_site_data, NYU_train), axis=0)
    label_list = np.concatenate((PITT_site_label, UM_1_site_label, USM_site_label, YALE_site_label, UCLA_1_site_label, NYU_train_label), axis=0)

    # 데이터를 섞기 위한 랜덤 인덱스 생성
    indices = np.random.permutation(data_list.shape[0])

    # 생성된 인덱스를 이용하여 train_x와 train_label 데이터를 섞어줌
    train_x = data_list[indices]
    train_label = label_list[indices]
    train_label = np.squeeze(train_label)

    train_loss, train_acc, train_sen, train_spe, train_f1, \
    test_loss, test_acc, test_sen, test_spe, test_f1 = trainer(args, seed, train_x, NYU_test, train_label, NYU_test_label)

    return args, test_loss, test_acc, test_sen, test_spe, test_f1

if __name__ == "__main__":
    total_result = [[0 for j in range(5)] for i in range(4)]


    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    seed_list = [42, 56, 96, 100, 777]
    for s in seed_list:
        args, test_loss, test_acc, test_sen, test_spe, test_f1 = main(s, timestamp)


    # save seed initialization
    with open(f'{args.root_dir}results/model{args.timestamp}/all_seed_result{args.timestamp}.csv', 'w') as f:
        f.write('fold,acc,sen,spec,f1\n')


    n_seed = 0
    for s in seed_list:
        # load testset result
        result_csv = pd.read_csv(f'{args.root_dir}results/model{args.timestamp}/train_info{args.timestamp}_{s}_test.csv')
        # get latest result
        test_result = result_csv.iloc[-1]
        # seed, acc, sen, spe, f1
        with open(f'{args.root_dir}results/model{args.timestamp}/all_seed_result{args.timestamp}.csv', 'a') as f:
            # round to 3 decimal places
            f.write(f'{s}, \
                    {round(test_result.iloc[2], 3) * 100}, \
                    {round(test_result.iloc[3], 3) * 100}, \
                    {round(test_result.iloc[4], 3) * 100}, \
                    {round(test_result.iloc[5], 3) * 100}\n')       
            
        total_result[0][n_seed] = round(test_result.iloc[2], 3)
        total_result[1][n_seed] = round(test_result.iloc[3], 3)
        total_result[2][n_seed] = round(test_result.iloc[4], 3)
        total_result[3][n_seed] = round(test_result.iloc[5], 3)
        n_seed += 1

        print(f"seed{s},{round(test_result.iloc[2], 3):.2f},{round(test_result.iloc[3], 3):.2f},{round(test_result.iloc[4], 3):.2f},{round(test_result.iloc[5], 3):.2f}")

    with open('Data/results/model{}/all_seed_result{}.csv'.format(args.timestamp, args.timestamp), 'a') as f:
        f.write(f'avg, \
                {np.mean(total_result[0]) * 100:.2f}, \
                {np.mean(total_result[1]) * 100:.2f}, \
                {np.mean(total_result[2]) * 100:.2f}, \
                {np.mean(total_result[3]) * 100:.2f}\n')
        f.write(f'std, \
                {np.nanstd(total_result[0], ddof=1) * 100:.2f}, \
                {np.nanstd(total_result[1], ddof=1) * 100:.2f}, \
                {np.nanstd(total_result[2], ddof=1) * 100:.2f}, \
                {np.nanstd(total_result[3], ddof=1) * 100:.2f}')
    wandb.log({"avg_acc": np.mean(total_result[0]) * 100, "avg_sen": np.mean(total_result[1]) * 100, "avg_spe": np.mean(total_result[2]) * 100, "avg_f1": np.mean(total_result[3]) * 100})
    wandb.log({"std_acc": np.nanstd(total_result[0], ddof=1) * 100, "std_sen": np.nanstd(total_result[1], ddof=1) * 100, "std_spe": np.nanstd(total_result[2], ddof=1) * 100, "std_f1": np.nanstd(total_result[3], ddof=1) * 100})
    wandb.finish()

