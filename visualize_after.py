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
from graph_preprocessing import t_test, define_node_edge,flatten_fc

from data_preprocessing import data_loader, cls_data_loader, augment_data_loader
from metric import accuracy, sensitivity, specificity, get_clf_eval
from model import GATR
from seed import set_seed
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def visualize_tsne(model, dataset, device, batch_size=16, num_groups=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, h = model(batch)           # h: [batch_size, hidden_dim]
            all_embs.append(h.cpu().numpy())
            all_labels.append(batch.g_y.cpu().numpy())

    embs = np.vstack(all_embs)            # [총_샘플, hidden_dim]
    labels = np.hstack(all_labels)        # [총_샘플,]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, random_state=42)
    embs_2d = tsne.fit_transform(embs)    # [총_샘플, 2]

    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("viridis", len(num_groups))
    
    name = ['NYU', 'PITT', 'UM_1', 'USM', 'YALE', 'UCLA_1']

    for g, n in zip(num_groups, name):
        # 해당 그룹의 인덱스 찾기
        idx = np.where(np.array(labels) == g)
        plt.scatter(embs_2d[idx, 0], embs_2d[idx, 1], 
                    c=[cmap(g)], label=f'{n}', s=50)
        
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of Graph-level Features by Group")
    plt.legend()

    # 서버에서 실행하는 경우 plt.show() 대신 파일로 저장
    plt.savefig("./analysis/multi-site-node-distribution_after" ,dpi=300, bbox_inches="tight")
    plt.close()

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

    site_data = [NYU_train, PITT_site_data, UM_1_site_data, USM_site_data, YALE_site_data, UCLA_1_site_data]
    site_label = [NYU_train_label, PITT_site_label, UM_1_site_label, USM_site_label, YALE_site_label, UCLA_1_site_label]

    num_groups = [i for i in range(len(site_data))]

    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=NYU_test, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)


    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)

    train_g_label = torch.LongTensor(train_group).to(args.device)

    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i],  g_y=train_g_label[i], edge_index=edge, edge_attr=attr))

    model = GATR(args, 128, 2, 1).to(args.device)
    model.load_state_dict(torch.load(f"./weight/model_{args.detail}.pt"))

    visualize_tsne(model, train_dataset, args.device, batch_size=args.batch_size, num_groups=num_groups)


    return

if __name__ == "__main__":

    seed = 96
    main(seed)
