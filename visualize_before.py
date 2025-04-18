import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
import torch
from sklearn.model_selection import train_test_split
from seed import set_seed
from graph_preprocessing import t_test, define_node_edge,flatten_fc
from sklearn.manifold import TSNE

from data_preprocessing import data_loader

class Graph:
    def __init__(self, x, group):
        self.x = x  # shape: [num_nodes, feature_dim]
        self.group = group

def main(seed):
    ## config
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a GNN')

    # pytorch base
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # cuda 0, 1
    parser.add_argument("--device", default=device)

    # training hyperparameter
    # parser.add_argument('--seed', type=int, default=500, help='random seed')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
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

    args = parser.parse_args('')
    print('Arguments: \n', args)

    # seed fix
    print(f"seed: {seed}")
    set_seed(seed)

    ## Load dataset
    # 122 subjects on NYU_train
    NYU_site_data = np.load(f'{args.root_dir}NYU_site_data.npy')
    NYU_site_label = np.load(f'{args.root_dir}NYU_site_label.npy')
    # 56 subjects
    PITT_site_data = np.load(f'{args.root_dir}PITT_site_data.npy')
    PITT_site_label = np.load(f'{args.root_dir}PITT_site_label.npy')
    # 106 subjects
    UM_1_site_data = np.load(f'{args.root_dir}UM_1_site_data.npy')
    UM_1_site_label = np.load(f'{args.root_dir}UM_1_site_label.npy')
    # 71 subjects
    USM_site_data = np.load(f'{args.root_dir}USM_site_data.npy')
    USM_site_label = np.load(f'{args.root_dir}USM_site_label.npy')
    # 56 subjects 
    YALE_site_data = np.load(f'{args.root_dir}YALE_site_data.npy')
    YALE_site_label = np.load(f'{args.root_dir}YALE_site_label.npy')
    # 72 subjects
    UCLA_1_site_data = np.load(f'{args.root_dir}UCLA_1_site_data.npy')
    UCLA_1_site_label = np.load(f'{args.root_dir}UCLA_1_site_label.npy')

    NYU_train, NYU_test, NYU_train_label, NYU_test_label = train_test_split(NYU_site_data, NYU_site_label, test_size=0.3, random_state=seed)  # 7:3

    site_data = [NYU_train, PITT_site_data, UM_1_site_data, USM_site_data, YALE_site_data, UCLA_1_site_data]
    site_label = [NYU_train_label, PITT_site_label, UM_1_site_label, USM_site_label, YALE_site_label, UCLA_1_site_label]

    num_groups = [i for i in range(len(site_data))]

    graphs = []


    for idx, data, label in zip(num_groups, site_data, site_label):
        
        topology = t_test(args, ROI=args.numROI, train_data=data, train_label=label)

        train_static_edge, \
        valid_static_edge = define_node_edge(train_data=data, test_data=NYU_test, t=topology, p_value=args.p_value,
                                            edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

        
        train_Node_list = data
        train_A_list = torch.FloatTensor(train_static_edge)
        train_label = label

        output = np.multiply(data,train_static_edge)
        # output = train_static_edge
        
        
        # (x, 200, 200) -> (x, 19900)
        fc_features = flatten_fc(output)


        for g in fc_features:
            graphs.append(Graph(g, group=idx))

    graph_features = []   # 각 그래프의 feature vector 저장
    graph_labels = []     # 각 그래프의 그룹(label) 저장
    for graph in graphs:
        # 평균 풀링: 모든 노드의 피처 평균 (axis=0)
        graph_features.append(graph.x)  # numpy로 변환
        graph_labels.append(graph.group)


    # PCA를 통해 차원을 50으로 축소
    pca = PCA(n_components=50, random_state=42)
    graph_features_pca = pca.fit_transform(graph_features)


    # Step 2: t-SNE로 2차원 임베딩
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, random_state=42)
    graph_features_2d = tsne.fit_transform(graph_features_pca)

    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("viridis", len(num_groups))
    
    name = ['NYU', 'PITT', 'UM_1', 'USM', 'YALE', 'UCLA_1']

    for g, n in zip(num_groups, name):
        # 해당 그룹의 인덱스 찾기
        idx = np.where(np.array(graph_labels) == g)
        plt.scatter(graph_features_2d[idx, 0], graph_features_2d[idx, 1], 
                    c=[cmap(g)], label=f'{n}', s=50)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of Graph-level Features by Group")
    plt.legend()

    # 서버에서 실행하는 경우 plt.show() 대신 파일로 저장
    plt.savefig("./analysis/multi-site-node-distribution_before" ,dpi=300, bbox_inches="tight")
    plt.close()

# start
if __name__ == "__main__":
    # set seed
    seed = 42
    main(seed=seed)