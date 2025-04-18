import os, sys
import numpy as np
import csv
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from graph_preprocessing import t_test, define_node_edge, define_each_node_edge

def train_data_loader(args, train_x, train_label, train_g_label):
    # make graph topology with only train data (1, 200, 200)
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    # define edge
    train_static_edge = define_each_node_edge(train_data=train_x, t=topology, p_value=args.p_value, edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    # load train data
    # define node
    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)
    train_g_label = torch.LongTensor(train_g_label).to(args.device)

    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], g_y=train_g_label[i], edge_index=edge, edge_attr=attr))

    return train_dataset

def test_data_loader(args, valid_x, valid_label):

    topology = t_test(args, ROI=args.numROI, train_data=valid_x, train_label=valid_label)

    valid_static_edge = define_each_node_edge(train_data=valid_x, t=topology, p_value=args.p_value, edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)
    
    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))

    # full batch
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))
    
    return validLoader

def data_loader(args, train_x, train_label, valid_x, valid_label):
    # make graph topology with only train data (1, 200, 200)
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    # define edge
    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    # load train data
    # define node
    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)
    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], edge_index=edge, edge_attr=attr))

    if args.batch_size > len(train_dataset):
        trainLoader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    else:
        # here
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    # full batch
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))

    return trainLoader, validLoader

def cls_data_loader(args, train_x, train_label, train_g_label, valid_x, valid_label):
    # make graph topology with only train data (1, 200, 200)
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    # define edge
    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    # load train data
    # define node
    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)

    train_g_label = torch.LongTensor(train_g_label).to(args.device)


    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], g_y=train_g_label[i], edge_index=edge, edge_attr=attr))


    if args.batch_size > len(train_dataset):
        trainLoader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    else:
        # here
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    # full batch
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))

    return trainLoader, validLoader

def augment_data_loader(args, train_x, train_label, train_g_label, valid_x, valid_label):
    # make graph topology with only train data (1, 200, 200)
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    # define edge
    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    # load train data
    # define node
    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)

    train_g_label = torch.LongTensor(train_g_label).to(args.device)


    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], g_y=train_g_label[i], edge_index=edge, edge_attr=attr))

    num_augments = 3
    transform = EdgeDropoutTransform(drop_ratio=0.1)

    augmented_dataset = []
    for d in train_dataset:
        # 원본도 포함
        # augmented_dataset.append(d)
        for _ in range(num_augments):
            # 복제한 후 증강 적용
            d_aug = d.clone()
            d_aug = transform(d_aug)
            augmented_dataset.append(d_aug)

    if args.batch_size > len(augmented_dataset):
        trainLoader = DataLoader(augmented_dataset, batch_size=len(augmented_dataset), shuffle=True, drop_last=True)
    else:
        # here
        trainLoader = DataLoader(augmented_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    # full batch
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))

    return trainLoader, validLoader

def edge_dropout(data, drop_ratio=0.1):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_ratio
    new_edge_index = edge_index[:, mask]
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_edge_attr = data.edge_attr[mask]
        data.edge_attr = new_edge_attr
    data.edge_index = new_edge_index
    return data

class EdgeDropoutTransform(object):
    def __init__(self, drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self, data):
        return edge_dropout(data, self.drop_ratio)
    
def group_data_loader(args, train_x, train_label, train_g_label, valid_x, valid_label):
    # make graph topology with only train data (1, 200, 200)
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    # define edge
    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    # load train data
    # define node
    train_Node_list = torch.FloatTensor(train_x).to(args.device)

    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)

    train_label = torch.LongTensor(train_label).to(args.device)

    train_g_label = torch.LongTensor(train_g_label).to(args.device)


    train_dataset = []

    for i, sub in enumerate(train_A_list):
        # returns edge_index [[x indices], [y indices]] and edge_attr [values]
        edge, attr = dense_to_sparse(sub)

        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], g_y=train_g_label[i], edge_index=edge, edge_attr=attr))


    if args.batch_size > len(train_dataset):
        trainLoader = DataLoader(train_dataset, batch_size=len(augmented_dataset), shuffle=True, drop_last=True)
    else:
        # here
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    # full batch
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))

    return trainLoader, validLoader
