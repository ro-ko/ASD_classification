import numpy as np
import torch
from scipy import stats

# input data is npy file
def flatten_fc(data):
    if isinstance(data, torch.Tensor):
        data = data.unsqueeze(0) if int(data.ndim) == 2 else data
    # here
    else:
        # (122, 200, 200)
        # if data is 2D, add batch dimension
        data = np.expand_dims(data, axis=0) if int(data.ndim) == 2 else data
    # upper triangular indices of the matrix excluding the diagonal
    # x : [0, 0, 0, ...], y : [1, 2, 3, ...]
    x, y = np.triu_indices(data.shape[1], k=1)
    # flatten the data to 2D array
    # data : from row 1 to row 122, 200x200 matrix -> FC_flatten : from row 1 to row 122, 19900 column
    FC_flatten = data[:, x, y]

    return FC_flatten


def flatten2dense(flatten, ROI):
    if isinstance(flatten, torch.Tensor):
        flatten = flatten.unsqueeze(0) if int(flatten.ndim) == 1 else flatten
    else:
        # here
        # (19900,) -> (1, 19900)
        flatten = np.expand_dims(flatten, axis=0) if int(flatten.ndim) == 1 else flatten

    B, F = flatten.shape
    # (200, 200)
    # upper triangular indices of the matrix excluding the diagonal
    x, y = np.triu_indices(ROI, k=1)
    if isinstance(flatten, torch.Tensor):
        sym = torch.zeros([B, ROI, ROI], device=flatten.device)
        sym[:, x, y] = flatten
        sym = sym + torch.transpose(sym, -1, -2)
    else:
        # here
        # (1, 200, 200)
        sym = np.zeros((B, ROI, ROI))
        sym[:, x, y] = flatten
        sym = sym + np.transpose(sym, (0, 2, 1))
    return sym

# t-test for pValueMasking
def t_test(args, ROI=200, train_data=None, train_label=None):
    # (122, 200, 200) -> (122, 19900)
    flatten = flatten_fc(train_data)
    # 1 : ASD vs 0 : TC
    ASDflatten = flatten[np.squeeze(train_label) == 1]
    TCflatten = flatten[np.squeeze(train_label) == 0]
    # welch t-test (19900,)
    p_value = stats.ttest_ind(ASDflatten, TCflatten, equal_var=False).pvalue
    # p_value : (1, 200, 200), diagonal is 0
    p_value = flatten2dense(p_value, ROI)
    # p_value가 큰 값은 0으로 masking: p_value가 낮을수록 유의미한 연결관계
    p_value[p_value > args.p_value] = 0
    p_value[p_value!=0] = 1
    print(f"Edge ratio: {round((p_value.sum() / flatten.shape[1])*100, 5)}%")

    return p_value

# edge topology mask
def pValueMasking(feature, t_test, p_value, binary=False, abs=True):
    data = feature.copy()
    # t_test : topology (1, 200, 200), p_value : 0.1
    mask = t_test <= p_value
    # true
    if abs:
        data = np.abs(data)
    # masking
    data = data * mask
    # true
    if binary:
        data[data != 0] = 1

    return data

# define edge
def define_node_edge(train_data, test_data, t, p_value, edge_binary, node_binary, edge_abs, node_abs, node_mask=False):
    train_static_edge = pValueMasking(feature=train_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)
    test_static_edge = pValueMasking(feature=test_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)

    return train_static_edge, test_static_edge
