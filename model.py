import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, GATConv, GINConv

class GIN(torch.nn.Module):
    def __init__(self, args, hid):
        super(GIN, self).__init__()
        self.args = args
        self.hid = hid

        mlp1 = nn.Sequential(
            nn.Linear(args.numROI, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.hid)
        )
        self.conv1 = GINConv(mlp1)
        mlp2 = nn.Sequential(
            nn.Linear(self.hid, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.hid)
        )
        self.conv2 = GINConv(mlp2)

        self.fc1 = nn.Linear(args.numROI*self.hid, args.numROI)
        self.bn1 = nn.BatchNorm1d(num_features=args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)
        self.dropout = nn.Dropout(p=args.dropout_ratio)

        self.domain_classifier = nn.Sequential(
            nn.Linear(args.numROI, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        out = x.reshape(-1, self.args.numROI*self.hid)
        out = F.dropout(F.relu(self.bn1(self.fc1(out))), p=self.args.dropout_ratio)
        logits = self.fc2(out)

        # rev_x = grad_reverse(out, 1.0)
        # cls_logits = self.domain_classifier(rev_x)

        return logits#, cls_logits

# GAT
class GAT(torch.nn.Module):
    def __init__(self, args, hid, in_head, out_head):
        super(GAT, self).__init__()
        self.args = args
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head

        self.conv1 = GATConv(args.numROI, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, self.hid * self.in_head, concat=False,
                             heads=self.out_head, dropout=0.6)

        self.fc1 = nn.Linear(args.numROI*self.hid * self.in_head, args.numROI)
        self.bn1 = nn.BatchNorm1d(num_features=args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)
        self.dropout = nn.Dropout(p=args.dropout_ratio)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        out = x.reshape(-1, self.args.numROI*self.hid*self.in_head)
        out = F.dropout(F.elu(self.bn1(self.fc1(out))), p=self.args.dropout_ratio)
        logits = self.fc2(out)

        return logits

# GCN
class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.numROI, args.embCh[0])
        self.conv2 = GCNConv(args.embCh[0], args.embCh[1])

        # readout : MLP
        self.read_out_mlp = nn.Linear(args.embCh[1], 8)
        self.fc1 = nn.Linear(8 * args.numROI, args.numROI)
        self.bn1 = nn.BatchNorm1d(args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)

    def forward(self, load):
        x, edge, attr, batch = load.x, load.edge_index, load.edge_attr, load.batch

        x = F.mish(self.conv1(x, edge, attr))
        x = F.mish(self.conv2(x, edge, attr))

        out = F.mish(self.read_out_mlp(x))
        out_features = out.reshape(-1, self.args.numROI*8)

        logits = self.fc2(F.mish(self.bn1(self.fc1(out_features))))

        return logits
    
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

# GATR
class GATR(torch.nn.Module):
    def __init__(self, args, hid, in_head, out_head):
        super(GATR, self).__init__()
        self.args = args
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head

        self.conv1 = GATConv(args.numROI, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, self.hid * self.in_head, concat=False,
                             heads=self.out_head, dropout=0.6)

        self.fc1 = nn.Linear(args.numROI*self.hid * self.in_head, args.numROI)
        self.bn1 = nn.BatchNorm1d(num_features=args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)
        self.dropout = nn.Dropout(p=args.dropout_ratio)

        self.domain_classifier = nn.Sequential(
            nn.Linear(args.numROI, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        out = x.reshape(-1, self.args.numROI*self.hid*self.in_head)
        out = F.dropout(F.elu(self.bn1(self.fc1(out))), p=self.args.dropout_ratio)
        logits = self.fc2(out)

        rev_x = grad_reverse(out, 1.0)
        cls_logits = self.domain_classifier(rev_x)

        return logits, cls_logits, out

# GCN
class GCNR(nn.Module):
    def __init__(self, args):
        super(GCNR, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.numROI, args.embCh[0])
        self.conv2 = GCNConv(args.embCh[0], args.embCh[1])

        # readout : MLP
        self.read_out_mlp = nn.Linear(args.embCh[1], 8)
        self.fc1 = nn.Linear(8 * args.numROI, args.numROI)
        self.bn1 = nn.BatchNorm1d(args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)

        self.domain_classifier = nn.Sequential(
            nn.Linear(args.numROI, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )


    def forward(self, load):
        x, edge, attr, batch = load.x, load.edge_index, load.edge_attr, load.batch

        x = F.mish(self.conv1(x, edge, attr))
        x = F.mish(self.conv2(x, edge, attr))
        out = F.mish(self.read_out_mlp(x))
        out_features = out.reshape(-1, self.args.numROI*8)

        middle_logits = F.mish(self.bn1(self.fc1(out_features)))
        logits = self.fc2(middle_logits)

        rev_x = grad_reverse(middle_logits, 1.0)
        cls_logits = self.domain_classifier(rev_x)

        return logits, cls_logits

class ChebyNetR(nn.Module):
    def __init__(self, args):
        super(ChebyNetR, self).__init__()
        self.args = args
        self.numROI = args.numROI
        self.conv1 = ChebConv(args.numROI, args.embCh[0], K=2)
        self.conv2 = ChebConv(args.embCh[0], args.embCh[1], K=2)

        # readout : MLP
        self.read_out_mlp = nn.Linear(args.embCh[1], 8)
        self.fc1 = nn.Linear(8 * args.numROI, args.numROI)
        self.bn1 = nn.BatchNorm1d(args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)

        self.domain_classifier = nn.Sequential(
            nn.Linear(args.numROI, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, load):
        x, edge, attr, batch = load.x, load.edge_index, load.edge_attr, load.batch

        x = F.mish(self.conv1(x, edge, attr))
        x = F.mish(self.conv2(x, edge, attr))
        out = F.mish(self.read_out_mlp(x))
        out_features = out.reshape(-1, self.args.numROI*8)

        middle_logits = F.mish(self.bn1(self.fc1(out_features)))
        logits = self.fc2(middle_logits)

        rev_x = grad_reverse(middle_logits, 1.0)
        cls_logits = self.domain_classifier(rev_x)

        return logits, cls_logits


# ChebyNet
class ChebyNet(nn.Module):
    def __init__(self, args):
        super(ChebyNet, self).__init__()
        self.args = args
        self.numROI = args.numROI
        self.conv1 = ChebConv(args.numROI, args.embCh[0], K=2)
        self.conv2 = ChebConv(args.embCh[0], args.embCh[1], K=2)

        # readout : MLP
        self.read_out_mlp = nn.Linear(args.embCh[1], 8)
        self.fc1 = nn.Linear(8 * args.numROI, args.numROI)
        self.bn1 = nn.BatchNorm1d(args.numROI)
        self.fc2 = nn.Linear(args.numROI, 2)

    def forward(self, load):
        x, edge, attr, batch = load.x, load.edge_index, load.edge_attr, load.batch

        x = F.mish(self.conv1(x, edge, attr))
        x = F.mish(self.conv2(x, edge, attr))
        out = F.mish(self.read_out_mlp(x))
        out_features = out.reshape(-1, self.args.numROI*8)
        logits = self.fc2(F.mish(self.bn1(self.fc1(out_features))))

        return logits

