import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch_geometric.nn import GINConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import TAGConv
from torch_geometric.nn import SGConv
import torch_geometric


import networkx as nx
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from tqdm import *
from sgm_utils import part_constructor_evaluator_sgm
from tools import *
from model import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='ca', help="synthetic(ws / cir / ba / grid / kron / ff) or empirical(cora / ca / bios / biod)")
parser.add_argument('--hid_dim', type=int, default=64, help="hidden dim")
parser.add_argument('--out_dim', type=int, default=64, help="output dim")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-5, help="weight decay")
parser.add_argument('--missing', type=float, default=0.2, help="missing percent")
parser.add_argument('--cuda', type=int, default=0, help="cuda")
parser.add_argument('--epoch', type=int, default=15000, help="weight decay")
parser.add_argument('--ps', type=float, default=0.02, help="probability scale")
parser.add_argument('--si', type=int, default=200, help="sample interval")
parser.add_argument('--seed', type=int, default=2050, help='Random seed.')
parser.add_argument('--gnn', type=str, default='gin', help='which kind of gnn to use')
parser.add_argument('--note', type=str, default='test', help="not for experiments")
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# hyper para configuration
HID_DIM = args.hid_dim
OUT_DIM = args.hid_dim
LR = args.lr
WD = args.wd
NET = args.net

# config cuda
dev_id = args.cuda
device = torch.device("cuda:"+str(dev_id) if torch.cuda.is_available() else "cpu")


# load data
adj = load_adj_data(NET)
adj = torch.from_numpy(adj).long().to(device)

# interrupt the adj by shuffing the order of nodes
interrupt = 1
adj = interrupt_adj(adj,device)

# delete last nodes
dn = int(adj.shape[0]*args.missing) # del number
noden = adj.shape[0] # node number
temp = np.zeros(adj.shape)
temp[:noden-dn,:noden-dn] = 1
adj_train = adj*torch.from_numpy(temp).to(device)
adj_train = adj_train.long()

# estimate edge num in unknown part by density
a1 = (noden-dn)**2 # known part area size
a2 = noden**2 - (noden-dn)**2 # unknown part area size
e1 = torch.sum(adj_train) #known part edge number
e2 = e1/a1*a2

# model
FEA_DIM = adj.shape[0]
encoder = Encoder(input_dim = FEA_DIM, hidden_dim = HID_DIM, output_dim = OUT_DIM, n_layers = 3,gnn=args.gnn).to(device)
decoder = InnerProductDecoder()

# optimizer
optimizer = optim.Adam(encoder.parameters(), lr=LR,weight_decay=WD)

# loss function
n1 = torch.sum(adj_train)
n0 = (noden-dn)**2
loss_func = nn.NLLLoss(ignore_index=-1,weight = torch.Tensor([n1/n0,n0/n1,0.])).to(device)

# initial features
feature = torch.eye(adj.shape[0]).to(device)

# recording the training process
losses = []
aucs_all = []
aucs_knun = []
aucs_unun = []
# auc baseline
aucs_allb = []
aucs_knunb = []
aucs_ununb = []

# start training
adj_used = adj_train.long()
edge_index = torch_geometric.utils.dense_to_sparse(adj_used)[0].long().to(device)

# training loop
for e in range(args.epoch):
    # encode and decode
    hid_embd = encoder(feature, edge_index)
    padj = decoder(hid_embd)

    # prediction and target
    padj_part = padj[:noden-dn,:noden-dn].reshape(-1).unsqueeze(1)
    pred = torch.cat([1-padj_part,padj_part,torch.ones(padj_part.shape).to(device)*0.001],dim=1)
    target = torch.triu(adj[:noden-dn,:noden-dn]) - torch.triu(torch.ones(adj[:noden-dn,:noden-dn].shape)).t().to(device) # make diagnal and lower -1
    eps = 1e-6
    target = target.reshape(-1)

    # loss caculation and gradicent decent
    optimizer.zero_grad()
    loss = loss_func(torch.log(pred+eps),target.long().view(-1))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    
    if e % 100 == 0:
        print(e,'/',args.epoch)
        print('loss:',loss.item())
        # metrics
        auu,aku,aall,apuu,apku,apall = part_constructor_evaluator_sgm(padj.cpu().detach(),1,adj.cpu().float(),noden,dn)
        # metrics
        print("auc all:", aall)
        print("auc knun:", aku)
        print("auc unun:", auu)
        print("ap all:", apall)
        print("ap knun:", apku)
        print("ap unun:", apuu)

        # baseline model: random guess
        padj1 = decoder(torch.randn(hid_embd.shape))
        # metrics of baseline model
        auub,akub,aallb,apuub,apkub,apallb = part_constructor_evaluator_sgm(padj1,1,adj.cpu().float(),noden,dn)
        print("auc all baseline:", aallb)
        print("auc knun baseline:", akub)
        print("auc unun baseline:", auub)
        print("ap all baseline:", apallb)
        print("ap knun baseline:", apkub)
        print("ap unun baseline:", apuub)

    # update the estimated matrix
    if e % args.si == 0 and e > 0:
        # get probability scaling factor
        ps = get_ps_by_density(padj,e2,dn,device)
        print('probability scale factor:',ps.item())
        pmat = padj*ps
        sam_mat = (torch.sign(pmat - torch.Tensor(pmat.shape).uniform_(0,1).to(device))+1)/2
        
        # update the adj
        temp0 = torch.ones(pmat.shape).to(device)
        temp1 = torch.zeros(pmat.shape).to(device)
        temp0[:noden-dn,:noden-dn] = 0
        temp1[:noden-dn,:noden-dn] = 1
        adj_updated = sam_mat * temp0 + adj_train * temp1
        adj_updated = adj_updated * (1-torch.eye(adj_updated.shape[0]).to(device)).long()
        adj_used = adj_updated
        edge_index = torch_geometric.utils.dense_to_sparse(adj_used)[0].long()
