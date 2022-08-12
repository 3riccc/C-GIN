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

# model
# Definition of Encoder
class Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, n_layers,gnn):
        super(Encoder,self).__init__()
        
        self.convs = torch.nn.ModuleList()
        
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        for _ in range(n_layers):
            if gnn == 'gin':
                self.convs.append(GINConv(nn.Linear(hidden_dim,hidden_dim)))
            elif gnn == 'gine':
                self.convs.append(GINEConv(nn.Linear(hidden_dim,hidden_dim)))
            elif gnn == 'gcn':
                self.convs.append(GCNConv(in_channels=hidden_dim,out_channels=hidden_dim))
            elif gnn == 'gat':
                self.convs.append(GATConv(in_channels=hidden_dim,out_channels=hidden_dim))        
            elif gnn == 'tag':
                self.convs.append(TAGConv(in_channels=hidden_dim,out_channels=hidden_dim))        
            elif gnn == 'sgc':
                self.convs.append(SGConv(in_channels=hidden_dim,out_channels=hidden_dim))        
        self.out_proj = torch.nn.Linear((n_layers+1)*hidden_dim, output_dim)

    def forward(self,x,edge_index):
        x = self.in_proj(x)
        hidden_states = [x]
        for layer in self.convs:
            x = layer(x,edge_index)
            hidden_states.append(x)
        x = torch.cat(hidden_states, dim=1)
        x = self.out_proj(x)
        return x

# Definition of Decoder
class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder,self).__init__()
    def forward(self,z):
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj
