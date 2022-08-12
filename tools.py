import networkx as nx
import pickle
import numpy as np
import random
import torch

def load_adj_data(net):
	# synthetic graphs
	if net == 'ws':
	    G = nx.watts_strogatz_graph(1000, 6, 0.1)
	    adj = nx.convert_matrix.to_numpy_array(G)
	elif net == 'cir':
	    G = nx.circulant_graph(1000, [1,3])
	    adj = nx.convert_matrix.to_numpy_array(G)
	elif net == 'ba':
	    G = nx.random_graphs.barabasi_albert_graph(1024, 2)
	    adj = nx.convert_matrix.to_numpy_array(G)
	elif net == 'grid':
	    G = nx.grid_2d_graph(33, 33)
	    adj = nx.convert_matrix.to_numpy_array(G)
	elif net == 'kron':
		adj_address = './data/kron_net.pkl'
		with open(adj_address,'rb') as f:
			adj = pickle.load(f)
	elif net == 'ff':
		f = open('./data/ff_net.txt')
		lines = f.readlines()
		# create graph and add edges
		G = nx.Graph()
		for line in lines:
		    if '#' in line:
		        continue
		    vs = line.strip().split('\t')
		    sv = int(vs[0])
		    ev = int(vs[1])
		    G.add_edge(sv,ev)
		adj = nx.convert_matrix.to_numpy_array(G)

	# empirial graphs
	elif net == 'cora':
		adj_address = './data/seed2051cora270810000-adjmat.pickle'
		with open(adj_address,'rb') as f:
			adj = pickle.load(f)
	elif net == 'bios':
		adj_address = './data/real_net_bio-CE-GT.pkl' #sparse
		with open(adj_address,'rb') as f:
			adj = pickle.load(f)
	elif net == 'biod':
		adj_address = './data/real_net_bio-SC-TS.pkl' #dense
		with open(adj_address,'rb') as f:
			adj = pickle.load(f)
	elif net == 'ca':
		adj_address = './data/real_net_ca-netscience.pkl' #sparse
		with open(adj_address,'rb') as f:
			adj = pickle.load(f)
	return adj


# interrupt adj by repermuting the row an cols
def interrupt_adj(adj,device):
	object_matrix = adj.cpu().numpy()
	noden = object_matrix.shape[0]
	order = np.arange(noden)
	random.shuffle(order)

	mat = np.zeros(object_matrix.shape)
	for i in range(object_matrix.shape[0]):
	    mat[i,order[i]] = 1

	object_matrix = np.matmul(np.matmul(mat,object_matrix),mat.T)
	adj = object_matrix
	adj = torch.from_numpy(adj).long().to(device)
	return adj



# get probability scaling factor by density estimation
def get_ps_by_density(pmat,edge_num,dn,device):
    t = torch.ones(pmat.shape).to(device)
    noden = pmat.shape[0]
    t[:noden-dn,:noden-dn] = 0
    unknownp = torch.sum(pmat*t)
    ps = edge_num / unknownp
    return ps
