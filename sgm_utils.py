import torch
import numpy as np
import pickle

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc
from sklearn.metrics import average_precision_score

# 已知和未知部分进行标记
def partial_mask(sz,del_num):
    '''mask the known part and un know part'''
    kn_mask = torch.zeros(sz,sz)
    kn_mask[:-del_num,:-del_num] = 1
    
    un_un_mask = torch.zeros(sz,sz)
    un_un_mask[-del_num:,-del_num:] = 1
    un_un_mask = un_un_mask - torch.diag(torch.diag(un_un_mask))
    
    left_mask = torch.ones(sz,sz)
    left_mask[:-del_num,:-del_num] = 0
    left_mask = left_mask-torch.diag(torch.diag(left_mask))
    kn_un_mask = left_mask - un_un_mask
    
    return kn_mask,left_mask,un_un_mask,kn_un_mask



def get_equal_mask(adj_true, test_mask, thresh=0):
    """create a mask which gives equal number of positive and negtive edges"""
    adj_true = adj_true > thresh
    pos_link_mask = adj_true * test_mask
    num_links = int(pos_link_mask.sum().item())

    if num_links > 0.5 * test_mask.sum().item():
        raise ValueError('test nodes over connected!')

    neg_link_mask = (~adj_true) * test_mask
    neg_link_mask = neg_link_mask.numpy()
    row, col = np.where(neg_link_mask > 0)
    new_idx = np.random.permutation(len(row))
    row, col = row[new_idx][:num_links], col[new_idx][:num_links]
    neg_link_mask *= 0
    neg_link_mask[row, col] = 1
    neg_link_mask = torch.from_numpy(neg_link_mask)

    assert((pos_link_mask * neg_link_mask).sum().item() == 0)
    assert(neg_link_mask.sum().item() == num_links)
    assert(((pos_link_mask + neg_link_mask) * test_mask != (pos_link_mask + neg_link_mask)).sum().item() == 0)
    return pos_link_mask + neg_link_mask


def cal_auc(pre,true_adj,un_mask):
    # print(1,un_mask)
    pre_un = pre[un_mask.bool()].cpu().detach().numpy()
    true_un = true_adj[un_mask.bool()].cpu().detach().numpy()
    fpr,tpr,threshold = roc_curve(true_un,pre_un)
    roc_auc = auc(fpr,tpr)
    return roc_auc
def aucs(pre_adj,object_matrix,un_mask,un_un_mask,kn_un_mask,kn_mask=[]):
    roc_auc = cal_auc(pre_adj,object_matrix,un_mask)   
    un_un_roc_auc = cal_auc(pre_adj,object_matrix,un_un_mask)     
    kn_un_roc_auc = cal_auc(pre_adj,object_matrix,kn_un_mask)
    if len(kn_mask) != 0:
        kn_roc_auc = cal_auc(pre_adj,object_matrix,kn_mask)
        return roc_auc,un_un_roc_auc,kn_un_roc_auc,kn_roc_auc
    return roc_auc,un_un_roc_auc,kn_un_roc_auc

def cal_ap(pre,true_adj,un_mask):
    pre_un = pre[un_mask.bool()].cpu().detach().numpy()
    true_un = true_adj[un_mask.bool()].cpu().detach().numpy()
    ap = average_precision_score(true_un,pre_un)
    return ap
def aps(pre_adj,object_matrix,un_mask,un_un_mask,kn_un_mask,kn_mask=[]):
    all_ap = cal_ap(pre_adj,object_matrix,un_mask)   
    un_un_ap = cal_ap(pre_adj,object_matrix,un_un_mask)     
    kn_un_ap = cal_ap(pre_adj,object_matrix,kn_un_mask)
    if len(kn_mask) != 0:
        kn_ap = cal_ap(pre_adj,object_matrix,kn_mask)
        return all_ap,un_un_ap,kn_un_ap,kn_ap
    return all_ap,un_un_ap,kn_un_ap




def sgraphmatch(A,B,m,iteration):
    # m,seed 节点的个数 iteration 迭代的个数   
    
    totv = A.shape[0]
    n = totv-m
    start = torch.ones(n,n)*(1/n)

    if m!= 0:
        # 标识未知与已知
        A12 = A[:m,m:totv]
        A21 = A[m:totv,:m]
        B12 = B[:m,m:totv]
        B21 = B[m:totv,:m]
    
    if m == 0:
        A12 = A21 = B12 = B21 = torch.zeros_like(n,n)
    
    if n==1:
        A12 = A12.T
        A21 = A21.T
        B12 = B12.T
        B21 = B21.T

    # 标识 未知与未知
    A22 = A[m:totv,m:totv]
    B22 = B[m:totv,m:totv]

    tol = 1

    patience = iteration
    P = start  #start 是初始选择的节点

    toggle = 1
    iter = 0

#     print(A21)
#     print(B21)
#     d()
    x = torch.mm(A21,B21.T)
    y = torch.mm(A12.T,B12)

    while (toggle == 1 and iter < patience):

        iter = iter + 1

        z = torch.mm(torch.mm(A22,P),B22.T)
        w = torch.mm(torch.mm(A22.T,P),B22)
        Grad = x + y + z + w  # 目标函数关于P 的一阶导
        
        mm = abs(Grad).max() 
        
        
        obj = Grad+torch.ones([n,n])*mm

        _,ind = linear_sum_assignment(-obj.cpu())

        Tt = torch.eye(n)
        Tt = Tt[ind] # 按照ind 的顺序排列矩阵
        
        wt = torch.mm(torch.mm(A22.T,Tt),B22)
        
        

        c = torch.sum(torch.diag(torch.mm(w,P.T)))
        
           
        d = torch.sum(torch.diag(torch.mm(wt,P.T)))+torch.sum(torch.diag(torch.mm(wt,Tt.T))) 
        e = torch.sum(torch.diag(torch.mm(wt,Tt.T)))

        
        u = torch.sum(torch.diag(torch.mm(P.T,x) + torch.mm(P.T,y)))
        v = torch.sum(torch.diag(torch.mm(Tt.T,x)+torch.mm(Tt.T,y)))
            
                
        if (c - d + e == 0 and d - 2 * e + u - v == 0):
            alpha = 0
        else: 
            alpha = -(d - 2 * e + u - v)/(2 * (c - d + e))


        f0 = 0
        f1 = c - e + u - v

        falpha = (c - d + e) * alpha**2 + (d - 2 * e + u - v) * alpha

        if (alpha < tol and alpha > 0 and falpha > f0 and falpha > f1):

            P = alpha * P + (1 - alpha) * Tt

        elif f0 > f1:
            P = Tt
            
        else: 
            P = Tt
            toggle = 0
        break
    

    D = P
    _,corr = linear_sum_assignment(-P.cpu()) # matrix(solve_LSAP(P, maximum = TRUE))# return matrix P 

    
    corr = torch.LongTensor(corr)
    P = torch.eye(n)
    
    

    ccat = torch.cat([torch.eye(m),torch.zeros([m,n])],1)
    P = torch.index_select(P,0,corr)
    
   
    rcat = torch.cat([torch.zeros([n,m]),P],1)

    
    P =  torch.cat((ccat,rcat),0)
    # P =  np.vstack([np.hstack([torch.eye(m),torch.zeros([m,n])]),np.hstack([np.zeros([n,m]),P[corr]])]) 
    corr = corr 
    
    return corr,P



def part_constructor_evaluator_sgm(pmat,tests,obj_matrix,sz,del_num,re_permute = True):
    kn_nodes = sz-del_num
    precision = []
    
    kn_mask,un_mask,un_un_mask,kn_un_mask = partial_mask(sz,del_num)

    pre_adj = pmat
    if re_permute:
        index_order,P = sgraphmatch(obj_matrix,pre_adj,kn_nodes,iteration=200)
    else:
        index_order = np.arange(obj_matrix.shape[0])
        P = torch.eye(obj_matrix.shape[1])
    pre_adj = torch.mm(torch.mm(P,pre_adj),P.T)

    # equal mask
    # kn_mask = get_equal_mask(obj_matrix,kn_mask)
    # un_mask = get_equal_mask(obj_matrix,un_mask)
    # un_un_mask = get_equal_mask(obj_matrix,un_un_mask)
    # kn_un_mask = get_equal_mask(obj_matrix,kn_un_mask)


    auc_net = aucs(pre_adj,obj_matrix,un_mask,un_un_mask,kn_un_mask,kn_mask)
    ap_net = aps(pre_adj,obj_matrix,un_mask,un_un_mask,kn_un_mask,kn_mask)
    return auc_net[1],auc_net[2],auc_net[0],ap_net[1],ap_net[2],ap_net[0]
    # return auc_net[1],auc_net[2],auc_net[0],auc_net[3],ap_net[1],ap_net[2],ap_net[0],ap_net[3]

