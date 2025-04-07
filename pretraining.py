
import numpy as np
import torch
import esm
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, JumpingKnowledge, BatchNorm, LayerNorm
from torch import nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import scipy
from torch_geometric.transforms import NormalizeScale, NormalizeRotation
import igraph as ig
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tabpfn import TabPFNRegressor
import h5py


mp = {
    'A': "ALA",
    'C': "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL"

}


rev_mp = dict(zip(mp.values(), mp.keys()))
rev_mg = dict(zip(mp.keys(), [a for a in range(len(mp.keys()))]))

edge_as_feats = False
redo = True
use_global_feats = False
pii = 4

def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def build_viteoris_rips(points, filt):
    # out : E * 2
    edge_index = []
    edge_typ = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i == j:
                continue
            if point_distance(points[i], points[j]) <= filt:
                edge_index.append([i,j])
                edge_index.append([j,i])
                #x = inf[i][1] + 2 * inf[i][2] + 4 * inf[i][3] + inf[j][1] + 2 * inf[j][2] + 4 * inf[j][3]
                x = 0
                edge_typ.append(x)       
                edge_typ.append(x)
                

                
    return np.array(edge_index).T, np.array(edge_typ)


def build_viteoris_rips2(points, edges):
    # out : E * 2
    edge_index = []
    edge_typ = []
    used = {}
    print(len(edges))
    for edge in edges.keys():
        #edge_index.append([edge[0], edge[1]])
        #edge_index.append([edge[1], edge[0]])
        r = edges[edge]
        z = -1
        for j in [0,1,2,3,5,6,7,8,9,10,11,12,13,14,4]:
            if r[j] == 1:
                z = j
                break
 
        #if z == 4:
            #z = 0
        #else:
            #continue
        edge_index.append([edge[0], edge[1]])
        # used[(edge[0], edge[1])] = True
        edge_typ.append(z)
        # for j in range(15):
        #     if r[j] == 1:
        #         edge_index.append([edge[0], edge[1]])
        #         edge_typ.append(j)

    # for i in range(points.shape[0]):
    #     for j in range(points.shape[0]):
    #         if i == j or (i,j) in used:
    #             continue
    #         if point_distance(points[i], points[j]) <= 5:
    #             edge_index.append([i,j])

    #             #x = inf[i][1] + 2 * inf[i][2] + 4 * inf[i][3] + inf[j][1] + 2 * inf[j][2] + 4 * inf[j][3]
    #             x = 4
    #             edge_typ.append(x)       


    return np.array(edge_index).T, np.array(edge_typ)


def per_residue_foldx(fn):
    al = []
    with open(fn, "r") as f:
        trk = {}
        for line in f.readlines():
            if len(line) < 50:
                continue
            ln = line.split()
            chain, res = ln[2], ln[3]
            #if len(chain) > 1:
            #    chain, res = chain[0], chain[1:]
            # print(f"{chain}/{res}")
            ou = ln[4:7] + ln[8:-1]
            trk[f"{chain}/{res}"] = [float(x) for x in ou]
            al.append([float(x) for x in ou])
            # print(trk[f"{chain}/{res}"])
    return trk, np.array(al)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=16, heads=1, dropout=0.0, concat=False, bias=True):
        super(GATConv, self).__init__(aggr='add', node_dim=0)  # "add" aggregation (can be changed to mean, max, etc.)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Node feature transformation
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Edge feature transformation
        self.W_e = nn.Linear(edge_dim, heads * out_channels, bias=False)

        # Attention mechanisms
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_channels))  # Left attention
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_channels))  # Right attention
        self.attn_e = nn.Parameter(torch.Tensor(heads, out_channels))  # Edge attention

        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)

        # Bias term
        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(out_channels * out_channels))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.xavier_uniform_(self.attn_e)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        x: Node feature matrix of shape (N, in_channels)
        edge_index: Edge index matrix of shape (2, E)
        edge_attr: Edge feature matrix of shape (E, edge_dim)
        """
        H, C = self.heads, self.out_channels
        
        # Transform node features
        x = self.W(x).view(-1, H, C)  # (N, H, C)

        # Transform edge features
        edge_attr = self.W_e(edge_attr).view(-1, H, C)  # (E, H, C)

        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        """
        x_j: Neighbor node features (E, H, C)
        x_i: Target node features (E, H, C)
        edge_attr: Edge features (E, H, C)
        """
        # Compute attention scores
        alpha = (x_i * self.attn_l + x_j * self.attn_r + edge_attr * self.attn_e).sum(dim=-1)  # (E, H)
        alpha = F.silu(alpha)

        # Normalize attention scores
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_dropout(alpha)

        # Apply attention scores
        return x_j * alpha.unsqueeze(-1)  # (E, H, C)

    def update(self, aggr_out):
        """Final aggregation step"""
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)  # Concatenate heads
        else:
            aggr_out = aggr_out.mean(dim=1)  # Average heads

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

class RXConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, edge_dim=1, heads=1, dropout=0.0, concat=True, bias=True):
        super(RGATConv, self).__init__(aggr='add', node_dim=0)  # "add" aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations  # Number of edge types
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Relation-specific weight matrices
        self.W_r = nn.ModuleList([
            nn.Linear(in_channels, heads * out_channels, bias=False)
            for _ in range(num_relations)
        ])

        # Edge feature transformation per relation
        self.W_e = nn.ModuleList([
            nn.Linear(edge_dim, heads * out_channels, bias=False)
            for _ in range(num_relations)
        ])

        # Attention parameters per relation
        self.attn_l = nn.ParameterList([
            nn.Parameter(torch.Tensor(heads, out_channels)) for _ in range(num_relations)
        ])
        self.attn_r = nn.ParameterList([
            nn.Parameter(torch.Tensor(heads, out_channels)) for _ in range(num_relations)
        ])
        self.attn_e = nn.ParameterList([
            nn.Parameter(torch.Tensor(heads, out_channels)) for _ in range(num_relations)
        ])

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Bias term
        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(out_channels * out_channels))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W_r:
            nn.init.xavier_uniform_(w.weight)
        for w in self.W_e:
            nn.init.xavier_uniform_(w.weight)
            # Iterate over all attention parameters separately
        for attn in self.attn_l:
            nn.init.xavier_uniform_(attn)
        for attn in self.attn_r:
            nn.init.xavier_uniform_(attn)
        for attn in self.attn_e:
            nn.init.xavier_uniform_(attn)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, edge_type):
        """
        x: Node feature matrix (N, in_channels)
        edge_index: Edge index tensor (2, E)
        edge_attr: Edge feature tensor (E, edge_dim)
        edge_type: Edge type tensor (E,) indicating relation indices
        """
        # print(edge_index.shape)
        # print(edge_attr.shape)
        # print(edge_type.shape)
        H, C = self.heads, self.out_channels

        # Initialize output
        out = torch.zeros(x.size(0), H, C, device=x.device)

        # Loop over each relation
        for r in range(self.num_relations):
            mask = edge_type == r  
            
            if mask.sum() == 0:
                continue  

            # Select relevant edges
            edge_index_r = edge_index[:, mask]
            edge_attr_r = edge_attr[mask]

            # Transform node & edge features for relation r
            x_r = self.W_r[r](x).view(-1, H, C)  # (N, H, C)
            edge_attr_r = self.W_e[r](edge_attr_r).view(-1, H, C)  # (E, H, C)

            # Propagate messages for relation r
            out += self.propagate(edge_index_r, x=x_r, edge_attr=edge_attr_r, relation=r)

        # Concatenate or average attention heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i, relation):
        """
        x_j: Neighbor node features (E, H, C)
        x_i: Target node features (E, H, C)
        edge_attr: Edge features (E, H, C)
        relation: Current relation index
        """
        # Compute attention coefficients
        alpha = (x_i * self.attn_l[relation] + 
                 x_j * self.attn_r[relation] + 
                 edge_attr * self.attn_e[relation]).sum(dim=-1)  # (E, H)
        
        alpha = F.silu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)  # Normalize attention
        alpha = self.attn_dropout(alpha)

        return x_j * alpha.unsqueeze(-1)  # Apply attention

    def update(self, aggr_out):
        """Final aggregation step"""
        return aggr_out




from torch_geometric.data import Data
from torch.nn.functional import normalize
class CombinedModel(nn.Module):
    def __init__(self, input_dim, num_relations=15, num_layers=4, hidden_channels=32, dropout=0.10, jumping=False, num_heads=1, global_feats=0, clf=False, supcon=False):
        super(CombinedModel, self).__init__()
        print(input_dim) 

        self.relational = not edge_as_feats
        self.jumping = jumping
        self.residual = True
        self.edge_attr = True
        self.clf = clf
        self.supcon = supcon
        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()


                
        if self.relational:
            self.convs1.append(RGATConv(input_dim, hidden_channels, num_relations=num_relations, dropout=dropout, concat=False))
            #self.convs1.append(ExtendedRGCNConv(input_dim, hidden_channels, num_relations=num_relations))
        else:
            self.convs1.append(GATConv(input_dim, hidden_channels, dropout=dropout))
        self.bns1.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            if self.relational:
                self.convs1.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))
                #self.convs1.append(ExtendedRGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            else:
                self.convs1.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=False))
            self.bns1.append(LayerNorm(hidden_channels))
        
        
        self.convs2 = nn.ModuleList()
        
        if self.relational:
            self.convs2.append(RGATConv(input_dim, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))
            #self.convs2.append(ExtendedRGCNConv(input_dim, hidden_channels, num_relations=num_relations))
        else:
            self.convs2.append(GATConv(input_dim, hidden_channels, dropout=dropout))
        self.bns2.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            if self.relational:
                #self.convs2.append(ExtendedRGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
                self.convs2.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))
            else:
                self.convs2.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=False))
            self.bns2.append(LayerNorm(hidden_channels))
        # self.ln1 = LayerNorm(hidden_channels)
        self.jk1 = JumpingKnowledge("cat")
        self.jk2 = JumpingKnowledge("cat")
        #self.cross_att = GATConv(hidden_channels, hidden_channels, heads=2)
        ca_size = hidden_channels
        if self.jumping:
            ca_size = hidden_channels*num_layers
        self.cross_attn = nn.MultiheadAttention(embed_dim=ca_size, num_heads=1)
        
        lin_size = hidden_channels
        if self.jumping:
            lin_size *= num_layers
        lin_size += global_feats
        hc2 = 256
        self.lin1 = nn.Linear(lin_size, hc2)
        self.mid_lins = nn.ModuleList()
        self.mid_lnn = nn.ModuleList()
        self.mid_lin_cnt = 0
        for _ in range(self.mid_lin_cnt):
            # self.mid_lins.append(nn.Linear(hc2, hc2))
            self.mid_lins.append(nn.MultiheadAttention(hc2,  num_heads=2))
            # self.mid_lnn.append(LayerNorm(hc2))
        self.lin2 = nn.Linear(hc2, 1)
        self.lin4 = nn.Linear(hc2, hc2)
        self.num_layers = num_layers
        self.norm1 = NormalizeScale()
        self.norm2 = NormalizeRotation()

        self.ln1 = LayerNorm(input_dim-3)
        self.ln2 = LayerNorm(input_dim-3)
      

    
    def forward(self, data_batch_1, data_batch_2, edge_index_1, 
            edge_index_2, edge_type_1, edge_type_2,
            ew1=None, ew2=None, global_feats=None):
        # n * m
        #print(data_batch_1.shape)
        # ew1, ew2 = ew1.view(-1, 1).T, ew2.view(-1, 1).T
        # print(ew1.shape)
        # print(edge_index_1.shape)

        x1s = []
        dat1 = Data(pos=data_batch_1[:,:3])
        norm = self.norm2(self.norm1(dat1))
        

        xyz = norm.pos
        
        concatenated_x1 = torch.cat([xyz, self.ln1(data_batch_1[:,3:])], axis=1)
        #print(concatenated_x1.shape) 
        for i in range(self.num_layers):
            if self.relational:
                #concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), edge_type_1.to(torch.int64))
                if self.edge_attr:
                    concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), edge_type_1.to(torch.int64))
                else:
                    concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), edge_type_1.to(torch.int64))      
            else:
                #concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64))
                concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), ew1)
            if len(x1s) > 0 and self.residual:
                concatenated_x1 = concatenated_x1 + x1s[-1]
            concatenated_x1 = self.bns1[i](concatenated_x1)
            concatenated_x1 = F.silu(concatenated_x1)

            x1s.append(concatenated_x1)
        
        if self.jumping:
            concatenated_x1 = self.jk1(x1s)

        
        # dat2 = Data(pos=data_batch_2[:,:3])
        # norm = self.norm2(self.norm1(dat2))
        # xyz = norm.pos

        # concatenated_x2 = torch.cat([xyz, self.ln2(data_batch_2[:,3:])], axis=1)
        
        # x2s = []

        # for i in range(self.num_layers):
        #     if self.relational:
        #         if self.edge_attr:
        #         #concatenated_x2 = self.convs1[i](concatenated_x2, edge_index_2.to(torch.int64), edge_type_2.to(torch.int64))
        #             concatenated_x2 = self.convs1[i](concatenated_x2, edge_index_2.to(torch.int64), ew2, edge_type_2.to(torch.int64))
        #         else:
        #             concatenated_x2 = self.convs1[i](concatenated_x2, edge_index_2.to(torch.int64), edge_type_2.to(torch.int64))
        #     else:
        #         #concatenated_x2 = self.convs1[i](concatenated_x2, edge_index_2.to(torch.int64))
        #         concatenated_x2 = self.convs1[i](concatenated_x2, edge_index_2.to(torch.int64), ew2)
        #     if len(x2s) > 0 and self.residual:
        #         concatenated_x2 = concatenated_x2 + x2s[-1]
        #     concatenated_x2 = self.bns1[i](concatenated_x2)
        #     concatenated_x2 = F.silu(concatenated_x2)

        #     x2s.append(concatenated_x2)
        
        # if self.jumping:
        #     concatenated_x2 = self.jk2(x2s)

        #print(concatenated_x2.shape) 
        # Concatenate all tensors along the last dimension
        
        # attn_output, attn_weights = self.cross_attn(query=concatenated_x1, key=concatenated_x2, value=concatenated_x2)
        
        # x = torch.cat([
        #     attn_output.mean(0),
        #     # torch.max(attn_output, 0)[0]

        #     ],
        #   dim=0)
        x3 = concatenated_x1.mean(0)
        if use_global_feats:
            x3 = torch.cat([global_feats, x3], axis=0)
        # print(x.shape)
        #x = torch.sub(concatenated_x1.mean(1), concatenated_x2.mean(1))
        x = F.silu(self.lin1(x3)).unsqueeze(0)
        lins = [x]
        for i in range(self.mid_lin_cnt):
            #x = F.silu(self.mid_lins[i](x))
            if self.residual:
                x = x + lins[-1]
            # x = F.silu(self.mid_lnn[i](x))
            x = F.silu(self.mid_lins[i](x, key=x, value=x)[0])
            lins.append(x)
        #x0 = F.silu(self.lin1(x))
        
        #x = self.lin4(x)
        if self.clf:
            x = self.lin2(x)
            x1 = F.sigmoid(x)
            #return x1, torch.cat(x1s, axis=1).mean(0)
            return x1, x3
        if self.supcon:
            x = self.lin4(x)
            x1 = torch.norm(x, 2)
            return x/x1, x3 
        return x, lins[-1]


def ipython_graph(point_info, edges):
    g = ig.Graph(len(point_info), edges.keys())
    point_cols = ['idx', 'id', 'x', 'y', 'z', 'res']
    for j, point in enumerate(point_info):
        for k, col in enumerate(point_cols):
            g.vs[j][col] = point[k] 
            
    for j, edge in enumerate(edges.keys()):
        for k in range(15):
            g.es[j][f"Attr_{k}"] = edges[edge][k]
    return g

def find_mut_site(pdb, site, chains=[]):
    with open(pdb, "r") as f:
        for line in f.readlines():

            ln = line.split()
            if line[:4] != "ATOM":
                continue
            if line[21] == site[0] and line[22:27].strip() == site[3:-1]:
                return (float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()))

def get_binding_residues(filename):
    ok = False
    bind_res = []
    bind_chains = {}
    for line in open(filename, "r"):
        if 'interface residues' in line:
            ok = True
            continue
        if ok:
            resd = line.split()
            for res in resd:
                r, chain, num = res[0], res[1], res[2:]
                bind_res.append(res)
                if not chain in bind_chains:
                    bind_chains[chain] = []
                bind_chains[chain].append(num)

    return bind_res, bind_chains

def per_residue_foldx(fn):
    al = []
    with open(fn, "r") as f:
        trk = {}
        for line in f.readlines():
            if len(line) < 50:
                continue
            ln = line.split("\t")
            chain, res = ln[2], ln[3]
            #print(f"{chain}/{res}")
            
            ou = ln[4:7] + ln[8:-2]
            ou = [float(x) for x in ou]
            if len(chain) > 1:
                chain, res = chain[0], chain[1:]
                ou = ln[3:6] + ln[7:-2]
            trk[f"{chain}/{res}"] = [float(x) for x in ou]
            al.append([float(x) for x in ou])
            # print(trk[f"{chain}/{res}"])
    return trk, np.array(al)

def point_distance(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def build_atom_graph(fn, arpeg_fn, limit, mut=None, interfaceResidues=None):
    point_info = []
    j = 0
    id_to_num = {}
    edges = {}

    z = 0

    with open(fn, "r") as f:
        for i, line in enumerate(f.readlines()):
            ln = line.split()
            if line[:4] != 'ATOM':
                continue
            pt = (float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()))
            if point_distance(pt, mut) > limit:
                continue

            resNo = line[22:26].strip()
            atomName = line[12:16].strip()
            chainId = line[21]
            id =  chainId + "/" + resNo + "/" + atomName
            id2 = chainId + "/" + resNo
            #if interfaceResidues and not id2 in interfaceResidues:
            #    continue
            #if j < 10:
            #    print(id)

            res = line[17:20].strip()
            if not res in aa3to1:
                kr = 20
            else:
                kr = rev_mg[aa3to1[line[17:20].strip()]]
            id_to_num[id] = j
            point_info.append([j, id, pt[0], pt[1], pt[2], kr])
            j += 1
    dists = []
    with open(arpeg_fn, "r") as f:
        for line in f.readlines():
            ln = line.split()
            #if not ln[-1] == "INTRA_SELECTION":
            #    continue
            a,b = ln[0], ln[1]
            a2, b2 = "/".join(ln[0].split("/")[:2]), "/".join(ln[1].split("/")[:2])
            #a2,b2 = "/".join(ln[0].split("/")[:2]), "/".join(ln[1].split("/")[:2]
            if not a in id_to_num or not b in id_to_num:
                continue
            if interfaceResidues and (not a2 in interfaceResidues and not b2 in interfaceResidues):
                continue
            ed = (id_to_num[a],id_to_num[b])
            ed2 = (ed[1], ed[0])
            co1 = point_info[id_to_num[a]][2:]
            co2 = point_info[id_to_num[b]][2:]
            

            if not ed in edges:
                edges[ed] = np.zeros((15))
                edges[ed2] = np.zeros((15))
            edges[ed] += np.array([float(x) for x in ln[2:17]])
            edges[ed2] += np.array([float(x) for x in ln[2:17]])
            
            #dist = point_distance(co1, co2)
            dist = [point_distance(co1, co2)]
            if edge_as_feats:
                dist.extend(edges[ed].tolist())
            dists.append(dist)
            dists.append(dist)

    print(len(edges))
    return ipython_graph(point_info, edges), edges, np.array(dists)

aa3to1={
        'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
        'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
        'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
        'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
        'MSE':'M', 'GLX':'Q',
    }

def build_res_graph(fn, arpeg_fn,  limit, mut=None):
    point_info = []
    j = 0
    id_to_num = {}
    edges = {}
    seq = ""
    with open(fn, "r") as f:
        for i, line in enumerate(f.readlines()):
            ln = line.split()
            if line[:4] != 'ATOM': 
                continue
            if line[13] != 'N':
                continue
            pt = (float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()))
            if point_distance(pt, mut) > limit:
                continue
            id = ln[4] + "/" + ln[5]

            if not id in id_to_num:
                if line[17:20] not in aa3to1:
                    seq += "<mask>"
                else:
                    seq += aa3to1[line[17:20]]
                id_to_num[id] = j
                point_info.append([j, id, pt[0], pt[1], pt[2]])
                j += 1


    with open(arpeg_fn, "r") as f:
        for line in f.readlines():
            ln = line.split()
            if not ln[-1] == "INTRA_SELECTION":
                continue
            a,b = "/".join(ln[0].split("/")[:2]), "/".join(ln[1].split("/")[:2])
            if not a in id_to_num or not b in id_to_num:
                continue
            ed = (id_to_num[a],id_to_num[b])
            ed2 = (ed[1], ed[0])
            if not ed in edges:
                edges[ed] = np.zeros((15))
                edges[ed2] = np.zeros((15))
            edges[ed] += np.array([float(x) for x in ln[2:17]])
            edges[ed2] += np.array([float(x) for x in ln[2:17]])
    return ipython_graph(point_info, edges)

def get_node_features(g):
    
    x = [
    g.authority_score(),

    # g.hub_score(),

    g.betweenness(),

    #g.bibcoupling(),
    
    g.closeness(),

    g.constraint(),

    #g.cocitation(),
    

    g.coreness(), 

    g.eccentricity(),

    g.eigenvector_centrality(),

    g.harmonic_centrality(),

    g.pagerank(),

    g.personalized_pagerank(),

    g.strength(),

    g.transitivity_local_undirected(),

    ]

    # for p in x:
    #     print(p[0])

    return np.nan_to_num(np.array(x).T, 0)
from rdkit import Chem
def get_rdkit_feat(fn):
    feats = []
    mol = Chem.MolFromPDBFile(fn, flavor=2, sanitize=False, removeHs=False)
    print(f"rdkit points: {len(mol.GetAtoms())}")
    for atom in mol.GetAtoms():
        implicit_valence = atom.GetImplicitValence()
        # Get the implicit valence for each atom
        charge = atom.GetFormalCharge()
        feats.append([implicit_valence, charge])
    
    return np.array(feats)

def get_universe_points(fn, limit, mut, res_foldx=None, interfaceResidues=None):
    points = []
    point_info = []
    info = []
    j = 0
    k = 0
    id_to_num = {}
    edges = {}
    seq = ""
    ind = []

    aa3to1={
        'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
        'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
        'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
        'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
        'MSE':'M', 'GLX':'E',
    
    }

    atypes = {
    'CA': 0,
    'CB': 1,
    'CG': 2,
    'CD': 3,
    'CE': 4,
    'CZ': 5,
    'CH': 6,
    'OD': 7,
    'OE': 8,
    'OG': 9,
    'OH': 10,
    'ND': 11,
    'NE': 12,
    'NH': 13,
    'NZ': 14,
    'SD': 15,
    'SG': 16,
    'C': 17,
    'N': 18,
    'O': 19,
    'S': 20,
    'H': 21,
    'P': 22,
    'K': 23,
    'F': 24,
    'I': 25,
    'B': 26
}
    seqs = []
    prev_chain = None
    seq = ""
    atom_seq = []
    u = -1
    prev_res = None
    with open(fn, "r") as f:
        for i, line in enumerate(f.readlines()):
            ln = line.split()
            if line[:4] != 'ATOM':
                continue
            #if line[13] != 'N':
                #continue
            pt = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
            if point_distance(pt, mut) > limit:
                continue
            id = line[21] + "/" + line[22:27].strip()
            if interfaceResidues and not id in interfaceResidues:
                continue
            #if id in id_to_num:
                #continue 
            #pt = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
            xx = line[13:15].strip()
            if not xx in atypes:
                atype = 27
            else:
                atype = atypes[xx]
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            res = line[17:20].strip()
            temp = float(line[60:65].strip())
            if not res in aa3to1:
                kr = 20
            else:
                kr = rev_mg[aa3to1[line[17:20].strip()]]
            #res = rev_mg[aa3to1[line[17:20].strip()]]
            uatype = [0 for _ in range(28)]
            ukr = [0 for _ in range(21)]
            uatype[atype] = 1
            ukr[kr] = 1
            
            lz = [x, y, z, temp]
            lz.extend(uatype)
            lz.extend(ukr)
            


            if res_foldx:
                ix = id
                if not ix[-1].isdigit():
                    ix = id[:-1]
                if ix in res_foldx and len(res_foldx[ix]) == 31:
                    lz.extend(res_foldx[ix])
                else:
                    print(ix)
                    lz.extend([0 for _ in range(31)])
            info.append(lz)
            
            if line[21] != prev_res:
                prev_res = line[21]
                u += 1
            atom_seq.append(u)

            #if point_distance(pt, mut) > limit:
            #    continue
            if not id in id_to_num:
                id_to_num[id] = j
                #point_info.append([j, id, ln[6], ln[7], ln[8]])
                if prev_chain is None:
                    prev_chain = line[21]
                if line[21] != prev_chain:
                    seqs.append((prev_chain, seq))
                    seq = ""
                    prev_chain = line[21]
                if line[17:20] not in aa3to1:
                    seq += "<mask>"
                else:
                    seq += aa3to1[line[17:20]]
                j += 1
            
            points.append(pt)
            k += 1
            ind.append(int(line[7:12].strip()) - 1)
    if seq != "":
        seqs.append((prev_chain, seq))
    print(f"universe points: {len(points)}") 
    return np.array(points), seqs, atom_seq, info, ind

with open("./log.txt", "w") as f:
    pass

def get_ar(a, idx):
    out = []
    for i in idx:
        out.append(a[i])
    return out


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        return -mean_log_prob_pos.mean()





def mutaters(pdb_list, cnt):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for pdb in pdb_list:
        with open(f"./skempi_pdb/{pdb}.pdb", "r") as f:
            lns = f.readlines()
            mutations = []
            flen = len(lns)
            for _ in cnt:
                ok = False
                while not ok:
                    a = random.randint(0, flen)
                    if lns[a][:4] != "ATOM":
                        continue
                    line = lns[a]
                    new = random.choice(amino_acids)
                    pos = line[22:26].strip()
                    chain = line[21]
                    if not res in aa3to1:
                        continue
                    else:
                        orig = rev_mg[aa3to1[res]]
                    ok = True
                    mutations.append((chain, pos, orig, new))
            
            for mut in mutations:
                mut = chain, pos, orig, new
                with open("individual_list.txt", "w") as f:
                    f.write(f"{chain}{pos}{orig}{new}")
                yatta = f"{pdb}_{mut[0]}_{mut[1]}_{mut[2]}_{mut[3]}"
                os.system(f"./foldx --command=BuildModel --pdb={pdb}.pdb --pdb-dir=./skempi_pdb/ --output-dir=./foldx_pdbs/ --output-file={yatta} --mutant-file=individual_list.txt")

max_feat_len = 60
def train_model(model, data, ys, val_data, val_ys, epochs=1):
    import random
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    batch_size = 8
    print_amt = 1500
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)


    # if os.path.exists(f"./kfold_{fn}_{epocha}_model.pt"):
    #     try:
    #         model.load_state_dict(torch.load(f"./kfold_{fn}_{epocha}_model.pt", weights_only=True))
    #     except:
    #         pass
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, 10, decay=0.8, eta_min=1e-5)
    model.train()
    
    best = 1000
    best2 = 1000
    ecount = 0
    preval = val_model(model, val_data, val_ys, epoch=f"-1_train")
    pear = np.sqrt(mean_squared_error(preval, val_ys)) 
    pear2 = -scipy.stats.pearsonr(preval, val_ys).statistic
    if pear < best:
        best = pear
    if pear2 < best2:
        best2 = pear2 
    #data, ys, val_data, val_ys = np.array(data), np.array(ys), np.array(val_data), np.array(val_ys)

    #scaler = StandardScaler()
    #data = scaler.fit_transform(data)
    #val_data = scaler.transform(val_data)


    print(f"Train Len: {len(data)}")
    print(f"Val Len: {len(val_data)}")
    es = EarlyStopping(patience=1000, min_delta=0.01)
    for ep in range(epochs):
        running_loss = 0.
        last_loss = 0.
        model.train()
        idx = [a for a in range(len(data))]
        random.shuffle(idx)
        # idx, vidx = train_test_split(idx, test_size=0.1)
        for i in idx:
            #print(i)
            #print(f"sh1: {data[i][0].shape}")
            #print(f"sh2: {data[0][0].shape}")
            
            inputs, labels = data[i], ys[i].to(device)
     
            data_batch_1, data_batch_2 = inputs[0], inputs[1]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            
            
            
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None
            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]
            # print(global_feats)
            # print(labels)
            outputs, rep = model(data_batch_1, data_batch_2, edge_index_1, edge_index_2, edge_type_1, edge_type_2,
                    ew1 = edge_weights_1, ew2 = edge_weights_2, global_feats=global_feats)
            
            #if ep == 0:
                #reps.append(rep.cpu())
            # print(outputs)
            # print(labels)
            loss = loss_fn(outputs.flatten(), labels)
            loss = loss/batch_size
            loss.backward()
            running_loss += loss.item()*batch_size
            if ((i+1) % batch_size == 0) or (i+1 == len(data)):
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            if i>0 and i%(batch_size*print_amt)==0:
                print(running_loss/i)
                """
                with open("./log.txt", "a") as f:
                    f.write(f"{i+1}:{running_loss/(print_amt*batch_size)}\n")
                running_loss = 0.
                """
        preval = val_model(model, data, ys, epoch=f"{i}_train")
        pear = np.sqrt(mean_squared_error(preval, ys)) 

        if pear < best:
            best = pear
            torch.save(model.state_dict(), f"./kfold_{fn}_{epocha}_model.pt")
            

    return True



def train_model_clf(model, data, ys, epochs=1):
    import random
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    batch_size = 8
    print_amt = 40

    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, 10, decay=0.8, eta_min=1e-5)
    model.train()
    
    best = 1000
    best2 = 1000
    ecount = 0

    es = EarlyStopping(patience=1000, min_delta=0.01)
    for ep in range(epochs):
        print(ep)
        running_loss = 0.
        last_loss = 0.
        model.train()
        idx = [a for a in range(len(data))]
        random.shuffle(idx)
        # idx, vidx = train_test_split(idx, test_size=0.1)
        for i, xx in enumerate(idx):

            #print(f"sh1: {data[i][0].shape}")
            #print(f"sh2: {data[0][0].shape}")
            mm = data[xx]

            with open(mm, 'rb') as handle:
                inputs = list(pickle.load(handle))
                for j in range(len(inputs)):
                    inputs[j] = torch.Tensor(inputs[j]).to(device)

            xx = random.randint(0,1)
            data_batch_1, data_batch_2 = inputs[0][:,:max_feat_len], inputs[1][:,:max_feat_len]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None
            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
                if edge_weights_1.shape[0] != edge_index_1.shape[1] or edge_weights_2.shape[0] != edge_index_2.shape[1]:
                    continue
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]
            
            if edge_type_1.dim() == 2:
                edge_type_1 = torch.Tensor([4,4]).to(device)
                edge_index_1 = torch.Tensor([[1,0],[0,1]]).to(device)
            if edge_type_2.dim() == 2:
                edge_type_2 = torch.Tensor([4,4]).to(device)
                edge_index_2 = torch.Tensor([[1,0],[0,1]]).to(device)

            

            if xx == 0:
                labels = torch.Tensor([0]).to(device)
                outputs, rep = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1,
                        ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats)
                loss = loss_fn(outputs.flatten(), labels)
                loss = loss/batch_size
                loss.backward()
            else:
                labels = torch.Tensor([1]).to(device)


                outputs, rep = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2,
                        ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats)
                loss = loss_fn(outputs.flatten(), labels)
                loss = loss/batch_size
                loss.backward()

            running_loss += loss.item()*batch_size
            if ((i+1) % batch_size == 0) or (i+1 == len(data)):
                optimizer.step()
                optimizer.zero_grad()
            if i>0 and i%(batch_size*print_amt)==0:
                print(running_loss/i)
                """
                with open("./log.txt", "a") as f:
                    f.write(f"{i+1}:{running_loss/(print_amt*batch_size)}\n")
                running_loss = 0.
                """
    torch.save(model.state_dict(), f"./pretrained_model.pt")
    knn_evaluate(model, data, len(max(pdb_dict.keys())) + 1)
    
    return True

pdb_dict = {}


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
@torch.no_grad()
def knn_evaluate(model, data, k=320):
    model.eval()
    X_train, y_train = [], []
    X_val, y_val = [], []

    # --- Get embeddings for training set (reference points)
    for mm in data:
        with open(mm, 'rb') as handle:
            inputs = list(pickle.load(handle))
            for j in range(len(inputs)):
                inputs[j] = torch.Tensor(inputs[j]).to(device)
        pdb = mm.split("/")[-1].split("_")[0]

        data_batch_1, data_batch_2 = inputs[0][:,:max_feat_len], inputs[1][:,:max_feat_len]
        edge_index_1, edge_index_2 = inputs[2], inputs[3]
        edge_type_1, edge_type_2 = inputs[4], inputs[5]
        edge_weights_1, edge_weights_2, global_feats = None, None, None
        if len(inputs) > 6:
            edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
            if edge_weights_1.shape[0] != edge_index_1.shape[1] or edge_weights_2.shape[0] != edge_index_2.shape[1]:
                continue
        if len(inputs) > 8 and use_global_feats:
            global_feats = inputs[8]
        
        if edge_type_1.dim() == 2:
            edge_type_1 = torch.Tensor([4,4]).to(device)
            edge_index_1 = torch.Tensor([[1,0],[0,1]]).to(device)
        if edge_type_2.dim() == 2:
            edge_type_2 = torch.Tensor([4,4]).to(device)
            edge_index_2 = torch.Tensor([[1,0],[0,1]]).to(device)
        labels = torch.Tensor([pdb_dict[pdb], pdb_dict[pdb]]).to(device)
            
        outputs1, rep1 = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1,
                ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats)

        outputs2, rep2 = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2,
                ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats)

        X_train.append(rep1.cpu())
        y_train.append(pdb_dict[pdb])
        X_train.append(rep2.cpu())
        y_train.append(pdb_dict[pdb])

    
    X_train = torch.stack(X_train).numpy()
    y_train = np.array(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


    # --- Run k-NN
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    visualize(X_val, y_val)
    print(y_val)
    print(f"ACC: {acc}")
    return acc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def visualize(embeddings, labels):
    # Replace this with your real embeddings and labels
    num_samples = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
    num_classes = np.max(labels) + 1


    # Perform 3D t-SNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)

    # k-NN clustering
    knn = KNeighborsClassifier(n_neighbors=num_classes)
    knn.fit(embeddings, labels)
    knn_preds = knn.predict(embeddings)

    # Plot 3D scatter with k-NN cluster colors
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    rev_pdb = dict(zip(pdb_dict.values(), pdb_dict.keys()))
    sc = ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=knn_preds, cmap='tab10', edgecolors='k', alpha=0.8
    )

    legend = ax.legend(*sc.legend_elements(), title="k-NN Clusters", loc="upper right")
    ax.add_artist(legend)

    ax.set_title("3D t-SNE of Projection Head Embeddings with k-NN Clusters")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.tight_layout()
    plt.show()

def train_model_supcon(model, data, ys, epochs=1):
    import random
    loss_fn = SupConLoss(0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    batch_size = 128
    print_amt = 1

    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, 10, decay=0.8, eta_min=1e-5)
    model.train()
    
    best = 1000
    best2 = 1000
    ecount = 0

    anchor = None
    bd = 0
    cur_labels = None
    cur_features = None
    uz = 0
    closs = 0.
    cdenom = 0.
    

    for ep in range(epochs):
        print(ep)
        running_loss = 0.
        last_loss = 0.
        model.train()
        idx = [a for a in range(len(data))]
        random.shuffle(idx)
        # idx, vidx = train_test_split(idx, test_size=0.1)
        loss = 0.
        for i, xx in enumerate(idx):
  

            #print(f"sh1: {data[i][0].shape}")
            #print(f"sh2: {data[0][0].shape}")
            mm = data[xx]
            pdb = mm.split("/")[-1].split("_")[0]

            with open(mm, 'rb') as handle:
                inputs = list(pickle.load(handle))
                for j in range(len(inputs)):
                    inputs[j] = torch.Tensor(inputs[j]).to(device)

            data_batch_1, data_batch_2 = inputs[0][:,:max_feat_len], inputs[1][:,:max_feat_len]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None
            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
                if edge_weights_1.shape[0] != edge_index_1.shape[1] or edge_weights_2.shape[0] != edge_index_2.shape[1]:
                    continue
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]

            if edge_type_1.dim() == 2:
                edge_type_1 = torch.Tensor([4,4]).to(device)
                edge_index_1 = torch.Tensor([[1,0],[0,1]]).to(device)
            if edge_type_2.dim() == 2:
                edge_type_2 = torch.Tensor([4,4]).to(device)
                edge_index_2 = torch.Tensor([[1,0],[0,1]]).to(device)
            labels = torch.Tensor([pdb_dict[pdb], pdb_dict[pdb]]).to(device)
            
            outputs1, rep = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1,
                    ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats)

            outputs2, rep = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2,
                    ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats)
            
            # print(cur_features.shape)
            # print(outputs.shape)
            outputs = torch.cat([outputs1, outputs2], axis=0)
            if not anchor:
                current_labels = labels
                cur_features = outputs
                anchor = True
            else:
                current_labels = torch.cat([current_labels, labels])
                cur_features = torch.cat([cur_features, outputs], axis=0)
            del labels
            del outputs

            bd += 2

            if bd % batch_size == 0:
                optimizer.zero_grad()
                loss = loss_fn(cur_features, current_labels)
                print(loss)
                loss.backward()
                closs += loss * batch_size
                anchor = None
                cur_features = None
                cur_labels = None
            if batch_size * print_amt == bd: 
                optimizer.step()
                bd = 0

                
   

        print(f"Loss: {closs/len(idx)}")
        closs = 0.
        _, udata = train_test_split(data, test_size=0.2)
            
        torch.save(model.state_dict(), f"./pretrained_model.pt")
        knn_evaluate(model, udata, len(max(pdb_dict.keys())) + 1)
    return True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)
torch.cuda.synchronize()
#esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
# esm_model.to(device)

# batch_converter = alphabet.get_batch_converter()
def get_esm_repr(seqs, aseq):

    batch_labels, batch_strs, batch_tokens = batch_converter(seqs)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    af = esm_model(batch_tokens.to(device), repr_layers=[30], return_contacts=False)
    token_representations = af["representations"][30].cpu().detach().numpy().squeeze()

    if token_representations.shape[0] > 8:
        xx = token_representations[1:token_representations.shape[0]-1]
    else:
        sequence_representations = []
        for i, tok_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tok_len - 1])
            print(tok_len)
            print(sequence_representations[i].shape)
        xx = np.concatenate(sequence_representations,axis=0)

    atom_rep = []
    for j in aseq:
        
        atom_rep.append(xx[j])

    return np.array(atom_rep)

def expand_weights(W_small, target_shape):
    W_large = torch.zeros(*target_shape)
    # Copy overlapping part
    min_shape = tuple(min(s, l) for s, l in zip(W_small.shape, target_shape))
    slices = tuple(slice(0, m) for m in min_shape)
    W_large[slices] = W_small[slices]
    return W_large

def transfer_weights(small_layer, large_layer):
    small_state = small_layer.state_dict()
    large_state = large_layer.state_dict()
    new_state = deepcopy(large_state)

    for name, param in small_state.items():
        if name in large_state:
            if param.shape != large_state[name].shape:
                # Expand if different shape
                new_param = expand_weights(param, large_state[name].shape)
                print(f"Expanding {name} from {param.shape} to {large_state[name].shape}")
                new_state[name] = new_param
            else:
                # Direct copy
                new_state[name] = param
        else:
            print(f"Skipping {name} (not in large layer)")
    
    large_layer.load_state_dict(new_state)

def val_model(model, data, ys, epoch=-1, get_reps=False):
    model.eval()
    fin = []
    reps = []
    with torch.no_grad():
        for i in range(len(data)):
            inputs, labels = data[i], ys[i].to(device)
            data_batch_1, data_batch_2 = inputs[0][:,:max_feat_len], inputs[1][:,:max_feat_len]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None

            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]

            # print(f"batch : {edge_weights_1.shape}")
            outputs, rep = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1, ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats )
            outputs, rep2 = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2, ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats )
            #outputs, rep = model(data_batch_1, data_batch_2, edge_index_1, edge_index_2 )
            if get_reps:
                rep = torch.cat([rep, rep2])
                reps.append(rep.flatten().cpu().numpy())
            fin.append(outputs.flatten().cpu().detach().item())
    
    print(f"RMSE: {np.sqrt(mean_squared_error(fin, ys))}\n")
    print(f"Pearson: {scipy.stats.pearsonr(fin, ys)}\n")
    # with open(f"./log.txt", "a") as f:
    #     f.write(f"Epoch {epoch}\n")
    #     f.write(f"RMSE: {np.sqrt(mean_squared_error(fin, ys))}\n")
    #     f.write(f"Pearson: {scipy.stats.pearsonr(fin, ys)}\n")
    if get_reps:
        return reps
    return fin

fn = "s8338"
df = pd.read_csv(f"./{fn.upper()}.csv")
models = []
rdkit1 = []
rdkit2 = []

if fn == "s645":
    mut_col = 'Mutation'
    y_col = "ddG(kcal/mol)"
    chain_col = "Partners(A_B)"
elif fn == "s8338":
    mut_col = 'chain.mut'
    y_col = "ddG"
    chain_col = "biounit_chain"
elif fn == "s1131":
    mut_col = 'mutation'
    y_col = 'DDG'
    chain_col = "Partners(A_B)"

if fn == "s1131":
    pro_col = "protein"
if fn == "s645":
    pro_col = "#PDB"
if fn == "s4169":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[pro_col] = df['#PDB'].apply(lambda x : x.split("_")[0])
    df = df.loc[:4168]
if fn == "s8338":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[pro_col] = df['#PDB'].apply(lambda x : x.split("_")[0])
    df = df.loc[:8338]


#numm = int(fn[1:])
# df[pro_col] = df[pro_col].apply(lambda x : x[:2])
numm = 8338
pdbs = df[pro_col].unique()
print(df.shape)
import sys
if len(sys.argv) > 1:
    epocha = int(sys.argv[1])
else:
    epocha = None
import time
print(epocha)

kf = KFold(10, shuffle=True, random_state=42)
import pickle
import os

protein_map = df.reset_index().groupby(pro_col)['index'].first()
print(protein_map)
for filt in np.arange(20.0, 20.5, 1.0):
    #model = CombinedModel(16+31).to(device)
    preda = np.zeros(df.shape[0])
    inputs = []
    outputs = []
    reg = [f"./pretrain_pdbs1/" + fn for fn in os.listdir(f"./pretrain_pdbs1/")]
    reg = reg + [f"./pretrain_pdbs2/" + fn for fn in os.listdir(f"./pretrain_pdbs2/")]
    reg = reg + [f"./pretrain_pdbs3/" + fn for fn in os.listdir(f"./pretrain_pdbs3/")]
    reg = reg + [f"./pretrain_pdbs4/" + fn for fn in os.listdir(f"./pretrain_pdbs4/")]
    reg =  reg + [f"./pretrain_pdbs5/" + fn for fn in os.listdir(f"./pretrain_pdbs5/")]
    reg = []


    for z, idx2 in enumerate(reg):
        if not redo:
            continue
        if ".pdb" not in idx2:
            continue
        pii = idx2.split("/")[1][-1]
        idx = idx2.split("/")[2]
        pref = idx.split(".")[-2]
  
        if len(pref.split("_")) != 5:
            continue
        print(idx)
        
        try:
            pdb, orig, chain, pos, new = pref.split("_")

            # if idx >= 4169:
            #     repr1, repr2, edges1, edges2, ei1, ei2, dists1, dists2 = inputs[idx-4169]
            #     inputs.append((repr2, repr1, edges2, edges1, ei2, ei1, dists2, dists1))
            #     outputs.append(-df.loc[idx-4169, y_col])
            #     continue
            
            #fn1 = f"./clean_pdbs/{at1}.clean.pdb"
            #fn2 = f"./clean_pdbs/{at2}.clean.pdb"
            if os.path.exists(f"./deta/{pref}.pkl"):
                inputs.append((f"./deta/{pref}.pkl", pdb))
                continue
            
            at1 = f"{fn}_{protein_map[pdb]}_0"
            
            
            fn1 = f"./foldx_pdbs/{at1}_Repair.pdb"
            fn2 = f"./pretrain_pdbs{pii}/{pref}.pdb"
            
            """
            br1, bc1 = get_binding_residues(f"./foldx_pdbs/Interface_Residues_{at1}_Repair.clean_AC.fxout")
            br2, bc2 = get_binding_residues(f"./foldx_pdbs/Interface_Residues_{at1}_Repair_1.clean_AC.fxout") 
            br1 = [g[1] + "/" + g[2:] for g in br1]
            br2 = [g[1] + "/" + g[2:] for g in br2]
            """
            

                
            apg1 = f"./appreg_feat/{at1}/{at1}_Repair.clean.contacts"
            apg2 = f"./pretrain_pdbs{pii}/{pref}.clean.contacts"
                

            #if idx >= 4169:
            #    fn1 = f"./clean_pdbs/s8338_{idx-4169}_1.clean.pdb"
            #    fn2 = f"./clean_pdbs/s8338_{idx-4169}_0.clean.pdb"
            
            foldx1, _ = per_residue_foldx(f"./foldx_pdbs/SD_{at1}_Repair.fxout")

            foldx2, _ = per_residue_foldx(f"./pretrain_pdbs{pii}/SD_{pref}.fxout")
            muta = chain + "." + orig + pos + new



            if orig == new:
                continue
            print(f"PDB : {pdb}")
            
            mu = find_mut_site(fn1, muta, chains=[])
            print(f"mu: {mu}")

            if mu is None:
                print('SKIPs')
                continue



            
            points1, seq1, aseq1, inf1, ind1 = get_universe_points(fn1, filt, mu, 
                    foldx1, 
                    )

            points2, seq2, aseq2, inf2, ind2 = get_universe_points(fn2, filt, mu, 
                    foldx2, 
                    )

            
            atom1, edges1, dists1 = build_atom_graph(fn1, apg1, filt, mut=mu)
            atom2, edges2, dists2 = build_atom_graph(fn2, apg2, filt, mut=mu)
            if len(edges1) == 0:
                edges1[(0,1)] = edges1[(1,0)] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,]
            if len(edges2) == 0:
                edges2[(0,1)] = edges2[(1,0)] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
            repr1 = get_node_features(atom1)
            repr2 = get_node_features(atom2)
            print(f"edges1: {len(edges1)}")
            print(f"edges2: {len(edges2)}")
            print(np.array(inf1).shape)
            print(np.array(inf2).shape)

            assert np.array(inf1).shape[0] == repr1.shape[0]
            repr1 = np.concatenate([
                np.array(inf1),
                repr1,

                ], 
                axis=1)
            repr2 = np.concatenate([
                np.array(inf2),
                repr2,
                #rexx2
                ], 
                axis=1)

            edges1, ei1 = build_viteoris_rips2(points1, edges1)
            edges2, ei2 = build_viteoris_rips2(points2, edges2)

            print(dists1.shape)
            print(dists2.shape)
            assert points1.shape[0] == repr1.shape[0]
            #assert repr1.shape[1] == 48 and repr2.shape[1] == 48
            with open(f"./deta/{pref}.pkl", "wb") as h:
                inp = (repr1, repr2, edges1, edges2, ei1, ei2, dists1, dists2)
                pickle.dump(inp, h, protocol=pickle.HIGHEST_PROTOCOL)
            

            outputs.append(-1)
        except Exception as e:
            print(e)
            pass
    import pickle
    from numpy import dot
    from numpy.linalg import norm
    # with open(f"esm2gcn_inputs_pretrain{pii}_{filt}.pkl", "wb") as h:
    #     pickle.dump(inputs, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    inputs = [f"./deta/" + fn for fn in os.listdir(f"./deta/") if '.' in fn]
    uz = 0
    for i, xx in enumerate(inputs):
        mm = inputs[i]
        pdb = mm.split("/")[-1].split("_")[0]
        if not pdb in pdb_dict:
            pdb_dict[pdb] = uz
            uz += 1
    print(f"LEN: {len(inputs)}")
    print(f"PDB_DICT: {len(pdb_dict)}")

    def get_ar(a, idx):
        out = []
        for i in idx:
            out.append(a[i])
        return out
    # for i in range(len(inputs)):
    #     inputs[i] = list(inputs[i])
    #     for j in range(len(inputs[i])):
    #         inputs[i][j] = torch.Tensor(inputs[i][j]).to(device)
            # if j < 2:
            #     inputs[i][j] = (inputs[i][j] - global_mean )/global_std

    
    #y1 = np.array(outputs)

    y1 = df[y_col]


    outputs = torch.tensor(outputs, dtype=torch.float32, device='cpu')
    with open(f"./deta/1AK4_G_A_109_H.pkl", 'rb') as handle:
        ex = pickle.load(handle)
    

    model = CombinedModel(ex[0][:,:max_feat_len].shape[1], hidden_channels=32, supcon=True).to(device)
    if os.path.exists(f"./pretrained_model.pt"):
        try:
            #m2 = CombinedModel(ex[0][:,:max_feat_len].shape[1], hidden_channels=32, supcon=True).to(device)
            model.load_state_dict(torch.load(f"./pretrained_model.pt", weights_only=True))
            #transfer_weights(model, m2)
        except Exception as e:
            print(e)
            pass
    train_model_supcon(model, inputs, outputs ,epochs=100)
    # train_model_clf(model,inputs,outputs,epochs=30)
