
import numpy as np
import torch
import esm
import os
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from tabpfn import TabPFNRegressor

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

from torch_geometric.data import Data
from torch.nn.functional import normalize

from foldx_func import foldx_calc
from arpeg_func import read_arpeggio
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

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
redo = False
use_global_feats = False

def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def build_viteoris_rips2(points, edges, dists):
    # out : E * 2
    edge_index = []
    edge_typ = []
    used = {}

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
        used[(edge[0], edge[1])] = True
        # for j in range(15):
        #     if r[j] == 1:
        #         edge_index.append([edge[0], edge[1]])
        #         edge_typ.append(j)
    # z = 0
    # for i in range(points.shape[0]):
    #     for j in range(points.shape[0]):
    #         if i == j or (i,j) in used:
    #             continue
    #         if point_distance(points[i], points[j]) <= 5:
    #             edge_index.append([i,j])
    #             if (i, j) in used:
    #                 z += 1

    #             #x = inf[i][1] + 2 * inf[i][2] + 4 * inf[i][3] + inf[j][1] + 2 * inf[j][2] + 4 * inf[j][3]
    #             x = 4
    #             edge_typ.append(x)
    #             dists.append([point_distance(points[i], points[j])])       

    return np.array(edge_index).T, np.array(edge_typ), np.array(dists)


def per_residue_foldx(fn):
    al = []
    with open(fn, "r") as f:
        trk = {}
        for line in f.readlines():
            if len(line) < 50:
                continue
            ln = line.split()
            chain, res = ln[2], ln[3]
            ou = ln[4:7] + ln[8:-1]
            trk[f"{chain}/{res}"] = [float(x) for x in ou]
            al.append([float(x) for x in ou])

    return trk, np.array(al)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, num_relations=15, num_layers=4, dropout=0.10, jumping=False, num_heads=1, global_feats=0, clf=False, supcon=False):
        super(CombinedModel, self).__init__()
       
        hidden_channels = 32
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


        x1s = []
        dat1 = Data(pos=data_batch_1[:,:3])
        norm = self.norm2(self.norm1(dat1))
        

        xyz = norm.pos
        
        concatenated_x1 = torch.cat([xyz, self.ln1(data_batch_1[:,3:])], axis=1)
        #print(concatenated_x1.shape) 
        for i in range(self.num_layers):
            if self.relational:
                if self.edge_attr:
                    concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), edge_type_1.to(torch.int64))
                else:
                    concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), edge_type_1.to(torch.int64))      
            else:
                concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64), ew1)
            if len(x1s) > 0 and self.residual:
                concatenated_x1 = concatenated_x1 + x1s[-1]
            concatenated_x1 = self.bns1[i](concatenated_x1)
            concatenated_x1 = F.silu(concatenated_x1)

            x1s.append(concatenated_x1)
        
        if self.jumping:
            concatenated_x1 = self.jk1(x1s)

        x3 = concatenated_x1.mean(0)
        if use_global_feats:
            x3 = torch.cat([global_feats, x3], axis=0)
        # print(x.shape)
        #x = torch.sub(concatenated_x1.mean(1), concatenated_x2.mean(1))
        x = F.silu(self.lin1(x3)).unsqueeze(0)
        lins = [x]
        for i in range(self.mid_lin_cnt):

            if self.residual:
                x = x + lins[-1]
            x = F.silu(self.mid_lins[i](x, key=x, value=x)[0])
            lins.append(x)

        if self.clf:
            x = self.lin2(x)
            x1 = F.sigmoid(x)
            return x1, torch.cat(x1s, axis=1).mean(0)
            #return x1, x3
        if self.supcon:
            x = self.lin4(x)
            x1 = torch.norm(x, 2)
            return x/x1, torch.cat(x1s, axis=1).mean(0)
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
            if len(ln) < 5 or line[13] != 'N':
                continue
            if line[21] == site[0] and line[22:27].strip() == site[3:-1]:
                return (float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()))

def get_node_stats(a):
    return np.concatenate([
        np.max(a, axis=0),
        # np.min(a, axis=0),
        np.mean(a, axis=0),
        # np.std(a, axis=0),
        np.sum(a, axis=0),

    ])



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

def build_atom_graph(fn, arpeg_fn, limit, backbone_comb=[2,2], mut=None, interfaceResidues=None, edge_subset=None ):
    point_info = []
    j = 0
    id_to_num = {}
    edges = {}

    z = 0

    backbone = ['N', 'CA', 'C', 'O', 'OXT', 'H', 'HA']


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
            aname1,aname2 = ln[0].split("/")[2], ln[1].split("/")[2]
            #a2,b2 = "/".join(ln[0].split("/")[:2]), "/".join(ln[1].split("/")[:2]
            if backbone_comb[0] == 0 and not aname1 in backbone:
                continue
            if backbone_comb[0] == 1 and aname1 in backbone:
                continue
            if backbone_comb[1] == 0 and not aname2 in backbone:
                continue
            if backbone_comb[1] == 1 and aname2 in backbone:
                continue

            if not a in id_to_num or not b in id_to_num:
                continue
            if interfaceResidues and (not a2 in interfaceResidues and not b2 in interfaceResidues):
                continue
            if edge_subset:
                pedge = False
                for ix in edge_subset:
                    if ln[ix] == '1':
                        pedge = True
                if not pedge:
                    continue
                    
            ed = (id_to_num[a],id_to_num[b])
            ed2 = (ed[1], ed[0])
            co1 = point_info[id_to_num[a]][2:]
            co2 = point_info[id_to_num[b]][2:]
            

            if not ed in edges:
                edges[ed] = np.zeros((15))
            edges[ed] += np.array([float(x) for x in ln[2:17]])

            dist = [point_distance(co1, co2)]
            if edge_as_feats:
                dist.extend(edges[ed].tolist())
            dists.append(dist)
            #dists.append(dist)
    #point_info, edges = reconv(point_info, edges)
    print(f"Atom graph edges: {len(edges)}")
    print(point_info[0])
    return ipython_graph(point_info, edges), edges, dists

def reconv(point_info, edges):
    i = 0
    used = {}
    for (a,b) in edges.keys():
        used[a] = 1
        used[b] = 1
    npi = []
    ned = {}
    conv_out = {}
    for u in sorted(used.keys()):
        lm = point_info[u] 
        lm[0] = i
        conv_out[u] = i
        i += 1
        npi.append(lm)
        
    for (a,b) in edges.keys():
        ned[(conv_out[a], conv_out[b])] = edges[(a,b)]
    
    return npi, ned

        
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
            res = line[17:20].strip()
            if not res in aa3to1:
                kr = 20
            else:
                kr = rev_mg[aa3to1[line[17:20].strip()]]

            if not id in id_to_num:
                if line[17:20] not in aa3to1:
                    seq += "<mask>"
                else:
                    seq += aa3to1[line[17:20]]
                id_to_num[id] = j
                point_info.append([j, id, pt[0], pt[1], pt[2], kr])
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

    g.betweenness(),

    
    g.closeness(),

    g.constraint(),

    g.coreness(), 

    g.eccentricity(),

    g.eigenvector_centrality(),

    g.harmonic_centrality(),

    g.pagerank(),

    g.transitivity_local_undirected(),

    ]

    return np.nan_to_num(np.array(x).T, 0)

from rdkit import Chem
def get_rdkit_feat(fn):
    feats = []
    mol = Chem.MolFromPDBFile(fn, flavor=2, sanitize=False, removeHs=False)
    # print(f"rdkit points: {len(mol.GetAtoms())}")
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
    closest = []
    clo = 1000000
    clor = None
    with open(fn, "r") as f:
        for i, line in enumerate(f.readlines()):
            ln = line.split()
            if line[:4] != 'ATOM':
                continue
            #if line[13] != 'N':
                #continue
            pt = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
            pdist = point_distance(pt, mut)
            if pdist > limit:
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
            if not res in aa3to1:
                kr = 20
            else:
                kr = rev_mg[aa3to1[line[17:20].strip()]]

            uatype = [0 for _ in range(28)]
            ukr = [0 for _ in range(21)]
            uatype[atype] = 1
            ukr[kr] = 1
            temp = float(line[60:65].strip())
            
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
                    lz.extend([0 for _ in range(31)])
            info.append(lz)
            
            if line[21] != prev_res:
                prev_res = line[21]
                u += 1
            atom_seq.append(u)

            if not id in id_to_num:
                id_to_num[id] = j
                if prev_chain is None:
                    prev_chain = line[21]
                if line[21] != prev_chain:
                    seqs.append((prev_chain, seq))
                    seq = ""
                    prev_chain = line[21]
                    closest.append(clor)
                if line[17:20] not in aa3to1:
                    seq += "A"
                else:
                    seq += aa3to1[line[17:20]]
                if pdist < clo:
                    clo = pdist
                    clor = len(seq) - 1
                j += 1
            
            points.append(pt)
            k += 1
            ind.append(int(line[7:12].strip()) - 1)
    if seq != "":
        seqs.append((prev_chain, seq))
        closest.append(clor)
    
    for i, (chain, seq) in enumerate(seqs):
        clor = closest[i]
        if chain != mut[0]:
            continue
        seqs[i] = (chain, seq[max(0, clor-50):clor+50])

    

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

def perform_foldx(clm, clm2, mu, chains,  bw=False):
    if not os.path.exists(f"./foldx_pdbs/Interaction_{pdb}_Repair_AC.fxout"):
        workdir = "foldx_pdbs"
        print("FOLDX STARTING")

        comm = './foldx --command=AnalyseComplex --pdb={}  --output-dir={} --pdb-dir={}'.format(\
                                    pdb + "_Repair.pdb",   f'./foldx_pdbs', f'./foldx_pdbs')

        os.system(comm)
    
    if not os.path.exists(f"./foldx_pdbs/Interaction_{clm2}_AC.fxout"):
        comm = './foldx --command=AnalyseComplex --pdb={}   --output-dir={} --pdb-dir={}'.format(\
                                clm2 + ".pdb", "foldx_pdbs", f'./foldx_pdbs')
        os.system(comm)
        print("FOLDX ENDING")

    chainid = mu[0]
    extra_chains = [c for c in chains if not c == chainid]

    wild = foldx_calc(f"./foldx_pdbs/Interaction_{pdb}_Repair_AC.fxout", chainid, extra_chains)
    mut = foldx_calc(f"./foldx_pdbs/Interaction_{clm2}_AC.fxout", chainid, extra_chains)    
    

    feat = np.concatenate([wild, mut, wild-mut])
    if bw:
        feat = np.concatenate([mut, wild, mut-wild])

    return feat

def perform_appreg(apg1, apg2, mu, chains, bw=False):

    chainid = mu[0]
    extra_chains = [c for c in chains if not c == chainid]

    ap1 = read_arpeggio(apg1, [chainid], extra_chains)
    ap2 = read_arpeggio(apg2, [chainid], extra_chains)

    feat = np.concatenate([ap1,ap2,ap1-ap2])
    if bw:
        feat = np.concatenate([ap2,ap1,ap2-ap1])


    return feat

def get_global_features(g, a):
    # g = g.to_networkx()

    # Calculate the adjacency matrix of the graph
    adj_matrix = np.array(g.get_adjacency().data)

    # Calculate the eigenvalues of the adjacency matrix
    adj_eigenvalues = np.linalg.eigvalsh(adj_matrix)
    adj_eigenvalues = sorted(adj_eigenvalues, reverse=True)  # Sort in descending order

    # Get the first and second largest eigenvalues of the adjacency matrix
    first_adj_eigenvalue = adj_eigenvalues[0]
    second_adj_eigenvalue = adj_eigenvalues[1]
    graph_energy = np.sum(adj_eigenvalues)
    print("Adjacency Matrix:")
    print("First eigenvalue:", first_adj_eigenvalue)
    print("Second eigenvalue:", second_adj_eigenvalue)

    # Calculate the Laplacian matrix of the graph
    laplacian_matrix = np.array(g.laplacian())

    # Calculate the eigenvalues of the Laplacian matrix
    lap_eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    lap_eigenvalues = sorted(lap_eigenvalues)  # Sorted in ascending order

    # Get the first and second smallest eigenvalues of the Laplacian matrix
    first_lap_eigenvalue = lap_eigenvalues[0]
    j = 1
    while j < len(lap_eigenvalues) - 1 and lap_eigenvalues[j] == 0:
        j += 1

    second_lap_eigenvalue = lap_eigenvalues[j]
    x = [
        # nx.wiener_index(g),
        # nx.schultz_index(g),
        # nx.gutman_index(g),
        first_adj_eigenvalue,
        second_adj_eigenvalue,
        graph_energy,
        second_lap_eigenvalue,


        a.shape[0]

    ]
    return np.array(x)

def perform_node_features(fn, fn2, fn_apg, fn2_apg, mu, limit=5, rev=False):


    atom_graph_1, _ , _ = build_atom_graph(fn, fn_apg, limit=limit, mut=mu)
    atom_graph_2, _ , _ = build_atom_graph(fn2, fn2_apg, limit=limit, mut=mu)
    res_graph_1  = build_res_graph(fn, fn_apg, limit=limit, mut=mu)
    res_graph_2 = build_res_graph(fn2, fn2_apg, limit=limit, mut=mu)
    atom_cnt_1 = len(atom_graph_1.vs)
    atom_cnt_2 = len(atom_graph_2.vs)
    res_cnt_1 = len(res_graph_1.vs)
    res_cnt_2 = len(res_graph_2.vs)
    atom = np.concatenate([
        get_node_features(atom_graph_1)], axis=1)
    atom2 = np.concatenate([

            get_node_features(atom_graph_2)], axis=1)
    res = np.concatenate([
        
        get_node_features(res_graph_1)], axis=1)
    res2 = np.concatenate([

        get_node_features(res_graph_2)], axis=1)

    wild = np.concatenate([
        get_node_stats(atom),
        get_node_stats(res),
        get_global_features(atom_graph_1, atom),
        get_global_features(res_graph_1, res),
    ])

    mut = np.concatenate([
        get_node_stats(atom2),
        get_node_stats(res2), 
        get_global_features(atom_graph_2, atom2),
        get_global_features(res_graph_2, res2),
    ])
    
    if rev:
        y = np.concatenate([mut, wild])
    else:
        y = np.concatenate([wild, mut])
        
    return y

def has_mutation(fn, mut, renum=False, last_resort=False):
    chains = {}
    pre_chain = {}
    chain = mut[0]
    wildtype = mut[2]
    num = mut[3:-1]
    mutant = mut[-1]
    print(renum)
    found = False
    with open(fn, 'r') as f:
        for j, ln in enumerate(f.readlines()):
            if ln[:4] != "ATOM":
                continue
            if renum:
                if not ln[21] in chains:
                    chains[ln[21]] = 0
                    pre_chain[ln[21]] = 0
                if pre_chain[ln[21]] != ln[22:27].strip():
                    chains[ln[21]] += 1
                    pre_chain[ln[21]] = ln[22:27].strip()

            if ln[21] == chain:
                #print(chains[ln[21]])
                if (renum and str(chains[ln[21]]) == num) or (not renum and ln[22:27].strip() == num):
                #print(rev_mp[ln[17:20]])
                #if rev_mp[ln[17:20]] != wildtype:
                    num = ln[22:27].strip()
                    if not rev_mp[ln[17:20]].strip() == wildtype:
                        if last_resort:
                            wildtype = rev_mp[ln[17:20]].strip()
                    else:
                        found = True
                    break
    return found, chain + ":" + wildtype + filter_letter(num) + mutant
def filter_letter(x):
    if x[-1].isalpha():
        return x[:-1]
    return x

def train_model_reg(model, data, ys, epochs=1):
    import random
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    batch_size = 8
    print_amt = 60

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

        for cm, i in enumerate(idx):


            inputs, labels = data[i], ys[i].to(device)
     
            data_batch_1, data_batch_2 = inputs[0], inputs[1]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None
            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]
    
            outputs, rep = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2,
                    ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats)
            loss = loss_fn(outputs.flatten(), labels)
            loss = loss/batch_size
            loss.backward()



            running_loss += loss.item()*batch_size
            if ((cm+1) % batch_size == 0) or (cm+1 == len(data)):
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            if cm>0 and cm%(batch_size*print_amt)==0:
                print(running_loss/cm)
                """
                with open("./log.txt", "a") as f:
                    f.write(f"{i+1}:{running_loss/(print_amt*batch_size)}\n")
                running_loss = 0.
                """

            
    torch.save(model.state_dict(), f"./pretrained_model.pt")
    return True

def train_model_clf(model, data, ys, epochs=1):
    import random
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    batch_size = 8
    print_amt = 60

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
        for cm, i in enumerate(idx):


            if i >= 4169:
                inputs, labels = data[i], torch.Tensor([1]).to(device)
            else:
                inputs, labels = data[i], torch.Tensor([0]).to(device)
     
            data_batch_1, data_batch_2 = inputs[0][:,:max_feat_len], inputs[1][:,:max_feat_len]
            edge_index_1, edge_index_2 = inputs[2], inputs[3]
            
            edge_type_1, edge_type_2 = inputs[4], inputs[5]
            edge_weights_1, edge_weights_2, global_feats = None, None, None
            if len(inputs) > 6:
                edge_weights_1, edge_weights_2 =  inputs[6], inputs[7]
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]
            if not edge_index_1.dim() == 2:
                edge_index_1 = edge_index_1.unsqueeze(1)
            if not edge_index_2.dim() == 2:
                edge_index_2 = edge_index_2.unsqueeze(1)
            outputs, rep = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1,
                    ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats)
            loss = loss_fn(outputs.flatten(), labels)
            loss = loss/batch_size
            loss.backward()

            if i >= 4169:
                inputs, labels = data[i], torch.Tensor([0]).to(device)
            else:
                inputs, labels = data[i], torch.Tensor([1]).to(device)


            outputs, rep = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2,
                    ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats)
            loss = loss_fn(outputs.flatten(), labels)
            loss = loss/batch_size
            loss.backward()
            

            
            running_loss += loss.item()*batch_size
            if ((cm+1) % batch_size == 0) or (cm+1 == len(data)):
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            if cm>0 and cm%(batch_size*print_amt)==0:
                print(running_loss/cm)


            
    torch.save(model.state_dict(), f"./pretrained_model.pt")
    return True


max_feat_len = 60
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

        xx = np.concatenate(sequence_representations,axis=0)

    atom_rep = []
    for j in aseq:
        
        atom_rep.append(xx[j])

    return np.array(atom_rep)



@torch.no_grad()
def val_model(model, data, ys, epoch=-1, get_reps=False):
    model.eval()
    fin = []
    reps = []

    with torch.no_grad():
        for i in range(len(data)):
            mm, rev = data[i]

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
            if len(inputs) > 8 and use_global_feats:
                global_feats = inputs[8]
            if edge_type_1.dim() == 2:
                edge_type_1 = torch.Tensor([4,4]).to(device)
                edge_index_1 = torch.Tensor([[1,0],[0,1]]).to(device)
            if edge_type_2.dim() == 2:
                edge_type_2 = torch.Tensor([4,4]).to(device)
                edge_index_2 = torch.Tensor([[1,0],[0,1]]).to(device)

            x = edge_weights_1.shape[0]
            x2 = edge_weights_2.shape[0]
            # print(f"batch : {data_batch_1.shape}")
            # print(f"batch : {data_batch_2.shape}")
            # print(f"batch : {edge_type_1.shape}")
            # print(f"batch : {edge_type_2.shape}")
            # print(f"batch : {edge_index_1.shape}")
            # print(f"batch : {edge_index_2.shape}")
            # print()
       


            outputs, rep = model(data_batch_1, data_batch_1, edge_index_1, edge_index_1, edge_type_1, edge_type_1, ew1 = edge_weights_1, ew2 = edge_weights_1, global_feats=global_feats )
            outputs, rep2 = model(data_batch_2, data_batch_2, edge_index_2, edge_index_2, edge_type_2, edge_type_2, ew1 = edge_weights_2, ew2 = edge_weights_2, global_feats=global_feats )

            if get_reps:
                if rev == 0:
                    rep = torch.cat([rep, rep2])
                else:
                    rep = torch.cat([rep2, rep])

                reps.append(rep.flatten().cpu().numpy())


    if get_reps:
        return reps
    return fin


def find_mut(mut, pos):
    if fn == "t56":
        pos = str(int(pos) - 9)
        mut = chain + "." + orig + pos + new

    if os.path.exists(f"./foldx_pdbs/{pdb}_Repair.pdb"):
        found, m2 = has_mutation(f"./foldx_pdbs/{pdb}_Repair.pdb", mut,  renum=False)
        if found:
            return True, m2
        found, m2 = has_mutation(f"./foldx_pdbs/{pdb}_Repair.pdb", mut,  renum=True)
        if found:
            return True, m2

    found, m2 = has_mutation( f"./skempi_pdb/{pdb}.pdb", mut, renum=False)
    if found:
        return False, m2 
    found, m2 = has_mutation( f"./skempi_pdb/{pdb}.pdb", mut, renum=True)
    if found:
        return False, m2
    found, m2 = has_mutation( f"./skempi_pdb/{pdb}.pdb", mut, renum=True,last_resort=True)
    return False, m2

    
import shutil
fn = "t56"
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
if fn == "spike-ace2":
    mut_col = 'chain.mut'
    y_col = "enrichment_ratio"
    chain_col = "biounit_chain"
if fn == "t55":
    mut_col = 'chain.mut'
    y_col = "ddG"
    chain_col = "biounit_chain"
if fn == "t56":
    mut_col = 'chain.mut'
    y_col = "ddG"
    chain_col = "biounit_chain"


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
if fn == "spike-ace2":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[chain_col] = "FG"
if fn == "t55":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[pro_col] = "3r2x"
    df[chain_col] = "ABC"
if fn == "t56":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[pro_col] = "26capri"
    
    df[chain_col] = "ABCDEFGHI"



numm = df.shape[0]
pdbs = df[pro_col].unique()
kf = KFold(10)

df = df[:numm]
print(df[y_col])
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)



def get_esm_repr2(seq):
    chain = mut[0]
    batch_labels, batch_strs, batch_tokens = batch_converter(seqs)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    af = esm_model(batch_tokens.to(device), repr_layers=[30], return_contacts=False)
    token_representations = af["representations"][30].cpu().detach().numpy().squeeze()

    if token_representations.shape[0] > 8:
        xx = token_representations[1:token_representations.shape[0]-1].squeeze()
    else:
        sequence_representations = []
        for i, tok_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tok_len - 1])
            print(tok_len)
            print(sequence_representations[i].shape)
        xx = np.concatenate(sequence_representations,axis=0).squeeze()
    yy = xx.mean(axis=0)
    if yy.shape[0] != 640:
        return yy.mean(axis=0)
    return yy

def hydrogen_graphs():
    out2 = []
    for comb in [[0,0], [0,1], [1,0], [1,1]]:
        super_hgraph, _, _ = build_atom_graph(fn1, apg1, filt, backbone_comb=comb, mut=mu, interfaceResidues=None, edge_subset=[7,8])
        sub_hgraph, _, _ = build_atom_graph(fn1, apg1, filt, backbone_comb=comb,  mut=mu, interfaceResidues=mut[0] + "/" + mut[3:-1], edge_subset=[7,8])
        super_hgraph2, _, _ = build_atom_graph(fn2, apg2, filt, backbone_comb=comb, mut=mu, interfaceResidues=None, edge_subset=[7,8])
        sub_hgraph2, _, _ = build_atom_graph(fn2, apg2, filt, backbone_comb=comb, mut=mu, interfaceResidues=mut[0] + "/" + mut[3:-1], edge_subset=[7,8])
        out = np.concatenate([get_node_features(super_hgraph).mean(axis=0), np.array([len(super_hgraph.vs), len(super_hgraph.es)]),
                            get_node_features(sub_hgraph).mean(axis=0), np.array([len(sub_hgraph.vs), len(sub_hgraph.es)]),
                                get_node_features(super_hgraph2).mean(axis=0),  np.array([len(super_hgraph2.vs), len(super_hgraph2.es)]),
                                get_node_features(sub_hgraph2).mean(axis=0), np.array([len(sub_hgraph2.vs), len(sub_hgraph2.es),])])
        out2.append(out)
    out2 = np.concatenate(out2)
    print(out2.shape)
    print(out2[:15])
    return out

yunex = {
    'H': 0,
    'B': 1,
    'E': 2,
    'G': 3,
    'I': 4,
    'T': 5,
    'S': 6,
    '-': 7,  # coil
    'P': 8
}
def get_indp_feats(fn1, mut, mu, rev=False):

    if rev:
        orig, chain, pos, new = mut[2], mut[0], mut[3:-1], mut[-1]
    else:
        new, chain, pos, orig = mut[2], mut[0], mut[3:-1], mut[-1]
    muttype = rev_mg[orig] * 20 + rev_mg[new]
    instab = []
    for rng in np.arange(5.0, 21.0, 5.0):
        c = 0.0
        d = 0.0
        with open(fn1, "r") as f:
            for i, line in enumerate(f.readlines()):
                ln = line.split()
                if line[:4] != 'ATOM':
                    continue
                pt = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
                temp = float(line[60:65].strip())
                if point_distance(pt, mu) > rng:
                    continue
                c += temp
                d += 1
            instab.append(c/d)

    print(f"MUT: {mut}")
    parser = PDBParser()
    structure = parser.get_structure("protein", fn1)
    with open(fn1, "r") as f:
        lines = f.readlines()

    if not lines[0].startswith("HEADER"):
        lines.insert(0, "HEADER    DUMMY HEADER LINE\n")
        with open(fn1, "w") as f:
            f.writelines(lines)
    model = structure[0]

    dssp = DSSP(model, fn1, dssp='mkdssp')

    # Pick a residue by (chain_id, resseq)
    chain_id = chain
    resseq = pos  # for example
    key = (chain_id, int(resseq))
    try:
        # Get DSSP data
        residue_data = dssp[key]
        sasa = residue_data[3]              # Relative ASA
        secondary_structure = yunex[residue_data[2]]  # H=alpha-helix, E=beta-strand, etc.
    except:
        sasa = -1
        secondary_structure = -1
        residue_data = [-1 for _ in range(6)]


    return np.array([muttype] + instab + [sasa] + [secondary_structure, residue_data[4] , residue_data[5] ])

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import get_surface, ResidueDepth
def biopython_feats(fn1):


    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("example", fn1)
    model = structure[0]  # Use first model
    dssp = DSSP(model, fn1, dssp='mkdssp')
    for key in dssp.keys():
        res_info = dssp[key]
        ss = res_info[2] 
        rsa = res_info[3] 
    surface = get_surface(model)

    # Calculate residue depth
    rd = ResidueDepth(model, surface)

    for chain in model:
        for residue in chain:
            try:
                depth = rd[residue]
                print(f"{residue}: depth = {depth}")
            except KeyError:
                continue 

# clusters = cluster()
glob = []
filt = 20.0

preda = np.zeros(df.shape[0])
inputs = []
outputs = []
hydrogens = []
import sys


def mutate():
    with open("individual_list.txt", "w") as f:
        f.write(f"{orig}{chain}{pos}{new};")
    try:
        if not os.path.exists(f"./foldx_pdbs/Dif_{at2}.fxout"):
            os.system(f"./foldx --command=BuildModel --pdb={pdb}_Repair.pdb --pdb-dir=./foldx_pdbs/ --output-dir=./foldx_pdbs/ --mutant-file=individual_list.txt")
            os.rename(f"./foldx_pdbs/Dif_{pdb}_Repair.fxout", f"./foldx_pdbs/Dif_{at2}.fxout")
            os.rename(f"./foldx_pdbs/{pdb}_Repair_1.pdb", fn2)
        with open(f"./foldx_pdbs/Dif_{at2}.fxout", "r") as f:
            r = float(f.readlines()[-1].split()[1])
        
    except:

        if not os.path.exists(f"./foldx_pdbs/Dif_{at2}.fxout"):
            os.system(f"./foldx --command=BuildModel --pdb={pdb}.pdb --pdb-dir=./skempi_pdb/ --output-dir=./foldx_pdbs/ --mutant-file=individual_list.txt")
            os.rename(f"./foldx_pdbs/Dif_{pdb}.fxout", f"./foldx_pdbs/Dif_{at2}.fxout")
            os.rename(f"./foldx_pdbs/{pdb}_1.pdb", fn2)
        with open(f"./foldx_pdbs/Dif_{at2}.fxout", "r") as f:
            r = float(f.readlines()[-1].split()[1])

    return r

foosa = []
for i in range(0,min(numm, df.shape[0])):
    

    
    if i >= 4169:
        idx = i - 4169
    else:
        idx = i 
    at1 = f"{fn}_{idx}_0"
    at2 = f"{fn}_{idx}_1"
    mut = df.loc[idx, mut_col]
    


    
    

    print(mut)
    pdb, orig, chain, pos, new = df.loc[idx, pro_col], mut[2], mut[0], mut[3:-1], mut[-1]
    found, mut = find_mut(mut, pos)

    pdb, orig, chain, pos, new = df.loc[idx, pro_col], mut[2], mut[0], mut[3:-1], mut[-1]
    print(f"Found: {found}")

    if os.path.exists(f"./foldx_pdbs/{at1}_Repair.pdb"):
        shutil.copy(f"./foldx_pdbs/{at1}_Repair.pdb", f"./foldx_pdbs/{pdb}_Repair.pdb")

    if not os.path.exists(f"./foldx_pdbs/{pdb}_Repair.pdb"):
        os.system(f"./foldx --command=RepairPDB --pdb={pdb}.pdb --pdb-dir=./skempi_pdb/ --output-dir=./foldx_pdbs/ --output-file={at1}_Repair")

    pref = f"{pdb}_{orig}_{chain}_{pos}_{new}"
    fn1 = f"./foldx_pdbs/{pdb}_Repair.pdb"
    if not found:
        fn1 = f"./skempi_pdb/{pdb}.pdb"
    fn2 = f"./foldx_pdbs/{at2}.pdb"
    fx1 = f"./foldx_pdbs/SD_{pdb}_Repair.fxout"
    fx2 = f"./foldx_pdbs/SD_{at2}.fxout"

    apg1 = f"./foldx_pdbs/{pdb}_Repair.clean.contacts"
    apg2 = f"./foldx_pdbs/{at2}.clean.contacts"
    try:
        mu = find_mut_site(f"./foldx_pdbs/{pdb}_Repair.pdb", mut.upper(), chains=df.loc[idx, chain_col].replace("_", ""))
    except:
        mu = find_mut_site(f"./skempi_pdb/{pdb}.pdb", mut.upper(), chains=df.loc[idx, chain_col].replace("_", ""))
    if mu is None:
        raise Exception("MISSING")
    if i <= 4169:
        foldx_ddg = mutate()
        foosa.append(foldx_ddg)
    else:
        foosa.append(-foosa[i-4169])



    if not os.path.exists(fx1):
        os.system(f"./foldx --command=SequenceDetail --pdb={pdb}_Repair.pdb --pdb-dir=./foldx_pdbs/ --output-dir=./foldx_pdbs/")
    if not os.path.exists(apg1):
        os.system(f"python3 ./arpeggio-master/arpeggio-master/clean_pdb.py ./foldx_pdbs/{pdb}_Repair.pdb")
        os.system(f"python3 ./arpeggio-master/arpeggio-master/arpeggio.py ./foldx_pdbs/{pdb}_Repair.clean.pdb")

    if not os.path.exists(fn2):
        with open("individual_list.txt", "w") as f:
            f.write(f"{orig}{chain}{pos}{new};")
        try:
            os.system(f"./foldx --command=BuildModel --pdb={pdb}_Repair.pdb --pdb-dir=./foldx_pdbs/ --output-dir=./foldx_pdbs/ --mutant-file=individual_list.txt")
            os.rename(f"./foldx_pdbs/{pdb}_Repair_1.pdb", fn2)
        except:
            os.system(f"./foldx --command=BuildModel --pdb={pdb}.pdb --pdb-dir=./skempi_pdb/ --output-dir=./foldx_pdbs/ --mutant-file=individual_list.txt")
            os.rename(f"./foldx_pdbs/{pdb}_1.pdb", fn2)

    if not os.path.exists(fx2):
        os.system(f"./foldx --command=SequenceDetail --pdb={at2}.pdb --pdb-dir=./foldx_pdbs/ --output-dir=./foldx_pdbs/")
    if not os.path.exists(apg2):
        os.system(f"python3 ./arpeggio-master/arpeggio-master/clean_pdb.py ./foldx_pdbs/{at2}.pdb")
        os.system(f"python3 ./arpeggio-master/arpeggio-master/arpeggio.py ./foldx_pdbs/{at2}.clean.pdb")
    if i >= 4169:
        inputs.append((inputs[i-4169][0], 1))
        outputs.append(-df.loc[i-4169, y_col])
        feat = glob[i-4169]
        foldx = feat[:87]
        apg = feat[87:87+90]
        gph = feat[87+90:87+90+128]
        ot = feat[87+90+128:]
        ot[0] = rev_mg[new] * 20 + rev_mg[orig]
        w, m, d = foldx[:29], foldx[29:58], foldx[58:]
        foldx = np.concatenate([m, w, -d])
        w, m, d = apg[:30], apg[30:60], apg[60:]
        apg = np.concatenate([m, w, -d])
        w, m = gph[:64], gph[64:]
        gph = np.concatenate([m,w])
        ff = np.concatenate([foldx, apg, gph, ot])

        glob.append(ff)

        continue

    if not os.path.exists(f"./out_feats/{pref}_rev.npy"):
        try:
            idp = get_indp_feats(f"./skempi_pdb/{pdb}.pdb", mut, mu)
        except:
            idp = get_indp_feats(f"./foldx_pdbs/{pdb}_Repair.clean.pdb", mut, mu)
        feat = np.concatenate([
            perform_foldx(at1, at2, mut, df.loc[idx, chain_col], bw=False),
            perform_appreg(apg1, apg2, mut, df.loc[idx, chain_col], bw=False),
            perform_node_features(fn1, fn2, apg1, apg2,  mu, limit=filt, rev=False),
            idp
        ])
        np.save(f"./out_feats/{pref}.npy", feat)
        
        try:
            idp = get_indp_feats(f"./skempi_pdb/{pdb}.pdb", mut, mu, rev=True)
        except:
            idp = get_indp_feats(f"./foldx_pdbs/{pdb}_Repair.clean.pdb", mut, mu, rev=True)
        feat2 = np.concatenate([
            perform_foldx(at1, at2, mut, df.loc[idx, chain_col], bw=True),
            perform_appreg(apg1, apg2, mut, df.loc[idx, chain_col], bw=True),
            perform_node_features(fn1, fn2, apg1, apg2,  mu, limit=filt, rev=True),
            idp
        ])

        np.save(f"./out_feats/{pref}_rev.npy", feat2)
    else:

        feat = np.load(f"./out_feats/{pref}.npy")



    glob.append(feat)
            
        


    
    
    print(idx)
    print(pref)

    if os.path.exists(f"./deta/{pref}.pkl") and not redo:
        inputs.append((f"./deta/{pref}.pkl", 0))
        continue

    
    foldx1, _ = per_residue_foldx(f"./foldx_pdbs/SD_{pdb}_Repair.fxout")
    foldx2, _ = per_residue_foldx(f"./foldx_pdbs/SD_{at2}.fxout")
    
    
    points1, seq1, aseq1, inf1, ind1 = get_universe_points(fn1, filt, mu, 
            foldx1, 
            )
    
    points2, seq2, aseq2, inf2, ind2 = get_universe_points(fn2, filt, mu, 
            foldx2, 
            )

    atom1, edges1, dists1 = build_atom_graph(fn1, apg1, filt, mut=mu)
    atom2, edges2, dists2 = build_atom_graph(fn2, apg2, filt, mut=mu)

    repr1 = get_node_features(atom1)
    repr2 = get_node_features(atom2)

    assert np.array(inf1).shape[0] == repr1.shape[0]
    repr1 = np.concatenate([
        np.array(inf1),
        repr1,

        ], 
        axis=1)
    repr2 = np.concatenate([
        np.array(inf2),
        repr2,

        ], 
        axis=1)

    edges1, ei1, dists1 = build_viteoris_rips2(points1, edges1, dists1)
    edges2, ei2, dists2 = build_viteoris_rips2(points2, edges2, dists2)
    if len(edges1) == 0:
        edges1 = np.array([[1,0], [0,1]])
        ei1 = np.array([[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,]])
        dists1 = np.array([5,5])
    if len(edges2) == 0:
        edges2 = np.array([[1,0], [0,1]])
        ei2 = np.array([[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,]])
        dists2 = np.array([5,5])
    print(f"edges1: {len(edges1)}")
    print(f"edges2: {len(edges2)}")
    print(np.array(inf1).shape)
    print(np.array(repr1).shape)

    print(edges1.shape)
    print(dists1.shape)

    assert points1.shape[0] == repr1.shape[0]
    assert edges1.shape[1] == dists1.shape[0]
    assert edges2.shape[1] == dists2.shape[0]
    with open(f"./deta/{pref}.pkl", "wb") as h:
        inp = (repr1, repr2, edges1, edges2, ei1, ei2, dists1, dists2)
        inputs.append((f"./deta/{pref}.pkl", 0))
        print("SAVED!")
        pickle.dump(inp, h, protocol=pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(f"./deta/{pref}.pkl"):
        print(pref)
        raise Exception("???")


np.save(f"foldx_{fn}_ddg.npy", np.array(foosa))

if not os.path.exists(f"{fn}_X.npy"):
    np.save(f"{fn}_X.npy", np.array(glob))

glob = np.load(f"{fn}_X.npy")


print(f"GLOBAL: {glob.shape}")
    
def get_ar(a, idx):
    out = []
    for i in idx:
        out.append(a[i])
    return out

y1 = df[y_col][:numm]

outputs = torch.tensor(y1, dtype=torch.float32, device='cpu')
with open(f"./deta/1AK4_G_A_109_H.pkl", 'rb') as handle:
    ex = pickle.load(handle)

model = CombinedModel(ex[0][:,:max_feat_len].shape[1], supcon=True).to(device)
if os.path.exists(f"./pretrained_model.pt"):
    try:
        model.load_state_dict(torch.load(f"./pretrained_model.pt", weights_only=True))
        print("MODEL LOADED")
    except Exception as e:
        print(e)
        pass
reps = val_model(model, inputs[:numm], outputs[:numm], get_reps=True)
reps = np.array(reps)


a1 = np.nan_to_num(np.concatenate([
np.load(f"{fn}_X.npy")[:numm], 

reps[:numm,:],
], axis=1), 0)
np.save(f"{fn}_aa.npy", a1)





    
