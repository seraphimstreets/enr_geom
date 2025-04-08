
from torch import nn
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, JumpingKnowledge, BatchNorm, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch.nn.functional import normalize

class CombinedModel(nn.Module):
    def __init__(self, input_dim, num_relations=15, num_layers=4, dropout=0.10, jumping=False, num_heads=1, global_feats=0, supcon=False):
        super(CombinedModel, self).__init__()
       
        hidden_channels = 32
        self.relational = not edge_as_feats
        self.jumping = jumping
        self.residual = True
        self.edge_attr = True
        self.supcon = supcon
        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()


        if self.relational:
            self.convs1.append(RGATConv(input_dim, hidden_channels, num_relations=num_relations, dropout=dropout, concat=False))

        else:
            self.convs1.append(GATConv(input_dim, hidden_channels, dropout=dropout))
        self.bns1.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            if self.relational:
                self.convs1.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))

            else:
                self.convs1.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=False))
            self.bns1.append(LayerNorm(hidden_channels))
        
        
        self.convs2 = nn.ModuleList()
        
        if self.relational:
            self.convs2.append(RGATConv(input_dim, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))
        else:
            self.convs2.append(GATConv(input_dim, hidden_channels, dropout=dropout))
        self.bns2.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            if self.relational:
                self.convs2.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, heads=num_heads, dropout=dropout, concat=False))
            else:
                self.convs2.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=False))
            self.bns2.append(LayerNorm(hidden_channels))

        self.jk1 = JumpingKnowledge("cat")
        self.jk2 = JumpingKnowledge("cat")

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
            self.mid_lins.append(nn.MultiheadAttention(hc2,  num_heads=2))
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

        if self.supcon:
            x = self.lin4(x)
            x1 = torch.norm(x, 2)
            return x/x1, torch.cat(x1s, axis=1).mean(0)
        return x, lins[-1]