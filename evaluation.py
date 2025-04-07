import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.decomposition import PCA
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
import random
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from tabpfn import TabPFNRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
import smogn
from sklearn.feature_selection import RFECV, VarianceThreshold,SelectFromModel

def preprocess(x):
    x = np.nan_to_num(x, 0, posinf=1000, neginf=-1000)
    hf = int(x.shape[1]/2)
    #return x[:,:hf] - x[:,hf:]
    out = []
    for j in range(x.shape[1]):


        out.append(x[:,j:j+1])
    return np.concatenate(out, axis=1)
def preprocess0(x):
    out = []

    for i in range(x.shape[0]):
        feat = x[i]
        foldx = feat[:87]
        apg = feat[87:87+90]
        gph = feat[87+90:87+90+128]
        ot = feat[87+90+128:87+90+137]
        gu = feat[87+90+137:]
        out.append(np.concatenate([foldx[54:], apg[60:], gph[64:] - gph[:64], ot, gu[:128] - gu[128:]]))
    return np.array(out)

def blind_test(train_X, train_y, test_X, test_y):
    mod = CatBoostRegressor( iterations=10000,  verbose=0).fit(train_X, train_y)
    pred = mod.predict(test_X)
    get_metrics(test_y, pred)
def get_metrics(test_y, pred):
    RMSD = np.sqrt(mean_squared_error(test_y,pred))
    pearsonr = scipy.stats.pearsonr(test_y,pred)
    kendall = scipy.stats.kendalltau(test_y,pred)
    print(pearsonr)
    print(kendall)
    print(RMSD)

def cluster():
    cluster_dic = {}

    # Core binding factor beta, CBF
    cluster_dic[1] = ['1E50']
    # 4-helical cytokines
    cluster_dic[2] = ['1A22', '1BP3', '1IAR', '3G6D', '3L5X', '3LB6', '3MZG', '3N06', '3N0P', '3NCB', '3NCC', '3S9D', '3SE3', '3SE4',
                    '4I77', '4NKQ', '4RS1']
    # Leucine-rich repeats
    cluster_dic[3] = ['1A4Y', '1N8Z', '1YY9', '1Z7X', '3N85', '4KRL', '4KRO', '4KRP', '4Y61']
    # CI-2 family of serine protease inhibitors
    cluster_dic[4] = ['1ACB', '1CSE', '1SBN', '1SIB', '1TM1', '1TM3', '1TM3', '1TM4', '1TM5', '1TM7', '1TMG', '1TO1', '1Y1K', '1Y33', '1Y34',
                    '1Y3C', '1Y3B', '1Y3D', '1Y48']
    # Immunoglobulin-related
    cluster_dic[5] = ['1AHW', '1DVF', '1JRH', '2U82', '4BFI', '4MYW', '4OFY', '4X4M', '4YFD', '4YH7', '3U82']
    # Retrovirus capsid protein N-terminal domain
    cluster_dic[6] = ['1AK4', '1M9E']
    # MHC antigen-recognition domain
    cluster_dic[7] = ['1AO7', '1BD2', '1LP9', '1MI5', '1OGA', '1QSE', '2AK4', '2BNQ', '2BNR', '2E7L', '2J8U', '2JCC', '2OI9', '2P5E', 
                    '2UWE', '2VLR', '3D3V', '3H9S', '3PWP', '3QDG', '3QDJ', '3QIB', '4FTV', '4JFD', '4JFF', '4L3E', '4MNQ', '4N8V',
                    '4OZG', '4P23', '4P5T', '5E9D', '3C60', '3QFJ','4JFE']
    # Microbial ribonucleases
    cluster_dic[8] = ['1B2S', '1B3S', '1BRS', '1X1W', '1X1X', '1B2U']
    # Snake toxin-like
    cluster_dic[9] = ['1B41', '1FSS', '1KTZ', '1MAH', '1REW', '3BT1']
    # Cystine-knot cytokines
    cluster_dic[10] = ['1BJ1', '1CZ8', '2B0U', '3B4V', '3SEK', '4HSA']
    # Ubiquitin-related
    cluster_dic[11] = ['1C1Y', '1LFD', '1XD3', '2OOB', '2REX', '3KUD', '3M62', '3M63', '3Q3J', '4G0N', '5CXB', '5CYK', '5E6P']
    # UBC-like
    cluster_dic[12] = ['1C4Z', '1S1Q']
    # BPTI-like
    cluster_dic[13] = ['1CBW', '2FTL']
    # Kazal-type serine protease inhibitors
    cluster_dic[14] = ['1CHO', '1CSO', '1CT0', '1CT4', '1CT2', '1PPF', '1R0R', '1SGD', '1SGE', '1SGN', '1SGP', '1SGQ', '1SGY', 
                    '2NU0', '2NU1', '2NU2', '2NU4', '2SGP', '2SGQ', '3HH2', '3SGB']
    # RIFT-related
    cluster_dic[15] = ['1DAN', '1EAW', '1F5R', '1FY8', '1SMF', '2B42', '3BN9', '3BTD', '3BTE', '3BTF', '3BTG', '3BTH', '3BTM', 
                    '3BTT', '3BTQ' ,'3BTW', '3TGK', '3VR6']
    # Lysozyme-like
    cluster_dic[16] = ['1DQJ', '1KIR', '1KIQ', '1KIP', '1MLC', '1VFB', '1XGP', '1XGQ', '1XGU', '1XGT', '1XGR', '1YQV', '2I26',
                    '3HFM']
    # SH3
    cluster_dic[17] = ['1EFN', '1GCQ', '1YCS']
    # His-Me finger endonucleases
    cluster_dic[18] = ['1EMV', '1FR2', '2GYK', '2VLN', '2VLO', '2VLP', '2VLQ', '2WPT', '']
    # Cell-division protein ZipA, C-terminal domain / peptide
    cluster_dic[19] = ['1F47', '3EQS', '3EQY', '3LNZ', '3RF3', '3UIG', '3UII']
    # Bacterial immunoglobulin/albumin-binding domains
    cluster_dic[20] = ['1FC2', '3MZW']
    # Immunoglobulin-binding domains
    cluster_dic[21] = ['1FCC', ]
    # Class I glutamine amidotransferase-like
    cluster_dic[22] = ['1FFW', ]
    # Ribosomal protein L31e/gp120 outer domain
    cluster_dic[23] = ['1GC1', '4JPK']
    # PMP inhibitors
    cluster_dic[24] = ['1GL0', '1GL1']
    # P-loop domains-related
    cluster_dic[25] = ['1GRN', '1GUA', '1K8R', '3EG5', '4EKD', '4GNK', '4PWX', '5TAR', '5XCO', '1E96']
    # Core binding factor beta, CBF
    cluster_dic[26] = ['1H9D', '1HE8']
    # Bacterial enterotoxins
    cluster_dic[27] = ['1JCK', '1SBB', '3W2D']
    # beta-propeller
    cluster_dic[28] = ['1JTD', '1NCA', '1NMB', '3NVN', '3NVQ', '3QHY', '4YEB']
    # a+b domain in beta-lactamase/transpeptidase-like proteins
    cluster_dic[29] = ['1JTG', ]
    # Domain in virus attachment proteins
    cluster_dic[30] = ['1KAC', '2J12', '2J1K']
    # Nuclear receptor coactivator ACTR
    cluster_dic[31] = ['1KBH', ]
    # HAD domain-related
    cluster_dic[32] = ['1MHP', '2B2X']
    # Ecotin, trypsin inhibitor
    cluster_dic[33] = ['1N8O', ]
    # Domain in virus attachment proteins
    cluster_dic[34] = ['1P69', '1P6A']
    # beta-lactamase-inhibitor protein, BLIP
    cluster_dic[35] = ['1S0W', '2G2U', '2G2W']
    # SMAD/FHA domain
    cluster_dic[36] = ['1U7F', ]
    # Inhibitor of vertebrate lysozyme, Ivy
    cluster_dic[37] = ['1UUZ', ]
    # Insulin-like
    cluster_dic[38] = ['1WQJ', '2DSQ']
    # ADP-ribosylation
    cluster_dic[82] = ['2A9K', ]
    # Coronavirus spike protein receptor-binding domain
    cluster_dic[39] = ['2AJF', '4ZS6']
    # EGF-related
    cluster_dic[40] = ['2AJF', '2AW2']
    # Cytochrome c
    cluster_dic[41] = ['2B0Z', '2B10', '2B11', '2B12', '2PCB', '2PCC']
    # IL8-related
    cluster_dic[42] = ['2BDN', ]
    # profilin-like
    cluster_dic[43] = ['2BTF', ]
    # ARM repeat
    cluster_dic[44] = ['2C0L', '3SF4', '4FZA', '4G2V', '4JGH', '4NZW', '4OR7', '4WND', '4O27']
    # Concanavalin A-like
    cluster_dic[45] = ['2C5D', '2HLE', '4L0P', '4RA0']
    # Ankyrin repeat
    cluster_dic[46] = ['2DVW', '3AAA', '4HRN']
    # Efb C-domain-like
    cluster_dic[47] = ['2GOX', '3D5R', '3D5S']
    # Glutathione S-transferase (GST)-C
    cluster_dic[48] = ['2HRK', ]
    # TIMP-like
    cluster_dic[49] = ['2J0T', ]
    # HPr-like
    cluster_dic[50] = ['2JEL', ]
    # SAM/DNA-glycosylase
    cluster_dic[51] = ['2KSO', ]
    # Metalloproteases ("zincins") catalytic domain
    cluster_dic[52] = ['2NYY', '2NZ9', '3KBH']
    # Nuclease A inhibitor (NuiA)-related
    cluster_dic[53] = ['2O3B', ]
    # Subtilisin inhibitor
    cluster_dic[54] = ['2SIC', ]
    # Viral protein domain
    cluster_dic[55] = ['2VIR', '2VIS', '3LZF', '4GXU', '4NM8']
    # Fibronectin type I module-like
    cluster_dic[56] = ['3BK3', ]
    # Glucose permease domain IIB
    cluster_dic[57] = ['3BP8', ]
    # beta-Trefoil
    cluster_dic[58] = ['3BX1', ]
    # Serpins
    cluster_dic[59] = ['3F1S', ]
    # alpha-helical domain in beta-lactamase/transpeptidase-like proteins
    cluster_dic[60] = ['3N4I', ]
    # gp120 inner domain
    cluster_dic[61] = ['3NGB', '3SE8', '3SE9']
    # FMN-binding split barrel
    cluster_dic[62] = ['3NPS', ]
    # Nucleic acid-binding proteins
    cluster_dic[63] = ['3Q8D', ]
    # C-terminal domain in some PLP-dependent transferases
    cluster_dic[64] = ['3R9A', ]
    # Globin-like
    cluster_dic[65] = ['3SZK', ]
    # Carbamate kinase-like
    cluster_dic[66] = ['3WWN', ]
    # FimD N-terminal domain-like
    cluster_dic[67] = ['4B0M', ]
    # omega toxin-related
    cluster_dic[68] = ['4CPA', ]
    # Bifunctional inhibitor/lipid-transfer protein/seed storage 2S albumin/Protein HNS-dependent expression A HdeA
    cluster_dic[69] = ['4CVW', ]
    # bacterioferritin-associated ferredoxin
    cluster_dic[70] = ['4E6K', ]
    # Rap2b (SMA2266)
    cluster_dic[71] = ['4HFK', ]
    # Rossmann-related
    cluster_dic[72] = ['4HRA', ]
    # Hedgehog/intein
    cluster_dic[73] = ['4J2L', ]
    # t-snare proteins
    cluster_dic[74] = ['4JEU', ]
    # Serum albumin-like
    cluster_dic[75] = ['4K71', ]
    # EDD domain
    cluster_dic[76] = ['4LRX', ]
    # Poxvirus L1 protein
    cluster_dic[77] = ['4U6H', ]
    # Viral glycoprotein ectodomain-like
    cluster_dic[78] = ['5C6T', ]
    # Frizzled cysteine-rich domain-related
    cluster_dic[79] = ['5F4E', ]
    # EF-hand-related
    cluster_dic[80] = ['5K39', '5M2O']
    # Chromo domain-like
    cluster_dic[81] = ['5UFQ', '1KNE', '5UFE']
    return cluster_dic

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

def plot_scatters(predictions, titles, actuals):
    # predictions = [predicted_A, predicted_B, predicted_C]
    # titles = ["A", "B", "C"]
    # actuals = [actual_A, actual_B, actual_C]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (pred, title) in enumerate(zip(predictions, titles)):
        ax = axes[i]
        actual = actuals[i]
        sns.scatterplot(x=actual, y=pred, ax=ax, s=10)
        ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color='black', linewidth=1)
        r, _ = pearsonr(actual, pred)
        ax.set_title(f"{title}")
        ax.set_xlabel(r"$\Delta\Delta G_{\text{WT Actual}}$")
        ax.set_ylabel(r"$\Delta\Delta G_{\text{WT Predicted}}$")
        ax.text(0.05, 0.9, f"Pearson: {r:.2f}", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()



nm = "s8338"
rs = 0
lim = 4169
df = pd.read_csv(f"{nm.upper()}.csv",encoding="latin-1").reset_index(drop=True)[:lim]

X = preprocess0(np.nan_to_num(np.load(f"{nm}_aa.npy"),0,))
X2 = preprocess0(np.nan_to_num(np.load(f"t55_aa.npy"),0,))
X = np.concatenate([X, np.load(f"foldx_{nm}_ddg.npy").reshape(-1,1)], axis=1)
X2 = np.concatenate([X2, np.load(f"foldx_t55_ddg.npy").reshape(-1,1)], axis=1)



y = pd.read_csv(f"./{nm.upper()}.csv",encoding="latin-1").reset_index(drop=True)['ddG']
y2 = pd.read_csv(f"./t55.csv",encoding="latin-1").reset_index(drop=True)['ddG']


selector = SelectFromModel(CatBoostRegressor(), threshold="1.25*mean")
selector = selector.fit(X, y)
fimpt = selector.get_support()
Xa = X[:,fimpt]
Xb = X2[:,fimpt]

# X = X[:,:-128]
# if not os.path.exists("feat_impt.npy"):
#     selector = SelectFromModel(CatBoostRegressor())
#     selector = selector.fit(X, y)
#     np.save("feat_impt.npy", selector.get_support())
# fimpt = np.load("feat_impt.npy")
# X = X[:,fimpt]


blind_test(Xa, y, Xb, y2)
print(X.shape)

cam = [a for a in range(15,14,5)]                                                   
for limit in cam:   
    r = preprocess(np.load(f"{nm}_X_{limit}_grph.npy"))
    hf = int(r.shape[1]/2)   
    r = r[:,:]
    X = np.concatenate([X, r], axis=1)

dat = pd.read_csv(f"./{nm.upper()}.csv",encoding="latin-1").reset_index(drop=True)
    # dat = pd.read_csv(f"./corrected_mmCSM_AB.csv")
if nm == "s1131":
    pro_col = "protein"
    ar = np.zeros(X.shape[0])
    with open("./divided-ecod-folds-hgroup.txt", "r") as f:
        hfolds = eval(f.read())
        i = 0
        for _, tes in hfolds:
            ar[tes] = i
            i += 1
        
    df['hgroup_folds'] = ar
if nm == "s645":
    import pickle
    pro_col = "#PDB"
    ar = np.zeros(X.shape[0])
    f = open('./divided-folds.pkl', 'rb')
    divid = pickle.load(f)
    f.close()
    split_folds = []
    for key in divid.keys():
        split_folds.append((divid[key][0], divid[key][1]))
    i = 0
    for _, tes in split_folds:
        ar[tes] = i
        i += 1
        
    df['hgroup_folds'] = ar
if nm == "s8338":
    pro_col = "#PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    df[pro_col] = df['#PDB'].apply(lambda x : x.split("_")[0])
cdic = cluster()
rdic = {}
for k in cdic.keys():
    for v in cdic[k]:
        rdic[v] = k

X = X[:lim]
#X = np.concatenate([X[:lim,:300], X[:lim,1280:1280+300]], axis=1)
Y = y[:lim]
strat = "kfold"
print(X.shape)


#X = np.concatenate([X, df['DGCddG'].values.reshape(-1,1)], axis=1)
n_num = X.shape[0]
result = np.zeros(n_num)[:n_num]
smote = False


def stack_model(tr_idx, val_idx, X, Y):
    tr_mod = []
    val_mod = []
    
    cam = [a for a in range(10,31,5)]
    for limit in cam:

        r = preprocess(np.load(f"{nm}_X_{limit}_grph.npy"))
        hf = int(r.shape[1]/2)
        r = r[:,:]
        X = np.concatenate([X, r], axis=1)

        X_tr, X_val, Y_tr, Y_val = X[tr_idx], X[val_idx], Y[tr_idx], Y[val_idx]
        #mod = ExtraTreesRegressor(n_estimators=300, min_samples_split=3).fit(X_tr, Y_tr)
        #mod = GradientBoostingRegressor(n_estimators=1000, n_iter_no_change=20, validation_fraction=0.1).fit(X_tr, Y_tr)
        mod = CatBoostRegressor(iterations=100,loss_function='RMSE',verbose=False).fit(X_tr, Y_tr)
        pred = mod.predict(X_val)
        RMSD = np.sqrt(mean_squared_error(Y_val,pred))
        pearsonr = scipy.stats.pearsonr(Y_val,pred)
        print(f"Limit {limit}")
        # print(RMSD)
        print(pearsonr)
        tr_mod.append(mod.predict(X_tr))
        val_mod.append(pred)
    
    tr_mod = np.array(tr_mod).T
    val_mod = np.array(val_mod).T
    print(tr_mod.shape)
    smod = CatBoostRegressor(loss_function='RMSE', verbose=False).fit(tr_mod, Y_tr)
    pred = smod.predict(val_mod)
    
    #pred = np.mean(np.array(tr_mod))
    RMSD = np.sqrt(mean_squared_error(Y_val,pred))
    pearsonr = scipy.stats.pearsonr(Y_val,pred)
    print("Stacked model")
    print(RMSD)
    print(pearsonr)
    result[val_idx] = pred
    


        
def stack_model2(tr_idx, val_idx):
    tr_mod = []
    val_mod = []
    
    Y = np.load(f"{nm}_Y.npy")
    inv = 100
    for a in range(0, X.shape[1], inv):
        X_tr, X_val, Y_tr, Y_val = X[tr_idx,a:a+inv], X[val_idx,a:a+inv], Y[tr_idx], Y[val_idx]
        mod = ExtraTreesRegressor(n_estimators=400).fit(X_tr, Y_tr)
        pred = mod.predict(X_val)
        RMSD = np.sqrt(mean_squared_error(Y_val,pred))
        pearsonr = scipy.stats.pearsonr(Y_val,pred)
        print(f"Limit {limit}")
        # print(RMSD)
        print(pearsonr)
        tr_mod.append(mod.predict(X_tr))
        val_mod.append(pred)
    tr_mod = np.array(tr_mod).T
    val_mod = np.array(val_mod).T
    print(tr_mod.shape)
    smod = ExtraTreesRegressor(n_estimators=100).fit(tr_mod, Y_tr)
    pred = smod.predict(val_mod)
    RMSD = np.sqrt(mean_squared_error(Y_val,pred))
    pearsonr = scipy.stats.pearsonr(Y_val,pred)
    print("Stacked model")
    print(RMSD)
    print(pearsonr)
    result[val_idx] = pred
    

        



class MultiTypeEdgeRGCN(MessagePassing):
    def __init__(self, num_node_features, num_relations, num_classes):
        super(MultiTypeEdgeRGCN, self).__init__(aggr='add')  # Add aggregation
        self.node_fc = torch.nn.Linear(num_node_features, 16)  # Node feature transformation
        self.edge_fc = torch.nn.Linear(num_relations, 16)  # Edge type embedding
        self.classifier = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_type):
        # x: node features
        # edge_index: edge index for the graph
        # edge_type: multi-hot encoding for edge types (e.g., [1, 0, 1] for "friend" and "colleague")

        # Message passing
        return self.propagate(edge_index, x=x, edge_type=edge_type)

    def message(self, x_j, edge_type):
        # Apply linear transformations to node and edge features
        edge_weight = self.edge_fc(edge_type)  # Convert multi-type edge feature into weight
        return x_j * edge_weight  # Weighted sum based on edge features

        return F.relu(aggr_out)


from collections import Counter
cnt = Counter()
def fold_pred(train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index].astype(float), Y[test_index].astype(float)
    if smote:
        try:
            X_train = pd.DataFrame(X_train, columns=[f"X_{i}" for i in range(X_train.shape[1])])
            Y_train = pd.DataFrame(Y_train, columns=['Y'])
            reb = pd.concat([X_train,Y_train], axis=1)
            rog = smogn.smoter(data=reb, y="Y")
            X_train, Y_train = rog.iloc[:,:-1], rog.iloc[:,-1]
        except:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

    
    # print(X_train.shape[0])
    # X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    # # CatBoost Pool (optional, but allows for categorical features, etc.)
    # train_pool = Pool(X_train, y_train)
    # val_pool = Pool(X_val, y_val)
    # mod = CatBoostRegressor(    iterations=10000,
    # learning_rate=0.05,
    # depth=8,
    # eval_metric="RMSE",
    # loss_function="RMSE",  # or "MAE", "RMSE", etc.
    # random_seed=42,
    # early_stopping_rounds=50,
    # use_best_model=True,
    # verbose=0).fit(train_pool, eval_set=val_pool)
    selector = SelectFromModel(CatBoostRegressor(verbose=0))
    selector = selector.fit(X_train, Y_train)
    fim = selector.get_support()
    X_train = X_train[:,fim]
    X_test = X_test[:,fim]
    mod = TabPFNRegressor(ignore_pretraining_limits=True).fit(X_train, Y_train)
    #mod = CatBoostRegressor( task_type="CPU", iterations=10000, verbose=0).fit(X_train, Y_train)
    
    #mod = ExtraTreesRegressor(n_estimators=400).fit(X_train, Y_train)
    
    #imp = mod0.get_feature_importance()
    #mp = sorted([(imp[i], i) for i in range(len(imp))])[::-1]
    
    #num_feats = 200
    #chosen = [a[1] for a in mp[:num_feats]] 
    #mod = ExtraTreesRegressor(n_estimators=200).fit(X_train[:,chosen], Y_train) 
    #mod = CatBoostRegressor(iterations=500,
    #        loss_function='RMSE',
    #        verbose=False).fit(X_train[:,chosen], Y_train) 
    #mod = xgb.XGBRegressor(n_estimators=1000).fit(X_train, Y_train)


    # imp = mod.feature_importances_
    # spot = sorted([(im, i) for i, im in enumerate(imp)])[::-1][:100]
    # idx = [a[1] for a in spot]
    # X_train, X_test = X_train[:,idx], X_test[:,idx]
    # mod = ExtraTreesRegressor(n_estimators=100).fit(X_train, Y_train)
    pr = mod.predict(X_test) 
    result[test_index] = pr
    RMSD = np.sqrt(mean_squared_error(Y_test,pr))
    
    #pearsonr = scipy.stats.pearsonr(Y_test,pr)
    print(RMSD)
    #print(pearsonr)
    """
    diff = np.abs(Y_test - pr)
    regs = sorted([(d, i) for i, d in enumerate(diff)])[::-1]
    #print(regs)
    culp = [a[1] for a in regs[:100]]
    for c in culp:
        cnt[df.iloc[c][pro_col]] += 1
    if pr.shape[0] > 1:
        RMSD = np.sqrt(mean_squared_error(Y_test,pr))
        pearsonr = scipy.stats.pearsonr(Y_test,pr)
        print(RMSD)
        print(pearsonr)
        return RMSD, pearsonr
    """
if strat == "mayhem":
    result = df['DGCddG']
if strat == "kfold":
    a = 0
    reps = 10
    for _ in range(reps):
        n_splits = 10
        kf = KFold(n_splits=n_splits,
                shuffle=True,

                )
        # z = [0 for _ in range(4169)] + [1 for _ in range(4169)]
        """
        pro_col = "PDB"
        df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
        
        kf = StratifiedKFold(n_splits=n_splits,
        shuffle=True,random_state=rs
        ) 
        """
        for train_index, test_index in kf.split(X):
            fold_pred(train_index, test_index)
            #stack_model(train_index, test_index,X,Y)
        RMSD = np.sqrt(mean_squared_error(Y,result))
        pearsonr = scipy.stats.pearsonr(Y,result)
        print('RMSD=',RMSD)
        print('pearsonr=',pearsonr)
        a += pearsonr.statistic
        print(cnt)
    print(a/reps)
    print(cnt)

if strat == "hfold":

    # z = [0 for _ in range(4169)] + [1 for _ in range(4169)]
    """
    pro_col = "PDB"
    df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
    
    kf = StratifiedKFold(n_splits=n_splits,
    shuffle=True,random_state=rs
    ) 
    """
    for cluster in df['hgroup_folds'].unique():
        train_index, test_index = df[df['hgroup_folds'] != cluster].index.values,  df[df['hgroup_folds'] == cluster].index.values
        
        fold_pred(train_index, test_index)
        #stack_model(train_index, test_index,X,Y)
    RMSD = np.sqrt(mean_squared_error(Y,result))
    pearsonr = scipy.stats.pearsonr(Y,result)
    print('RMSD=',RMSD)
    print('pearsonr=',pearsonr)


if strat == "oop":
    dat = pd.read_csv(f"./{nm.upper()}.csv",encoding="latin-1").reset_index(drop=True)
    # dat = pd.read_csv(f"./corrected_mmCSM_AB.csv")
    if nm == "s1131":
        pro_col = "protein"
    if nm == "s645":
        pro_col = "#PDB"
    if nm == "s8338":
        pro_col = "#PDB"
        df[pro_col] = df['pdb_chains'].apply(lambda x : x.split("_")[0])
        df[pro_col] = df['#PDB'].apply(lambda x : x.split("_")[0])
    pdbs = df[pro_col].unique()
    tt1 = []

    
    print(len(pdbs))
    for p0 in pdbs:
        print(p0)
        train_index, test_index = df[df[pro_col] != p0].index.values.flatten(), df[df[pro_col] == p0].index.values.flatten()
        # print(train_index.shape)
        # print(test_index.shape)
        # train_index = np.concatenate([train_index, test_index[-5:]])
        # test_index = test_index[:-5]
        # if len(test_index) == 0:
        #     continue
        # test_index = test_index[:-1]
        fold_pred(train_index, test_index)
        tt1.extend(test_index.tolist())


if strat == "unseen":
    # dat = pd.read_csv(f"./corrected_mmCSM_AB.csv")
    dat = pd.read_csv(f"./GeoPPI/data/benchmarkDatasets/S8338.csv",encoding="latin-1")
    dat['d2'] = dat['chain.mut'].apply(lambda x: x.replace(".", ":"))
    dat['PDB'] = dat['pdb_chains'].apply(lambda x : x.split("_")[0])

    dat2 = pd.read_csv(f"./GeoPPI/data/benchmarkDatasets/{nm.upper()}.csv",encoding="latin-1")

    # print(list(dat2['protein']))
    # idx = dat2[dat2['#PDB'].isin(dat['PDB'])]
    
    # print(idx.shape[0])
    # idx = dat[dat['source']!='SKE4MPI2'].index.values
    X_tr = np.load("S8338_X.npy")
    Y_tr = -np.load("S8338_Y.npy")

    # X = np.load("S8338_X.npy")[~idx]
    # Y = np.load("S8338_Y.npy")[~idx]
    print(X_tr.shape)
    mod = ExtraTreesRegressor(n_estimators=400).fit(X_tr, Y_tr)
    result = mod.predict(X)

np.save(f"{nm}_results.npy", result)
RMSD = np.sqrt(mean_squared_error(Y,result))
pearsonr = scipy.stats.pearsonr(Y,result)
print('RMSD=',RMSD)
print('pearsonr=',pearsonr)

c1 = (result > 0.0)
c2 = (result < 0.0)

b1 = (Y > 0.0)
b2 = (Y < 0.0)

print(f"Destabilizing acc: {accuracy_score(b1, c1)}")
print(f"Stabilizing acc: {accuracy_score(b2, c2)}")
print(f"Destabilizing precision: {precision_score(b1, c1)}")
print(f"Stabilizing precision: {precision_score(b2, c2)}")
print(f"Destabilizing recall: {recall_score(b1, c1)}")
print(f"Stabilizing recall: {recall_score(b2, c2)}")
