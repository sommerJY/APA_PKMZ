

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import norm
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_hex
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import networkx as nx
import community
from community import community_louvain
from gprofiler import GProfiler
from collections import defaultdict
from pyvis.network import Network
from sklearn.decomposition import PCA
from tqdm import tqdm
import copy
import colorsys
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42 

# notations for new methods in figures
s_xi = r'$\xi$'
s_xicor = r'$\tilde\xi$'
s_pcc = r'$\rho_p$'
s_scc = r'$\rho_s$'
s_zeta = r'$Xi\rho$'
s_zeta_t = r'$\tilde{Xi\rho}$'

# path check 
datapath = './data/'
plotpath = './figures/'


def xi_cor(x, y, method):
    n = len(x)
    rank_x = stats.rankdata(x, method=method) # check rank
    rank_y = stats.rankdata(y, method=method) # check rank 
    sorted_rank_y = np.array([rank_y[i] for i in np.argsort(rank_x)]) # rerank according to x 
    A = np.sum(np.abs(np.diff(sorted_rank_y)))
    l = n + 1 - rank_y  
    D = 2 * np.sum(l * (n - l))
    xi = 1 - (n * A) / D
    if xi >= 0 : 
        return xi
    else : 
        return 0 


##### xi scaling function because of the small number of samples 

def scaling_model(x, c):
    return x[1] / (1 - c / x[0])

obs_res = []
for i in range(1, 11) : 
    N = 7*i
    x_linear = np.linspace(0, 10, N)
    y_linear = x_linear
    observed_xicor = xi_cor(x_linear, y_linear, 'dense')  
    pearsonr = stats.pearsonr(x_linear, y_linear)[0]
    obs_res.append((N, observed_xicor, pearsonr))

# set the values 
N_values = np.array([x[0] for x in obs_res])
observed_xicor_values = np.array([x[1] for x in obs_res])
answer = np.array([x[2] for x in obs_res])


# Fitting value C with curve_fit
popt, pcov = curve_fit(scaling_model, (N_values, observed_xicor_values), answer)

def get_new_xi (old_xi) : 
    new = scaling_model((14, old_xi), popt[0])
    if new >1 : 
        return(1)
    elif new <0 : 
        return (0)
    else :
        return(new)


all_deg = pd.read_csv(datapath+'02.allDEG.csv', index_col = 0 )

DG_deg = all_deg[(all_deg.tissue =='DG') & (all_deg.comparison=='yoked vs. trained')]
DEG_list = list(DG_deg.gene)

RNA_DG = pd.read_csv(datapath+'03.EXP_PC1_merge.DG.csv', index_col = 0)
RNA_CA3 = pd.read_csv(datapath+'03.EXP_PC1_merge.CA3.csv', index_col = 0)
RNA_CA1 = pd.read_csv(datapath+'03.EXP_PC1_merge.CA1.csv', index_col = 0)

colnames = list(RNA_DG.columns)
RNA_DG_genes = colnames[:colnames.index('RNAseqID')]

# selected memory related genes
candidategenes = ["Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1","Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci"]



# check Rayleigh distribution availability for new score Z

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh


def pval_zeta_rayleigh(xi, rho, n):
    # Z-scoring 
    z_xi = xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n) , statistical z scoring 
    z_rho = rho * np.sqrt(n - 1)  # Spearman ~ N(0,1/(n−1)), t-dist z scoring 
    zeta_stat = np.sqrt(z_xi**2 + z_rho**2)
    # Rayleigh 분포의 survival function (1 - CDF)
    p_value = np.exp(-zeta_stat**2 / 2)
    return zeta_stat, p_value # only for pvalue calculation 


def XI_PC_sub_parallel(MY_LOG, MY_LOG_gene, method='dense'):
    X_re = MY_LOG[MY_LOG_gene]  # gene exp 
    Y_re = MY_LOG['PC1']        # PC1 as reference
    n = len(X_re)
    # linear correlations
    PCOR, P_pv = stats.pearsonr(X_re, Y_re)
    SCOR, S_pv = stats.spearmanr(X_re, Y_re)
    # xi score 
    original_xi = xi_cor(X_re, Y_re, method)
    new_xi = get_new_xi(original_xi)
    z_xi = original_xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n) 
    XI_pv = 1 - stats.norm.cdf(z_xi)  # one-side according to original paper
    # zeta score 2: using original xi vs using scaled xi
    ZCOR_ori = np.sqrt(SCOR**2 + original_xi**2)
    ZCOR = np.sqrt(SCOR**2 + new_xi**2)
    z_Z, Z_pv = pval_zeta_rayleigh(original_xi, SCOR, n) # Z pvalue ~ Rayleigh 
    tmp = {
        'gene'  : MY_LOG_gene,
        'PCOR'  : PCOR,
        'P_pv'  : P_pv,
        'SCOR'  : SCOR,
        'S_pv'  : S_pv,
        'XI_ori' : original_xi,
        'XI_new' : new_xi,
        'XI_pv' : XI_pv,
        'ZCOR_ori': ZCOR_ori,
        'ZCOR'  : ZCOR,
        'Z_pv'  : Z_pv 
    }
    return tmp


results_all = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_PC_sub_parallel)(RNA_DG, gene) for gene in tqdm(RNA_DG_genes) 
)

results_all_df = pd.DataFrame(results_all)

results_all_df['PCOR2'] = np.abs(results_all_df.PCOR)
results_all_df['SCOR2'] = np.abs(results_all_df.SCOR)

results_all_df.to_csv(datapath+'04.all_relationship.csv')







# check distribution
results_all_df = pd.read_csv(datapath+'04.all_relationship.csv', index_col =0)

Z_total_ori = list(results_all_df['ZCOR_ori'])
Z_total_new = list(results_all_df['ZCOR'])

# Remove too low values for fitting
Z_total_ori_filtered = [x for x in Z_total_ori if x > 1e-6]
Z_total_new_filtered = [x for x in Z_total_new if x > 1e-6]

# Fit data to Rayleigh dist -> floc 0 : standardized
param_ori = rayleigh.fit(Z_total_ori_filtered, floc=0)
param_new = rayleigh.fit(Z_total_new_filtered, floc=0)

# Select 500 spots in linspace
x_ori = np.linspace(0, np.max(Z_total_ori_filtered), 500)
x_new = np.linspace(0, np.max(Z_total_new_filtered), 500)

# PDF calculation
pdf_ori = rayleigh.pdf(x_ori, loc=0, scale=param_ori[1])
pdf_new = rayleigh.pdf(x_new, loc=0, scale=param_new[1])

# Combine 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

sns.histplot(Z_total_ori, stat='density', bins=50, alpha=0.3, ax=axes[0, 0], label='Observed')
sns.lineplot(x=x_ori, y=pdf_ori, lw=2, color='red', label=f'Rayleigh fit (scale={param_ori[1]:.2f})', ax=axes[0, 0])
sns.kdeplot(Z_total_ori, ax=axes[0, 0], color='blue', lw=2, label='Original PDF')
axes[0, 0].set_title(s_zeta+ ', with '+s_xi)
axes[0, 0].set_xlabel(s_zeta)
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True)
axes[0, 0].legend()

sns.histplot(Z_total_new, stat='density', bins=50, alpha=0.3, ax=axes[0, 1], label='Observed')
sns.lineplot(x=x_new, y=pdf_new, lw=2, color='red', label=f'Rayleigh fit (scale={param_new[1]:.2f})', ax=axes[0, 1])
sns.kdeplot(Z_total_new, ax=axes[0, 1], color='blue', lw=2, label='Original PDF')
axes[0, 1].set_title(s_zeta_t+ ', with  '+s_xicor)
axes[0, 1].set_xlabel(s_zeta_t)
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True)
axes[0, 1].legend()

# Bottom row: Q-Q plots
stats.probplot(Z_total_ori_filtered, dist=rayleigh, sparams=(0, param_ori[1]), plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot; "+s_zeta)
axes[1, 0].grid(True)

stats.probplot(Z_total_new_filtered, dist=rayleigh, sparams=(0, param_new[1]), plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot; "+s_zeta_t)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(plotpath+'04.Rayleigh_check.png', dpi=300)
plt.savefig(plotpath+'04.Rayleigh_check.pdf', dpi=300)
plt.show()

from scipy.stats import kstest, rayleigh
import numpy as np

# K-S 검정 수행
D_ori, p_ori = kstest(Z_total_ori_filtered, 'rayleigh', args=(0, param_ori[1]))
D_new, p_new = kstest(Z_total_new_filtered, 'rayleigh', args=(0, param_new[1]))

# 결과 출력
print(f"[Original J] KS statistic = {D_ori:.4f}, p-value = {p_ori:.4f}")
# KS stat : 0.12497660576090514
# pval : 1.3336089300890033e-224

print(f"[New J]      KS statistic = {D_new:.4f}, p-value = {p_new:.4f}")
# KS stat : 0.12423695829443823
# pval : 6.020518907360288e-222






# Figure - scatter
from matplotlib.lines import Line2D


results_all_df = pd.read_csv(datapath+'04.all_relationship.csv', index_col = 0)
custom_colors = [ '#6A00A8',"#00ffff","#fff678"] 
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)

color_norm = mcolors.Normalize(vmin=results_all_df["Z_pv"].min(), vmax=0.05)
results_all_df["color"] = results_all_df["Z_pv"].apply(lambda x: mcolors.to_hex(custom_cmap(color_norm(x))))  # HEX 변환


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 1) XI & SCOR2
sns.scatterplot(
    x='SCOR2', y='XI_new', 
    color=results_all_df["color"].tolist(), ax=axes[0], 
    data=results_all_df, legend=False, s=10, alpha=0.6)  

for _, row in results_all_df.iterrows():
    if row["gene"] in DEG_list:
        axes[0].scatter(row["SCOR2"], row["XI_new"], color=row["color"], edgecolors='orangered', s=15, alpha=1)

axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].set_xlabel('|'+s_scc+'|')
axes[0].set_ylabel(s_xicor)
axes[0].set_title('a', loc='left', fontsize=20)
axes[0].set_aspect('equal', adjustable='box')  # rectangle shape fit 

# 2) SCOR2 & PCOR2
sns.scatterplot(
    x='SCOR2', y='PCOR2', 
    color=results_all_df["color"].tolist(), ax=axes[1], 
    data=results_all_df, legend=False, s=10, alpha=0.6)

for _, row in results_all_df.iterrows():
    if row["gene"] in DEG_list:
        axes[1].scatter(row["SCOR2"], row["PCOR2"], color=row["color"], edgecolors='orangered', s=15, alpha=1)

axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].set_xlabel('|'+s_scc+'|')
axes[1].set_ylabel('|'+s_pcc+'|')
axes[1].set_title('b', loc='left', fontsize=20)
axes[1].set_aspect('equal', adjustable='box')  

plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.4)

# colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=custom_cmap), ax=axes, shrink=0.4, pad=0.02)
cbar.set_label(s_zeta+" p-value", fontsize=8)

# 원형 legend 표시
circle_deg = Line2D([], [], marker='o', color='w', markerfacecolor='white',
                    markeredgecolor='orangered', markersize=6, markeredgewidth=1.5)

axes[1].legend(
    handles=[circle_deg],  
    labels=['DEG'],
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    frameon=False,
    prop={'size': 8}
)

plt.savefig(plotpath + '04.scatter_all.png', dpi = 300)
plt.savefig(plotpath + '04.scatter_all.pdf', dpi = 300)









# gene-gene scoring 

g_g_df = pd.read_csv(datapath + '04.all_relationship.csv', index_col = 0)

selected_gene = g_g_df[g_g_df.Z_pv<=0.05] # 702

target_genes = list(np.unique(list(selected_gene.gene) + candidategenes)) # 714
from itertools import combinations
gg_combi_list = list(combinations(target_genes, 2))
# 254541



def XI_pair_sub_parallel(MY_LOG, geneA, geneB, method='dense'):
    X_re = MY_LOG[geneA]  # gene A 
    Y_re = MY_LOG[geneB]  # gene B 
    n = len(X_re)
    # linear correlations
    PCOR, P_pv = stats.pearsonr(X_re, Y_re)
    SCOR, S_pv = stats.spearmanr(X_re, Y_re)
    # xi score 
    original_xi_A = xi_cor(X_re, Y_re, method)
    original_xi_B = xi_cor(Y_re, X_re, method)
    original_xi = max(original_xi_A, original_xi_B)
    new_xi = get_new_xi(original_xi)
    z_xi = original_xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n) 
    XI_pv = 1 - stats.norm.cdf(z_xi)  # one-side according to original paper
    # zeta score 2: using original xi vs using scaled xi
    ZCOR_ori = np.sqrt(SCOR**2 + original_xi**2)
    ZCOR = np.sqrt(SCOR**2 + new_xi**2)
    z_Z, Z_pv = pval_zeta_rayleigh(original_xi, SCOR, n) # Z pvalue ~ Rayleigh 
    tmp = {
        'geneA'  : geneA,
        'geneB'  : geneB,
        'PCOR'  : PCOR,
        'P_pv'  : P_pv,
        'SCOR'  : SCOR,
        'S_pv'  : S_pv,
        'XI_ori' : original_xi,
        'XI_new' : new_xi,
        'XI_pv' : XI_pv,
        'ZCOR_ori': ZCOR_ori,
        'ZCOR'  : ZCOR,
        'Z_pv'  : Z_pv 
    }
    return tmp


results_pair = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(RNA_DG, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_df = pd.DataFrame(results_pair)

results_pair_df['SCOR2'] = np.abs(results_pair_df['SCOR'])

max_check = []
for i in tqdm(range(results_pair_df.shape[0])) :
    max_check.append(max(results_pair_df.iloc[i]['XI_new'], results_pair_df.iloc[i]['SCOR2']))

results_pair_df['MAX'] = max_check

results_pair_df.to_csv(datapath + '04.all_relationship_GG.csv')

# results_pair_df = pd.read_csv(datapath + '04.all_relationship_GG.csv', index_col = 0)





# multiple seed iteration for RGB check 
genes = np.unique(list(results_pair_df.geneA) + list(results_pair_df.geneB)).tolist()

G = nx.Graph()
for _, row in results_pair_df.iterrows():
    G.add_edge(row['geneA'], row['geneB'], weight=row['ZCOR'])


nodes = list(G.nodes())
node_index = {node: i for i, node in enumerate(nodes)}
n = len(nodes)
n_iter = 1000







# (1) Check 1000 iteration and take it to PCA 3PC and give RGB to each methods 

co_matrix = np.zeros((n, n))
size_check = []

for seed in tqdm(range(n_iter)): # 1000 seed 1 hour 
    partition = community.best_partition(G, random_state=seed)
    cluster_to_nodes = defaultdict(list)
    # each cluster mapping
    for node, cid in partition.items():
        cluster_to_nodes[cid].append(node)
    # 
    for cluster_nodes in cluster_to_nodes.values():
        for i in range(len(cluster_nodes)):
            for j in range(i+1, len(cluster_nodes)):
                idx1 = node_index[cluster_nodes[i]]
                idx2 = node_index[cluster_nodes[j]]
                co_matrix[idx1, idx2] += 1
                co_matrix[idx2, idx1] += 1
    size_check.append(len(set(partition.values())))

items = list(set(size_check))
[(a, size_check.count(a)) for a in items] # num cluster check 
# 100 iter : [(3, 4), (4, 93), (5, 3)]
# 1000 iter : [(3, 64), (4, 922), (5, 14)]

# to probability 
co_matrix2 = co_matrix/ n_iter
co_matrix3 = pd.DataFrame(co_matrix2)
co_matrix3.columns = nodes
co_matrix3.index = nodes

co_matrix3.to_csv(datapath + '04.Louvain_1_iter1000.csv')

# to PCA 
pca = PCA(n_components=3)
embedding = pca.fit_transform(co_matrix)

# RGB
rgb_colors = (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))
rgb_dict = {node: tuple(rgb_colors[i]) for i, node in enumerate(nodes)}
rgb_dict_df = pd.DataFrame(rgb_dict).T
rgb_dict_df.columns = ['R','G','B']

# to hex 
hex_val = [to_hex(rgb) for gene, rgb in rgb_dict.items()]

def brighten_by_hls(rgb, lightness=0.4):
    h, l, s = colorsys.rgb_to_hls(*rgb) # too dark 
    l = max(l, lightness)
    return colorsys.hls_to_rgb(h, l, s)

hls_val = [to_hex(brighten_by_hls(rgb)) for gene, rgb in rgb_dict.items()]

rgb_dict_df['hex_val'] = hex_val
rgb_dict_df['hls_val'] = hls_val

rgb_dict_df.to_csv(datapath + '04.Louvain_1_RGB.csv')


# top rank genes check in two ways 
# 1) rank according to one category

method_1_R_200 = rgb_dict_df.sort_values('R', ascending = False).iloc[0:200]
method_1_G_200 = rgb_dict_df.sort_values('G', ascending = False).iloc[0:200]
method_1_B_200 = rgb_dict_df.sort_values('B', ascending = False).iloc[0:200]

method_1_R_20 = rgb_dict_df.sort_values('R', ascending = False).iloc[0:20]
method_1_G_20 = rgb_dict_df.sort_values('G', ascending = False).iloc[0:20]
method_1_B_20 = rgb_dict_df.sort_values('B', ascending = False).iloc[0:20]

method_1_R_20['L1_M1'] = 'R'
method_1_G_20['L1_M1'] = 'G'
method_1_B_20['L1_M1'] = 'B'

method_1_top20 = pd.concat([method_1_R_20, method_1_G_20, method_1_B_20])
method_1_top20['gene'] = list(method_1_top20.index)

method_1_top20.to_csv(datapath + '04.Louvain_1_RGB_M1_top20.csv')




# 2) rank according to purity 

pure_R = np.array([1, 0, 0])
pure_G = np.array([0, 1, 0])
pure_B = np.array([0, 0, 1])

all_rgbs = np.array([rgb_dict[gene] for gene in nodes])

sim_R = cosine_similarity([pure_R], all_rgbs)[0]
sim_G = cosine_similarity([pure_G], all_rgbs)[0]
sim_B = cosine_similarity([pure_B], all_rgbs)[0]

rgb_dict_df2 = copy.deepcopy(rgb_dict_df)
rgb_dict_df2['method2_R'] = sim_R
rgb_dict_df2['method2_G'] = sim_G
rgb_dict_df2['method2_B'] = sim_B

method_2_R_200 = rgb_dict_df2.sort_values('method2_R', ascending = False).iloc[0:200]
method_2_G_200 = rgb_dict_df2.sort_values('method2_G', ascending = False).iloc[0:200]
method_2_B_200 = rgb_dict_df2.sort_values('method2_B', ascending = False).iloc[0:200]

method_2_R_20 = rgb_dict_df2.sort_values('method2_R', ascending = False).iloc[0:20]
method_2_G_20 = rgb_dict_df2.sort_values('method2_G', ascending = False).iloc[0:20]
method_2_B_20 = rgb_dict_df2.sort_values('method2_B', ascending = False).iloc[0:20]

method_2_R_20['L1_M2'] = 'R'
method_2_G_20['L1_M2'] = 'G'
method_2_B_20['L1_M2'] = 'B'

method_2_top20 = pd.concat([method_2_R_20, method_2_G_20, method_2_B_20])
method_2_top20['gene'] = list(method_2_top20.index)

method_2_top20.to_csv(datapath + '04.Louvain_1_RGB_M2_top20.csv')




# Step 2: anchor gene + RGB + (Fmr1 dimming)

anchor_rgb_genes = {
    'Arc':    (1.0, 0.0, 0.0),   # Red
    'Prkcz':  (0.0, 0.0, 1.0),   # Blue
    'Nsf':    (0.0, 1.0, 0.0),   # Green
}
darkening_gene = 'Fmr1'

# Step 3: 클러스터 동시 할당 횟수 기록
coassign_counts = {gene: {anchor: 0 for anchor in list(anchor_rgb_genes) + [darkening_gene]} for gene in genes}

size_check2 = []

for seed in tqdm(range(n_iter)):
    partition = community.best_partition(G, random_state=seed)
    #
    cluster_to_genes = {}
    for gene, cid in partition.items():
        cluster_to_genes.setdefault(cid, []).append(gene)
    #
    for cluster_genes in cluster_to_genes.values():
        gene_set = set(cluster_genes)
        for anchor in coassign_counts[genes[0]]:
            if anchor in gene_set:
                for g in cluster_genes:
                    coassign_counts[g][anchor] += 1
    size_check2.append(len(set(partition.values())))

coassign_counts_df = pd.DataFrame(coassign_counts).T
coassign_counts_df['gene'] = list(coassign_counts_df.index)

coassign_counts_df.to_csv(datapath + '04.Louvain_2_iter1000.csv')


node_rgb = {}
for gene in genes:
    rgb = np.zeros(3)
    # add color according to anchor 
    for anchor, color in anchor_rgb_genes.items():
        w = coassign_counts[gene][anchor] / n_iter
        rgb += w * np.array(color)
    rgb = np.clip(rgb, 0, 1)
    # darkening with fmr1 
    w_dark = coassign_counts[gene][darkening_gene] / n_iter
    rgb = rgb * (1 - w_dark)  # 밝기 조절
    node_rgb[gene] = rgb

L2_rgb_dict_df = pd.DataFrame(node_rgb).T
L2_rgb_dict_df.columns = ['R','G','B']

L2_hex_val = [to_hex(rgb) for gene, rgb in node_rgb.items()]
L2_hls_val = [to_hex(brighten_by_hls(rgb)) for gene, rgb in node_rgb.items()]

L2_rgb_dict_df['hex_val'] = L2_hex_val
L2_rgb_dict_df['hls_val'] = L2_hls_val
L2_rgb_dict_df['gene'] = list(L2_rgb_dict_df.index)

L2_rgb_dict_df.to_csv(datapath+"Louvain_2_RGB.csv", index=False)



L2_R_200 = L2_rgb_dict_df.sort_values('R', ascending = False).iloc[0:200]
L2_G_200 = L2_rgb_dict_df.sort_values('G', ascending = False).iloc[0:200]
L2_B_200 = L2_rgb_dict_df.sort_values('B', ascending = False).iloc[0:200]

L2_R_20 = L2_rgb_dict_df.sort_values('R', ascending = False).iloc[0:20]
L2_G_20 = L2_rgb_dict_df.sort_values('G', ascending = False).iloc[0:20]
L2_B_20 = L2_rgb_dict_df.sort_values('B', ascending = False).iloc[0:20]

L2_R_20['L2'] = 'R'
L2_G_20['L2'] = 'G'
L2_B_20['L2'] = 'B'

L2_top20 = pd.concat([L2_R_20, L2_G_20, L2_B_20])
L2_top20['gene'] = list(L2_top20.index)

L2_top20.to_csv(datapath + '04.Louvain_2_RGB_top20.csv')



# GO term analysis with 200s 

def gprof(gene_list, name, NUM) : 
    gp = GProfiler(return_dataframe=True)
    gp_res = gp.profile(organism='mmusculus',
                query=gene_list,
                no_evidences = False,
                no_iea = False)
    gp_res['min_logP'] = -np.log(gp_res['p_value'])
    gp_res['perc']=gp_res['intersection_size']/gp_res['term_size']
    gp_res.to_csv(datapath+'04.GP_res_'+name+'_'+str(NUM)+'.csv')
    #
    gp_res = gp_res[gp_res.term_size < 5000] # remove too big term
    return(gp_res)


# bar plot with GO terms  
def cluster_go_RGB (gene_list, name, num, lims = 1000) :
    cluster_res = gprof(gene_list, name, num)
    cluster_filt = cluster_res[cluster_res.source.isin(['GO:MF', 'GO:CC', 'GO:BP', 'KEGG'])]
    cluster_filt2 = cluster_filt[cluster_filt.term_size <= lims ]
    cluster_filt2 = cluster_filt2.iloc[0:10]
    cluster_filt2['cluster'] = num
    #
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    sns.barplot(
        ax = axes,
        y = 'name', x = 'min_logP',
        data = cluster_filt2, alpha = 1, saturation = 1, 
        hue = 'cluster', legend = False, orient='h',
        palette = {
            'R': '#fc746a', # '#FF87C8', 
            'G': '#ccff85', # '#A5E594',
            'B': '#7593ff', # '#33CCFF'
            }
        )
    #
    sns.despine(ax = axes, right=True, top=True)
    axes.set_ylabel('')
    axes.set_xticks(axes.get_xticks())
    axes.set_xticklabels(axes.get_xticklabels(), ha='left')
    axes.set_xlabel('-log10(Pval)')
    plt.savefig(plotpath + '04.cluster.{}.{}.png'.format(name,num), bbox_inches='tight')
    plt.savefig(plotpath + '04.cluster.{}.{}.pdf'.format(name,num), bbox_inches='tight',transparent=True)
    # for pretty figure 
    axes.invert_xaxis() 
    axes.tick_params(axis='y', labelright=True)
    plt.savefig(plotpath + '04.cluster_inv.{}.{}.png'.format(name,num), bbox_inches='tight')
    plt.savefig(plotpath + '04.cluster_inv.{}.{}.pdf'.format(name,num), bbox_inches='tight',transparent=True)
    plt.close()    


cluster_go_RGB(list(method_1_R_200.index), 'L1_METHOD1', 'R')
cluster_go_RGB(list(method_1_G_200.index), 'L1_METHOD1', 'G')
cluster_go_RGB(list(method_1_B_200.index), 'L1_METHOD1', 'B')

cluster_go_RGB(list(method_2_R_200.index), 'L1_METHOD2', 'R')
cluster_go_RGB(list(method_2_G_200.index), 'L1_METHOD2', 'G')
cluster_go_RGB(list(method_2_B_200.index), 'L1_METHOD2', 'B')

cluster_go_RGB(list(L2_R_200.index), 'L2', 'R')
cluster_go_RGB(list(L2_G_200.index), 'L2', 'G')
cluster_go_RGB(list(L2_B_200.index), 'L2', 'B')


