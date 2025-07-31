

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
import os 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter



datapath = './data/'
plotpath = './figures/'

# for pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42   # EPS 

# Arial 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


# notations for new methods in figures
s_xi = r'$\xi_c$'
s_xicor = r'$\tilde\xi_c$'
s_pcc = r'$r_p$'
s_scc = r'$\rho_s$'
s_Sigma = r'$\Sigma$'
s_Sigma_t = r'$\tilde{\Sigma}$'




def xi_cor(x, y, method='dense'):
    n = len(x)
    rank_x = stats.rankdata(x, method=method)
    order = np.argsort(rank_x)
    sorted_y = np.asarray(y)[order]
    r_sorted_y = stats.rankdata(sorted_y, method=method)
    A = np.sum(np.abs(np.diff(r_sorted_y)))
    # tie-aware l_i calculation: number of "Y_j >= Y_i" in original Y 
    l = np.array([np.sum(y >= yi) for yi in y]) #  
    D = 2 * np.sum(l * (n - l))
    xi = 1 - (n * A) / D
    return float(np.clip(xi, 0.0, 1.0))



##### Xi scaling function because of the small number of samples 

def scaling_model(x, c):
    return x[1] / (1 - c / x[0])

obs_res = []
for i in range(1, 11) : 
    N = 7*i # used 7 sample wise to see the trend, and used to fit the curve 
    x_linear = np.linspace(0, 10, N)
    y_linear = x_linear
    observed_xicor = xi_cor(x_linear, y_linear, 'dense')  
    pearsonr = stats.pearsonr(x_linear, y_linear)[0]
    obs_res.append((N, observed_xicor, pearsonr))

# Set the values 
N_values = np.array([x[0] for x in obs_res])
observed_xicor_values = np.array([x[1] for x in obs_res])
answer = np.array([x[2] for x in obs_res])


# Fitting value C with curve_fit
popt, pcov = curve_fit(scaling_model, (N_values, observed_xicor_values), answer)

def get_new_xi (n, old_xi) : 
    new = scaling_model((n, old_xi), popt[0])
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


# Selected memory related genes
candidategenes = ["Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1","Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci"]



# see simple PCA to see train yoked spot 
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

# axes.scatter(data = RNA_DG, x = 'PC1', y= 'PC2', 
#              c =RNA_DG.training.map({'yoked': 'gray','trained':'orangered'}), 
#              alpha = 0.7, s = 100 )

# axes.set_xticklabels([])
# axes.set_yticklabels([])
# axes.set_xlabel('PC_mem')
# plt.tight_layout()
# plt.savefig(plotpath+'04.scat_only_ovlap.png', dpi=300)
# plt.savefig(plotpath+'04.scat_only_ovlap.eps', dpi=300)
# plt.savefig(plotpath+'04.scat_only_ovlap.pdf', dpi=300)
# plt.close()






# RNA_DG_cop = copy.deepcopy(RNA_DG)
# RNA_DG_cop = RNA_DG_cop.sort_values('PC1')
# RNA_DG_cop['numbering'] = [a for a in range(14)]

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
# axes.scatter(data = RNA_DG_cop, x = 'PC1', y= 'PC2', 
#              c = RNA_DG_cop.training.map({'yoked': 'gray','trained':'orangered'}), 
#              alpha = 0.5, s = 100)

# for i in range(14) :
#     axes.text(x = RNA_DG_cop.loc[i,'PC1'], y= RNA_DG_cop.loc[i,'PC2'], 
#             s = RNA_DG_cop.loc[i,'numbering'], 
#             ha='center', va='center')
#             #transform=axes.transAxes)


# axes.set_xticklabels([])
# axes.set_yticklabels([])
# axes.set_xlabel('PC_mem')
# plt.tight_layout()
# plt.savefig(plotpath+'04.scat_only_ovlap2.png', dpi=300)
# plt.savefig(plotpath+'04.scat_only_ovlap2.eps', dpi=300)
# plt.savefig(plotpath+'04.scat_only_ovlap2.pdf', dpi=300)
# plt.close()




# Check Rayleigh distribution availability for new score Z

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
    new_xi = get_new_xi(n, original_xi)
    z_xi = original_xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n) 
    XI_pv = 1 - stats.norm.cdf(z_xi)  # one-side according to original paper
    # zeta score 2: using original xi vs using scaled xi
    SIGMA_ori = np.sqrt(SCOR**2 + original_xi**2)
    SIGMA = np.sqrt(SCOR**2 + new_xi**2)
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
        'SIGMA_ori': SIGMA_ori,
        'SIGMA'  : SIGMA,
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

Z_total_ori = list(results_all_df['SIGMA_ori'])
Z_total_new = list(results_all_df['SIGMA'])

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
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))

sns.histplot(Z_total_ori, stat='density', bins=50, alpha=0.3, ax=axes[0, 0], label='Observed')
sns.lineplot(x=x_ori, y=pdf_ori, lw=2, color='red', label=f'Rayleigh(scale={param_ori[1]:.2f})', ax=axes[0, 0])
sns.kdeplot(Z_total_ori, ax=axes[0, 0], color='blue', lw=2, label='Original PDF')
axes[0, 0].set_title(s_Sigma+ ', with '+s_xi )
axes[0, 0].set_xlabel(s_Sigma)
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True)
axes[0, 0].legend(
    loc='upper left', bbox_to_anchor=(1.05, 1.0), 
    fontsize=6, frameon=False,
    handlelength=1.0, handletextpad=0.4, borderaxespad=0.2, markerscale=0.7
)



sns.histplot(Z_total_new, stat='density', bins=50, alpha=0.3, ax=axes[0, 1], label='Observed')
sns.lineplot(x=x_new, y=pdf_new, lw=2, color='red', label=f'Rayleigh(scale={param_new[1]:.2f})', ax=axes[0, 1])
sns.kdeplot(Z_total_new, ax=axes[0, 1], color='blue', lw=2, label='Original PDF')
axes[0, 1].set_title(s_Sigma_t+ ', with  '+s_xicor)
axes[0, 1].set_xlabel(s_Sigma_t)
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True)
axes[0, 1].legend(
    loc='upper left', bbox_to_anchor=(1.05, 1.0),  
    fontsize=6, frameon=False,
    handlelength=1.0, handletextpad=0.4, borderaxespad=0.2, markerscale=0.7
)


# Bottom row: Q-Q plots
stats.probplot(Z_total_ori_filtered, dist=rayleigh, sparams=(0, param_ori[1]), plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot; "+s_Sigma)
axes[1, 0].grid(True)

stats.probplot(Z_total_new_filtered, dist=rayleigh, sparams=(0, param_new[1]), plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot; "+s_Sigma_t)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(plotpath+'04.Rayleigh_check.png', dpi=300)
plt.savefig(plotpath+'04.Rayleigh_check.pdf', dpi=300)
plt.savefig(plotpath+'04.Rayleigh_check.eps', dpi=300)
plt.savefig(plotpath+'04.Rayleigh_check.tiff', dpi=300)

plt.close()




from scipy.stats import kstest, rayleigh
import numpy as np

# K-S test in case..? 
D_ori, p_ori = kstest(Z_total_ori_filtered, 'rayleigh', args=(0, param_ori[1]))
D_new, p_new = kstest(Z_total_new_filtered, 'rayleigh', args=(0, param_new[1]))

# 
print(f"[Original J] KS statistic = {D_ori:.4f}, p-value = {p_ori:.4f}")
print(f"[New J]      KS statistic = {D_new:.4f}, p-value = {p_new:.4f}")







# Figure - scatter
from matplotlib.lines import Line2D

results_all_df = pd.read_csv(datapath+'04.all_relationship.csv', index_col = 0)
custom_colors = [ '#6A00A8',"#00ffff","#fff678"] 
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)

color_norm = mcolors.Normalize(vmin=results_all_df["Z_pv"].min(), vmax=0.05)
results_all_df["color"] = results_all_df["Z_pv"].apply(lambda x: mcolors.to_hex(custom_cmap(color_norm(x))))  # HEX 변환



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))

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
axes[0].set_xlabel('|'+s_scc+'|', fontsize = 10)
axes[0].set_ylabel(s_xicor, fontsize = 10)
axes[0].set_xticks(axes[0].get_xticks())
axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize = 8)
axes[0].set_yticks(axes[0].get_yticks())
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize = 8)
axes[0].set_title('a', loc='left', fontsize=12)
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
axes[1].set_xticks(axes[1].get_xticks())
axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize = 8)
axes[1].set_yticks(axes[1].get_yticks())
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize = 8)
axes[1].set_xlabel('|'+s_scc+'|', fontsize =10 )
axes[1].set_ylabel('|'+s_pcc+'|', fontsize =10 )
axes[1].set_title('b', loc='left', fontsize=12)
axes[1].set_aspect('equal', adjustable='box')  

plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, wspace=0.4)

# colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=custom_cmap), ax=axes, shrink=0.4, pad=0.02)
cbar.set_label(s_Sigma+" p-value", fontsize=10)
cbar.set_ticks([0.001, 0.01, 0.02, 0.03, 0.04, 0.05])
cbar.set_ticklabels(['0.001', '0.01', '0.02', '0.03', '0.04', '0.05'])
cbar.ax.tick_params(labelsize=8)

# Circular legend 
circle_deg = Line2D([], [], marker='o', color='w', markerfacecolor='white',
                    markeredgecolor='orangered', markersize=6, markeredgewidth=1.5)

axes[1].legend(
    handles=[circle_deg],  
    labels=['DEG'],
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    frameon=False,
    prop={'size': 10}
)

plt.savefig(plotpath + '04.scatter_all.png', dpi = 300, bbox_inches='tight')
plt.savefig(plotpath + '04.scatter_all.pdf', dpi = 300, bbox_inches='tight')
plt.savefig(plotpath + '04.scatter_all.eps', dpi = 300, bbox_inches='tight')
plt.savefig(plotpath + '04.scatter_all.tiff', dpi = 300, bbox_inches='tight')
plt.close()
# not gonna be width 7 inch 








# gene-gene scoring 

g_g_df = pd.read_csv(datapath + '04.all_relationship.csv', index_col = 0)

selected_gene = g_g_df[g_g_df.Z_pv<=0.05] # 702

target_genes = list(np.unique(list(selected_gene.gene) + candidategenes)) # 714
len(target_genes)
from itertools import combinations
gg_combi_list = list(combinations(target_genes, 2)) # 254541 pairs



def XI_pair_sub_parallel(MY_LOG, geneA, geneB, method='dense'):
    X_re = MY_LOG[geneA]  # gene A 
    Y_re = MY_LOG[geneB]  # gene B 
    n = len(X_re)
    # linear correlations
    PCOR, P_pv = stats.pearsonr(X_re, Y_re)
    SCOR, S_pv = stats.spearmanr(X_re, Y_re) 
    PCOR2 = np.abs(PCOR)
    SCOR2 = np.abs(SCOR)
    if np.isnan(SCOR) :
        SCOR2 = 0 
        S_pv = 1 
    if np.isnan(PCOR) :
        PCOR2 = 0 
        P_pv = 1 
    #
    original_xi_A = xi_cor(X_re, Y_re, method)
    original_xi_B = xi_cor(Y_re, X_re, method)
    original_xi = max(original_xi_A, original_xi_B)
    new_xi = get_new_xi(n, original_xi)
    z_xi = original_xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n)
    XI_pv = 1 - stats.norm.cdf(z_xi)  # one-side according to original paper
    # zeta score 2: using original xi vs using scaled xi
    SIGMA_ori = np.sqrt(SCOR2**2 + original_xi**2)
    # if SCOR errors, it means all value same. that case, must make xicor also 0
    if np.isnan(SCOR):
        SIGMA = 0 
        MAX_val = 0
        Z_pv = 1 
    else:
        SIGMA = np.sqrt(SCOR2**2 + new_xi**2)
        MAX_val = max(new_xi, SCOR2)
        z_Z, Z_pv = pval_zeta_rayleigh(original_xi, SCOR2, n) # Z pvalue ~ Rayleigh 
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
        'SIGMA_ori': SIGMA_ori,
        'SIGMA'  : SIGMA,
        'Z_pv'  : Z_pv ,
        'PCOR2' : PCOR2,
        'SCOR2' : SCOR2,
        'MAX' : MAX_val
    }
    return tmp


# total merged
results_pair = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(RNA_DG, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_df = pd.DataFrame(results_pair)
results_pair_df.to_csv(datapath + '04.all_relationship_GG.csv')

# results_pair_df = pd.read_csv(datapath + '04.all_relationship_GG.csv', index_col = 0)





# Yoked only 
DG_yoked = RNA_DG[RNA_DG.training=='yoked']

results_pair_yoked = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(DG_yoked, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_yoked_df = pd.DataFrame(results_pair_yoked)

results_pair_yoked_df.to_csv(datapath + '04.all_relationship_GG_YOKED.csv')



# trained only 
DG_trained = RNA_DG[RNA_DG.training=='trained']

results_pair_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(DG_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_trained_df = pd.DataFrame(results_pair_trained)

results_pair_trained_df.to_csv(datapath + '04.all_relationship_GG_TRAINED.csv')















# Graph making 

def process_seed(seed, G, node_index, n):
    local_co_matrix = np.zeros((n, n), dtype=np.uint16)
    partition = community.best_partition(G, random_state=seed)
    cluster_to_nodes = defaultdict(list)
    # 
    for node, cid in partition.items():
        cluster_to_nodes[cid].append(node)
    #
    for cluster_nodes in cluster_to_nodes.values():
        for i in range(len(cluster_nodes)):
            for j in range(i+1, len(cluster_nodes)):
                idx1 = node_index[cluster_nodes[i]]
                idx2 = node_index[cluster_nodes[j]]
                local_co_matrix[idx1, idx2] += 1
                local_co_matrix[idx2, idx1] += 1
    n_clusters = len(set(partition.values()))
    return local_co_matrix, n_clusters



def graph_process(data, out_filename: str, n_iter: int = 1000, n_jobs: int = None):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['geneA'], row['geneB'], weight=row['SIGMA'])
    #
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    if n_jobs is None:
        n_jobs = min(8, os.cpu_count())
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_seed)(seed, G, node_index, n) for seed in tqdm(range(n_iter))
    )
    aggregate = np.zeros((n, n), dtype=np.uint32)
    cluster_sizes = []
    for local_mat, csize in results:
        aggregate += local_mat
        cluster_sizes.append(csize)
    #
    co_prob = aggregate / n_iter
    co_matrix_df = pd.DataFrame(co_prob, index=nodes, columns=nodes)
    full_out = os.path.join(datapath, out_filename)
    co_matrix_df.to_csv(full_out)
    unique_sizes = sorted(set(cluster_sizes))
    count_by_size = [(size, cluster_sizes.count(size)) for size in unique_sizes]
    print(count_by_size)
    return co_matrix_df



n_iter = 1000 # 1000 iterations for the final result, takes some time 
#n_iter = 10 for testing 
n_jobs = min(8, os.cpu_count())

datasets_1 = [
    # # Merged dataset
    ('04.all_relationship_GG.csv', '04.Louvain_1_iter1000.csv'),
    # # Yoked only dataset
    ('04.all_relationship_GG_YOKED.csv', '04.Louvain_1_iter1000_Yoked.csv'),
    # Trained only dataset
    ('04.all_relationship_GG_TRAINED.csv', '04.Louvain_1_iter1000_Trained.csv'),
]

# this takes a while. each process takes ~10 minutes.
for data in datasets_1 :
    data_read = pd.read_csv(datapath + data[0], index_col = 0)
    out_filename = data[1]
    graph_process(data_read, out_filename, n_iter, n_jobs)


# Total merged 
# final : [(3, 64), (4, 922), (5, 14)]

# Yoked only 
# final : [(9, 231), (10, 717), (11, 52)]

# Trained only
# [(30, 20), (31, 908), (32, 72)]







# Put RGB color 

def brighten_by_hls(rgb, lightness=0.4):
    h, l, s = colorsys.rgb_to_hls(*rgb) # too dark 
    l = max(l, lightness)
    return colorsys.hls_to_rgb(h, l, s)


def get_RGB(df, min_brightness=0.3): # too dark if they are all black 
    rgb = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    rgb = rgb * (1 - min_brightness) + min_brightness
    # 
    hex_val = [to_hex(rgb.loc[gene]) for gene in rgb.index]
    hls_val = [ to_hex(brighten_by_hls(rgb.loc[gene].values)) 
                for gene in rgb.index ]
    rgb['hex_val'] = hex_val
    rgb['hls_val'] = hls_val
    return rgb



# Re read data in case stopped 

clust_all = pd.read_csv(datapath + '04.Louvain_1_iter1000.csv', index_col = 0)
clust_Yoked = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked.csv', index_col = 0)
clust_Trained = pd.read_csv(datapath + '04.Louvain_1_iter1000_Trained.csv', index_col = 0)

genes = list(clust_all.index)
for i in genes :
    clust_all.at[i,i] = 1
    clust_Yoked.at[i,i] = 1
    clust_Trained.at[i,i] = 1


pca_all = PCA() # 
scores_all = pca_all.fit_transform(clust_all) 

# PC spectrum check 
evs = pca_all.explained_variance_ # Eigenvalue 
evr = pca_all.explained_variance_ratio_     # Variance ratio of each PC

pcs = np.arange(1, len(evs)+1)

num_lim = 10  # for plotting 
fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize= (8,3))
axes[0].plot(pcs[0:num_lim], evs[0:num_lim], 'o-')
axes[0].set_xticks(pcs[0:num_lim])
axes[0].set_xlabel('PC')
axes[0].set_ylabel('Eigenvalue')
axes[0].grid(axis = 'y')


axes[1].bar(pcs[0:num_lim], evr[0:num_lim], alpha=0.6)
axes[1].plot(pcs[0:num_lim], np.cumsum(evr)[0:num_lim], 'r--', marker='o', label='Cumulative')
axes[1].set_xticks(pcs[0:num_lim])
axes[1].set_xlabel('PC')
axes[1].set_ylabel('Variance Ratio')
axes[1].grid(axis = 'y')

plt.tight_layout()
plt.savefig(plotpath+'04.Cluster_PC_spectrum.png', dpi = 300)
plt.savefig(plotpath+'04.Cluster_PC_spectrum.eps', dpi = 300)
plt.savefig(plotpath+'04.Cluster_PC_spectrum.pdf', dpi = 300)
plt.close()
# checked the 3 PC is enough to explain the variance 



pca_all = PCA(n_components=3) # 
scores_all = pca_all.fit_transform(clust_all) 

scores_YOKED  = pca_all.transform(clust_Yoked)
scores_TRAINED  = pca_all.transform(clust_Trained)



scores_all_df = pd.DataFrame(scores_all, index = list(clust_all.index), columns = ['R','G','B'])
scores_Yo_df = pd.DataFrame(scores_YOKED, index = list(clust_Yoked.index), columns = ['R','G','B'])
scores_Tr_df = pd.DataFrame(scores_TRAINED, index = list(clust_Trained.index), columns = ['R','G','B'])


scores_Yo_df2 = get_RGB(scores_Yo_df)
scores_Tr_df2 = get_RGB(scores_Tr_df)

scores_Tr_df_colchange = copy.deepcopy(scores_Tr_df)
scores_Tr_df_colchange['G'] = 1-scores_Tr_df_colchange['G'] # used this to change the color, it was too dark. doesn't matter though not used in the final figure 

scores_Tr_df2 = get_RGB(scores_Tr_df_colchange)


col_train = list(scores_Tr_df2.hex_val)


def plot_3d_rgb (this_df, R_ratio, G_ratio, B_ratio) : 
    this_df['candy'] = ['O' if a in candidategenes else 'X' for a in list(this_df.index)]
    #
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    mask_candy = this_df['candy'] == 'O'
    for index, row in this_df.loc[mask_candy].iterrows():
        ax.scatter(
            row['R'],
            row['G'],
            row['B'],
            c=row['hex_val'],
            marker='d',  # diamond
            s=50,
            linewidth=0.5,
            edgecolor='black',
            alpha=0.7,
            label='Candigene'
        )
        ax.text(row['R'], row['G'], row['B'], index, size=10, zorder=1, ha='center')
    #
    mask_not_candy = this_df['candy'] == 'X'
    ax.scatter(
        this_df.loc[mask_not_candy, 'R'],
        this_df.loc[mask_not_candy, 'G'],
        this_df.loc[mask_not_candy, 'B'],
        c=this_df.loc[mask_not_candy, 'hex_val'],
        marker='o',  # circle
        s=30,
        linewidth=0,
        edgecolor='None',
        alpha=0.5,
        label='X'
    )
    ax.set_xlabel('PC1 ({:.2f}%)'.format(R_ratio*100))
    ax.set_ylabel('PC2 ({:.2f}%)'.format(G_ratio*100))
    ax.set_zlabel('PC3 ({:.2f}%)'.format(B_ratio*100))
    ax.view_init(elev=30, azim=90)
    plt.tight_layout()
    plt.close()


plot_3d_rgb(scores_Yo_df2, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])
plot_3d_rgb(scores_Tr_df2, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])


# with original PC values
alls = copy.deepcopy(scores_all_df)
alls['hex_val'] = scores_Tr_df2['hex_val']
plot_3d_rgb(alls, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])

Yo = copy.deepcopy(scores_Yo_df)
Yo['hex_val'] = scores_Tr_df2['hex_val']
Yo['candy'] = ['O' if a in candidategenes else 'X' for a in list(Yo.index)]
plot_3d_rgb(Yo, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])

Tr = copy.deepcopy(scores_Tr_df)
Tr['hex_val'] = scores_Tr_df2['hex_val']
Tr['candy'] = ['O' if a in candidategenes else 'X' for a in list(Tr.index)]
plot_3d_rgb(Tr, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])




def plot_yoked_vs_trained(df_yk, df_tr, R_ratio, G_ratio, B_ratio, title,
                          elev=30, azim=90):
    def on_key(event):
        if event.key == 'j':
            fig.savefig(plotpath+title+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
            fig.savefig(plotpath+title+'.eps', format='eps', dpi=300, bbox_inches='tight')
            print("Saved")
    # 1) global range 
    r_min = min(df_yk['R'].min(), df_tr['R'].min())
    r_max = max(df_yk['R'].max(), df_tr['R'].max())
    g_min = min(df_yk['G'].min(), df_tr['G'].min())
    g_max = max(df_yk['G'].max(), df_tr['G'].max())
    b_min = min(df_yk['B'].min(), df_tr['B'].min())
    b_max = max(df_yk['B'].max(), df_tr['B'].max())
    fig = plt.figure(figsize=(16, 6))
    # left yoked 
    ax1 = fig.add_subplot(121, projection='3d')
    df_yk_O = df_yk[df_yk['candy'] == 'O']
    df_yk_X = df_yk[df_yk['candy'] == 'X']
    ax1.scatter(
        df_yk_O['R'], df_yk_O['G'], df_yk_O['B'],
        c=df_yk_O['hex_val'],
        marker='d',  alpha=0.8,
        label='Candigene',
        s = 120,  
        edgecolor='black', 
        linewidth=1
    )
    ax1.scatter(
        df_yk_X['R'], df_yk_X['G'], df_yk_X['B'],
        c=df_yk_X['hex_val'],
        marker='o',  alpha=0.5,
        label=None,
        s = 50,
        edgecolor='none', 
        linewidth=0  
    )
    for gene in candidategenes:
        if gene in df_yk_O.index:
            x, y, z = df_yk_O.loc[gene, ['R','G','B']]
            ax1.text(x, y, z, gene, color='k', fontsize=8)
    ax1.set_title('Yoked')
    ax1.set_xlabel(f'PC1 ({R_ratio*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({G_ratio*100:.1f}%)')
    ax1.set_zlabel(f'PC3 ({B_ratio*100:.1f}%)')
    # 2) same range 
    ax1.set_xlim(r_min, r_max)
    ax1.set_ylim(g_min, g_max)
    ax1.set_zlim(b_min, b_max)
    ax1.view_init(elev=elev, azim=azim)
    ax1.legend(loc='upper left')
    # right trained 
    ax2 = fig.add_subplot(122, projection='3d')
    df_tr_O = df_tr[df_tr['candy'] == 'O']
    df_tr_X = df_tr[df_tr['candy'] == 'X']
    ax2.scatter(
        df_tr_O['R'], df_tr_O['G'], df_tr_O['B'],
        c=df_tr_O['hex_val'],
        marker='d',  alpha=0.8,
        label='Candigene',
        edgecolor='black', 
        s = 120, 
        linewidth=1  
    )
    ax2.scatter(
        df_tr_X['R'], df_tr_X['G'], df_tr_X['B'],
        c=df_tr_X['hex_val'],
        marker='o',  alpha=0.5,
        label=None,
        edgecolor='none', 
        s = 50, 
        linewidth=0  
    )
    for gene in candidategenes:
        if gene in df_tr_O.index:
            x, y, z = df_tr_O.loc[gene, ['R','G','B']]
            ax2.text(x, y, z, gene, color='k', fontsize=8)
    ax2.set_title('Trained')
    ax2.set_xlabel(f'PC1 ({R_ratio*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({G_ratio*100:.1f}%)')
    ax2.set_zlabel(f'PC3 ({B_ratio*100:.1f}%)')
    # 동일한 range 강제 설정
    ax2.set_xlim(r_min, r_max)
    ax2.set_ylim(g_min, g_max)
    ax2.set_zlim(b_min, b_max)
    ax2.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.close()


plot_yoked_vs_trained(
    Yo, Tr,
    pca_all.explained_variance_ratio_[0],
    pca_all.explained_variance_ratio_[1],
    pca_all.explained_variance_ratio_[2], 
    '04.both_rotated3d'
)







def save_transition_gif(scores_start, scores_end, colors, gene_list, candidategenes,
                        output_path, n_frames=60, interval=100, figsize=(6,6), dpi=100,elev=20, azim=30):
    # Precompute frames
    frames = [scores_start + (scores_end - scores_start) * (i / (n_frames - 1))
              for i in range(n_frames)]
    # Axis limits
    all_pts = np.vstack([scores_start, scores_end])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    # Setup figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    scat = ax.scatter(frames[0][:,0], frames[0][:,1], frames[0][:,2],
                      c=colors, alpha=0.6, s=20)
    # Create text artists
    texts = []
    for gene in candidategenes:
        idx = gene_list.index(gene)
        x, y, z = frames[0][idx]
        txt = ax.text(x, y, z, gene, color='black', fontsize=8)
        texts.append(txt)
    # Set axes
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # Update function
    def update(i):
        pts = frames[i]
        scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        for txt, gene in zip(texts, candidategenes):
            idx = gene_list.index(gene)
            x, y, z = pts[idx]
            txt.set_position((x, y))
            txt.set_3d_properties(z, 'z')
        return [scat] + texts
    # Create animation and save
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    writer = PillowWriter(fps=1000/interval)
    ani.save(output_path, writer=writer)
    plt.close(fig)

# Usage: generate two GIFs
save_transition_gif(np.array(scores_Yo_df2[['R','G','B']]), np.array(scores_Tr_df2[['R','G','B']]), col_train, list(scores_Tr_df2.index), candidategenes,
                    output_path=plotpath+'04.PCall_yotr.gif')









# cluster map 

clust_all = pd.read_csv(datapath + '04.Louvain_1_iter1000.csv', index_col = 0)
clust_Yoked = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked.csv', index_col = 0)
clust_Trained = pd.read_csv(datapath + '04.Louvain_1_iter1000_Trained.csv', index_col = 0)

genes = list(clust_all.index)
for i in genes :
    clust_all.at[i,i] = 1
    clust_Yoked.at[i,i] = 1
    clust_Trained.at[i,i] = 1


matrices = {
    'all': clust_all,
    'Yoked': clust_Yoked,
    'Trained': clust_Trained,

}


def plot_clustermap(matrix, filename, vmin=0, vmax=1, center=None, cmap='Blues', candy_labels=None):
    g = sns.clustermap(matrix, figsize=(7,7), vmin=vmin, vmax=vmax, center=center,
                       method='average', metric='correlation', cmap=cmap,
                       linewidths=0, linecolor='white', cbar=True)
    #
    if candy_labels is not None:
        new_xticklabels = [label if label in candy_labels else '' for label in g.data2d.columns]
        new_yticklabels = [label if label in candy_labels else '' for label in g.data2d.index]
        #
        g.ax_heatmap.set_xticks(np.arange(len(new_xticklabels)))
        g.ax_heatmap.set_yticks(np.arange(len(new_yticklabels)))
        g.ax_heatmap.set_xticklabels(new_xticklabels, rotation=90)
        g.ax_heatmap.set_yticklabels(new_yticklabels, rotation=0)
    #
    plt.savefig(plotpath + filename+'.png', dpi=300)
    plt.savefig(plotpath + filename+'.pdf', dpi=300)
    plt.savefig(plotpath + filename+'.eps', dpi=300)
    plt.close()


plot_clustermap(clust_Yoked, '04.clustermap_YOKED', candy_labels=candidategenes)
plot_clustermap(clust_Trained, '04.clustermap_Trained', candy_labels=candidategenes)

# see the difference 
diff_mat = clust_Trained - clust_Yoked
plot_clustermap(diff_mat, f'04.clustermap_TR_min_YO', vmin=-1, vmax=1, center=0, cmap='bwr', candy_labels=candidategenes)

# filter only the candidate genes 
diff_candy = diff_mat.loc[candidategenes, candidategenes]
plot_clustermap(diff_candy, f'04.clustermap_TR_min_YO.candy', vmin=-1, vmax=1, center=0, cmap='bwr')



# Gprofiler to check some cluster to see the GO term 

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cut_tree

dist = pdist(diff_mat, metric='correlation')
L = linkage(dist, method='average')

labels = fcluster(L, t=3, criterion='maxclust') # same 
# labels = cut_tree(L, n_clusters=3).squeeze()

idx = list(diff_mat.index).index('Prkcz')
target_label = labels[idx]
members = [g for g, lbl in zip(diff_mat.index, labels) if lbl == target_label]



gp = GProfiler(return_dataframe=True)

res = gp.profile(
    organism='mmusculus',        
    query=members,                 
    user_threshold=0.05,        
    significance_threshold_method='fdr',    
    no_evidences=False          
)

res_1000 = res[(res.term_size <5000) & (res.term_size >5)]
res_1000['pval'] = -np.log10(res_1000['p_value'])
res_1000['perc'] = np.round((res_1000['intersection_size'] / res_1000['term_size'] ) * 100,2)

plt.figure(figsize=(7,6))
sns.barplot(
    x='pval',
    y='name',
    data=res_1000,
    color = 'blue',
    alpha = 0.5
    )

plt.xlabel('-log10(pval)')
plt.ylabel('GO term')
plt.tight_layout()
plt.savefig(plotpath+'04.GO_for_same_module.png', dpi = 300)
plt.savefig(plotpath+'04.GO_for_same_module.eps', dpi = 300)
plt.savefig(plotpath+'04.GO_for_same_module.pdf', dpi = 300)
plt.close()








# get cosine similarity values 

# get Δv
delta = scores_TRAINED - scores_YOKED  # shape = (714, 3)

# method 1) change quantification 
norms = np.linalg.norm(delta, axis=1) # euclidean norm. 
top_norms = pd.Series(norms, index=genes)


# method 2 ) centroid wsimilarity 
# check total shift (centroid change) and normalize it 
centroid_yoked   = scores_YOKED.mean(axis=0)
centroid_trained = scores_TRAINED.mean(axis=0)
u = centroid_trained - centroid_yoked # direction 
u = u / np.linalg.norm(u)  # normlizing 
contrib = delta.dot(u)   # train-yoked difference dot! 

# check contribution 
top_contrib = pd.Series(contrib, index=genes)

# make cosine similarity 
cos_align = (delta * u).sum(axis=1) / np.linalg.norm(delta, axis=1)
cos_align = pd.Series(cos_align, index = genes)


# method 3) gene similarity 
dotprod = (scores_YOKED * scores_TRAINED).sum(axis=1)
top_dot   = pd.Series(dotprod, index=genes)

# row-wise norm
norm_Y = np.linalg.norm(scores_YOKED, axis=1)
norm_T = np.linalg.norm(scores_TRAINED, axis=1)

# cosine similarity
cos_sim = dotprod / (norm_Y * norm_T)
cos_sim = pd.Series(cos_sim, index = genes)

# merge all 
top_all = pd.concat([top_norms, top_contrib, top_dot, cos_align, cos_sim], axis = 1)
top_all.columns = ['norm','centroid','PCdot', 'cent_sim', 'gene_sim']
top_all['candy'] = ['O' if a in candidategenes else 'X' for a in list(top_all.index)]
top_all['hex_val'] = list(scores_Tr_df2['hex_val'])



# visualization 
def simple_2d_plot(df, x_col, y_col, palette, plotpath, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    # only dots 
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=df.index,       
        palette=palette,
        marker='o',
        s=50,
        alpha=0.5,
        legend=False,
        ax=ax
    )
    # candy=='O' only text 
    for gene, row in df[df['candy']=='O'].iterrows():
        ax.text(
            row[x_col],
            row[y_col],
            gene,
            fontsize=12,
            ha='center',
            va='center',
            color='black'
        )
    # 
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    ax.figure.savefig(os.path.join(plotpath, filename + '.png'), dpi=300)
    ax.figure.savefig(os.path.join(plotpath, filename + '.pdf'), dpi=300)
    ax.figure.savefig(os.path.join(plotpath, filename + '.eps'), dpi=300)
    plt.close()



top_all['gene'] = list(top_all.index)  
hex_palette = { gene: top_all.loc[gene, 'hex_val'] for gene in top_all.index }

simple_2d_plot(
    df=top_all,
    x_col='norm',
    y_col='gene_sim',
    palette=hex_palette,
    plotpath=plotpath,    
    filename='04.norm_vs_gene_sim'
)

simple_2d_plot(
    df=top_all,
    x_col='norm',
    y_col='cent_sim',
    palette=hex_palette,
    plotpath=plotpath,
    filename='04.norm_vs_centsim'
)

simple_2d_plot(
    df=top_all,
    x_col='cent_sim',
    y_col='gene_sim',
    palette=hex_palette,
    plotpath=plotpath,
    filename='04.centsim_vs_genesim'
)




##### how about 3D plot 


def on_key2(event):
    if event.key == 'j':
        fig.savefig(plotpath+'04.rotated_3d_cossimcent.pdf', format='pdf', dpi=300, bbox_inches='tight')
        fig.savefig(plotpath+'04.rotated_3d_cossimcent.eps', format='eps', dpi=300, bbox_inches='tight')
        print("Saved as rotated3d.pdf")


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mask_candy = top_all['candy'] == 'O'
for index, row in top_all.loc[mask_candy].iterrows():
    ax.scatter(
        row['norm'],
        row['cent_sim'],
        row['gene_sim'],
        c=row['hex_val'],
        marker='d',  # diamond
        s=50,
        alpha=0.7,
        edgecolor='black',
        linewidth = 1,
        label='Candigene'
    )
    ax.text(row['norm'], row['cent_sim'], row['gene_sim'], index, size=10, zorder=1, ha='center')


mask_not_candy = top_all['candy'] == 'X'
for index, row in top_all.loc[mask_not_candy].iterrows():
    ax.scatter(
        row['norm'],
        row['cent_sim'],
        row['gene_sim'],
        c=row['hex_val'],
        marker='o',  
        s=10,
        alpha=0.4,
        label='NO'
    )

ax.set_xlabel('norm')
ax.set_ylabel('cent_sim')
ax.set_zlabel('gene_sim')

fig.canvas.mpl_connect('key_press_event', on_key2) # saved with j key 
plt.close()







# TSNE trial  
from sklearn.manifold import TSNE

# Prepare data for t-SNE
t_X = top_all[['norm', 'cent_sim', 'gene_sim']]
t_y = top_all['candy']
genes = list(top_all.index)

# option parameters 
perplexities = [10, 30, 50]
random_states = [24, 42, 5]

# empty slot 
tsne_results = [[None for _ in random_states] for _ in perplexities]

for i, perplexity in enumerate(perplexities):
    for j, random_state in enumerate(random_states):
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state) # , init='pca' if you want to use PCA initialization
        X_2d = reducer.fit_transform(t_X)
        df2d = pd.DataFrame(X_2d, index=genes, columns=['comp1', 'comp2'])
        df2d['candy'] = t_y.values
        df2d['norm'] = top_all['norm'].values
        df2d['cent_sim'] = top_all['cent_sim'].values
        df2d['gene_sim'] = top_all['gene_sim'].values
        df2d['hex_val'] = top_all['hex_val'].values
        #
        # 
        norm_min, norm_max = df2d['norm'].min(), df2d['norm'].max()
        cmap = plt.get_cmap('viridis')
        df2d['norm_col'] = list(cmap((df2d['norm'] - norm_min) / (norm_max - norm_min)))
        #
        tsne_results[i][j] = df2d



# 1) norm_col
fig1, axs1 = plt.subplots(len(perplexities), len(random_states), figsize=(10, 8))
for i in range(len(perplexities)):
    for j in range(len(random_states)):
        df2d = tsne_results[i][j]
        ax = axs1[i, j]
        # candy=='O'
        CandyO = df2d[df2d['candy'] == 'O']
        ax.scatter(
            CandyO['comp1'], CandyO['comp2'],
            c=CandyO['norm_col'],
            marker='d',
            s=CandyO['norm'] * 10,
            alpha=0.9,
            linewidths = 0,
            edgecolors = None
        )
        for gene, row in CandyO.iterrows():
            ax.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
        # candy=='X'
        CandyX = df2d[df2d['candy'] == 'X']
        ax.scatter(
            CandyX['comp1'], CandyX['comp2'],
            c=CandyX['norm_col'],
            marker='o',
            s=CandyX['norm'] * 10,
            alpha=0.6,
            linewidths = 0,
            edgecolors = None
        )
        ax.set_title(f"perp={perplexities[i]}, rs={random_states[j]}")
        ax.set_xticks([]); ax.set_yticks([])


plt.tight_layout()
plt.savefig(os.path.join(plotpath, '04.tsne_norm_col.png'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.tsne_norm_col.eps'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.tsne_norm_col.pdf'), dpi=300)
plt.close(fig1)


# 2) Train color 
fig2, axs2 = plt.subplots(len(perplexities), len(random_states), figsize=(10, 8))
for i in range(len(perplexities)):
    for j in range(len(random_states)):
        df2d = tsne_results[i][j]
        ax = axs2[i, j]
        candyO = df2d[df2d['candy'] == 'O']
        ax.scatter(
            candyO['comp1'], candyO['comp2'],
            c=candyO['hex_val'],
            marker='d',
            s=candyO['norm'] * 10,
            alpha=0.9,
            linewidths = 0,
            edgecolors = None
        )
        for gene, row in candyO.iterrows():
            ax.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
        candyX = df2d[df2d['candy'] == 'X']
        ax.scatter(
            candyX['comp1'], candyX['comp2'],
            c=candyX['hex_val'],
            marker='o',
            s=candyX['norm'] * 10,
            alpha=0.6,
            linewidths = 0,
            edgecolors = None
        )
        ax.set_title(f"perp={perplexities[i]}, rs={random_states[j]}")
        ax.set_xticks([]); ax.set_yticks([])


plt.tight_layout()
plt.savefig(os.path.join(plotpath, '04.tsne_hex_col.png'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.tsne_hex_col.eps'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.tsne_hex_col.pdf'), dpi=300)
plt.close(fig2)






# individual plot 

def tsne_plot(random_state, perplexity, t_X) : 
    tsne_reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_2d = tsne_reducer.fit_transform(t_X)
    df2d = pd.DataFrame(X_2d, index=genes, columns=['comp1', 'comp2'])
    df2d['candy'] = t_y.values
    df2d['norm'] = top_all['norm'].values
    norm_min, norm_max = df2d['norm'].min(), df2d['norm'].max()
    cmap = plt.get_cmap('viridis')
    df2d['norm_col'] = list(cmap((df2d['norm'] - norm_min) / (norm_max - norm_min)))
    #
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    # candy=='O'
    CandyO = df2d[df2d['candy'] == 'O']
    axs.scatter(
        CandyO['comp1'], CandyO['comp2'],
        c=CandyO['norm_col'],
        marker='d',
        s=CandyO['norm'] * 5,
        alpha=0.7
    )
    for gene, row in CandyO.iterrows():
        axs.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
    # candy=='X'
    CandyX = df2d[df2d['candy'] == 'X']
    axs.scatter(
        CandyX['comp1'], CandyX['comp2'],
        c=CandyX['norm_col'],
        marker='o',
        s=CandyX['norm'] * 5,
        alpha=0.4
    )
    axs.set_title(f"random_state={random_state}, perplexity={perplexity}")
    axs.set_xticks([]); axs.set_yticks([])
    plt.close()


tsne_plot(42, 60, t_X)
tsne_plot(30, 70, t_X)






# Save only one with specific color 
# used the custom color for ToC figure, however, journal figure will use default color 

import matplotlib.cm as cm


def TSNE_final (TSNE_select, col_select) : 
    col_min = TSNE_select[col_select].min()
    col_max = TSNE_select[col_select].max()
    norm = mcolors.Normalize(vmin=col_min, vmax=col_max)
    #custom_colors = [ '#404040', '#f4a582', '#ca0020'] 
    custom_colors = [ '#440154', '#31688e', '#21918c', '#35b779','#fde725'] 
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # 
    #
    fig, ax = plt.subplots(figsize=(10, 6))
    mask_not_candy = TSNE_select['candy'] == 'X'
    for index, row in TSNE_select.loc[mask_not_candy].iterrows():
        color = custom_cmap((row[col_select] - col_min) / (col_max - col_min))
        ax.scatter(
            row['comp1'],
            row['comp2'],
            color=color,
            marker='o', 
            edgecolor=None,
            linewidth=0, 
            s=70,
            alpha=0.5,
            label='NO'
        )
    mask_candy = TSNE_select['candy'] == 'O'
    for index, row in TSNE_select.loc[mask_candy].iterrows():
        color = custom_cmap((row[col_select] - col_min) / (col_max - col_min))
        ax.scatter(
            row['comp1'],
            row['comp2'],
            color=color,
            marker='d',  #
            s=80,  # 
            edgecolor=None,
            linewidth=2,
            alpha=0.7,
            label='Candigene'
        )
        ax.text(row['comp1'], row['comp2'], index, size=10, zorder=1, ha='center')  
    # 
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(col_select)
    plt.savefig(plotpath+'04.TSNE_col_{}.png'.format(col_select), dpi = 300)
    plt.savefig(plotpath+'04.TSNE_col_{}.eps'.format(col_select), dpi = 300)
    plt.savefig(plotpath+'04.TSNE_col_{}.pdf'.format(col_select), dpi = 300)
    plt.show()



TSNE_select = tsne_results[1][1]

TSNE_final(TSNE_select, 'norm')
TSNE_final(TSNE_select, 'cent_sim')
TSNE_final(TSNE_select, 'gene_sim')
















from umap import UMAP

# Apply UMAP
n_neighbors = [5, 15, 45]
min_dists = [0.1, 0.5, 0.9]

# empty slot 
umap_results = [[None for _ in min_dists] for _ in n_neighbors]

for i, n_neighbor in enumerate(n_neighbors):
    for j, min_dist in enumerate(min_dists):
        umap_reducer = UMAP(n_components = 2, random_state = 42, n_neighbors=n_neighbor, min_dist=min_dist, metric='euclidean')
        X_2d = umap_reducer.fit_transform(t_X)
        df2d = pd.DataFrame(X_2d, index=genes, columns=['comp1', 'comp2'])
        df2d['candy'] = t_y.values
        df2d['norm'] = top_all['norm'].values
        df2d['cent_sim'] = top_all['cent_sim'].values
        df2d['gene_sim'] = top_all['gene_sim'].values
        df2d['hex_val'] = top_all['hex_val'].values
        #
        norm_min, norm_max = df2d['norm'].min(), df2d['norm'].max()
        #custom_colors = [ '#404040', '#f4a582', '#ca0020'] 
        custom_colors = [ '#440154', '#31688e', '#21918c', '#35b779','#fde725'] 
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
        df2d['norm_col'] = list(custom_cmap((df2d['norm'] - norm_min) / (norm_max - norm_min)))
        #
        umap_results[i][j] = df2d




# 1) norm_col
fig1, axs1 = plt.subplots(len(n_neighbors), len(min_dists), figsize=(10, 8))
for i in range(len(n_neighbors)):
    for j in range(len(min_dists)):
        df2d = umap_results[i][j]
        ax = axs1[i, j]
        # candy=='O'
        CandyO = df2d[df2d['candy'] == 'O']
        ax.scatter(
            CandyO['comp1'], CandyO['comp2'],
            c=CandyO['norm_col'],
            marker='d',
            s=CandyO['norm'] * 10,
            alpha=0.9,
            linewidths = 0,
            edgecolors = None
        )
        for gene, row in CandyO.iterrows():
            ax.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
        # candy=='X'
        CandyX = df2d[df2d['candy'] == 'X']
        ax.scatter(
            CandyX['comp1'], CandyX['comp2'],
            c=CandyX['norm_col'],
            marker='o',
            s=CandyX['norm'] * 10,
            alpha=0.6,
            linewidths = 0,
            edgecolors = None
        )
        ax.set_title(f"n_neigh={n_neighbors[i]}, min_dist={min_dists[j]}")
        ax.set_xticks([]); ax.set_yticks([])


plt.tight_layout()
plt.savefig(os.path.join(plotpath, '04.umap_norm_col.png'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.umap_norm_col.eps'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.umap_norm_col.pdf'), dpi=300)
plt.close(fig1)


# 2) Train color 
fig2, axs2 = plt.subplots(len(n_neighbors), len(min_dists), figsize=(10, 8))
for i in range(len(n_neighbors)):
    for j in range(len(min_dists)):
        df2d = umap_results[i][j]
        ax = axs2[i, j]
        candyO = df2d[df2d['candy'] == 'O']
        ax.scatter(
            candyO['comp1'], candyO['comp2'],
            c=candyO['hex_val'],
            marker='d',
            s=candyO['norm'] * 10,
            alpha=0.9,
            linewidths = 0,
            edgecolors = None
        )
        for gene, row in candyO.iterrows():
            ax.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
        candyX = df2d[df2d['candy'] == 'X']
        ax.scatter(
            candyX['comp1'], candyX['comp2'],
            c=candyX['hex_val'],
            marker='o',
            s=candyX['norm'] * 10,
            alpha=0.6,
            linewidths = 0,
            edgecolors = None
        )
        ax.set_title(f"n_neigh={n_neighbors[i]}, min_dist={min_dists[j]}")
        ax.set_xticks([]); ax.set_yticks([])


plt.tight_layout()
plt.savefig(os.path.join(plotpath, '04.umap_hex_col.png'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.umap_hex_col.eps'), dpi=300)
plt.savefig(os.path.join(plotpath, '04.umap_hex_col.pdf'), dpi=300)
plt.close(fig2)




# individual plot 
def umap_plot(random_state, n_neighbors, min_dist, t_X) : 
    umap_reducer = UMAP(n_components = 2, random_state = random_state, n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean')
    X_2d = umap_reducer.fit_transform(t_X)
    df2d = pd.DataFrame(X_2d, index=genes, columns=['comp1', 'comp2'])
    df2d['candy'] = t_y.values
    df2d['norm'] = top_all['norm'].values
    norm_min, norm_max = df2d['norm'].min(), df2d['norm'].max()
    #custom_colors = [ '#404040', '#f4a582', '#ca0020'] 
    custom_colors = [ '#440154', '#31688e', '#21918c', '#35b779','#fde725'] 
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
    df2d['norm_col'] = list(custom_cmap((df2d['norm'] - norm_min) / (norm_max - norm_min)))
    #
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    # candy=='O'
    CandyO = df2d[df2d['candy'] == 'O']
    axs.scatter(
        CandyO['comp1'], CandyO['comp2'],
        c=CandyO['norm_col'],
        marker='d',
        s=CandyO['norm'] * 5,
        alpha=0.7
    )
    for gene, row in CandyO.iterrows():
        axs.text(row['comp1'], row['comp2'], gene, fontsize=10, ha='center')
    # candy=='X'
    CandyX = df2d[df2d['candy'] == 'X']
    axs.scatter(
        CandyX['comp1'], CandyX['comp2'],
        c=CandyX['norm_col'],
        marker='o',
        s=CandyX['norm'] * 5,
        alpha=0.4
    )
    axs.set_title(f"n_neigh={n_neighbors}, min_dist={min_dist}")
    axs.set_xticks([]); axs.set_yticks([])
    plt.show()


umap_plot(64, 30, 0.5, t_X) 
umap_plot(20, 30, 0.5, t_X) 
umap_plot(200, 30, 0.5, t_X) 








