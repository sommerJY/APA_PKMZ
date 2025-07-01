# 04_1.anchor_based_method.py


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



Anchor_genes = ['Arc','Nsf','Prkcz','Fmr1']
anchor_index = {node: idx for idx, node in enumerate(Anchor_genes)}


n_iter = 1000
#n_iter = 10
n_jobs = min(8, os.cpu_count())

data = ('04.all_relationship_GG.csv', '04.Anchor_Louvain_1_iter1000.csv')



data_read = pd.read_csv(datapath + data[0], index_col = 0)
out_filename = data[1]
data_read = data_read.fillna(0)

G = nx.Graph()
for _, row in data_read.iterrows():
    G.add_edge(row['geneA'], row['geneB'], weight=row['SIGMA'])

nodes = list(G.nodes())
n = len(nodes)
node_index = {node: idx for idx, node in enumerate(nodes)}
if n_jobs is None:
    n_jobs = min(8, os.cpu_count())


def Anchor_process_seed(seed, G, node_index, n, Anchor_genes):
    local_co_matrix = np.zeros((n, 4), dtype=np.uint16)
    partition = community.best_partition(G, random_state=seed)
    cluster_to_nodes = defaultdict(list)
    # 
    for node, cid in partition.items():
        cluster_to_nodes[cid].append(node)
    #
    for cluster_nodes in cluster_to_nodes.values():
        for anchor in Anchor_genes:
            if anchor in cluster_nodes:
                for node in cluster_nodes:
                    if node != anchor:
                        idx1 = anchor_index[anchor]
                        idx2 = node_index[node]
                        local_co_matrix[idx2, idx1] += 1
    return local_co_matrix


results = Parallel(n_jobs=n_jobs)(
    delayed(Anchor_process_seed)(seed, G, node_index, n, Anchor_genes) for seed in tqdm(range(n_iter))
)


results_sum = np.sum(results, axis = 0)


co_prob = results_sum / n_iter
co_matrix_df = pd.DataFrame(co_prob, index=nodes, columns=Anchor_genes)

for anchor in Anchor_genes:
        co_matrix_df.loc[anchor, anchor] = 1.0


full_out = os.path.join(datapath, out_filename)

co_matrix_df.to_csv(full_out)
# co_matrix_df = pd.read_csv(full_out, index_col = 0)



# anchor & fmr1 darkening 
anchor_rgb_genes = {
    'Arc':   (1.0, 0.0, 0.0),   # Red
    'Nsf':   (0.0, 1.0, 0.0),   # Green
    'Prkcz': (0.0, 0.0, 1.0),   # Blue
}


node_rgb = {}
for gene in list(co_matrix_df.index):
    rgb = np.zeros(3)
    for anchor, color in anchor_rgb_genes.items():
        w = co_matrix_df.loc[gene][anchor] 
        rgb += w * np.array(color)
    rgb = np.clip(rgb, 0, 1)
    w_dark = co_matrix_df.loc[gene]['Fmr1']
    rgb = rgb * (1 - w_dark)  # 밝기 조절
    node_rgb[gene] = rgb


L2_rgb_dict_df = pd.DataFrame(node_rgb).T
L2_rgb_dict_df.columns = ['R','G','B']


def brighten_by_hls(rgb, lightness=0.7):
    h, l, s = colorsys.rgb_to_hls(*rgb) # too dark 
    l = max(l, lightness)
    return colorsys.hls_to_rgb(h, l, s)


L2_hex_val = [to_hex(rgb) for gene, rgb in node_rgb.items()]
L2_hls_val = [to_hex(brighten_by_hls(rgb)) for gene, rgb in node_rgb.items()]


L2_rgb_dict_df['hex_val'] = L2_hex_val
L2_rgb_dict_df['hls_val'] = L2_hls_val
L2_rgb_dict_df['gene'] = list(L2_rgb_dict_df.index)

plot_3d_rgb(L2_rgb_dict_df, 'hls_val')



L2_R_200 = L2_rgb_dict_df.sort_values('R', ascending = False).iloc[0:200]
L2_G_200 = L2_rgb_dict_df.sort_values('G', ascending = False).iloc[0:200]
L2_B_200 = L2_rgb_dict_df.sort_values('B', ascending = False).iloc[0:200]

L2_R_20 = list(L2_rgb_dict_df.sort_values('R', ascending = False).iloc[0:20].index)
L2_G_20 = list(L2_rgb_dict_df.sort_values('G', ascending = False).iloc[0:20].index)
L2_B_20 = list(L2_rgb_dict_df.sort_values('B', ascending = False).iloc[0:20].index)


rgb_to_save = L2_rgb_dict_df[['hls_val','gene','candy']]

rgb_to_save['top'] = ['R' if a in L2_R_20 else 'G' if a in L2_G_20 else 'B' if a in L2_B_20 else 'X' for a in list(rgb_to_save.index)]

rgb_to_save.to_csv(datapath+'04.L2_rgb_dict_df.csv')


#### 


def topGO_cluster(cluster_gene, name, color) :
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(
        organism='mmusculus',        
        query=cluster_gene,                 
        user_threshold=0.05,        
        #significance_threshold_method='fdr', #    
        no_evidences=False          
    )
    #
    res_1000 = res[(res.term_size <=1000) & (res.term_size >5)]#
    res_1000 = res_1000[res_1000.source.isin(['KEGG','GO:BP','GO:CC','GO:MF'])]
    res_1000['pval'] = -np.log10(res_1000['p_value'])
    res_1000['perc'] = np.round((res_1000['intersection_size'] / res_1000['term_size'] ) * 100,2)
    #
    plt.figure(figsize=(7,3))
    sns.barplot(
        x='pval',
        y='name',
        data=res_1000.iloc[0:10],
        color = color,
        alpha = 0.5
        )
    #
    plt.xlabel('-log10(pval)')
    plt.ylabel('GO term')
    plt.tight_layout()
    plt.savefig(plotpath+'04.GO_for_'+name+'.png', dpi = 300)
    plt.savefig(plotpath+'04.GO_for_'+name+'.eps', dpi = 300)
    plt.savefig(plotpath+'04.GO_for_'+name+'.pdf', dpi = 300)
    #plt.show()
    return res



res_R = topGO_cluster(list(L2_R_200.index), 'Arc', '#ff666b')
res_G = topGO_cluster(list(L2_G_200.index), 'Nsf', '#66ff68')
res_B = topGO_cluster(list(L2_B_200.index), 'Prkcz', '#666aff')

res_1000 = res_B[(res_B.term_size <=1000) & (res_B.term_size >5)]#
res_1000 = res_1000[res_1000.source.isin(['KEGG','GO:BP','GO:CC','GO:MF'])]