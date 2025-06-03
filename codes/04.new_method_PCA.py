pc-wise 


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
    new_xi = get_new_xi(MY_LOG.shape[0], original_xi)
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





# trained but no remember  

non_trained = RNA_DG[RNA_DG.PC1 < 0]
yes_trained = RNA_DG[RNA_DG.PC1 > 0]


results_pair_non_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(non_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_non_trained_df = pd.DataFrame(results_pair_non_trained)
results_pair_non_trained_df['SCOR2'] = np.abs(results_pair_non_trained_df['SCOR'])

max_check = []
for i in tqdm(range(results_pair_non_trained_df.shape[0])) :
    max_check.append(max(results_pair_non_trained_df.iloc[i]['XI_new'], results_pair_non_trained_df.iloc[i]['SCOR2']))

results_pair_non_trained_df['MAX'] = max_check

results_pair_non_trained_df.to_csv(datapath + '04.all_relationship_GG_NON_TRAINED.csv')




results_pair_yes_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel)(yes_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)

results_pair_yes_trained_df = pd.DataFrame(results_pair_yes_trained)
results_pair_yes_trained_df['SCOR2'] = np.abs(results_pair_yes_trained_df['SCOR'])

max_check = []
for i in tqdm(range(results_pair_yes_trained_df.shape[0])) :
    max_check.append(max(results_pair_yes_trained_df.iloc[i]['XI_new'], results_pair_yes_trained_df.iloc[i]['SCOR2']))

results_pair_yes_trained_df['MAX'] = max_check

results_pair_yes_trained_df.to_csv(datapath + '04.all_relationship_GG_YES_TRAINED.csv')



# louvain 


results_pair_non_trained_df = pd.read_csv(datapath + '04.all_relationship_GG_NON_TRAINED.csv', index_col = 0)
results_pair_yes_trained_df = pd.read_csv(datapath + '04.all_relationship_GG_YES_TRAINED.csv', index_col = 0)



# results_pair_df = copy.deepcopy(results_pair_df)
results_pair_df = copy.deepcopy(results_pair_non_trained_df)
results_pair_df = copy.deepcopy(results_pair_yes_trained_df)

results_pair_df = results_pair_df.fillna(0)



# multiple seed iteration for RGB check 
genes = np.unique(list(results_pair_df.geneA) + list(results_pair_df.geneB)).tolist()

G = nx.Graph()
for _, row in results_pair_df.iterrows():
    G.add_edge(row['geneA'], row['geneB'], weight=row['ZCOR'])


nodes = list(G.nodes())
node_index = {node: i for i, node in enumerate(nodes)}
n = len(nodes)
n_iter = 1000




from joblib import Parallel, delayed
import os

def process_seed(seed, G, node_index, n):
    local_co_matrix = np.zeros((n, n), dtype=np.uint16)
    partition = community.best_partition(G, random_state=seed)
    cluster_to_nodes = defaultdict(list)
    # cluster mapping
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


n_iter = 1000  # 예시
n = len(G.nodes)

co_matrix = np.zeros((n, n), dtype=np.uint16)
size_check = []

num_workers = min(8,  os.cpu_count())  # 최대 8개 코어 사용
results = Parallel(n_jobs=num_workers)(
    delayed(process_seed)(seed, G, node_index, n) for seed in tqdm(range(n_iter))
)

for local_co_matrix, n_clusters in results:
    co_matrix += local_co_matrix
    size_check.append(n_clusters)


items = list(set(size_check))
[(a, size_check.count(a)) for a in items] # num cluster check 

# non-trained
# 1000 : [(4, 4), (5, 555), (6, 434), (7, 7)]

# trained 
# 1000 : [(44, 61), (45, 620), (46, 319)]




# to probability 
co_matrix2 = co_matrix/ n_iter
co_matrix3 = pd.DataFrame(co_matrix2)
co_matrix3.columns = nodes
co_matrix3.index = nodes

co_matrix3.to_csv(datapath + '04.Louvain_1_iter1000_NON_trained.csv')
co_matrix3.to_csv(datapath + '04.Louvain_1_iter1000_YES_trained.csv')

# to PCA 
pca = PCA(n_components=3)
embedding = pca.fit_transform(co_matrix3*1000)


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

rgb_dict_df.to_csv(datapath + '04.Louvain_1_RGB.NON_trained.csv')
rgb_dict_df.to_csv(datapath + '04.Louvain_1_RGB.YES_trained.csv')


# rgb_dict_df.to_csv(datapath + '04.Louvain_1_RGB_Yoked.csv')
# rgb_dict_df.to_csv(datapath + '04.Louvain_1_RGB_Trained.csv')











# --- 예시 데이터 로드 ---
# 실제 데이터로 대체하세요.
neigh_NON = pd.read_csv(datapath + '04.Louvain_1_iter1000_NON_trained.csv', index_col = 0)  # Yoked neighbor frequency matrix
neigh_YES = pd.read_csv(datapath + '04.Louvain_1_iter1000_YES_trained.csv', index_col = 0)  # Trained neighbor frequency matrix




# --- 1) Yoked PCA 기준 ---
pca_NON = PCA(n_components=3)
scores_NON       = pca_NON.fit_transform(neigh_NON)   # Yoked 샘플의 PC1·2·3 점수
scores_YES_on_NON  = pca_NON.transform(neigh_YES)       # Trained 벡터를 Yoked PCA 축에 투영
# scores_T_on_Y = (neigh_T - pca_Y.mean_) @ pca_Y.components_.T 같은거임 

# --- 2) Trained PCA 기준 ---
pca_YES = PCA(n_components=3)
scores_YES       = pca_YES.fit_transform(neigh_YES)   # Trained 샘플의 PC1·2·3 점수
scores_NON_on_YES  = pca_YES.transform(neigh_NON)       # Yoked 벡터를 Trained PCA 축에 투영

gene_list = list(neigh_NON.index)
















rgb_dict_df_NON = pd.read_csv(datapath + '04.Louvain_1_RGB.NON_trained.csv', index_col = 0)
rgb_dict_df_YES = pd.read_csv(datapath + '04.Louvain_1_RGB.YES_trained.csv', index_col = 0)


colors_NON = list(rgb_dict_df_NON ['hex_val'])
colors_YES = list(rgb_dict_df_YES ['hex_val'])



# --- 시각화 ---
fig = plt.figure(figsize=(14, 12))

# 1) Yoked PCA: Yoked samples (원본 Yoked 색 사용)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(scores_NON[:, 0], scores_NON[:, 1], scores_NON[:, 2], c=colors_NON, alpha=0.6, s=20)
for gene in candidategenes:
    idx = gene_list.index(gene)
    x, y, z = scores_NON[idx]
    ax1.text(x, y, z, gene, color='black', fontsize=8)

ax1.set_title("Yoked PCA: Yoked Samples")
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")

# 2) Yoked PCA: Trained projected (Trained 색 사용)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(scores_YES_on_NON[:, 0], scores_YES_on_NON[:, 1], scores_YES_on_NON[:, 2], c=colors_NON, alpha=0.6, s=20)
for gene in candidategenes:
    idx = gene_list.index(gene)
    x, y, z = scores_YES_on_NON[idx]
    ax2.text(x, y, z, gene, color='black', fontsize=8)

ax2.set_title("Yoked PCA: Trained Projected")
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")

# 3) Trained PCA: Yoked projected (Yoked 색 사용)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(scores_NON_on_YES[:, 0], scores_NON_on_YES[:, 1], scores_NON_on_YES[:, 2], c=colors_YES, alpha=0.6, s=20)
for gene in candidategenes:
    idx = gene_list.index(gene)
    x, y, z = scores_NON_on_YES[idx]
    ax3.text(x, y, z, gene, color='black', fontsize=8)

ax3.set_title("Trained PCA: Yoked Projected")
ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")

# 4) Trained PCA: Trained samples (원본 Trained 색 사용)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(scores_YES[:, 0], scores_YES[:, 1], scores_YES[:, 2], c=colors_YES, alpha=0.6, s=20)
for gene in candidategenes:
    idx = gene_list.index(gene)
    x, y, z = scores_YES[idx]
    ax4.text(x, y, z, gene, color='black', fontsize=8)

ax4.set_title("Trained PCA: Trained Samples")
ax4.set_xlabel("PC1"); ax4.set_ylabel("PC2"); ax4.set_zlabel("PC3")

plt.tight_layout()
plt.show()




# Interactive 로 저장
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Create 2x2 subplot with 3D scatter
fig = make_subplots(rows=2, cols=2,
                    specs=[[{'type':'scene'},{'type':'scene'}],
                           [{'type':'scene'},{'type':'scene'}]],
                    subplot_titles=("Non-trained PCA : Non-trained samples" ,
                                    "Non-trained PCA : Yes-trained Projected",
                                    "Yes-trained PCA : Non-trained Projected",
                                    "Yes-trained PCA : Yes-trained Samples"))

# 1) Yoked PCA: Yoked samples
fig.add_trace(
    go.Scatter3d(x=scores_NON[:,0], y=scores_NON[:,1], z=scores_NON[:,2],
                 mode='markers+text',
                 marker=dict(color=colors_NON, size=4),
                 text=[g if g in candidategenes else "" for g in gene_list],
                 textposition="top center",
                 hovertext=gene_list),
    row=1, col=1
)

# 2) Yoked PCA: Trained projected
fig.add_trace(
    go.Scatter3d(x=scores_YES_on_NON[:,0], y=scores_YES_on_NON[:,1], z=scores_YES_on_NON[:,2],
                 mode='markers+text',
                 marker=dict(color=colors_NON, size=4),
                 text=[g if g in candidategenes else "" for g in gene_list],
                 textposition="top center",
                 hovertext=gene_list),
    row=1, col=2
)

# 3) Trained PCA: Yoked projected
fig.add_trace(
    go.Scatter3d(x=scores_NON_on_YES[:,0], y=scores_NON_on_YES[:,1], z=scores_NON_on_YES[:,2],
                 mode='markers+text',
                 marker=dict(color=colors_YES, size=4),
                 text=[g if g in candidategenes else "" for g in gene_list],
                 textposition="top center",
                 hovertext=gene_list),
    row=2, col=1
)

# 4) Trained PCA: Trained samples
fig.add_trace(
    go.Scatter3d(x=scores_YES[:,0], y=scores_YES[:,1], z=scores_YES[:,2],
                 mode='markers+text',
                 marker=dict(color=colors_YES, size=4),
                 text=[g if g in candidategenes else "" for g in gene_list],
                 textposition="top center",
                 hovertext=gene_list),
    row=2, col=2
)

# Update layout
fig.update_layout(height=900, width=1200,
                  title_text="Split Non vs Yes",
                  showlegend=False)

# Save as HTML
output_file = plotpath+"pca_int_NONYES.html"
fig.write_html(output_file)

















# 애니매이션을 위한 노오력
# yoked 기준

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def animate_transition(scores_start, scores_end, colors, gene_list, candidategenes, 
                       n_frames=60, pause_time=0.1, figsize=(6,6), dpi=100):
    #  interpolation frames
    frames = [
        scores_start + (scores_end - scores_start) * (i / (n_frames - 1))
        for i in range(n_frames)
    ]
    # axis limit 
    all_pts = np.vstack([scores_start, scores_end])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    # Interactive 
    plt.ion()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    scat = ax.scatter(frames[0][:,0], frames[0][:,1], frames[0][:,2],
                      c=colors, alpha=0.6, s=20)
    # text for candidate
    texts = []
    for gene in candidategenes:
        idx = gene_list.index(gene)
        x, y, z = frames[0][idx]
        txt = ax.text(x, y, z, gene, color='black', fontsize=8)
        texts.append(txt)
    # 
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    # 
    for pts in frames:
        scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        for txt, gene in zip(texts, candidategenes):
            idx = gene_list.index(gene)
            x, y, z = pts[idx]
            txt.set_position((x, y))
            txt.set_3d_properties(z, 'z')
        plt.draw()
        plt.pause(pause_time)
    plt.ioff()
    plt.show()


animate_transition(scores_NON, scores_YES_on_NON, colors_NON, gene_list, candidategenes)
animate_transition(scores_NON_on_YES, scores_YES, colors_YES, gene_list, candidategenes)





# to gif 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def save_transition_gif(scores_start, scores_end, colors, gene_list, candidategenes,
                        output_path, n_frames=60, interval=100, figsize=(6,6), dpi=100):
    # Precompute frames
    frames = [scores_start + (scores_end - scores_start) * (i / (n_frames - 1))
              for i in range(n_frames)]
    # Axis limits
    all_pts = np.vstack([scores_start, scores_end])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    # Setup figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')
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
save_transition_gif(scores_NON, scores_YES_on_NON, colors_NON, gene_list, candidategenes,
                    output_path=plotpath+'pc_NON_YES.gif')
save_transition_gif(scores_NON_on_YES, scores_YES, colors_YES, gene_list, candidategenes,
                    output_path=plotpath+'pc_YES_NON.gif')





# 혹시 각도도 돌리고 싶다면 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def save_transition_gif(scores_start, scores_end, colors, gene_list, candidategenes,
                        output_path, n_frames=60, interval=100, figsize=(6,6), dpi=100,
                        elev=30, azim_start=45, azim_end=405):
    # Precompute frames
    frames = [scores_start + (scores_end - scores_start) * (i / (n_frames - 1))
              for i in range(n_frames)]
    # Axis limits
    all_pts = np.vstack([scores_start, scores_end])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    # Setup figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')
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
    # Precompute azimuth angles
    azims = np.linspace(azim_start, azim_end, n_frames)
    # Update function
    def update(i):
        pts = frames[i]
        scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        # update text positions
        for txt, gene in zip(texts, candidategenes):
            idx = gene_list.index(gene)
            x, y, z = pts[idx]
            txt.set_position((x, y))
            txt.set_3d_properties(z, 'z')
        # rotate view
        ax.view_init(elev=elev, azim=azims[i])
        return [scat] + texts
    # Create animation and save
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    writer = PillowWriter(fps=1000/interval)
    ani.save(output_path, writer=writer)
    plt.close(fig)

# Example usage:
# save_transition_gif(scores_Y, scores_T_on_Y, colors_Y, gene_list, candidategenes,
#                    output_path='yoked_to_trained_rot.gif',
#                    azim_start=45, azim_end=405, elev=30)





# candidate pair

gene_order = list(rgb_dict_df_NON.index)

scores_NON_df = pd.DataFrame(scores_NON)
scores_YES_on_NON_df = pd.DataFrame(scores_YES_on_NON)

scores_NON_on_YES_df = pd.DataFrame(scores_NON_on_YES)
scores_YES_df = pd.DataFrame(scores_YES)

scores_NON_df.index = gene_order; scores_YES_on_NON_df.index = gene_order
scores_NON_on_YES_df.index = gene_order ; scores_YES_df.index = gene_order

scores_NON_df.columns = ['PC1','PC2','PC3']; scores_YES_on_NON_df.columns = ['PC1','PC2','PC3']
scores_NON_on_YES_df.columns = ['PC1','PC2','PC3'] ; scores_YES_df.columns = ['PC1','PC2','PC3']

from itertools import combinations
candi_pairs = list(combinations(candidategenes, 2))

from scipy.spatial import distance

candi_pair_res = []

for pairs in candi_pairs : 
    NON_pc_NON_dist = distance.euclidean(scores_NON_df.loc[pairs[0]], scores_NON_df.loc[pairs[1]])
    NON_pc_YES_dist = distance.euclidean(scores_YES_on_NON_df.loc[pairs[0]], scores_YES_on_NON_df.loc[pairs[1]])
    YES_pc_NON_dist = distance.euclidean(scores_NON_on_YES_df.loc[pairs[0]], scores_NON_on_YES_df.loc[pairs[1]])
    YES_pc_YES_dist = distance.euclidean(scores_YES_df.loc[pairs[0]], scores_YES_df.loc[pairs[1]])
    candi_pair_res.append((pairs[0], pairs[1], NON_pc_NON_dist, NON_pc_YES_dist, YES_pc_NON_dist, YES_pc_YES_dist))

candi_pair_res_df = pd.DataFrame(candi_pair_res)
candi_pair_res_df.columns = ['geneA','geneB','NON_pc_NON_dist','NON_pc_YES_dist','YES_pc_NON_dist','YES_pc_YES_dist']
candi_pair_res_df['gene_gene'] = candi_pair_res_df['geneA'] + '_' + candi_pair_res_df['geneB']
candi_pair_res_df['Prkcz'] = candi_pair_res_df.gene_gene.apply(lambda x : 'O' if 'Prkcz' in x else 'X')





# original total used pc
fig1, axes = plt.subplots(nrows=1, ncols=1, figsize= (7, 7))

sns.scatterplot(x='NON_pc_NON_dist', y='NON_pc_YES_dist', 
                data=candi_pair_res_df, ax = axes,
                hue = 'Prkcz',
                size = 50, legend = False)


xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()

axes.plot(
    [xmin, xmax],
    [ymin, ymax],
    linestyle='--',
    color='gray',
    linewidth=1
)

for idx in range(candi_pair_res_df.shape[0]):
    Prkcz = candi_pair_res_df.loc[idx, 'Prkcz']
    if Prkcz == 'O' : 
        x = candi_pair_res_df.loc[idx, 'NON_pc_NON_dist']
        y = candi_pair_res_df.loc[idx, 'NON_pc_YES_dist']
        gene_gene = candi_pair_res_df.loc[idx, 'gene_gene']
        gene_gene = gene_gene.replace('Prkcz', '').replace('_','')
        axes.text(x, y, gene_gene, color='black', fontsize=8)

axes.set_ylabel('Yes-trained')
axes.set_xlabel('Non-trained')

plt.tight_layout()
plt.savefig(plotpath + 'Prkcz_scatter_NONYES_NONPC.png', dpi = 300)
plt.show()




# original total used pc
fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

sns.scatterplot(x='YES_pc_NON_dist', y='YES_pc_YES_dist', 
                data=candi_pair_res_df, ax = axes,
                hue = 'Prkcz',
                size = 50, legend = False)


xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()

axes.plot(
    [xmin, xmax],
    [ymin, ymax],
    linestyle='--',
    color='gray',
    linewidth=1
)

for idx in range(candi_pair_res_df.shape[0]):
    Prkcz = candi_pair_res_df.loc[idx, 'Prkcz']
    if Prkcz == 'O' : 
        x = candi_pair_res_df.loc[idx, 'YES_pc_NON_dist']
        y = candi_pair_res_df.loc[idx, 'YES_pc_YES_dist']
        gene_gene = candi_pair_res_df.loc[idx, 'gene_gene']
        gene_gene = gene_gene.replace('Prkcz', '').replace('_','')
        axes.text(x, y, gene_gene, color='black', fontsize=8)

axes.set_ylabel('Yes-trained')
axes.set_xlabel('Non-trained')
plt.tight_layout()
plt.savefig(plotpath + 'Prkcz_scatter_NONYES_YESPC.png', dpi = 300)

plt.show()






# GO 확인하기 위해서 PCA 기준으로 clustering 하는거 제작 



gene_order = list(rgb_dict_df_NON.index)

scores_NON_df = pd.DataFrame(scores_NON)
scores_YES_on_NON_df = pd.DataFrame(scores_YES_on_NON)

scores_NON_on_YES_df = pd.DataFrame(scores_NON_on_YES)
scores_YES_df = pd.DataFrame(scores_YES)

scores_NON_df.index = gene_order; scores_YES_on_NON_df.index = gene_order
scores_NON_on_YES_df.index = gene_order ; scores_YES_df.index = gene_order

scores_NON_df.columns = ['PC1','PC2','PC3']; scores_YES_on_NON_df.columns = ['PC1','PC2','PC3']
scores_NON_on_YES_df.columns = ['PC1','PC2','PC3'] ; scores_YES_df.columns = ['PC1','PC2','PC3']




from sklearn.neighbors import NearestNeighbors

def get_neigh_GO (pca_df,name,num) : 
    coords = pca_df[['PC1','PC2','PC3']].values
    k = 200
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    #
    target_name = 'Prkcz'
    target_idx  = pca_df.index.get_loc(target_name)
    #
    neighbor_indices = indices[target_idx][1:]
    neighbor_names   = pca_df.index[neighbor_indices]
    #
    cluster_res = gprof(list(neighbor_names), name , num)
    # cluster_res = gprof(list(neighbor_names), 'YES' , 'PKMZ')
    cluster_filt = cluster_res[cluster_res.source.isin(['GO:MF', 'GO:CC', 'GO:BP', 'KEGG'])]
    cluster_filt2 = cluster_filt[cluster_filt.term_size <= 500 ]
    return(cluster_filt2)


GO_NON_PKMZ = get_neigh_GO(scores_NON_df, 'NON', 'PKMZ')
GO_YES_on_NON_PKMZ = get_neigh_GO(scores_YES_on_NON_df, 'YES_on_NON', 'PKMZ') # ER membrane 
GO_NON_on_YES_PKMZ = get_neigh_GO(scores_NON_on_YES_df, 'NON_on_YES', 'PKMZ')
GO_YES_PKMZ = get_neigh_GO(scores_YES_df, 'YES', 'PKMZ') # ER membrane 























