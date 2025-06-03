

이건 window 나눠서 보려고 한건데 
실패
장렬하게 실패



# gene-gene scoring 
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
import networkx as nx
from collections import defaultdict
from tqdm import tqdm




RNA_DG = pd.read_csv(datapath+'03.EXP_PC1_merge.DG.csv', index_col = 0)
RNA_CA3 = pd.read_csv(datapath+'03.EXP_PC1_merge.CA3.csv', index_col = 0)
RNA_CA1 = pd.read_csv(datapath+'03.EXP_PC1_merge.CA1.csv', index_col = 0)


# --- 기존에 정의된 함수들: xi_cor, get_new_xi, pval_zeta_rayleigh는 이미 있다고 가정 ---

def XI_pair_sub_parallel(MY_LOG, geneA, geneB, method='dense'):
    X_re = MY_LOG[geneA]
    Y_re = MY_LOG[geneB]
    n = len(X_re)
    PCOR, P_pv = stats.pearsonr(X_re, Y_re)
    SCOR, S_pv = stats.spearmanr(X_re, Y_re)
    original_xi_A = xi_cor(X_re, Y_re, method)
    original_xi_B = xi_cor(Y_re, X_re, method)
    original_xi = max(original_xi_A, original_xi_B)
    new_xi = get_new_xi(MY_LOG.shape[0], original_xi)
    z_xi = original_xi / np.sqrt(0.4 / n)
    XI_pv = 1 - stats.norm.cdf(z_xi)
    ZCOR = np.sqrt(SCOR**2 + new_xi**2)
    _, Z_pv = pval_zeta_rayleigh(original_xi, SCOR, n)
    return {
        'geneA': geneA,
        'geneB': geneB,
        'ZCOR': ZCOR,
        'Z_pv': Z_pv
    }



def calculate_gene_relationship(MY_LOG, gg_combi_list, n_jobs=8):
    results_pair = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(XI_pair_sub_parallel)(MY_LOG, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list)
    )
    return pd.DataFrame(results_pair).fillna(0)


def run_louvain_clustering(results_pair_df, n_iter=500, n_jobs=8):
    genes = np.unique(results_pair_df[['geneA', 'geneB']])
    G = nx.Graph()
    for _, row in results_pair_df.iterrows():
        G.add_edge(row['geneA'], row['geneB'], weight=row['ZCOR'])
    node_index = {node: i for i, node in enumerate(G.nodes())}
    n = len(node_index)
    #
    def process_seed(seed, G, node_index, n):
        local_co_matrix = np.zeros((n, n), dtype=np.uint16)
        partition = community.best_partition(G, random_state=seed)
        clusters = defaultdict(list)
        for node, cid in partition.items():
            clusters[cid].append(node)
        for cluster_nodes in clusters.values():
            for i in range(len(cluster_nodes)):
                for j in range(i+1, len(cluster_nodes)):
                    idx1, idx2 = node_index[cluster_nodes[i]], node_index[cluster_nodes[j]]
                    local_co_matrix[idx1, idx2] += 1
                    local_co_matrix[idx2, idx1] += 1
        return local_co_matrix
    #
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_seed)(seed, G, node_index, n) for seed in tqdm(range(n_iter))
    )
    co_matrix = sum(results)
    co_matrix_prob = co_matrix / n_iter
    co_matrix_df = pd.DataFrame(co_matrix_prob, index=G.nodes(), columns=G.nodes())
    return co_matrix_df




# --- 메인 모듈화 함수 ---
def clustering_probability_window(expr_df, target_genes, window_size=3, step=1, n_jobs=8, n_iter=500):
    n_samples = expr_df.shape[0]
    windows = [
        (start, start + window_size)
        for start in range(0, n_samples - window_size + 1, step)
    ]
    #
    gg_combi_list = list(combinations(target_genes, 2))
    window_results = {}
    #
    for start, end in windows:
        window_name = f"sample_{start}_{end-1}"
        print(f"\nProcessing window: {window_name}")
        window_expr = expr_df.iloc[start:end, :]
        results_pair_df = calculate_gene_relationship(window_expr, gg_combi_list, n_jobs=n_jobs)
        co_matrix_df = run_louvain_clustering(results_pair_df, n_iter=n_iter, n_jobs=n_jobs)
        window_results[window_name] = co_matrix_df
    return window_results



copy_DG = RNA_DG.sort_values('PC1').reset_index(drop = True)
# 사용 예시:
# expr_df는 샘플×유전자 형태의 DataFrame (예: 14×714)
# target_genes는 clustering 대상 유전자 리스트 (예: 길이 714)

g_g_df = pd.read_csv(datapath + '04.all_relationship.csv', index_col = 0)

selected_gene = g_g_df[g_g_df.Z_pv<=0.05] # 702

target_genes = list(np.unique(list(selected_gene.gene) + candidategenes)) # 714

window_size = 3
window_size = 4

window_results = clustering_probability_window(
    expr_df=copy_DG[target_genes], 
    target_genes=target_genes,
    window_size=window_size,
    step=1,
    n_jobs=8,
    n_iter=1000
)


for key in window_results.keys() :
    window_results[key].to_csv(datapath +'wind_{}.csv'.format(key))






# loading 여기부터 해주세요 

windows = [
        'sample_{}_{}'.format(start, start + window_size -1)
        for start in range(0, 14 - window_size + 1, 1)
    ]


window_results = {}
for windkey in windows: 
    window_results[windkey] = pd.read_csv(datapath +'wind_{}.csv'.format(windkey), index_col = 0)



clust_all = pd.read_csv(datapath + '04.Louvain_1_iter1000.csv', index_col = 0)

genes = list(clust_all.index)
for i in genes :
    clust_all.at[i,i] = 1


pca_all = PCA(n_components=3)
scores_all = pca_all.fit_transform(clust_all) 

clust_all_order = list(clust_all.index)

window_results_PC3 = {}

for windkey in windows: 
    window_results_PC3[windkey] = pca_all.transform(window_results[windkey].loc[clust_all_order,clust_all_order])




scores_TRAINED  = pca_all.transform(clust_Trained)
scores_Tr_df = pd.DataFrame(scores_TRAINED, index = list(clust_Trained.index), columns = ['R','G','B'])




from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def save_multistep_transition_gif(window_results_PC3, gene_list, candidategenes, colors,
                                   output_path, n_frames_per_step=15, interval=100, figsize=(6,6), dpi=100,
                                   elev=20, azim=30):
    # 정렬된 키 리스트
    keys = list(window_results_PC3.keys())
    # 각 상태의 3D 좌표를 numpy로 변환 (순서대로 쌓기)
    frames_all = []
    for i in range(len(keys) - 1):
        start = window_results_PC3[keys[i]]
        end = window_results_PC3[keys[i + 1]]
        # 프레임 보간
        for j in range(n_frames_per_step):
            alpha = j / (n_frames_per_step - 1)
            interp = start + (end - start) * alpha
            frames_all.append(interp)
    #
    # 전체 범위 설정
    all_concat = np.vstack(frames_all)
    mins, maxs = all_concat.min(axis=0), all_concat.max(axis=0)
    # Figure setup
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    scat = ax.scatter(frames_all[0][:,0], frames_all[0][:,1], frames_all[0][:,2],
                      c=colors, alpha=0.6, s=20)
    ax.set_title('0')
    #
    # 텍스트 라벨 준비
    texts = []
    for gene in candidategenes:
        idx = gene_list.index(gene)
        x, y, z = frames_all[0][idx]
        txt = ax.text(x, y, z, gene, color='black', fontsize=8)
        texts.append(txt)
    #
    # 축 설정
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # 업데이트 함수
    def update(i):
        pts = frames_all[i]
        scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        for txt, gene in zip(texts, candidategenes):
            idx = gene_list.index(gene)
            x, y, z = pts[idx]
            txt.set_position((x, y))
            txt.set_3d_properties(z, 'z')
        # 현재 transition 단계 추적
        transition_idx = i // n_frames_per_step
        within_step_idx = i % n_frames_per_step + 1  # 1부터 시작
        ax.set_title(f"{keys[transition_idx]} → {keys[transition_idx + 1]}  (frame {within_step_idx} / {n_frames_per_step})",
                     fontsize=10)
        return [scat] + texts
    #
    # 애니메이션 생성 및 저장
    ani = FuncAnimation(fig, update, frames=len(frames_all), interval=interval, blit=False)
    writer = PillowWriter(fps=1000/interval)
    ani.save(output_path, writer=writer)
    plt.close(fig)


# full overlap 
save_multistep_transition_gif(
    window_results_PC3=window_results_PC3,
    gene_list=list(clust_all_order),
    candidategenes=candidategenes,
    colors=col_train,  # 또는 개별 gene 색상 배열 (예: seaborn.color_palette("husl", n_genes))
    output_path=plotpath+'PC3_gene_trajectory_win4.gif',
    n_frames_per_step=20,
    interval=100
)



# selected only 
selected_keys = ['sample_0_2', 'sample_2_4', 'sample_4_6',  'sample_6_8', 'sample_8_10',  'sample_10_12', 'sample_11_13']  

selected_keys = ['sample_0_3', 'sample_2_5', 'sample_4_7', 'sample_6_9', 'sample_8_11', 'sample_10_13']
subset_window_results = {k: window_results_PC3[k] for k in selected_keys}

save_multistep_transition_gif(
    window_results_PC3=subset_window_results,
    gene_list=list(clust_all_order),
    candidategenes=candidategenes,
    colors=col_train,  #
    output_path=plotpath+'PC3_gene_trajectory2.gif',
    n_frames_per_step=20,
    interval=100
)

save_multistep_transition_gif(
    window_results_PC3=subset_window_results,
    gene_list=list(clust_all_order),
    candidategenes=candidategenes, 
    colors=col_train,  #
    output_path=plotpath+'PC3_gene_trajectory2_win4.gif',
    n_frames_per_step=20,
    interval=100
)



# Seperate PCA 

window_size = 4

windows = [
        'sample_{}_{}'.format(start, start + window_size -1)
        for start in range(0, 14 - window_size + 1, 1)
    ]

clust_all = pd.read_csv(datapath + '04.Louvain_1_iter1000.csv', index_col = 0)

genes = list(clust_all.index)
for i in genes :
    clust_all.at[i,i] = 1


pca_all = PCA(n_components=3)
scores_all = pca_all.fit_transform(clust_all) 

clust_all_order = list(clust_all.index)


window_results = {}
for windkey in windows: 
    window_results[windkey] = pd.read_csv(datapath +'wind_{}.csv'.format(windkey), index_col = 0)




window_results_PC3 = {}

for windkey in windows: 
    window_results_PC3[windkey] = pca_all.transform(window_results[windkey].loc[clust_all_order,clust_all_order])


key = 'sample_11_14'
frame_filt = window_results_PC3[key]

fig = plt.figure(figsize=(2,2), dpi=300)
ax = fig.add_subplot(projection='3d')

scat = ax.scatter(frame_filt[:,0], frame_filt[:,1], frame_filt[:,2],
                    c=col_train, alpha=0.6, s=20)

gene_list = list(clust_all_order)

texts = []
for gene in candidategenes:
    idx = gene_list.index(gene)
    x, y, z = frame_filt[idx]
    txt = ax.text(x, y, z, gene, color='black', fontsize=8)
    texts.append(txt)


ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')

plt.tight_layout()
plt.show()





