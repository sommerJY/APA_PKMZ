
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif as MIC
from numpy import random
import pandas as pd 
from sklearn.neighbors import KernelDensity
from scipy.special import kl_div
from tqdm import tqdm 
from scipy.stats import mannwhitneyu
from mpl_toolkits.mplot3d import Axes3D  # 반드시 import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib import rcParams


datapath = './data/'
plotpath = './figures/'

# for pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42   # EPS 저장할 경우도 함께

# Arial 폰트를 기본 sans-serif 폰트로 지정
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


# notations for new methods in figures
s_xi = r'$\xi$'
s_xicor = r'$\tilde\xi$'
s_pcc = r'$\rho_p$'
s_scc = r'$\rho_s$'
s_zeta = r'$\zeta$'
s_zeta_t = r'$\tilde\zeta$'


Normalised_DG = pd.read_csv(datapath+'03.EXP_PC1_merge.DG.csv', index_col = 0 )
Normalised_DG_cols = list(Normalised_DG.columns)

Normalised_DG_genes = Normalised_DG_cols[:Normalised_DG_cols.index('RNAseqID')]





# unlock log and filter for calculation of KL divergence
Normalised_DG_vals = 2**(Normalised_DG[Normalised_DG_genes]) -1 # 16467 genes 
nu_Normalised_DG = Normalised_DG_vals.loc[:,(Normalised_DG_vals==0).mean() <= 0.5] # 14429 genes

features = nu_Normalised_DG.select_dtypes(include=[float]).columns.to_list()
features = np.array(features)

trained_index = Normalised_DG['training'] == 'trained'
yoked_index = Normalised_DG['training'] == 'yoked'
Normalised_DG['training_class'] = Normalised_DG['training'].apply(lambda x : 1 if x =='trained' else 0)
nu_Normalised_DG['training_class'] = list(Normalised_DG['training_class'])

IEG = np.array(["Fos", "Fosl2", "Npas4", "Arc"]) # Immediate early genes
candidate = ["Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1","Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci" ]



# DEG input 
all_deg = pd.read_csv(datapath+'02.allDEG.csv', index_col = 0 )

DG_deg = all_deg[(all_deg.tissue =='DG') & (all_deg.comparison=='yoked vs. trained')]
DEG_list = list(DG_deg.gene)

feature_deg_index = np.where(np.isin(features, DEG_list))
feature_IEG_index = np.where(np.isin(features, IEG))
feature_candy_index = np.where(np.isin(features, candidate))


def evaluation_metric(top50_genes,IEG=IEG, CANDY=candidate, DEG=DEG_list):
    IEG_genes = np.where(np.isin(top50_genes,IEG))
    CANDY_genes = np.where(np.isin(top50_genes,CANDY))
    DEG_genes = np.where(np.isin(top50_genes,DEG))
    print(f'out of {top50_genes.shape[0]} # of IEG genes are', IEG_genes[0].shape[0])
    print(f'out of {top50_genes.shape[0]} # of Candidate genes are', CANDY_genes[0].shape[0])
    print(f'out of {top50_genes.shape[0]} # of DESeq2 genes are', DEG_genes[0].shape[0])




# 1) KL

def bandwidth2(data):
    n = len(data)
    iqr_d = (np.subtract(*np.percentile(data,[75,25])))/1.34
    return 0.9 * min(np.std(data),iqr_d)*(n**(-1/5))

def score(p,q,dx,b,a):
    n = b - a
    kld = np.sum(np.where(np.logical_and(q!=0,p!=0) , p*np.log(p/q),0))
    kld = kld*dx
    return kld/(np.log(n))

kld_score = np.zeros((len(features)))

for i,feature in tqdm(enumerate(features)):
    data = nu_Normalised_DG[feature].values.reshape(-1,1)
    ini = int(min(data))
    fi = int(max(data))
    bandwidth = bandwidth2(data)
    kde = KernelDensity(bandwidth=bandwidth,kernel='gaussian', algorithm='kd_tree').fit(data)
    b= fi+2.5*bandwidth ; a = ini-2.5*bandwidth
    dx = 1 
    values = np.arange(a,b,dx).reshape(-1,1)
    probabilities = kde.score_samples(values)
    probabilities = np.exp(probabilities)
    uni = np.ones_like(values)*(1/(b-a))
    uni = uni.T[0]
    kld_score[i] = score(uni,probabilities,dx,b,a)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
axes.plot(kld_score, '.', color='black', zorder=1)
axes.plot(kld_score[feature_deg_index], '*', color='blue', zorder=2)
axes.set_yscale('log')
axes.set_xlabel('Gene')
axes.set_ylabel('KLD_score')
plt.savefig(plotpath+'06.KLD_DEG_check.png', dpi = 300)
plt.close()



kld_index = np.argsort(kld_score)
evaluation_metric(features[kld_index[:500]])
np.where(np.isin(features[kld_index],IEG))
np.where(np.isin(features[kld_index],candidate))






# 2) MIC 
mi_all = np.zeros((len(features)))
for i,feature in enumerate(features):
    mutual_info = MIC(nu_Normalised_DG[feature].to_numpy().reshape(-1,1), nu_Normalised_DG['training_class'].to_numpy(), )
    mi_all[i] = mutual_info

mi_all_index =  np.argsort(mi_all)[::-1]

np.where(np.isin(features[mi_all_index],IEG))

evaluation_metric(features[mi_all_index[:500]])




# 3) Mann whitney u 
p_values = np.zeros((len(features)))
U_stat = np.zeros((len(features)))

for i,feature in enumerate(features):
    trained = nu_Normalised_DG[feature][trained_index]
    yoked = nu_Normalised_DG[feature][yoked_index]
    U_stat[i],p_values[i] = mannwhitneyu(trained, yoked, method='asymptotic', alternative='two-sided')

p_index = np.argsort(p_values)
np.where(np.isin(features[p_index],IEG))
evaluation_metric(features[p_index[:500]])






# merge

merged_all = pd.DataFrame({
    'gene' : features,
    'MIC' : mi_all,
    'KLD' : kld_score,
    'MWU' : p_values
    })

print(min(merged_all['MIC'])) ; print(max(merged_all['MIC']))
# 0.0
# 0.6544534830249111

print(min(merged_all['KLD'])) ; print(max(merged_all['KLD']))
# 0.0453771022039515
# inf

print(min(merged_all['MWU'])) ; print(max(merged_all['MWU']))
# 0.0023880570242425293
# 1.0

merged_all['MIC_log'] = np.log(merged_all['MIC'])
merged_all['KLD_log'] = np.log(merged_all['KLD'])
merged_all['MWU_log'] = -np.log(merged_all['MWU'])

merged_all.to_csv(datapath+'06.Robust3.csv')


# visualize 

fig1 = plt.figure(figsize=(7, 5))
ax = fig1.add_subplot(111, projection='3d')

cmap = plt.cm.get_cmap('viridis')
color_norm = mcolors.Normalize(vmin=merged_all["MWU_log"].min(), vmax=merged_all["MWU_log"].max())
merged_all["color"] = merged_all["MWU_log"].apply(lambda x: mcolors.to_hex(cmap(color_norm(x))))




# First scatter all general points
scatter = ax.scatter(
    merged_all["MIC"],
    merged_all["KLD_log"],
    merged_all["MWU_log"],
    c=merged_all["MWU_log"],  
    cmap=cmap,
    norm=color_norm,
    alpha=0.2,
    s=10
)

# Then scatter the specific gene categories for better visibility
for _, row in merged_all.iterrows():
    x, y, z = row["MIC"], row["KLD_log"], row["MWU_log"]
    color = row["color"]
    gene = row['gene']
    if gene in IEG:
        ax.scatter(x, y, z, color=color, alpha=1, s=100, linewidths=5, edgecolors='blue', marker='D')
    elif gene in candidate:
        ax.scatter(x, y, z, color=color, alpha=1, s=100, linewidths=5, edgecolors='black', marker='D')
        #if gene == "Prkcz": 
        #    ax.text(x, y+0.5, z, gene, fontsize=15, color='black')  # 위치 조절 가능
    elif gene in DEG_list:
        ax.scatter(x, y, z, color=color, alpha=1, s=80, edgecolors='red', marker='*')


#
cbar = plt.colorbar(scatter, ax=ax, shrink=0.3, pad=0.08)
cbar.set_label(r'$-\log(p_{\mathrm{MW}})$', fontsize=15)

ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=10)

ax.set_xlabel('MIC',fontsize=15)
ax.set_ylabel('log(KLD)',fontsize=15)
ax.set_zlabel(r'$-\log(p_{\mathrm{MW}})$',fontsize=15)
ax.set_facecolor('white')
ax.grid(True)

legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='DEG', markeredgewidth=2.5, 
           markerfacecolor='white', markeredgecolor='red', markersize=10),
    Line2D([0], [0], marker='D', color='w', label='IEG', markeredgewidth=2.5,
           markerfacecolor='white', markeredgecolor='blue', markersize=10, linewidth=0),
    Line2D([0], [0], marker='D', color='w', label='Candidate', markeredgewidth=2.5,
           markerfacecolor='white', markeredgecolor='black', markersize=10, linewidth=0)
]

ax.legend(handles=legend_elements, 
          loc='lower left',            
    bbox_to_anchor=(1.05, 0.7),
      frameon=False)

# plt.show()

plt.savefig(plotpath + '06.Robust_3D.png', dpi=300)
plt.close()




