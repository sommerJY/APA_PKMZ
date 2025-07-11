

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import rcParams
import seaborn as sns 

datapath = './data/'
plotpath = './figures/'

# for pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42   # in case of eps saving 

# Arial font as default sans-serif font
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# notations for new methods in figures
s_xi = r'$\xi$'
s_xicor = r'$\tilde\xi$'
s_pcc = r'$\rho_p$'
s_scc = r'$\rho_s$'
s_zeta = r'$Xi\rho$' 
s_zeta_t = r'$\tilde{Xi\rho}$'


# in here, we don't need to use the new xi, as we have sufficient data points 

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



######################################################################
######################################################################
######################################################################
# Xicor method experiment


# Create x-values and noise 
np.random.seed(42)
x = np.random.uniform(-10, 10, size=500)
noise = np.random.uniform(-10, 10, size=500)

add_data = pd.read_csv(datapath+'07.DatasaurusDozen.tsv', sep='\t')


# Create several y-values
y1 = x
y2 = x + noise
y3 = 3*x
y4 = 3*x + noise
y5 = x**2
y6 = x**2 + noise
y7 = 2**x
y8 = 2**x + 10 * noise
y9 = -3 * x**3 + 9 * x**2 + 6 * x + 53
y10 = -3 * x**3 + 9 * x**2 + 6 * x + 53 + 100 * noise


# Create functions to iterate and plot
ids = [id for id in range(1,11)]
y_funcs = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
x_funcs = [x,x,x,x,x,x,x,x,x,x]
titles = ["Y1 = X", 
          "Y2 = X + noise",
          "Y3 = 3X", 
          "Y4 = 3X + noise",
          "Y5 = X^2",
          "Y6 = X^2 + noise",
          "Y7 = 2^X",
          "Y8 = 2^X + 10*noise", 
          "Y9 = -3X^3 + ... + 53",
          "Y10 = -3X^3 + ... + 100*noise"]


# Create a grid plot
plt.figure(figsize=(30,3))
for i, x_vals, y_vals, title in zip(ids, x_funcs, y_funcs, titles):
    print(i)
    plt.subplot(1, 10, i)
    plt.scatter(x_vals, y_vals, alpha=0.25)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig(plotpath +'07.example_scatter.png', dpi = 300)
plt.savefig(plotpath +'07.example_scatter.pdf', dpi = 300)

plt.show()





# data 1 

methods = ['average','max','min','ordinal','dense']

scores = []
for x_vals, y_vals in zip(x_funcs, y_funcs):
    data_res = []
    pcor = stats.pearsonr(np.array(x_vals), np.array(y_vals))[0]
    scor = stats.spearmanr(np.array(x_vals), np.array(y_vals))[0]
    data_res.append(pcor)
    data_res.append(scor)
    for method in methods :
        xi_1 = xi_cor(np.array(x_vals), np.array(y_vals), method)
        xi_2 = xi_cor(np.array(y_vals), np.array(x_vals), method)
        data_res.append(max(xi_1, xi_2))
    scores.append(data_res)
    



# Plot the results
fig, ax = plt.subplots(figsize=(30, 3), constrained_layout=True)

x_vals = np.arange(1, 11, 1)
ax.bar(x=x_vals - .30, height=np.abs([a[0] for a in scores]), color="black", width=.1, label="Pearson")
ax.bar(x=x_vals - .20, height=np.abs([a[1] for a in scores]), color="grey", width=.1, label="Spearman")
ax.bar(x=x_vals - .10, height=np.abs([a[2] for a in scores]), color="red", width=.1, label="XI-Average")
ax.bar(x=x_vals - .00, height=np.abs([a[3] for a in scores]), color="green", width=.1, label="XI-Max")
ax.bar(x=x_vals + .10, height=np.abs([a[4] for a in scores]), color="blue", width=.1, label="XI-Min")
ax.bar(x=x_vals + .20, height=np.abs([a[5] for a in scores]), color="purple", width=.1, label="XI-Ordinal")
ax.bar(x=x_vals + .30, height=np.abs([a[6] for a in scores]), color="orange", width=.1, label="XI-Dense")

ax.set_xticks(x_vals)
ax.set_xticklabels([])

for loc in ["top", "right"]:
    ax.spines[loc].set_visible(False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(plotpath +'07.example_scatter_score.png', dpi = 300)
plt.savefig(plotpath +'07.example_scatter_score.pdf', dpi = 300)
plt.show()




# data 2
ids = [id for id in range(0,6)]
col_names = ['dino','x_shape','high_lines','dots','bullseye','slant_up']
col_renames = ['Dino','X shape','High lines','Dots','Bullseye','Slant up']

methods = ['average','max','min','ordinal','dense']


fig, ax = plt.subplots(figsize=(18, 3), nrows = 1, ncols = 6)

scores = []
for id, colname, renames in zip(ids, col_names, col_renames) :
    x_vals = add_data[add_data.dataset == colname]['x']
    y_vals = add_data[add_data.dataset == colname]['y']
    data_res = []
    pcor = stats.pearsonr(np.array(x_vals), np.array(y_vals))[0]
    scor = stats.spearmanr(np.array(x_vals), np.array(y_vals))[0]
    data_res.append(pcor)
    data_res.append(scor)
    for method in methods :
        xi_1 = xi_cor(np.array(x_vals), np.array(y_vals), method)
        xi_2 = xi_cor(np.array(y_vals), np.array(x_vals), method)
        data_res.append(max(xi_1, xi_2))
    scores.append(data_res)
    ax[id].scatter(x_vals, y_vals, alpha=0.25)
    ax[id].set_title(renames)
    ax[id].set_xticks([])
    ax[id].set_yticks([])


plt.tight_layout()
plt.savefig(plotpath +'07.data_scatter.png', dpi = 300)
plt.savefig(plotpath +'07.data_scatter.pdf', dpi = 300)


# Plot the results
fig, ax = plt.subplots(figsize=(18, 3), constrained_layout=True)

x_vals = np.arange(1, 7, 1)
ax.bar(x=x_vals - .30, height=np.abs([a[0] for a in scores]), color="black", width=.1, label="Pearson")
ax.bar(x=x_vals - .20, height=np.abs([a[1] for a in scores]), color="grey", width=.1, label="Spearman")
ax.bar(x=x_vals - .10, height=np.abs([a[2] for a in scores]), color="red", width=.1, label="XI-Average")
ax.bar(x=x_vals - .00, height=np.abs([a[3] for a in scores]), color="green", width=.1, label="XI-Max")
ax.bar(x=x_vals + .10, height=np.abs([a[4] for a in scores]), color="blue", width=.1, label="XI-Min")
ax.bar(x=x_vals + .20, height=np.abs([a[5] for a in scores]), color="purple", width=.1, label="XI-Ordinal")
ax.bar(x=x_vals + .30, height=np.abs([a[6] for a in scores]), color="orange", width=.1, label="XI-Dense")

ax.set_xticks(x_vals)
ax.set_xticklabels([])

for loc in ["top", "right"]:
    ax.spines[loc].set_visible(False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(plotpath +'07.data_scatter_score.png', dpi = 300)
plt.savefig(plotpath +'07.data_scatter_score.pdf', dpi = 300)
plt.show()


