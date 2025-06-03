
# 지금하는건 이 두개만 pdf 로 저장할 수 있게 하면 됨 
# 이미 한거를 다시 그냥 붙이는거라 문제는 없을 것으로 생각


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import scipy.spatial as ssp
from matplotlib import rcParams
import numpy as np
import seaborn as sns 

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
s_zeta = r'$Xi\rho$' # r'$\zeta$'
s_zeta_t = r'$\tilde{Xi\rho}$'


# SCALING fig 

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



# 부트스트랩 함수 정의
def bootstrap_xicor(x, y, n_bootstrap=1000):
    n = len(x)
    bootstrap_results = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        tmp_1 = xi_cor(x[indices], y[indices], 'dense')
        tmp_2 = xi_cor(y[indices], x[indices], 'dense')
        bootstrap_results.append(max(tmp_1, tmp_2))
    return np.mean(bootstrap_results), np.std(bootstrap_results)

# 데이터 포인트 개수 (7의 배수)
data_points = [7*i for i in range(1,11)]

# 결과 저장용
results_linear = []
results_non_linear = []
results_u_shape = []
results_go_up = []
results_go_dn = []

for n in data_points:
    # 
    x_linear = np.linspace(0, 10, n)
    y_linear = x_linear
    ori_xi = max(xi_cor(x_linear, y_linear, 'dense'), xi_cor(y_linear, x_linear, 'dense'))
    mean_linear, std_linear = bootstrap_xicor(x_linear, y_linear)
    results_linear.append((ori_xi, mean_linear, std_linear))
    # 
    x_non_linear = np.linspace(0, 10, n)
    y_non_linear = np.where(x_non_linear <= 3, 0, -(x_non_linear - 7)**2 + 16)  # y가 0에서 시작하도록 수정
    ori_xi = max(xi_cor(x_non_linear, y_non_linear, 'dense'), xi_cor(y_non_linear, x_non_linear, 'dense'))
    mean_non_linear, std_non_linear = bootstrap_xicor(x_non_linear, y_non_linear)
    results_non_linear.append((ori_xi, mean_non_linear, std_non_linear))
    # 
    x_u_shape = np.linspace(-5, 5, n)
    y_u_shape = x_u_shape**2
    ori_xi = max(xi_cor(x_u_shape, y_u_shape, 'dense'), xi_cor(y_u_shape, x_u_shape, 'dense'))
    mean_u_shape, std_u_shape = bootstrap_xicor(x_u_shape, y_u_shape)
    results_u_shape.append((ori_xi, mean_u_shape, std_u_shape))
    #
    x_go_up = np.linspace(0, 10, n)
    y_go_up = 2**x_go_up
    ori_xi = max(xi_cor(x_go_up, y_go_up, 'dense'), xi_cor(y_go_up, x_go_up, 'dense'))
    mean_go_up, std_go_up = bootstrap_xicor(x_go_up, y_go_up)
    results_go_up.append((ori_xi, mean_go_up, std_go_up))
    #
    x_go_dn = np.linspace(-5, 5, n)
    y_go_dn = -3*(x_go_dn)**3 + 9*(x_go_dn)**2+6*(x_go_dn) + 53
    ori_xi = max(xi_cor(x_go_dn, y_go_dn, 'dense'), xi_cor(y_go_dn, x_go_dn, 'dense'))
    mean_go_dn, std_go_dn = bootstrap_xicor(x_go_dn, y_go_dn)
    results_go_dn.append((ori_xi, mean_go_dn, std_go_dn))    
    # 
    if n == 7 :
        num_7_example_points = [(x_linear, y_linear), (x_non_linear, y_non_linear), (x_u_shape, y_u_shape), (x_go_up, y_go_up), (x_go_dn, y_go_dn)]
    #
    if n == 70 : 
        num_70_example_points = [(x_linear, y_linear), (x_non_linear, y_non_linear), (x_u_shape, y_u_shape), (x_go_up, y_go_up), (x_go_dn, y_go_dn)]

        

# fitting 결과 활용 

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



# obs_res에서 N과 observed_xicor 값을 분리
N_values = np.array([x[0] for x in obs_res])
observed_xicor_values = np.array([x[1] for x in obs_res])
answer = np.array([x[2] for x in obs_res])



# curve_fit을 사용하여 c 값을 fitting
popt, pcov = curve_fit(scaling_model, (N_values, observed_xicor_values), answer)


def get_new_xi (num, old_xi) : 
    new = scaling_model((num, old_xi), popt[0])
    if new >1 : 
        return(1)
    elif new <0 : 
        return (0)
    else :
        return(new)



fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18,6))

# 일단 rho 관련 추가 
labels = [7 * i for i in range(1, 11)]

xi_1_res = [] ; xi_2_res = []
rho_1_res = [] ; rho_2_res = []

for n in range(1, 15): 
    for p in range(100): 
        x = np.random.uniform(0, 10, size=7 * n) 
        noise = np.random.uniform(0, 5, size=7 * n) 
        y = 2 * x
        #
        xi_1 = xi_cor(x, y, 'dense')
        rho_1 = stats.pearsonr(x, y)[0]
        #
        xi_1_res.append((n, xi_1))    # 샘플 크기 정보 추가
        rho_1_res.append((n, rho_1))
        #
        if (n == 1) & (p == 0) :
            sns.scatterplot(x = x, y = y, ax = axes[0][0])


means_rho = []
stds_rho = []
means_xi = []
stds_xi = []

for n in range(1, 15):
    rho_vals = [r for (label, r) in rho_1_res if label == n]
    xi_vals = [x for (label, x) in xi_1_res if label == n]
    means_rho.append(np.mean(rho_vals))
    stds_rho.append(np.std(rho_vals))
    means_xi.append(np.mean(xi_vals))
    stds_xi.append(np.std(xi_vals))
    axes[1][0].text(np.mean(rho_vals), np.mean(xi_vals), str(7 * n), fontsize=7, color ='white',
            ha='center', va='center')

sns.scatterplot(x=means_rho, y=means_xi, ax=axes[1][0], s=150, color = 'black',  edgecolor='black', facecolors='black', legend=False)


# 축 설정
axes[1][0].set_ylim(0.6, 1)
axes[1][0].set_ylabel(s_xi)
axes[1][0].set_xlabel(s_pcc)



for i in range(5) :
    dataset_2 = num_70_example_points[i]
    sns.scatterplot(x = dataset_2[0] , y = dataset_2[1], ax = axes[0][i+1])

sns.lineplot(x = data_points, y = [r[0] for r in results_linear], label = 'original', ax = axes[1][1])
sns.lineplot(x = data_points, y = [results_linear[a][0]/(1-(5/((a+1)*7))) for a in range(10)], label = 'c=5', ax = axes[1][1])
sns.lineplot(x = data_points, y = [results_linear[a][0]/(1-(3/((a+1)*7))) for a in range(10)], label = 'c=3', ax = axes[1][1])
sns.lineplot(x = data_points, y = [results_linear[a][0]/(1-(popt.item()/((a+1)*7))) for a in range(10)], label = 'fitted', ax = axes[1][1])

sns.lineplot(x = data_points, y = [r[0] for r in results_non_linear], label = 'original', ax = axes[1][2])
sns.lineplot(x = data_points, y = [results_non_linear[a][0]/(1-(5/((a+1)*7))) for a in range(10)], label = 'c=5', ax = axes[1][2])
sns.lineplot(x = data_points, y = [results_non_linear[a][0]/(1-(3/((a+1)*7))) for a in range(10)], label = 'c=3', ax = axes[1][2])
sns.lineplot(x = data_points, y = [results_non_linear[a][0]/(1-(popt.item()/((a+1)*7))) for a in range(10)], label = 'fitted', ax = axes[1][2])

sns.lineplot(x = data_points, y = [r[0] for r in results_u_shape], label = 'original', ax = axes[1][3])
sns.lineplot(x = data_points, y = [results_u_shape[a][0]/(1-(5/((a+1)*7))) for a in range(10)], label = 'c=5', ax = axes[1][3])
sns.lineplot(x = data_points, y = [results_u_shape[a][0]/(1-(3/((a+1)*7))) for a in range(10)], label = 'c=3', ax = axes[1][3])
sns.lineplot(x = data_points, y = [results_u_shape[a][0]/(1-(popt.item()/((a+1)*7))) for a in range(10)], label = 'fitted', ax = axes[1][3])

sns.lineplot(x = data_points, y = [r[0] for r in results_go_up], label = 'original', ax = axes[1][4])
sns.lineplot(x = data_points, y = [results_go_up[a][0]/(1-(5/((a+1)*7))) for a in range(10)], label = 'c=5', ax = axes[1][4])
sns.lineplot(x = data_points, y = [results_go_up[a][0]/(1-(3/((a+1)*7))) for a in range(10)], label = 'c=3', ax = axes[1][4])
sns.lineplot(x = data_points, y = [results_go_up[a][0]/(1-(popt.item()/((a+1)*7))) for a in range(10)], label = 'fitted', ax = axes[1][4])

sns.lineplot(x = data_points, y = [r[0] for r in results_go_dn], label = 'original', ax = axes[1][5])
sns.lineplot(x = data_points, y = [results_go_dn[a][0]/(1-(5/((a+1)*7))) for a in range(10)], label = 'c=5', ax = axes[1][5])
sns.lineplot(x = data_points, y = [results_go_dn[a][0]/(1-(3/((a+1)*7))) for a in range(10)], label = 'c=3', ax = axes[1][5])
sns.lineplot(x = data_points, y = [results_go_dn[a][0]/(1-(popt.item()/((a+1)*7))) for a in range(10)], label = 'fitted', ax = axes[1][5])

for i in range(1,6) :
    axes[1][i].set_xlabel('# sample')
    axes[1][i].set_ylabel('mean('+s_xi+')')
    axes[1][i].set_ylim(0, 1.2)
    axes[1][i].grid()
    axes[1][i].set_xticks([a*7 for a in range(11)], [a*7 for a in range(11)])


plt.tight_layout()
plt.savefig(plotpath + '07.XI_scaling.png', dpi = 300)
plt.savefig(plotpath + '07.XI_scaling.pdf', dpi = 300)

plt.show()



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
plt.tight_layout()
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





