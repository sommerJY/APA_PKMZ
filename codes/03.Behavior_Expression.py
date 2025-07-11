
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import copy 
import scipy.stats as stats 

datapath = './data/'
plotpath = './figures/'

# For pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42   # EPS saving 

# Arial as default
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']



# To merge expression and behavioral PC1 
# I restarted separating the subfields, also to match with Asit code 

# Original count 

behav_PC1 = pd.read_csv(datapath + '00.NEW_PC1.csv', index_col = 0)
colData = pd.read_csv(datapath+ "00_colData.csv")

countData = pd.read_csv(datapath + "00_countData.csv", index_col = 0)
Outliers = ["146D-DG-3", "145A-CA3-2", "146B-DG-2", "146D-CA1-3", "148B-CA1-4"]
colData['class'] = [1 if a =='trained' else 0 for a in list(colData['training'])]
colData = colData[colData.RNAseqID.isin(Outliers)==False]

RNA_PC = pd.merge(colData, behav_PC1, on = ['ID', 'treatment'], how= 'left')
RNA_PC['training_class'] = RNA_PC.training.apply(lambda x : 1 if x=='trained' else 0)


DG_samples = list(colData[colData.subfield =='DG']['RNAseqID'])
CA3_samples = list(colData[colData.subfield =='CA3']['RNAseqID'])
CA1_samples = list(colData[colData.subfield =='CA1']['RNAseqID'])
all_samples = DG_samples + CA3_samples + CA1_samples


# Normalizing subfield wise 
def get_count (sub, all_count):  
    subs = ['all', 'DG', 'CA3', 'CA1']  
    all_count = all_count[all_samples]
    DG_count = all_count[DG_samples]
    CA3_count = all_count[CA3_samples]
    CA1_count = all_count[CA1_samples]
    #
    count_list = [all_count, DG_count, CA3_count, CA1_count]
    return (count_list[subs.index(sub)])


def normalizing (sub):
    count_matrix = get_count(sub, countData)
    sample_num = count_matrix.shape[1]
    size_factor = count_matrix.loc[(count_matrix!=0).all(axis=1)]
    size_factor.loc[:,'geo_mean'] = None
    def func_geomean(data):
        a = np.log(data)
        return np.exp(np.mean(a, axis=1))
    #
    size_factor['geo_mean'] = func_geomean(size_factor.drop(labels='geo_mean',axis=1))
    size_factor['geo_mean'].values
    size_factor.iloc[:,:sample_num] = size_factor.iloc[:,:sample_num].div(size_factor['geo_mean'], axis = 0)
    size_factor.loc['Normalisation_sample'] = None
    size_factor.loc['Normalisation_sample'] = size_factor.median(axis=0)
    Normalised = count_matrix.div(size_factor.loc['Normalisation_sample'], axis=1)
    Normalised = Normalised.drop('geo_mean', axis = 1)
    return (Normalised, list(Normalised.index))


out_all, out_all_genes = normalizing('all') 
out_DG, out_DG_genes = normalizing('DG') 
out_CA3, out_CA3_genes = normalizing('CA3')
out_CA1, out_CA1_genes = normalizing('CA1')


def filter_low (out, gene_list) : 
    norm_df_pre = out.T
    df_gene = gene_list
    filter = np.sum(norm_df_pre[df_gene]>np.min(norm_df_pre[df_gene]), axis = 0)>1
    filter_genes = list(filter[filter==True].index) 
    norm_df = np.log2(norm_df_pre[filter_genes]+1)
    norm_df['RNAseqID'] = list(norm_df.index)
    norm_df_info = pd.merge(norm_df, RNA_PC[['RNAseqID','treatment','training','training_class','PC1','PC2']], on = 'RNAseqID')
    print('num genes : {}'.format(len(filter_genes)))
    return norm_df_info, filter_genes


RNA_ALL, RNA_ALL_genes = filter_low(out_all, out_all_genes)  # 17413
RNA_DG, RNA_DG_genes = filter_low(out_DG, out_DG_genes) # 16467
RNA_CA3, RNA_CA3_genes = filter_low(out_CA3, out_CA3_genes) # 15778
RNA_CA1, RNA_CA1_genes = filter_low(out_CA1, out_CA1_genes) # 16171



   
candidategenes = [
    "Fos", "Fosl2", "Npas4", "Arc", 
    "Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1", 
    "Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci"]


astrocytegenes = ["Aldh1a1", "Aldh1l1", "Aldh1l2", "Slc1a2", "Gfap", "Gjb6", "Fgfr3", "Aqp4", "Aldoc"]


# Barplot and heatmap 

pc1_label = r'Pearson Coef. PC1$^{\mathrm{memory}}$'

def corr_matrix(sub_matrix, sub, candidategenes, title, titleA ='a', titleB ='b' ) :
    exp_matrix = copy.deepcopy(sub_matrix)
    corr_list = candidategenes+['PC1','PC2']
    corr_matrix = np.zeros((len(corr_list), len(corr_list)))
    #
    for i in range(0,len(corr_list)) : 
        for j in range(0,len(corr_list)) :
            corr_matrix[i,j] = stats.pearsonr(exp_matrix[corr_list[i]], exp_matrix[corr_list[j]])[0]
    #
    corr_matrix_df = pd.DataFrame(corr_matrix)
    corr_matrix_df.columns = corr_list
    corr_matrix_df.index = corr_list
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#2680ff", "white", "#ff4a26"])
    norm = plt.Normalize(-1, 1)  # 
    #
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), gridspec_kw={'width_ratios': [0.7, 1.3]})
    # Barplot 
    ax_bar = axes[0]
    tmp_df = corr_matrix_df.loc[candidategenes,'PC1']
    bar_colors = [custom_cmap(norm(val)) for val in tmp_df]
    sns.barplot(y=tmp_df.index, x=tmp_df.values, ax=ax_bar,  palette=bar_colors)
    ax_bar.set_title(titleA, loc = 'left', fontsize = 12)
    ax_bar.set_xlabel(pc1_label, fontsize = 11)
    ax_bar.set_ylabel(sub, fontsize= 11)
    ax_bar.set_xlim(-1,1)
    ax_bar.set_xticks(ax_bar.get_xticks())
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), fontsize = 8)
    ax_bar.set_yticks(ax_bar.get_yticks())
    ax_bar.set_yticklabels(ax_bar.get_yticklabels(), fontsize = 8)
    ax_bar.axvline(x=-0.7, color='grey', linestyle='--', linewidth=0.5)
    ax_bar.axvline(x=0.7, color='grey', linestyle='--', linewidth=0.5)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    # Heatmap 
    ax_heat = axes[1]
    tmp_df = corr_matrix_df.loc[candidategenes,candidategenes]
    annot_matrix = np.where((np.abs(tmp_df) >= 0.6) & (np.abs(tmp_df) < 0.99), tmp_df.round(2).astype(str), "")
    sns.heatmap(tmp_df, annot=annot_matrix, annot_kws={"color": "black", 'fontsize': 6},
                cmap=custom_cmap, center=0, ax=ax_heat,
                linewidths=0, vmin=-1, vmax=1, fmt="", cbar = False)
    ax_heat.set_title(titleB, loc = 'left', fontsize = 12)
    ax_heat.set_xlabel('')
    ax_heat.set_ylabel('')
    ax_heat.set_xticks(ax_heat.get_xticks())
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), fontsize = 8)
    ax_heat.set_yticks(np.arange(len(tmp_df)) + 0.5)
    ax_heat.set_yticklabels(tmp_df.index, rotation = 0 , fontsize = 8)
    plt.tight_layout()
    plt.savefig(plotpath + '03.bar_heat.{}.{}.png'.format(title, sub), dpi = 300)
    plt.savefig(plotpath + '03.bar_heat.{}.{}.pdf'.format(title, sub), dpi = 300)
    plt.savefig(plotpath + '03.bar_heat.{}.{}.eps'.format(title, sub), dpi = 300)
    plt.savefig(plotpath + '03.bar_heat.{}.{}.tiff'.format(title, sub), dpi = 300)
    plt.close()


corr_matrix(RNA_DG, 'DG', candidategenes, 'candy', titleA = 'a', titleB ='b')
corr_matrix(RNA_CA3,'CA3', candidategenes, 'candy', titleA = 'c', titleB ='d')
corr_matrix(RNA_CA1, 'CA1', candidategenes, 'candy', titleA = 'e', titleB ='f')

corr_matrix(RNA_DG, 'DG', astrocytegenes, 'astro', titleA = 'a', titleB ='b')
corr_matrix(RNA_CA3, 'CA3', astrocytegenes,'astro', titleA = 'c', titleB ='d')
corr_matrix(RNA_CA1,'CA1', astrocytegenes, 'astro', titleA = 'e', titleB ='f')


RNA_DG.to_csv(datapath+'03.EXP_PC1_merge.DG.csv')
RNA_CA3.to_csv(datapath+'03.EXP_PC1_merge.CA3.csv')
RNA_CA1.to_csv(datapath+'03.EXP_PC1_merge.CA1.csv')



# just to check pearson correlation difference
# RNA_DG_yo = RNA_DG[RNA_DG.training == 'yoked']
# RNA_DG_tr = RNA_DG[RNA_DG.training == 'trained']

# corr_list = candidategenes
# corr_matrix_yo = np.zeros((len(corr_list), len(corr_list)))
# #
# for i in range(0,len(corr_list)) : 
#     for j in range(0,len(corr_list)) :
#         corr_matrix_yo[i,j] = stats.pearsonr(RNA_DG_yo[corr_list[i]], RNA_DG_yo[corr_list[j]])[0]
# #
# corr_matrix_tr = np.zeros((len(corr_list), len(corr_list)))
# #
# for i in range(0,len(corr_list)) : 
#     for j in range(0,len(corr_list)) :
#         corr_matrix_tr[i,j] = stats.pearsonr(RNA_DG_tr[corr_list[i]], RNA_DG_tr[corr_list[j]])[0]

# corr_matrix_diff = corr_matrix_tr - corr_matrix_yo

# corr_matrix_df = pd.DataFrame(corr_matrix_diff)
# corr_matrix_df.columns = corr_list
# corr_matrix_df.index = corr_list
# custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#2680ff", "white", "#ff4a26"])
# norm = plt.Normalize(-1, 1)  # 

# fig, axes = plt.subplots(1, 1, figsize=(3, 3))
# ax_heat = axes
# tmp_df = corr_matrix_df.loc[candidategenes,candidategenes]
# annot_matrix = np.where((np.abs(tmp_df) >= 0.6) & (np.abs(tmp_df) < 0.99), tmp_df.round(2).astype(str), "")
# sns.clustermap(tmp_df, annot_kws={"color": "black", 'fontsize': 6},
#             cmap=custom_cmap, center=0, 
#             linewidths=0, vmin=-1, vmax=1, fmt="", cbar = False)

# ax_heat.set_xlabel('')
# ax_heat.set_ylabel('')
# ax_heat.set_xticks(ax_heat.get_xticks())
# ax_heat.set_xticklabels(ax_heat.get_xticklabels(), fontsize = 8)
# ax_heat.set_yticks(np.arange(len(tmp_df)) + 0.5)
# ax_heat.set_yticklabels(tmp_df.index, rotation = 0 , fontsize = 8)



###########
# Correlation of DEG and PCs
PC1_cor_list = []
PC2_cor_list = []

all_deg = pd.read_csv(datapath+'02.allDEG.csv', index_col = 0 )

DG_deg = all_deg[(all_deg.tissue =='DG') & (all_deg.comparison=='yoked vs. trained')]
DEG_list = list(DG_deg.gene)

for deg in DEG_list :
    pc, pv = stats.pearsonr(RNA_DG[deg], RNA_DG['PC1'])
    PC1_cor_list.append((deg, pc, pv))
    pc, pv = stats.pearsonr(RNA_DG[deg], RNA_DG['PC2'])
    PC2_cor_list.append((deg, pc, pv))


PC1_df = pd.DataFrame(PC1_cor_list)
PC2_df = pd.DataFrame(PC2_cor_list)

PC1_df.columns = ['gene','PC1 PCor','p-value']
PC2_df.columns = ['gene','PC1 PCor','p-value']

PC1_df['IEG'] = ['IEG' if a in candidategenes else '' for a in list(PC1_df.gene)]
PC1_df = PC1_df.sort_values('PC1 PCor', ascending = False)
PC1_df = PC1_df.reset_index(drop = True)

PC1_df.to_csv(datapath+'03.PC1_DEG_corr.csv')


PC1_df[PC1_df['p-value'] <= 0.05] # 247 

# to gprofiler web 
PC1_df_pos = PC1_df[PC1_df['PC1 PCor'] >0]

gprof_web_res = pd.read_csv(datapath+'03.gProfiler_DEG_PCpositive.csv')
gprof_web_res['intersection_re'] = gprof_web_res['intersections'].apply(lambda x : ', '.join([a[0].upper()+a[1:].lower() for a in  x.split(',')]))
gprof_web_res_re = gprof_web_res[['source','term_name','term_size','intersection_size', 'intersection_re']]
gprof_web_res_re.columns = ['Domain','Name','Total','Present', 'Genes']
gprof_web_res_re['Domain'] = gprof_web_res_re.Domain.apply(lambda x : x.split(':')[1])
gprof_web_res_re.to_csv(datapath + '03.gProfiler_DEG_PCpos_edited.csv')




########## To see whether 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


be_for_ano = copy.deepcopy(RNA_DG)

def check_gene_aov (gene, sub, be_for_ano) : 
    # just to check whether the gene can differ in 4 treatment 
    model = ols('{} ~ treatment'.format(gene), data=be_for_ano).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # ANOVA table
    ano_table = pd.DataFrame(anova_table.loc['treatment']).T
    ano_table['Gene'] = gene
    ano_table['Subfield'] = sub
    #print(anova_table)
    #
    tukey_result = pairwise_tukeyhsd(be_for_ano[gene], be_for_ano['treatment'])
    #print(tukey_result)
    tukey_df = pd.DataFrame(tukey_result.summary().data[1:],  
                        columns=tukey_result.summary().data[0])
    tukey_df['Group'] = tukey_df['group1'] + '.VS.' + tukey_df['group2']
    tukey_df['Subfield'] = sub
    tukey_df['Gene'] = gene
    tukey_df_mini = tukey_df[['Subfield','Gene','Group','meandiff','p-adj','reject']]
    #
    # check 2 training in t-test 
    tt = stats.ttest_ind(be_for_ano[be_for_ano.training=='yoked'][gene] , be_for_ano[be_for_ano.training=='trained'][gene])
    t_df = pd.DataFrame({
        'Subfield' : sub,
        'Gene' : gene, 
        't-stat' : np.round(tt.statistic,2),
        'pvalue' : np.round(tt.pvalue,2),
        'df' : np.round(tt.df,2),
        }, index = [0])
    return ano_table, tukey_df_mini, t_df
    

anova_list = [] 
tucky_list = []
tt_list = []

for gg in candidategenes :
    anovatable, tucky, tteest = check_gene_aov(gg,'DG', RNA_DG)
    anova_list.append(anovatable)
    tucky_list.append(tucky)
    tt_list.append(tteest)
    #
    anovatable, tucky, tteest = check_gene_aov(gg,'CA3', RNA_CA3)
    anova_list.append(anovatable)
    tucky_list.append(tucky)
    tt_list.append(tteest)
    #
    anovatable, tucky, tteest = check_gene_aov(gg,'CA1', RNA_CA1)
    anova_list.append(anovatable)
    tucky_list.append(tucky)
    tt_list.append(tteest)


anova_df = pd.concat(anova_list)
tucky_df = pd.concat(tucky_list)
tt_df = pd.concat(tt_list)

anova_df['Gene'] = pd.Categorical(anova_df['Gene'], categories=candidategenes) 
anova_df = anova_df.sort_values(['Subfield','Gene'], ascending = [False, True])
anova_df['sig'] = anova_df['PR(>F)'].apply(lambda x : '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ' ')
anova_df = anova_df[['Subfield','Gene','sum_sq','df','F','PR(>F)', 'sig']].reset_index(drop = True)

tucky_df['Gene'] = pd.Categorical(tucky_df['Gene'], categories=candidategenes) 
tucky_df = tucky_df.sort_values(['Subfield', 'Group', 'Gene'], ascending = [False, True, True])
tucky_df['sig'] = tucky_df['p-adj'].apply(lambda x : '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ' ')
tucky_df = tucky_df[['Subfield', 'Gene', 'Group', 'meandiff', 'p-adj', 'sig']]

tt_df['Gene'] = pd.Categorical(tt_df['Gene'], categories=candidategenes) 
tt_df = tt_df.sort_values(['Subfield','Gene'], ascending = [False, True])
tt_df['sig'] = tt_df['pvalue'].apply(lambda x : '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ' ')


anova_df = anova_df.reset_index(drop=True)
anova_df.to_csv(datapath + '03.candidate_ANOVA.csv')

tucky_df = tucky_df.reset_index(drop=True)
tucky_df.to_csv(datapath + '03.candidate_Tucky.csv')

tt_df = tt_df.reset_index(drop=True)
tt_df.to_csv(datapath + '03.candidate_Ttest.csv')







#####
# Fold change check 

fc_check = RNA_DG[RNA_DG_genes + ['training']].groupby('training').mean().T
fc_check['FC'] = fc_check['trained'] - fc_check['yoked']
fc_check['candy'] = ['O' if x in candidategenes else 'X' for x in list(fc_check.index)]
fc_check.to_csv(datapath + '03.candy_fc.csv')





