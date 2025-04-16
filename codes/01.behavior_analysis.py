
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams

from tqdm import tqdm
import copy
import seaborn as sns

from sklearn import decomposition
PCA = decomposition.PCA
import scipy.stats as stats 

datapath = './data/'
plotpath = './figures/'

# for pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 


# facor levels
levelstreatment =  ['standard.yoked' , 'standard.trained' ,
                     'conflict.yoked' ,  'conflict.trained' ]
levelstreatmentlegend = ['standard yoked' , 'standard trained' ,
                          'conflict yoked' ,  'conflict trained' ]
levelstraining = ["yoked", "trained"]
levelssubfield = ["DG", "CA3", "CA1"]


treatment_names = dict({
  'standard.yoked' : "standard yoked",
  'standard.trained' :"standard trained",
  'conflict.yoked' : "conflict yoked",
  'conflict.trained' : "conflict trained"
})


treatmentcolors = dict({ "standard.yoked" : "#404040", 
                      "standard.trained" : "#ca0020",
                      "conflict.yoked" : "#969696",
                      "conflict.trained" : "#f4a582"})


colorvalsubfield = dict({"DG" : "#d95f02", 
                      "CA3" : "#1b9e77", 
                      "CA1" : "#7570b3"})


trainingcolors =  dict({"trained" : "darkred", 
                     "yoked" : "black"})


allcolors = copy.deepcopy(treatmentcolors)


allcolors.update(colorvalsubfield)
allcolors.update(trainingcolors)
allcolors['NS'] = "#d9d9d9"

colData = pd.read_csv(datapath+ "00_colData.csv")

behavior = pd.read_csv(datapath + '00_behaviordata.csv')
behavior['trial'] = ['Pre' if a == 'Hab' else a for a in list(behavior['trial'])]
behavior[behavior.trial =='Pre'].groupby('treatment').count()

trialnameandnumber = behavior[['trial','trialNum']].drop_duplicates()
dfshocks = (
    behavior.groupby(['treatment', 'trial', 'trialNum'], as_index=False)
    .agg(
        m=('NumShock', lambda x: np.around(np.mean(x), 2)), # get the average 
        se=('NumShock', lambda x: np.around(np.std(x, ddof=1) / np.sqrt(len(x)),2))  # standard deviation 
    )
    .assign(measure="NumShock")  # to new column
)



# just new naming
groups = ['standard.yoked', 'standard.trained', 'conflict.yoked', 'conflict.trained']
colors = [treatmentcolors[a] for a in dfshocks['treatment']]
dfshocks['treatment_col'] = colors
reorders = ['Pre','T1','T2','T3','Retest','T4_C1','T5_C2', 'T6_C3', 'Retention']

dfshocks_save = copy.deepcopy(dfshocks)

dfshocks_save['treatment'] = pd.Categorical(dfshocks_save['treatment'], groups)
dfshocks_save['trial'] = pd.Categorical(dfshocks_save['trial'], reorders)

dfshocks_save = dfshocks_save.sort_values(['treatment','trial'])
dfshocks_save = dfshocks_save.reset_index(drop = True)

dfshocks_save[['treatment','trial','trialNum','m','se']].to_csv(datapath + '01.behav_table_stat.csv')




# plot for behavioral traits
fig1, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 10))
axes = axes.flatten()



for i, treatment in enumerate(groups):
    subset = dfshocks[dfshocks["treatment"] == treatment]    
    subset['trial'] = pd.Categorical(subset['trial'], categories=reorders, ordered=True)
    subset = subset.sort_values('trial')
    subset = subset.reset_index(drop=True)
    axes[i].plot(subset["trial"], subset["m"], label=treatment, 
                 color=treatmentcolors[treatment], linewidth=3)
    #for _, row in subset.iterrows():
    #    axes[i].errorbar(row["trial"], row["m"], yerr=row["se"], fmt="none", 
    #                     color=treatmentcolors[row["treatment"]], capsize=2, linewidth=0.8)
    for _, row in subset.iterrows():
        axes[i].text(row["trial"], row["m"] + 1, f"{int(np.around(row['m']))}",
                     ha='center', va='bottom', fontsize=14, color=treatmentcolors[row["treatment"]])
    #
    axes[i].set_xticklabels(["P", "T1", "T2", "T3", "Rt", "T4", "T5", "T6", "Rn"])
    axes[i].set_yticks([])
    axes[i].set_ylim(0, max(subset.m) + 3)
    axes[i].set_ylabel("")
    axes[i].set_xlabel("")
    axes[i].set_title(treatment.replace('.', ' '), loc = 'left')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)

axes[3].spines['bottom'].set_visible(True)  # additional edition

# Suptitle을 왼쪽 세로 정렬로 설정
fig1.suptitle("Number of Shocks", fontsize=14, x=0.02, y=0.5, rotation=90, va='center')

plt.tight_layout()
plt.savefig(plotpath + '01.bhv_shock.png', dpi = 300)
plt.savefig(plotpath + '01.bhv_shock.pdf', dpi = 300, bbox_inches='tight')
plt.show()




# Get PCA result from behavioral data 

behavior['train_col'] = [trainingcolors[a] for a in list(behavior['training'])]
behavior['treat_col'] = [treatmentcolors[a] for a in list(behavior['treatment'])]

ind_a = list(behavior.columns).index('TotalPath')
ind_b = list(behavior.columns).index('ShockPerEntrance')

behav_columns = list(behavior.columns)[ind_a : ind_b+1]

def makepcadf (behav_table) :
    Z = behav_table[behav_columns]
    Z = Z.loc[:, Z.var(skipna=True) != 0]
    Z_scaled = (Z - Z.mean()) / Z.std()
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(Z_scaled)
    PC1_ratio = pca.explained_variance_ratio_[0]
    PC2_ratio = pca.explained_variance_ratio_[1]
    pc_df = pd.DataFrame(principal_components)
    pc_df.columns = ['PC1','PC2']
    pc_df['ID'] = list(behav_table['ID'])
    pc_df['treatment'] = list(behav_table['treatment'])
    pc_df['training'] = list(behav_table['training'])
    pc_df['trialNum'] = list(behav_table['trialNum'])
    pc_df['Day'] = list(behav_table['Day'])
    return pc_df, PC1_ratio, PC2_ratio



pca_total = makepcadf(behavior)

pca_T3_all = makepcadf(behavior[behavior.trial=='T3'])
pca_T3_std = makepcadf(behavior[(behavior.trial=='T3') & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_T3_cft = makepcadf(behavior[(behavior.trial=='T3') & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])

pca_Rt_all = makepcadf(behavior[behavior.trial=='Retest'])
pca_Rt_std = makepcadf(behavior[(behavior.trial=='Retest') & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_Rt_cft = makepcadf(behavior[(behavior.trial=='Retest') & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])

pca_T3Rt_all = makepcadf(behavior[behavior.trial.isin(['T3','Retest'])])
pca_T3Rt_std = makepcadf(behavior[(behavior.trial.isin(['T3','Retest'])) & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_T3Rt_cft = makepcadf(behavior[(behavior.trial.isin(['T3','Retest'])) & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])

pca_T6_all = makepcadf(behavior[behavior.trial=='T6_C3'])
pca_T6_std = makepcadf(behavior[(behavior.trial=='T6_C3') & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_T6_cft = makepcadf(behavior[(behavior.trial=='T6_C3') & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])

pca_Rn_all = makepcadf(behavior[behavior.trial=='Retention'])
pca_Rn_std = makepcadf(behavior[(behavior.trial=='Retention') & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_Rn_cft = makepcadf(behavior[(behavior.trial=='Retention') & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])

pca_T6Rn_all = makepcadf(behavior[behavior.trial.isin(['T6_C3','Retention'])])
pca_T6Rn_std = makepcadf(behavior[(behavior.trial.isin(['T6_C3','Retention'])) & (behavior.treatment.isin(['standard.yoked','standard.trained']))])
pca_T6Rn_cft = makepcadf(behavior[(behavior.trial.isin(['T6_C3','Retention'])) & (behavior.treatment.isin(['conflict.yoked','conflict.trained']))])


this_pca_list = [pca_T3_all ,pca_T3_std ,pca_T3_cft ,pca_Rt_all ,pca_Rt_std ,pca_Rt_cft ,pca_T3Rt_all ,pca_T3Rt_std ,pca_T3Rt_cft ,pca_T6_all ,pca_T6_std ,pca_T6_cft ,pca_Rn_all ,pca_Rn_std ,pca_Rn_cft ,pca_T6Rn_all ,pca_T6Rn_std ,pca_T6Rn_cft ]
title_list = ["T3.all","T3.std","T3.cft","Rt.all","Rt.std","Rt.cft","T3Rt.all","T3Rt.std","T3Rt.cft","T6.all","T6.std","T6.cft","Rn.all","Rn.std","Rn.cft","T6Rn.all","T6Rn.std","T6Rn.cft"]
index_num = sum([[b+6*a for a in range(3)] for b in range(6)], [])



# original total used pc
fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

my_pca, pc1_perc, pc2_perc = pca_total

sns.scatterplot(x='PC1', y='PC2', data=my_pca, ax = axes,
                hue = 'treatment', palette = treatmentcolors,
                size = 30, legend = False)
axes.set_ylabel('PC2 : {}%'.format(round(pc2_perc*100, 1)))
axes.set_xlabel('PC1 : {}%'.format(round(pc1_perc*100, 1)))
axes.set_xticklabels('')
axes.set_yticklabels('')

plt.tight_layout()
plt.savefig(plotpath+'01.PC_original.png', dpi = 300)
plt.show()



# get total PC figures
fig1, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))
axes = axes.flatten()

for i in range(18):
    my_pca, pc1_perc, pc2_perc = this_pca_list[i]
    my_title = title_list[i]
    index = index_num[i]
    sns.scatterplot(x='PC1', y='PC2', data=my_pca, ax = axes[index],
                    hue = 'treatment', palette = treatmentcolors,
                    size = 30, legend = False)
    axes[index].set_ylabel('PC2 : {}%'.format(round(pc2_perc*100, 1)))
    axes[index].set_xlabel('PC1 : {}%'.format(round(pc1_perc*100, 1)))
    axes[index].set_title(my_title, loc = 'left')
    axes[index].set_xticklabels('')
    axes[index].set_yticklabels('')

plt.tight_layout()
plt.savefig(plotpath+'01.PC_series.png', dpi = 300)
plt.show()






#####  Use the best one which covers the most #####

pca_Rn_behav = behavior[behavior.trial=='Retention']
Z = pca_Rn_behav[behav_columns]
Z = Z.loc[:, Z.var(skipna=True) != 0]
Z_scaled = (Z - Z.mean()) / Z.std()
pca = PCA(n_components=2)
principal_components = pca.fit_transform(Z_scaled)
PC1_ratio = pca.explained_variance_ratio_[0]
PC2_ratio = pca.explained_variance_ratio_[1]
pc_df = pd.DataFrame(principal_components)
pc_df.columns = ['PC1','PC2']
pc_df['ID'] = list(pca_Rn_behav['ID'])
pc_df['treatment'] = list(pca_Rn_behav['treatment'])
pc_df['training'] = list(pca_Rn_behav['training'])
pc_df['trialNum'] = list(pca_Rn_behav['trialNum'])
pc_df['Day'] = list(pca_Rn_behav['Day'])

pc1_res = pca.components_[0]
pc2_res = pca.components_[1]
contribution_percentage_pc1 = (pc1_res**2) / np.sum(pc1_res**2) * 100
contribution_percentage_pc2 = (pc2_res**2) / np.sum(pc2_res**2) * 100

PC1_feature_perc = pd.DataFrame({
    'behav' : list(Z.columns), 
    'percent' : contribution_percentage_pc1 })

PC1_feature_perc = PC1_feature_perc.sort_values('percent', ascending = False)


PC2_feature_perc = pd.DataFrame({
    'behav' : list(Z.columns), 
    'percent' : contribution_percentage_pc2 })

PC2_feature_perc = PC2_feature_perc.sort_values('percent', ascending = False)

# check t-test within PC 
PC1_t = stats.ttest_ind(pc_df[pc_df.training=='trained']['PC1'] , pc_df[pc_df.training=='yoked']['PC1'])
PC2_t = stats.ttest_ind(pc_df[pc_df.training=='trained']['PC2'] , pc_df[pc_df.training=='yoked']['PC2'])
print(np.round(PC1_t, 4))
print(np.round(PC2_t, 4))



# make figure

def plot_behavior_feature(ax, titi,  varname, namestr, ylims):
    df = (behavior.groupby(['treatment', 'trial', 'trialNum'])
           .agg(
               m=(varname, 'mean'),
               se=(varname, lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # 표준 오차
           )
           .reset_index())
    df['measure'] = varname
    #print(dfa.describe())
    for treatment, group in df.groupby('treatment'):
        color = treatmentcolors[treatment]  # 색상 가져오기
        ax.errorbar(group['trialNum'], group['m'], yerr=group['se'], fmt='o', 
                    label=treatment, capsize=3, markersize=4, color=color)  # color 매개변수 추가
        sns.lineplot(data=group, x='trialNum', y='m', ax=ax, color=color, legend=None)
        sns.scatterplot(data=group, x='trialNum', y='m', ax=ax, s=30, color=color, legend=None)
    #
    ax.set_ylabel(namestr)
    ax.set_xlabel("trial")
    ax.set_xticks(list(range(1,10)))
    ax.set_xticklabels(["P", "T1", "T2", "T3", "Rt", "T4", "T5", "T6", "Rn"])
    ax.set_ylim(ylims)
    ax.set_title(titi, fontsize= 14, loc = 'left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #return ax, df



fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))

plot_behavior_feature(axes[0][0], 'a', 'NumEntrances', 'NumEntrances' ,  [0, 35])
plot_behavior_feature(axes[0][1], 'b','Time1stEntr', 'Time1stEntr (min)' ,  [0, 400])
plot_behavior_feature(axes[0][2], 'c','pTimeShockZone', 'pTimeShockZone' ,  [0, 0.35])

pc_df['treat_col'] = [treatmentcolors[a] for a in list(pc_df.treatment)]
sns.scatterplot(
    x='PC1', y='PC2', data=pc_df,
    ax=axes[1][0], hue='treatment', 
    palette=treatmentcolors,     
    s=100 , alpha = 0.5, legend = False )

axes[1][0].set_ylabel('PC2 : {}% variance explained'.format(round(PC2_ratio*100, 1)), fontsize= 10)
axes[1][0].set_xlabel('PC1 : {}% variance explained'.format(round(PC1_ratio*100, 1)), fontsize= 10)
axes[1][0].set_xticklabels('')
axes[1][0].set_yticklabels('')
axes[1][0].set_title('d', fontsize = 14, loc = 'left')


sns.barplot(
    y = 'percent', x = 'behav', 
    data = PC1_feature_perc.iloc[0:8],
    ax = axes[1][1]
    )

axes[1][1].set_xlabel('Estimates of memory', fontsize= 10)
axes[1][1].set_ylabel('PC1 % contrib.', fontsize= 10)
axes[1][1].set_xticks(list(range(8)))
axes[1][1].set_yticks([0,2,4,6,8])
axes[1][1].set_xticklabels(axes[1][1].get_xticklabels(), ha='right', fontsize = 8, rotation = 45)
axes[1][1].set_yticklabels([0,2,4,6,8],fontsize = 8)
axes[1][1].set_title('e', fontsize = 14, loc = 'left')

for spine in axes[1][1].spines.values():
    spine.set_visible(False)


sns.barplot(
    y = 'percent', x = 'behav', 
    data = PC2_feature_perc.iloc[0:8],
    ax = axes[1][2]
    )

axes[1][2].set_xlabel('Estimates of activity' , fontsize= 10)
axes[1][2].set_ylabel('PC2 % contrib.', fontsize= 10)
axes[1][2].set_xticks(list(range(8)))
axes[1][2].set_yticks([0,5,10,15,20])
axes[1][2].set_xticklabels(axes[1][2].get_xticklabels(), ha='right', fontsize = 8, rotation = 45)
axes[1][2].set_yticklabels([0,5,10,15,20],fontsize = 8)
axes[1][2].set_title('f', fontsize = 14, loc = 'left')

for spine in axes[1][2].spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(plotpath + '01.behav_bar.pdf', dpi = 300)
plt.savefig(plotpath + '01.behav_bar.png', dpi = 300)
plt.close()    






# now all the stats for supplementary materials
#-----------------

# Maybe it should be done like this: 
# Q1: are the groups different? 1-way ANOVA of groups on Pre 
# Q2: are the groups different during initial training T1-T3? 2-way ANOVA of groups X trial 
# Q3: do the groups differ in initial recall? 1-way ANOVA of groups on Rt 
# Q4: Do the groups differ in subsequent training? T4-T6 2-way ANOVA of groups X trial 
# Q5: do the groups differ in subsequent recall? 1-way ANOVA of groups on Rn


import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

# 특정 trial phase 선택

be_for_ano = copy.deepcopy(behavior)

def run_anova(be_for_ano, trial_filter, formulas):
    results = []
    trial_phase = be_for_ano[be_for_ano['trial'].isin(trial_filter)]
    trial_phase['treatment'] = pd.Categorical(trial_phase['treatment'])
    trial_phase['trial'] = pd.Categorical(trial_phase['trial'], categories = trial_filter)
    for name, formula in formulas.items():
        model = ols(formula, data=trial_phase).fit()
        anova_result = sm.stats.anova_lm(model, typ=2)
        anova_result['name'] = name
        results.append(anova_result)
    combined_result = pd.concat(results)
    combined_result['trial'] = '_'.join(trial_filter)
    return combined_result.drop('Residual', axis=0)

# Define the models and phases
formulas_single = {
    'NumEntrances ~ treatment': 'NumEntrances ~ C(treatment)',
    'pTimeShockZone ~ treatment': 'pTimeShockZone ~ C(treatment)',
    'Time1stEntr ~ treatment': 'Time1stEntr ~ C(treatment)'
}

formulas_double = {
    'NumEntrances': 'NumEntrances ~ C(trial) + C(treatment) + C(trial):C(treatment)',
    'pTimeShockZone': 'pTimeShockZone ~ C(trial) + C(treatment) + C(trial):C(treatment)',
    'Time1stEntr': 'Time1stEntr ~ C(trial) + C(treatment) + C(trial):C(treatment)'
}

phases_single = ['Pre', 'Retention', 'Retest']
phase_groups = [['T1', 'T2', 'T3'], ['T4_C1', 'T5_C2', 'T6_C3']]

# Run single ANOVA
results_single = []
for phase in phases_single:
    result = run_anova(be_for_ano, [phase], formulas_single)
    results_single.append(result.loc['C(treatment)'])

table_1 = pd.concat(results_single)
table_1['trial'] = ['Pre-training'] * 3 + ['Retention'] * 3 + ['Retest'] * 3

# Run two-way ANOVA
table_2 = pd.concat([run_anova(be_for_ano, phase, formulas_double) for phase in phase_groups])
table_2['trial'] = ['Initial Training (T1~T3)'] * 9 + ['Conflict Training (T4~T6)'] * 9

# Format and annotate tables
def format_anova_table(table):
    table['p'] = round(table['PR(>F)'], 3)
    table['F'] = round(table['F'], 2)
    table['df'] = table['df'].astype(int)
    table['sig'] = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' ' for p in table['p']]
    return table


table_1 = format_anova_table(table_1)[['trial', 'name', 'df', 'F', 'p', 'sig']]
table_2['name2'] = ['trial', 'treatment', 'trial*treatment'] * 6
table_2['name'] = table_2['name'] + ' ~ ' + table_2['name2']
table_2 = format_anova_table(table_2)[['trial', 'name', 'df', 'F', 'p', 'sig']]

# Merge tables
table_merge = pd.concat([table_1, table_2]).reset_index(drop=True)


# Run ANOVA for all PCs
def run_anova_pc(be_for_ano, formulas):
    results = []
    tmp_ano = copy.deepcopy(be_for_ano)
    tmp_ano['treatment'] = pd.Categorical(tmp_ano['treatment'])
    for name, formula in formulas.items():
        model = ols(formula, data=tmp_ano).fit()
        anova_result = sm.stats.anova_lm(model, typ=2) 
        # typ1 : order matters, typ2 : take only Main effect, typ3 : also check interaction among conditions
        anova_result['name'] = name
        results.append(anova_result)
    combined_result = pd.concat(results)
    print(combined_result)
    return combined_result.drop('Residual', axis=0)

pc_formulas = {'PC1 ~ treatment': 'PC1 ~ C(treatment)', 'PC2 ~ treatment': 'PC2 ~ C(treatment)'}
ori_pcs = run_anova_pc(pca_total[0], pc_formulas).loc['C(treatment)']
new_pcs = run_anova_pc(pca_Rn_all[0], pc_formulas).loc['C(treatment)']

table_pcs = pd.concat([ori_pcs, new_pcs])
table_pcs['trial'] = ['All trials', 'All trials', 'Retention only', 'Retention only']
table_pcs['name'] = ['PC1 ~ treatment', 'PC2 ~ treatment'] * 2
table_pcs = format_anova_table(table_pcs)

# Combine all tables for final output
table_merge2 = pd.concat([table_merge, table_pcs[['trial', 'name', 'df', 'F', 'p', 'sig']]]).reset_index(drop=True)
desired_order = ['Pre-training', 'Initial Training (T1~T3)', 'Retest', 'Conflict Training (T4~T6)', 'Retention', 'All trials', 'Retention only']
table_merge2['trial'] = pd.Categorical(table_merge2['trial'], categories=desired_order, ordered=True)
table_merge3 = table_merge2.sort_values(['trial', 'name']).reset_index(drop=True)


table_merge3.to_csv(datapath+'01.behav_anova.csv')


# tucky test for each phase 

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

trial_phase = copy.deepcopy(be_for_ano)
trial_phase['treatment2'] = list(trial_phase["treatment"])
trial_phase['trial_phase_trial'] = trial_phase["treatment2"]+'_'+trial_phase["trial"]
trial_phase['treatment'] = pd.Categorical(trial_phase['treatment'])
trial_phase['trial'] = pd.Categorical(trial_phase['trial'], categories = ['Pre', 'T1', 'T2', 'T3', 'Retest', 'T4_C1', 'T5_C2', 'T6_C3', 'Retention'])


phases = ['Pre', 'T1', 'T2', 'T3', 'Retest', 'T4_C1', 'T5_C2', 'T6_C3', 'Retention']

def tucky_df (trial_phase_filt, for_var, phph) : 
    tukey_result_v1 = pairwise_tukeyhsd(
        endog=trial_phase_filt[for_var],       # 종속 변수
        groups=trial_phase_filt["treatment"],           # 그룹 변수
        alpha=0.05                        # 유의수준
    )
    tukey_result_v1_re = pd.DataFrame(data=tukey_result_v1.summary().data[1:],  # 첫 행은 header이므로 [1:]부터
                            columns=tukey_result_v1.summary().data[0]) 
    #
    tukey_result_v1_re['group1_tr'] = tukey_result_v1_re.group1.apply(lambda x : x.split('.')[1])
    tukey_result_v1_re['group2_tr'] = tukey_result_v1_re.group2.apply(lambda x : x.split('.')[1])
    tukey_result_v1_re = tukey_result_v1_re[tukey_result_v1_re.group1_tr == tukey_result_v1_re.group2_tr]
    tukey_result_v1_re['name'] = for_var
    tukey_result_v1_re['phase'] = phph
    return(tukey_result_v1_re)


tucky_res_list = []
for phph in phases : 
    trial_phase_filt = trial_phase[trial_phase.trial == phph]
    trial_tucky_re1 = tucky_df(trial_phase_filt, 'NumEntrances', phph)
    trial_tucky_re2 = tucky_df(trial_phase_filt, 'pTimeShockZone', phph)
    trial_tucky_re3 = tucky_df(trial_phase_filt, 'Time1stEntr', phph)
    tucky_res_list.append(trial_tucky_re1)
    tucky_res_list.append(trial_tucky_re2)
    tucky_res_list.append(trial_tucky_re3)


tucky_res_df = pd.concat(tucky_res_list)
tucky_res_df = tucky_res_df[['phase','name','group1','group2','meandiff','p-adj','reject' ]]
tucky_res_df = tucky_res_df.reset_index(drop = True)
tucky_res_df['meandiff'] = np.round(tucky_res_df['meandiff'], 2)
tucky_res_df['p-adj'] = np.round(tucky_res_df['p-adj'], 4)
tucky_res_df['reject'] = ['*' if p < 0.05 else '' for p in list(tucky_res_df['p-adj']) ]
tucky_res_df.columns = ['trial', 'name', 'group1', 'group2', 'meandiff', 'p-adj', 'sig.']
phase_name = ['Pre-training', 'Initial Training 1','Initial Training 2','Initial Training 3', 'Retest', 'Conflict Training 4','Conflict Training 5','Conflict Training 6', 'Retention']
tucky_res_df['trial'] = [phase for phase in phase_name for _ in range(6)]

tucky_res_df.to_csv(datapath+'01.behav_tucky.csv')


# If want to check by adding phase to group name
tukey_result_1 = pairwise_tukeyhsd(
    endog=trial_phase["NumEntrances"],       
    groups=trial_phase["trial_phase_trial"],        
    alpha=0.05                       
)

tukey_result_2 = pairwise_tukeyhsd(
    endog=trial_phase["pTimeShockZone"],   
    groups=trial_phase["trial_phase_trial"],
    alpha=0.05                
)

tukey_result_3 = pairwise_tukeyhsd(
    endog=trial_phase["Time1stEntr"],   
    groups=trial_phase["trial_phase_trial"],  
    alpha=0.05   
)



# just to save and fix the PC1

NEW_PC1 = pca_Rn_all[0]
NEW_PC1 = NEW_PC1.sort_values('PC1')[['ID','treatment','trialNum','PC1','PC2']]
NEW_PC1 = NEW_PC1.reset_index(drop = True)

NEW_PC1.to_csv(datapath+'00.NEW_PC1.csv')






