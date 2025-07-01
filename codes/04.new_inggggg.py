trained only 


g_g_df = pd.read_csv(datapath + '04.all_relationship.csv', index_col = 0)

selected_gene = g_g_df[g_g_df.Z_pv<=0.05] # 702

target_genes = list(np.unique(list(selected_gene.gene) + candidategenes)) # 714
len(target_genes)
from itertools import combinations
gg_combi_list = list(combinations(target_genes, 2))
# 254541







testname = 'test3'

results_pair_yoked = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel2)(DG_yoked, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_yoked_df = pd.DataFrame(results_pair_yoked)
results_pair_yoked_df.to_csv(datapath + '04.all_relationship_GG_YOKED_' + testname + '.csv')


results_pair_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel2)(DG_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_trained_df = pd.DataFrame(results_pair_trained)
results_pair_trained_df.to_csv(datapath + '04.all_relationship_GG_TRAINED_' + testname + '.csv')





testname = 'test4'

results_pair_yoked = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel3)(DG_yoked, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_yoked_df = pd.DataFrame(results_pair_yoked)
results_pair_yoked_df.to_csv(datapath + '04.all_relationship_GG_YOKED_' + testname + '.csv')

results_pair_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel3)(DG_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_trained_df = pd.DataFrame(results_pair_trained)
results_pair_trained_df.to_csv(datapath + '04.all_relationship_GG_TRAINED_' + testname + '.csv')





testname = 'test5'

results_pair_yoked = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel4)(DG_yoked, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_yoked_df = pd.DataFrame(results_pair_yoked)
results_pair_yoked_df.to_csv(datapath + '04.all_relationship_GG_YOKED_' + testname + '.csv')


results_pair_trained = Parallel(n_jobs=10, backend="loky")(
    delayed(XI_pair_sub_parallel4)(DG_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
)
results_pair_trained_df = pd.DataFrame(results_pair_trained)
results_pair_trained_df.to_csv(datapath + '04.all_relationship_GG_TRAINED_' + testname + '.csv')










n_iter = 1000
#n_iter = 10
n_jobs = min(8, os.cpu_count())




datasets_1 = [
    #('04.all_relationship_GG_YOKED_' + 'test1' + '.csv', '04.Louvain_1_iter1000_Yoked_' + 'test1' + '.csv'),
    #('04.all_relationship_GG_TRAINED_' + 'test1' + '.csv', '04.Louvain_1_iter1000_Trained_' + 'test1' + '.csv'),
    # ('04.all_relationship_GG_YOKED_' + 'test2' + '.csv', '04.Louvain_1_iter1000_Yoked_' + 'test2' + '.csv'),
    # ('04.all_relationship_GG_TRAINED_' + 'test2' + '.csv', '04.Louvain_1_iter1000_Trained_' + 'test2' + '.csv'),
    # ('04.all_relationship_GG_YOKED_' + 'test3' + '.csv', '04.Louvain_1_iter1000_Yoked_' + 'test3' + '.csv'),
    # ('04.all_relationship_GG_TRAINED_' + 'test3' + '.csv', '04.Louvain_1_iter1000_Trained_' + 'test3' + '.csv'),
    # ('04.all_relationship_GG_YOKED_' + 'test4' + '.csv', '04.Louvain_1_iter1000_Yoked_' + 'test4' + '.csv'),
    # ('04.all_relationship_GG_TRAINED_' + 'test4' + '.csv', '04.Louvain_1_iter1000_Trained_' + 'test4' + '.csv'),
    # ('04.all_relationship_GG_YOKED_' + 'test5' + '.csv', '04.Louvain_1_iter1000_Yoked_' + 'test5' + '.csv'),
    # ('04.all_relationship_GG_TRAINED_' + 'test5' + '.csv', '04.Louvain_1_iter1000_Trained_' + 'test5' + '.csv'),

]


for data in datasets_1 :
    print(data[0])
    data_read = pd.read_csv(datapath + data[0], index_col = 0)
    out_filename = data[1]
    data_read = data_read.fillna(0)
    graph_process(data_read, out_filename, n_iter, n_jobs)



그러고 Suppl material 정리하면 될듯 
한스한테 그 그림을 내일까지 보내버리는걸로 하자 
내일 아침에는 강연이 있고 
내일 오후에는 3시에 시방임 
혹시 내일 교수님 일정 어떤지 확인해두기 




from sklearn.manifold import TSNE


for testname in ['test1', 'test2', 'test3', 'test4', 'test5'] : 
    clust_all = pd.read_csv(datapath + '04.Louvain_1_iter1000.csv', index_col = 0)
    clust_Yoked = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked_' + testname + '.csv', index_col = 0)
    clust_Trained = pd.read_csv(datapath + '04.Louvain_1_iter1000_Trained_' + testname + '.csv', index_col = 0)
    #
    genes = list(clust_all.index)
    for i in genes :
        clust_all.at[i,i] = 1
        clust_Yoked.at[i,i] = 1
        clust_Trained.at[i,i] = 1
    #
    pca_all = PCA(n_components=3) # 
    scores_all = pca_all.fit_transform(clust_all) 
    scores_YOKED  = pca_all.transform(clust_Yoked)
    scores_TRAINED  = pca_all.transform(clust_Trained)
    #
    # get Δv
    delta = scores_TRAINED - scores_YOKED  # shape = (714, 3)
    norms = np.linalg.norm(delta, axis=1) # euclidean norm. 
    top_norms = pd.Series(norms, index=genes)
    #
    centroid_yoked   = scores_YOKED.mean(axis=0)
    centroid_trained = scores_TRAINED.mean(axis=0)
    u = centroid_trained - centroid_yoked # direction 
    u = u / np.linalg.norm(u)  # normlizing 
    contrib = delta.dot(u)   # train-yoked difference dot! 
    top_contrib = pd.Series(contrib, index=genes)
    cos_align = (delta * u).sum(axis=1) / np.linalg.norm(delta, axis=1)
    cos_align = pd.Series(cos_align, index = genes)
    dotprod = (scores_YOKED * scores_TRAINED).sum(axis=1)
    top_dot   = pd.Series(dotprod, index=genes)
    norm_Y = np.linalg.norm(scores_YOKED, axis=1)
    norm_T = np.linalg.norm(scores_TRAINED, axis=1)
    cos_sim = dotprod / (norm_Y * norm_T)
    cos_sim = pd.Series(cos_sim, index = genes)
    top_all = pd.concat([top_norms, top_contrib, top_dot, cos_align, cos_sim], axis = 1)
    top_all.columns = ['norm','centroid','PCdot', 'cent_sim', 'gene_sim']
    top_all['candy'] = ['O' if a in candidategenes else 'X' for a in list(top_all.index)]
    t_X = top_all[['norm', 'cent_sim', 'gene_sim']]
    t_y = top_all['candy']
    genes = list(top_all.index)
    tsne_plot(5, 50, t_X)




clust_Yoked5 = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked_' + 'test5' + '.csv', index_col = 0)
clust_Yoked3 = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked_' + 'test3' + '.csv', index_col = 0)

pal_diff = clust_Yoked3-clust_Yoked5
pal_diff['gene'] = pal_diff.index
pal_diff.melt(id_vars=['gene'], var_name='test', value_name='value').sort_values(by='value', ascending=False)

502811    Cpeb4  Zfp804b  0.993
503369   Zscan2  Zfp804b  0.993
509786  Zfp804b   Zscan2  0.993
111374  Zfp804b    Cpeb4  0.993

clust_Yoked3.loc['Cpeb4','Zfp804b']
clust_Yoked5.loc['Cpeb4','Zfp804b']


rel_Yoked5 = pd.read_csv(datapath + '04.all_relationship_GG_YOKED_' + 'test5' + '.csv', index_col = 0)
rel_Yoked3 = pd.read_csv(datapath + '04.all_relationship_GG_YOKED_' + 'test3' + '.csv', index_col = 0)


rel_Yoked3[(rel_Yoked3.geneA=='Cpeb4') & (rel_Yoked3.geneB=='Zfp804b')]
             99128
geneA        Cpeb4
geneB      Zfp804b
PCOR           0.0
P_pv           1.0
SCOR           0.0
S_pv           1.0
XI_ori         NaN
XI_new         NaN
XI_pv          NaN
SIGMA_ori      NaN
SIGMA          NaN
Z_pv           NaN
SCOR2          0.0
MAX            NaN

rel_Yoked5[(rel_Yoked5.geneA=='Cpeb4') & (rel_Yoked5.geneB=='Zfp804b')]

             99128
geneA        Cpeb4
geneB      Zfp804b
PCOR           NaN
P_pv           1.0
SCOR           NaN
S_pv           1.0
XI_ori         NaN
XI_new         NaN
XI_pv          NaN
SIGMA_ori      NaN
SIGMA          0.0
Z_pv           1.0
PCOR2          0.0
SCOR2          0.0
MAX            0.0



rel_Yoked3[np.round(rel_Yoked3['SIGMA'], 3) != np.round(rel_Yoked5['SIGMA'], 3)]

rel_Yoked3[np.round(rel_Yoked3['SIGMA'].fillna(0), 3) != np.round(rel_Yoked5['SIGMA'].fillna(0), 3)]
# 254497




결국 틀린거 없음. 
앙드레만 나를 의심하게 생김 
ㅋㅋㅋㅋㅋㅋㅋ 어휴 몰라 
실수했다고 하고 그냥 모든 코드를 점검했다고 하지 뭐 








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
    plt.show()







#######

# # No Train only 
# non_trained = RNA_DG[RNA_DG.PC1 < 0]


# results_pair_non_trained = Parallel(n_jobs=10, backend="loky")(
#     delayed(XI_pair_sub_parallel)(non_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
# )

# results_pair_non_trained_df = pd.DataFrame(results_pair_non_trained)

# results_pair_non_trained_df.to_csv(datapath + '04.all_relationship_GG_NON_TRAINED.csv')





# # Yes Train only 
# yes_trained = RNA_DG[RNA_DG.PC1 > 0]

# results_pair_yes_trained = Parallel(n_jobs=10, backend="loky")(
#     delayed(XI_pair_sub_parallel)(yes_trained, geneA, geneB) for geneA, geneB in tqdm(gg_combi_list) 
# )

# results_pair_yes_trained_df = pd.DataFrame(results_pair_yes_trained)

# results_pair_yes_trained_df.to_csv(datapath + '04.all_relationship_GG_YES_TRAINED.csv')





original_yo = pd.read_csv(datapath+'04.Louvain_1_iter1000_Yoked_original.csv', index_col = 0)
this_yo = pd.read_csv(datapath+'04.Louvain_1_iter1000_Yoked.csv', index_col = 0)
another_yo = pd.read_csv(datapath+'04.Louvain_1_iter1000_Yoked2.csv', index_col = 0)
another_yo = pd.read_csv(datapath+'04.Louvain_1_iter1000_Yoked2.csv', index_col = 0)
this_yo = this_yo.where(~np.eye(this_yo.shape[0],dtype=bool), 1)
another_yo = another_yo.where(~np.eye(another_yo.shape[0],dtype=bool), 1)

diff = original_yo - this_yo
np.max(np.array(original_yo - this_yo))

    # NO only 
    #('04.all_relationship_GG_NON_TRAINED.csv', '04.Louvain_1_iter1000_NON_trained.csv'),
    # YES only 
    #('04.all_relationship_GG_YES_TRAINED.csv', '04.Louvain_1_iter1000_YES_trained.csv'),



# total merged 
# 1000 iter : [(3, 64), (4, 922), (5, 14)]
# re with SIGMA 0  [(3, 64), (4, 922), (5, 14)] # great..? but didn't have the diff from the first time 
# one more for xi change  # [(3, 64), (4, 922), (5, 14)]
# 그래 이게 바뀔 일은 없음. 714 가 바뀌는 것도 아님.
# 마지막 부분이 항상 문제인데 왜냐면 거기는 0 이 많으니까 
# xi change only : [(3, 64), (4, 922), (5, 14)]

# yoked only 
# 1000 iter : [(9, 231), (10, 717), (11, 52)] 
# re with SIGMA 0 [(4, 179), (5, 732), (6, 89)]... # 어쩌냐 
# one more for xi change [(4, 194), (5, 742), (6, 64)]
# 오 별 차이가 없어 ?
# xi change only : [(4, 194), (5, 742), (6, 64)]



# trained only 
# 1000 iter : [(32, 72), (30, 18), (31, 910)] # something happening 
# re [(3, 988), (4, 12)]
# one more for xi change : [(2, 12), (3, 954), (4, 34)]
# 오? 
# xi change only [(2, 12), (3, 954), (4, 34)]





# NON-trained
# 1000 : [(4, 4), (5, 555), (6, 434), (7, 7)]
#re 

# YES-trained 
# 1000 : [(44, 61), (45, 620), (46, 319)] 
#re 



# sample scor rechange 를 얘기 해야하나 



# clust_all.to_csv(datapath + '04.Louvain_1_iter1000_original.csv')
# clust_Yoked.to_csv(datapath + '04.Louvain_1_iter1000_Yoked_original.csv')
# clust_Trained.to_csv(datapath + '04.Louvain_1_iter1000_Trained_original.csv')

clust_all_original = pd.read_csv(datapath + '04.Louvain_1_iter1000_original.csv', index_col = 0)
clust_Yoked_original = pd.read_csv(datapath + '04.Louvain_1_iter1000_Yoked_original.csv', index_col = 0)
clust_Trained_original = pd.read_csv(datapath + '04.Louvain_1_iter1000_Trained_original.csv', index_col = 0)

#clust_NO = pd.read_csv(datapath + '04.Louvain_1_iter1000_NON_trained.csv', index_col = 0)
#clust_YES = pd.read_csv(datapath + '04.Louvain_1_iter1000_YES_trained.csv', index_col = 0)

# diff_check = clust_Yoked - clust_Yoked_re
# long_diff = (
#     diff_check
#     .stack()                                  # 멀티인덱스 Series (idx, col) → value
#     .reset_index(name='value')                # 인덱스를 컬럼으로 풀고, value 이름 지정
#     .rename(columns={'level_0':'index',       # 원래 인덱스 이름 바꾸기
#                      'level_1':'column'}))


#scores_NO  = pca_all.transform(clust_NO)
#scores_YES  = pca_all.transform(clust_YES)
#scores_No_df = pd.DataFrame(scores_NO, index = list(clust_NO.index), columns = ['R','G','B'])
#scores_Ys_df = pd.DataFrame(scores_YES, index = list(clust_YES.index), columns = ['R','G','B'])
#scores_No_df2 = get_RGB(scores_No_df)
#scores_Ys_df2 = get_RGB(scores_Ys_df)
#col_yes = list(scores_Ys_df2.hex_val)
#plot_3d_rgb(scores_No_df2, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])
#plot_3d_rgb(scores_Ys_df2, pca_all.explained_variance_ratio_[0], pca_all.explained_variance_ratio_[1], pca_all.explained_variance_ratio_[2])

save_transition_gif(np.array(scores_No_df2[['R','G','B']]), np.array(scores_Ys_df2[['R','G','B']]), col_yes, list(scores_No_df2.index), candidategenes,
                    output_path=plotpath+'04.PCall_noyes.gif')


#clust_NO = pd.read_csv(datapath + '04.Louvain_1_iter1000_NON_trained.csv', index_col = 0)
#clust_YES = pd.read_csv(datapath + '04.Louvain_1_iter1000_YES_trained.csv', index_col = 0)
    #'NO': clust_NO,
    #'YES': clust_YES,
#plot_clustermap(clust_NO, '04.clustermap_NO', candy_labels=candidategenes)
#plot_clustermap(clust_YES, '04.clustermap_YES', candy_labels=candidategenes)
