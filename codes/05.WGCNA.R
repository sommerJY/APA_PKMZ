
library(WGCNA)
library(ggplot2)

datapath = './data/'
plotpath = './figures/'

candidategenes = c("Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1","Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci")

DG_original = read.csv(file = paste0(datapath, '03.EXP_PC1_merge.DG.csv'), stringsAsFactors = FALSE, header = TRUE, check.names=FALSE)
gene_all = colnames(DG_original)[2:16468] # 16467
gene_rm = setdiff(gene_all, c("Prkcz", "Camk2a"))

data_all = DG_original[gene_all]
data_ext = DG_original[gene_rm]
sample_name = paste(DG_original$treatment, DG_original$RNAseqID, sep=".")

rownames(data_all) = sample_name
rownames(data_ext) = sample_name




traitData = DG_original[c('treatment','training','Prkcz','Camk2a','PC1')]
rownames(traitData) = sample_name

data_all_exp <- as.data.frame(data_all) # transforming the data.frame so columns now represent genes and rows represent samples
names(data_all_exp) <- colnames(data_all)
gsg <-goodSamplesGenes(data_all_exp)
summary(gsg)
gsg$allOK
#                            0610007P14Rik 0610009B22Rik 0610009L18Rik
# conflict.trained.143A-DG-1      5.608932      3.955310      2.275426
# conflict.yoked.143B-DG-1        5.641230      3.484059      3.194010
# standard.yoked.143D-DG-3        6.493728      1.619408      2.363165
# conflict.trained.144A-DG-2      5.882097      2.911210      0.000000

dim(data_all_exp)
#[1]    14 16467


data_ext_exp <- as.data.frame(data_ext) 
names(data_ext_exp) <- colnames(data_ext)
gsg <-goodSamplesGenes(data_ext_exp)
summary(gsg)
gsg$allOK
#                            0610007P14Rik 0610009B22Rik 0610009L18Rik
# conflict.trained.143A-DG-1      5.608932      3.955310      2.275426
# conflict.yoked.143B-DG-1        5.641230      3.484059      3.194010
# standard.yoked.143D-DG-3        6.493728      1.619408      2.363165
# conflict.trained.144A-DG-2      5.882097      2.911210      0.000000

dim(data_ext_exp)
# [1]    14 16465



traitData$treatment2 = as.numeric(
    sapply(
        traitData$treatment, switch, 
        'standard.trained'=1, 'standard.yoked'=2, 
        'conflict.trained'=3, 'conflict.yoked'=4 ))

traitData$training2 = as.numeric(
    sapply(
        traitData$training, switch, 
        'yoked'=1, 'trained'=2))

traitData$samples = rownames(traitData)

allTraits <- traitData[,c("Prkcz","Camk2a","treatment2","training2",'PC1')] 
colnames(allTraits) = c("Prkcz","Camk2a","4_treat","2_train",'PC1')

enableWGCNAThreads(nThreads = NULL)

########### 


# check threshold
# 이미 내 데이터가 있다는 것을 명시하면 된다고 함 
ST_all <- pickSoftThreshold(data_all_exp) 
ST_ext <- pickSoftThreshold(data_ext_exp) 



# 
scale_ind_plot = function(ST, name) {
    png(paste0(plotpath,"05.scale_independence_", name,".png"), width = 800, height = 600, res=300)  # PNG 파일 설정
    par(mar=c(0.2,1,0.4,0.2)) 
    par(oma=c(2,1,0,0))
    plot(ST$fitIndices[,1], ST$fitIndices[,2],
        xlab = "Soft Threshold (power)",
        ylab = "Scale Free Topology Model Fit, signed R^2",
        type = "n",
        main = paste0("Scale independence ",name),
        cex.lab=0.3, cex.axis=0.3, cex.main=0.3, cex.sub=0.3)
    text(ST$fitIndices[,1], ST$fitIndices[,2], col = "red", labels = ST$fitIndices[,1], cex = 0.8)
    abline(h = 0.80, col = "red")
    dev.off()  # 파일 저장 완료
}

scale_ind_plot(ST_all, 'ALL')
scale_ind_plot(ST_ext, 'EXT')


# 
mean_conn_plot = function(ST, name) {
    png(paste0(plotpath,"05.mean_connectivity_", name,".png"), width = 800, height = 600, res=300)  # PNG 파일 설정
    par(mar=c(0.2,1,0.4,0.2))  # 마진 조정 (기본값이 더 적절함)
    par(oma=c(2,1,0,0))
    plot(ST$fitIndices[,1], ST$fitIndices[,5],
        xlab = "Soft Threshold (power)",
        ylab = "Mean Connectivity",
        type = "n",
        main = paste0("Mean connectivity", name) ,
        cex.lab=0.3, cex.axis=0.3, cex.main=0.3, cex.sub=0.3)
    text(ST$fitIndices[,1], ST$fitIndices[,5], labels = ST$fitIndices[,1], col = "red", cex = 0.8)
    dev.off()  # 파일 저장 완료
}

mean_conn_plot(ST_all, 'ALL')
mean_conn_plot(ST_ext, 'EXT')




softPower <- 6



dendroCOL_plot = function(geneTree, ModuleColors, name){
    # Save as PNG
    png(paste0(plotpath,"05.dendroCOL_", name,".png"), width = 1200, height = 900, res = 300)
    plotDendroAndColors(
        geneTree, ModuleColors,"Module",
        dendroLabels = FALSE, hang = 0.03,
        addGuide = TRUE, guideHang = 0.05,
        main = paste0("Gene dendrogram and module colors ",'ALL'),
        cex.lab=0.3, cex.axis=0.3, cex.main=0.3, cex.sub=0.3)
    dev.off()
    # Save as PDF
    pdf(paste0(plotpath,"05.dendroCOL_", name,".pdf"), width = 12, height = 9)
    plotDendroAndColors(
        geneTree, ModuleColors,"Module",
        dendroLabels = FALSE, hang = 0.03,
        addGuide = TRUE, guideHang = 0.05,
        main = paste0("Gene dendrogram and module colors ",'ALL'),
        cex.lab=0.3, cex.axis=0.3, cex.main=0.3, cex.sub=0.3)
    dev.off()
}



moduleFact_plot_png = function(module.trait.correlation, textMatrix, allTraits, MEs, name) {
    png(paste0(plotpath,"05.Module_fact_", name,".png"), width = 1600, height = 1200, res = 300)
    par(mar=c(4,5,1,1))
    par(oma=c(2,2,1,1))
    labeledHeatmap(Matrix = module.trait.correlation,
    xLabels = names(allTraits),
    yLabels = gsub("^ME", "", names(MEs)),   # <--- ME 제거
    ySymbols = gsub("^ME", "", names(MEs)),
    colorLabels = FALSE,
    colors = blueWhiteRed(50),
    textMatrix = textMatrix,
    setStdMargins = FALSE,
    zlim = c(-1,1),
    cex.text = 0.8, cex.lab=0.8)
    dev.off()
}

moduleFact_plot_pdf = function(module.trait.correlation, textMatrix, allTraits, MEs, name) {
    pdf(paste0(plotpath,"05.Module_fact_", name,".pdf"), width = 6, height = 5)
    par(mar=c(4,5,1,1))
    par(oma=c(2,2,1,1))
    labeledHeatmap(
    Matrix = module.trait.correlation,
    xLabels = names(allTraits),
    yLabels = gsub("^ME", "", names(MEs)),   # <--- ME 제거
    ySymbols = gsub("^ME", "", names(MEs)),
    colorLabels = FALSE,
    colors = blueWhiteRed(50),
    textMatrix = textMatrix,
    setStdMargins = FALSE,
    zlim = c(-1,1),
    cex.text = 0.8, cex.lab=0.8)
    dev.off()
}




#########
# check our pair data

all_pair = read.csv(paste0(datapath,'04.all_relationship_GG.csv')) 
all_gene = unique(c(all_pair$geneA, all_pair$geneB))

# 714
n_genes = length(all_gene) ; print(length(all_gene))

data_filter = data_all[,all_gene]

data_filter_all = data_filter
data_filter_ext = data_filter[,!colnames(data_filter) %in% c('Prkcz', 'Camk2a')]

dim(data_filter_all)
# [1]  14 714
dim(data_filter_ext)
# [1]  14 712




# 1) Xicor matrix 
adjMatrix <- matrix(0, nrow = n_genes, ncol = n_genes, dimnames = list(all_gene, all_gene))

for (i in 1:nrow(all_pair)) {
    gene1 <- all_pair$geneA[i]
    gene2 <- all_pair$geneB[i]
    xiScore <- all_pair$XI_new[i]
    if (gene1 == gene2) {
        adjMatrix[gene1, gene2] <- 1  
    } else if (xiScore <0){
        adjMatrix[gene1, gene2] <- 0
        adjMatrix[gene2, gene1] <- 0  # make Symmetric matrix
    } else {
        adjMatrix[gene1, gene2] <- xiScore
        adjMatrix[gene2, gene1] <- xiScore  # Symmetric matrix        
    }
}

# I to 1 
for (i in 1:length(all_gene)) {
    gene1 <- all_gene[i]
    adjMatrix[gene1, gene1] = 1
}


# 2) SCC matrix 
SCC_matrix <- matrix(0, nrow = n_genes, ncol = n_genes, dimnames = list(all_gene, all_gene))

for (i in 1:nrow(all_pair)) {
    gene1 <- all_pair$geneA[i]
    gene2 <- all_pair$geneB[i]
    SCOR2_score <- all_pair$SCOR[i]
    if (gene1 == gene2) {
        SCC_matrix[gene1, gene2] <- 1  # 같은 gene인 경우 1을 넣음
    } else {
        SCC_matrix[gene1, gene2] <- SCOR2_score
        SCC_matrix[gene2, gene1] <- SCOR2_score  # Symmetric matrix        
    }
}

for (i in 1:length(all_gene)) {
    gene1 <- all_gene[i]
    SCC_matrix[gene1, gene1] = 1
}



# 3) In case of PCC matrix 
all_pair$PCOR2 <- abs(all_pair$PCOR)
PCC_matrix <- matrix(0, nrow = n_genes, ncol = n_genes, dimnames = list(all_gene, all_gene))

for (i in 1:nrow(all_pair)) {
    gene1 <- all_pair$geneA[i]
    gene2 <- all_pair$geneB[i]
    PCOR2_score <- all_pair$PCOR2[i]
    if (gene1 == gene2) {
        PCC_matrix[gene1, gene2] <- 1  # 같은 gene인 경우 1을 넣음
    } else {
        PCC_matrix[gene1, gene2] <- PCOR2_score
        PCC_matrix[gene2, gene1] <- PCOR2_score  # Symmetric matrix        
    }
}

for (i in 1:length(all_gene)) {
    gene1 <- all_gene[i]
    PCC_matrix[gene1, gene1] = 1
}


MAX_matrix <- pmax(adjMatrix, SCC_matrix)

SIGMA_matrix = sqrt(adjMatrix^2 + SCC_matrix^2) / sqrt(2)
# have to match max to 1 


# original version : start from adjacency 
ORI_version = function(name, data__exp, allTraits, softPower, deepSplit = 4, pval = 0.05, minClusterSize = 15) {
    # Calculate adjacency and TOM similarity
    adjacencys <- adjacency(data__exp, power = softPower)
    TOM_ORI <- TOMsimilarity(adjacencys)
    TOM_dissimilarity <- 1 - TOM_ORI
    # Hierarchical clustering
    geneTree <- hclust(as.dist(TOM_dissimilarity), method = "average")
    # Dynamic tree cut for module detection
    Modules <- cutreeDynamic(
        dendro = geneTree,
        distM = TOM_dissimilarity,
        deepSplit = deepSplit,
        pamRespectsDendro = FALSE,
        minClusterSize = minClusterSize
    )
    # Assign colors to modules
    ModuleColors <- labels2colors(Modules)
    dendroCOL_plot(geneTree, ModuleColors, name)
    # Calculate module eigengenes
    MElist <- moduleEigengenes(data__exp, colors = ModuleColors)
    MEs <- MElist$eigengenes
    ME_dissimilarity <- 1 - cor(MEs, use = "complete")
    METree <- hclust(as.dist(ME_dissimilarity), method = "average")
    # Create a dataframe for gene colors
    ORI_ColDF <- data.frame(
        gene = colnames(data__exp),
        cols = MElist$validColors
    )
    # Calculate module-trait correlations
    nSamples <- nrow(data__exp)
    module_trait_correlation <- cor(MEs, allTraits, use = "p")
    module_trait_Pvalue <- corPvalueStudent(module_trait_correlation, nSamples)
    # Mask correlations and p-values based on threshold
    module_trait_correlation_masked <- ifelse(abs(module_trait_Pvalue) >= pval, "", signif(module_trait_correlation, 2))
    module_trait_Pvalue_masked <- ifelse(abs(module_trait_Pvalue) >= pval, "", paste0('(', signif(module_trait_Pvalue, 2), ')'))
    textMatrix <- module_trait_correlation_masked
    # textMatrix <- paste(module_trait_correlation_masked, module_trait_Pvalue_masked, sep = "")
    dim(textMatrix) <- dim(module_trait_correlation_masked)
    # Plot module-trait relationships
    moduleFact_plot_png(module_trait_correlation, textMatrix, allTraits, MEs, name)
    moduleFact_plot_pdf(module_trait_correlation, textMatrix, allTraits, MEs, name)
    return(list(ORI_ColDF, MElist))
}




############## XIcor based version : start from similarity matrix 

XI_version = function(name, data_exp, this_adj, allTraits, deepSplit = 4, pval = 0.05, minClusterSize = 15) {
    # Calculate TOM similarity and dissimilarity
    TOM_MI <- TOMsimilarity(this_adj)
    TOM_dissimilarity <- 1 - TOM_MI
    # Hierarchical clustering
    geneTree <- hclust(as.dist(TOM_dissimilarity), method = "average")
    # Dynamic tree cut for module detection
    Modules <- cutreeDynamic(
        dendro = geneTree,
        distM = TOM_dissimilarity,
        deepSplit = deepSplit,
        pamRespectsDendro = FALSE,
        minClusterSize = minClusterSize
    )
    # Assign colors to modules
    ModuleColors <- labels2colors(Modules)
    dendroCOL_plot(geneTree, ModuleColors, name)
    # Calculate module eigengenes
    MElist <- moduleEigengenes(data_exp, colors = ModuleColors)
    MEs <- MElist$eigengenes
    ME_dissimilarity <- 1 - cor(MEs, use = "complete")
    METree <- hclust(as.dist(ME_dissimilarity), method = "average")
    # Create a dataframe for gene colors
    MI_ColDF <- data.frame(
        gene = colnames(data_exp),
        cols = MElist$validColors
    )
    # Calculate module-trait correlations
    nSamples <- nrow(data_exp)
    module_trait_correlation <- cor(MEs, allTraits, use = "p")
    module_trait_Pvalue <- corPvalueStudent(module_trait_correlation, nSamples)
    # Mask correlations and p-values based on threshold
    module_trait_correlation_masked <- ifelse(abs(module_trait_Pvalue) >= pval, "", signif(module_trait_correlation, 2))
    module_trait_Pvalue_masked <- ifelse(abs(module_trait_Pvalue) >= pval, "", paste0('(', signif(module_trait_Pvalue, 2), ')'))
    textMatrix <- module_trait_correlation_masked
    #textMatrix <- paste(module_trait_correlation_masked, module_trait_Pvalue_masked, sep = "")
    dim(textMatrix) <- dim(module_trait_correlation_masked)
    # Plot module-trait relationships
    moduleFact_plot_png(module_trait_correlation, textMatrix, allTraits, MEs, name)
    moduleFact_plot_pdf(module_trait_correlation, textMatrix, allTraits, MEs, name)
    return(list(MI_ColDF, MElist))
}


######################### Min cluster size 15 version 

power = 6


#1 Original version without prkcz and camk2a 
ori_EXT_power = ORI_version('ori_EXT_power_S15', data_filter_ext, allTraits, 6, deepSplit= 4, minClusterSize = 15)

#2 XI version 
adjMatrix_nopower_ext = adjMatrix[!rownames(adjMatrix) %in% c('Prkcz','Camk2a'),!colnames(adjMatrix) %in% c('Prkcz','Camk2a')]
adjMatrix_power_ext = adjMatrix_nopower_ext^power
XI_power_ext = XI_version('XI_power_ext_S15', data_filter_ext, adjMatrix_power_ext, allTraits, deepSplit = 4, minClusterSize = 15)

# 3  MAX based version 
MAX_nopower_ext = MAX_matrix[!rownames(MAX_matrix) %in% c('Prkcz','Camk2a'),!colnames(MAX_matrix) %in% c('Prkcz','Camk2a')]
MAX_power_ext = MAX_nopower_ext^power
MAX_power_ext = XI_version('MAX_power_ext_S15', data_filter_ext, MAX_power_ext, allTraits, deepSplit = 4, minClusterSize = 15)

# 4 SIGMA score version 
SIGMA_nopower_ext = SIGMA_matrix[!rownames(SIGMA_matrix) %in% c('Prkcz','Camk2a'),!colnames(SIGMA_matrix) %in% c('Prkcz','Camk2a')]
SIGMA_power_ext = SIGMA_nopower_ext^power
SIGMA_power_ext = XI_version('SIGMA_power_ext_S15', data_filter_ext, SIGMA_power_ext, allTraits, deepSplit = 4, minClusterSize = 15)




#1 Original version without prkcz and camk2a 
ori_EXT_power_30 = ORI_version('ori_EXT_power_S30', data_filter_ext, allTraits, 6, deepSplit= 4, minClusterSize = 30)

#2 XI version 
adjMatrix_nopower_ext_30 = adjMatrix[!rownames(adjMatrix) %in% c('Prkcz','Camk2a'),!colnames(adjMatrix) %in% c('Prkcz','Camk2a')]
adjMatrix_power_ext_30 = adjMatrix_nopower_ext_30^power
XI_power_ext_30 = XI_version('XI_power_ext_S30', data_filter_ext, adjMatrix_power_ext_30, allTraits, deepSplit = 4, minClusterSize = 30)

# 3  MAX based version 
MAX_nopower_ext = MAX_matrix[!rownames(MAX_matrix) %in% c('Prkcz','Camk2a'),!colnames(MAX_matrix) %in% c('Prkcz','Camk2a')]
MAX_power_ext_30 = MAX_nopower_ext^power
MAX_power_ext_30 = XI_version('MAX_power_ext_S30', data_filter_ext, MAX_power_ext_30, allTraits, deepSplit = 4, minClusterSize = 30)

# 4 SIGMA score version 
SIGMA_nopower_ext = SIGMA_matrix[!rownames(SIGMA_matrix) %in% c('Prkcz','Camk2a'),!colnames(SIGMA_matrix) %in% c('Prkcz','Camk2a')]
SIGMA_power_ext_30 = SIGMA_nopower_ext^power
SIGMA_power_ext_30 = XI_version('SIGMA_power_ext_S30', data_filter_ext, SIGMA_power_ext_30, allTraits, deepSplit = 4, minClusterSize = 30)



#1 Original version without prkcz and camk2a 
ori_EXT_power_D2S15 = ORI_version('ori_EXT_power_D2S15', data_filter_ext, allTraits, 6, deepSplit= 2, minClusterSize = 15)

#2 XI version 
adjMatrix_nopower_ext_D2S15 = adjMatrix[!rownames(adjMatrix) %in% c('Prkcz','Camk2a'),!colnames(adjMatrix) %in% c('Prkcz','Camk2a')]
adjMatrix_power_ext_D2S15 = adjMatrix_nopower_ext_D2S15^power
XI_power_ext_D2S15 = XI_version('XI_power_ext_D2S15', data_filter_ext, adjMatrix_power_ext_D2S15, allTraits, deepSplit = 2, minClusterSize = 15)

# 3  MAX based version 
MAX_nopower_ext_D2S15 = MAX_matrix[!rownames(MAX_matrix) %in% c('Prkcz','Camk2a'),!colnames(MAX_matrix) %in% c('Prkcz','Camk2a')]
MAX_power_ext_D2S15 = MAX_nopower_ext_D2S15^power
MAX_power_ext_D2S15 = XI_version('MAX_power_ext_D2S15', data_filter_ext, MAX_power_ext_D2S15, allTraits, deepSplit = 2, minClusterSize = 15)

# 4 SIGMA score version 
SIGMA_nopower_ext_D2S15 = SIGMA_matrix[!rownames(SIGMA_matrix) %in% c('Prkcz','Camk2a'),!colnames(SIGMA_matrix) %in% c('Prkcz','Camk2a')]
SIGMA_power_ext_D2S15 = SIGMA_nopower_ext_D2S15^power
SIGMA_power_ext_D2S15 = XI_version('SIGMA_power_ext_D2S15', data_filter_ext, SIGMA_power_ext_D2S15, allTraits, deepSplit = 2, minClusterSize = 15)




