
library(DESeq2) ## for gene expression analysis
library(png)
library(tidyverse) # for dataframe 
library(scales) # formation or the digits 
library(cowplot) ## for some easy to use themes
library(apaTables)
library(showtext)

library(BiocParallel)
register(MulticoreParam(8))

# heatmaps
library(RColorBrewer)
library(gplots)
library(ggplot2)


if (.Platform$OS.type == "windows") {
  font_add("Arial", regular = "C:/Windows/Fonts/arial.ttf")
} else {
  font_add("Arial", regular = "/Library/Fonts/Arial.ttf")
}

showtext_auto()

# Most codes from Rayna Harris github : https://github.com/raynamharris/IntegrativeProjectWT2015



datapath = './data/'
plotpath = './figures/'



# Prep col data,
outliers <- c("146D-DG-3", "145A-CA3-2", "146B-DG-2", "146D-CA1-3", "148B-CA1-4") # pre-defined outliers

colData <- read.csv(paste0(datapath, "/00_colData.csv"), header = T) %>%
    filter(!RNAseqID %in% outliers)
colData$training <- factor(colData$training, levels = c("yoked", "trained"))
colData$treatment <- factor(colData$treatment, levels = c('standard.yoked' , 'standard.trained' ,'conflict.yoked' ,  'conflict.trained' ))

# remove outliers
savecols <- as.character(colData$RNAseqID) #select the rowsname 
savecols <- as.vector(savecols) # make it a vector

countData <- read.csv(paste0(datapath, "00_countData.csv"), 
                        header = T, check.names = F, row.names = 1) %>%
    dplyr::select(one_of(savecols)) # select just the columns 
head(countData)

    ##               143A-CA3-1 143A-DG-1 143B-CA1-1 143B-DG-1 143C-CA1-1 143D-CA1-3
    ## 0610007P14Rik         85       112         60        48         38         28
    ## 0610009B22Rik         24        34         21        10         19          0
    ## 0610009L18Rik          4         9         10         8          2          0
    ## 0610009O20Rik         85       185         44        72         76         25
    ## 0610010F05Rik        142       155         54       117         57         39
    ## 0610010K14Rik         24        74         14        21         23         21
    #...
    ##               148A-DG-3 148B-CA3-4 148B-DG-4
    ## 0610007P14Rik       104        122        16
    ## 0610009B22Rik        15         45         2
    ## 0610009L18Rik        11         11         1
    ## 0610009O20Rik       226         70        18
    ## 0610010F05Rik       176        177        23
    ## 0610010K14Rik        17         39        10



# Get varience stabilized gene expression for each tissue

returnddstreatment <- function(mytissue){
  print(mytissue)
  colData <- colData %>% 
    filter(subfield %in% c(mytissue))  %>% 
    droplevels()
  
  savecols <- as.character(colData$RNAseqID) 
  savecols <- as.vector(savecols) 
  countData <- countData %>% dplyr::select(one_of(savecols)) 
  
  ## create DESeq object using the factors subfield and APA
  dds <- DESeqDataSetFromMatrix(countData = countData,
                                colData = colData,
                                design = ~ treatment)
  
  dds <- dds[ rowSums(counts(dds)) > 1, ]  # Pre-filtering genes with 0 counts
  dds <- DESeq(dds, parallel = TRUE)
  return(dds)
}



returnddstraining <- function(mytissue){
  print(mytissue)
  colData <- colData %>% 
    filter(subfield %in% c(mytissue))  %>% 
    droplevels()
  
  savecols <- as.character(colData$RNAseqID) 
  savecols <- as.vector(savecols) 
  countData <- countData %>% dplyr::select(one_of(savecols)) 
  
  ## create DESeq object using the factors subfield and APA
  dds <- DESeqDataSetFromMatrix(countData = countData,
                                colData = colData,
                                design = ~ training)
  
  dds <- dds[ rowSums(counts(dds)) > 1, ]  # Pre-filtering genes with 0 counts
  dds <- DESeq(dds, parallel = TRUE)
  return(dds)
}


# DEGs with looking at all four treatments individually
DGdds <- returnddstreatment("DG")  # DG ~ treatment DEG # 16992

CA3dds <- returnddstreatment("CA3") # CA3 ~ treatment DEG # 16481

CA1dds <- returnddstreatment("CA1") # CA1 ~ treatment DEG # 16840

# DEGs with looking at all grouped trained and yoked
DGdds2 <- returnddstraining("DG") 

CA3dds2 <- returnddstraining("CA3") 

CA1dds2 <- returnddstraining("CA1") 



# all 
alldds = returnddstreatment(c('DG','CA3','CA1'))
alldds2 = returnddstraining(c('DG','CA3','CA1'))



savevsds <- function(mydds, vsdfilename){
  dds <- mydds
  vsd <- vst(dds, blind=FALSE) ## variance stabilized
  print(head(assay(vsd),3))
  return(write.csv(assay(vsd), file = vsdfilename, row.names = T))
}


## fitting model and testing

savevsds(DGdds2, paste0(datapath, "02.DG_vsdtraining.csv"))

savevsds(CA3dds2, paste0(datapath, "02.CA3_vsdtraining.csv"))

savevsds(CA1dds2, paste0(datapath, "02.CA1_vsdtraining.csv"))


savevsds(DGdds, paste0(datapath, "02.DG_vsdtreat.csv"))

savevsds(CA3dds, paste0(datapath, "02.CA3_vsdtreat.csv"))

savevsds(CA1dds, paste0(datapath, "02.CA1_vsdtreat.csv"))




# Results to compare with volcano plots


res_summary_subfield <- function(mydds, mycontrast){
  res <- results(mydds, contrast = mycontrast, independentFiltering = T)
  print(mycontrast)
  print(sum(res$padj < 0.1, na.rm=TRUE))
  print(summary(res))
  cat("\n")
}


print("DG")

res_summary_subfield(DGdds2, c("training", "trained", "yoked"))

res_summary_subfield(DGdds, c("treatment", "conflict.trained", "standard.trained"))

res_summary_subfield(DGdds, c("treatment", "conflict.yoked", "standard.yoked"))

print("CA3")

res_summary_subfield(CA3dds2, c("training", "trained", "yoked"))

res_summary_subfield(CA3dds, c("treatment", "conflict.trained", "standard.trained"))

res_summary_subfield(CA3dds, c("treatment", "conflict.yoked", "standard.yoked"))


print("CA1")

res_summary_subfield(CA1dds2, c("training", "trained", "yoked"))

res_summary_subfield(CA1dds, c("treatment", "conflict.trained", "standard.trained"))

res_summary_subfield(CA1dds, c("treatment", "conflict.yoked", "standard.yoked"))




# Volcano plots

calculateDEGs = function(mydds, whichtissue, whichfactor, up, down){
  # calculate DEG results
  res <- results(mydds, contrast =c(whichfactor, up, down),
                 independentFiltering = T, alpha = 0.1)

  # create dataframe with pvalues and lfc
  data <- data.frame(gene = row.names(res),
                     padj = res$padj, 
                     logpadj = -log10(res$padj),
                     lfc = res$log2FoldChange)
  data <- na.omit(data)
  data <- data %>%
    dplyr::mutate(direction = ifelse(data$lfc > 0 & data$padj < 0.1, 
                                     yes = up, no = ifelse(data$lfc < 0 & data$padj < 0.1, 
                                                           yes = down, no = "NS"))) 
  data$direction <- factor(data$direction, levels = c(down, "NS", up))
  data$tissue <- whichtissue
  data$comparison <- paste0(down, " vs. ", up, sep = "" )
  data <- data %>% select(tissue, gene, lfc, padj, logpadj, comparison, direction) 
  return(data)
}  



save_all_df <-  function(mydds, whichtissue, whichfactor, up, down){
  # calculate DEG results
  res <- results(mydds, contrast =c(whichfactor, up, down),
                 independentFiltering = T, alpha = 0.1)
  
  # create dataframe with pvalues and lfc
  data <- data.frame(gene = row.names(res),
                     padj = res$padj, 
                     logpadj = -log10(res$padj),
                     lfc = res$log2FoldChange)
  data <- na.omit(data)
  return(data)
}  



# create data frame for making volcanos plots

DGa <-  calculateDEGs(DGdds, "DG", "treatment", "standard.trained", "standard.yoked") 
DGb <-  calculateDEGs(DGdds, "DG", "treatment", "conflict.trained", "conflict.yoked") 
DGc <-  calculateDEGs(DGdds, "DG", "treatment", "conflict.trained", "standard.trained") 
DGd <-  calculateDEGs(DGdds, "DG", "treatment", "conflict.yoked", "standard.yoked") 
DGe <- calculateDEGs(DGdds2, "DG", "training", "trained", "yoked") 

write_csv(DGe, paste0(datapath, '02.R_DEG_DG.csv'))


CA3a <-  calculateDEGs(CA3dds, "CA3", "treatment", "standard.trained", "standard.yoked") 
CA3b <-  calculateDEGs(CA3dds, "CA3", "treatment", "conflict.trained", "conflict.yoked") 
CA3c <-  calculateDEGs(CA3dds, "CA3", "treatment", "conflict.trained", "standard.trained") 
CA3d <-  calculateDEGs(CA3dds, "CA3", "treatment", "conflict.yoked", "standard.yoked") 
CA3e <- calculateDEGs(CA3dds2, "CA3", "training", "trained", "yoked") 

CA1a <-  calculateDEGs(CA1dds, "CA1", "treatment", "standard.trained", "standard.yoked") 
CA1b <-  calculateDEGs(CA1dds, "CA1", "treatment", "conflict.trained", "conflict.yoked") 
CA1c <-  calculateDEGs(CA1dds, "CA1", "treatment", "conflict.trained", "standard.trained") 
CA1d <-  calculateDEGs(CA1dds, "CA1", "treatment", "conflict.yoked", "standard.yoked") 
CA1e <- calculateDEGs(CA1dds2, "CA1", "training", "trained", "yoked") 

CA1b[CA1b$direction != 'NS',]


# save df with DEGs

allDEG <- rbind(DGa, DGb, DGc, DGd, DGe, # 1096 * 7
                CA3a, CA3b, CA3c, CA3d, CA3e, 
                CA1a, CA1b, CA1c, CA1d, CA1e) %>% 
    dplyr::filter(direction != "NS") %>%
    dplyr::mutate(lfc = round(lfc, 2),
                    padj = scientific(padj, digits = 3), # just to show the dgits in right way
                    logpadj = round(logpadj, 2)) %>%
    arrange(tissue, comparison, gene)

rownames(allDEG) = as.character(1:nrow(allDEG))
write.csv(allDEG, paste0(datapath, '02.allDEG.csv'))



# pca analysis and bar plots functions

# theme option pre-defined 

treatmentcolors <- c( "standard.yoked" = "#404040", 
                      "standard.trained" = "#ca0020",
                      "conflict.yoked" = "#969696",
                      "conflict.trained" = "#f4a582")

colorvalsubfield <- c("DG" = "#d95f02", 
                      "CA3" = "#1b9e77", 
                      "CA1" = "#7570b3")

trainingcolors <-  c("trained" = "darkred", 
                     "yoked" = "black")

allcolors <- c(treatmentcolors, 
               colorvalsubfield, 
               trainingcolors,
               "NS" = "#d9d9d9")

theme_ms <- function () { 
  theme_classic(base_size = 8) +
    theme(
      panel.grid.major  = element_blank(),  # remove major gridlines
      panel.grid.minor  = element_blank(),  # remove minor gridlines
      plot.title = element_text(hjust = 0, face = "bold", size = 7), # center & bold 
      plot.subtitle = element_text(hjust = 0, size = 7)
    )
}


pcadataframe <- function (object, intgroup = "condition", ntop = 500, returnData = FALSE) {
  rv <- rowVars(assay(object))
  select <- order(rv, decreasing = TRUE)[seq_len(min(ntop, 
                                                     length(rv)))]
  pca <- prcomp(t(assay(object)[select, ]))
  percentVar <- pca$sdev^2/sum(pca$sdev^2)
  if (!all(intgroup %in% names(colData(object)))) {
    stop("'intgroup' should specify columns of colData(dds)")
  }
  intgroup.df <- as.data.frame(colData(object)[, intgroup, 
                                               drop = FALSE])
  group <- if (length(intgroup) > 1) {
    factor(apply(intgroup.df, 1, paste, collapse = " : "))
  }
  else {
    colData(object)[[intgroup]]
  }
  d <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], PC3 = pca$x[, 3], 
                  PC4 = pca$x[, 4], PC5 = pca$x[, 5], PC6 = pca$x[, 6], 
                  PC7 = pca$x[, 7], PC8 = pca$x[, 8], PC9 = pca$x[, 9],
                  group = group, 
                  intgroup.df, name = colnames(object))
  if (returnData) {
    attr(d, "percentVar") <- percentVar[1:9]
    return(d)
  }
}



plotPCs <- function(mydds, mytitle){
  vsd <- vst(mydds, blind = FALSE)
  pcadata <- pcadataframe(vsd, intgroup = c("treatment", "training"), returnData = TRUE)
  percentVar <- round(100 * attr(pcadata, "percentVar"))
  #
  print(aov(PC1 ~ treatment, data = pcadata))
  apa1 <- apa.aov.table(aov(PC1 ~ treatment, data = pcadata))
  apa1 <- as.data.frame(apa1$table_body) 
  errodf <- apa1 %>% filter(Predictor == "Error") %>% pull(df)
  pvalue <- apa1 %>% filter(Predictor == "treatment") %>% pull(p)
  Fstat <- apa1 %>% filter(Predictor == "treatment") %>% pull(F)
  treatmentdf <- apa1 %>% filter(Predictor == "treatment")  %>% pull(df)
  #
  mynewsubtitle <- paste("F", treatmentdf, ",", errodf, "=", Fstat, ",p=", pvalue, sep = "")
  #
  PCA12 <- ggplot(pcadata, aes(pcadata$PC1, pcadata$PC2)) +
    geom_point(size = 2, alpha = 0.8, aes(color = treatment)) +
    stat_ellipse(aes(color = training)) +
    xlab(paste0("PC1: ", percentVar[1], "%")) +
    ylab(paste0("PC2: ", percentVar[2], "%")) +
    scale_color_manual(
      drop = FALSE,
      values = allcolors,
      breaks = c("standard.yoked", "standard.trained", 
                 "conflict.yoked", "conflict.trained", 
                 "yoked", "trained", "NS")
    ) +
    labs(subtitle = mynewsubtitle, title = mytitle) +
    theme_ms() +
    theme(
      legend.position = "none",
      axis.text = element_text(size = 6),
      axis.title = element_text(size = 8),
      plot.title = element_text(size = 8),
      plot.subtitle = element_text(size = 6),
      legend.text = element_text(size = 6),
      legend.title = element_text(size = 6)
    )
  return(PCA12)
}


plot.volcano <- function(data, mysubtitle){
  level_list = levels(data$direction)
  count_table <- table(data$direction)
  counts <- setNames(as.integer(count_table[level_list]), level_list)
  counts[is.na(counts)] <- 0  # NA 값을 0으로 변환
  filtered_levels <- level_list[level_list != "NS"]
  filtered_counts <- counts[names(counts) %in% filtered_levels]
  legend_labels <- paste0(filtered_counts)
  volcano <- data %>%
    ggplot(aes(x = lfc, y = logpadj, color = direction)) + 
    geom_point(size = 1, alpha = 1, na.rm = TRUE, show.legend = TRUE) +    
    theme_ms() +
    scale_color_manual(
      values = allcolors,
      name = "", 
      labels = legend_labels,
      breaks = filtered_levels
    ) +
    ylim(c(0, 12.5)) +  
    xlim(c(-8, 8)) +
    labs(
      y = NULL, x = NULL,
      caption = "log fold change",
      subtitle = mysubtitle
    ) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey", size = 0.5) +
    theme(
      legend.position = c(0.26, 0.9), 
      legend.spacing.y = unit(0.1, 'cm'),
      legend.key.size = unit(0.2, 'cm'),
      legend.background = element_blank(),
      legend.box.background = element_blank(), 
      axis.text = element_text(size = 6),
      axis.title = element_text(size = 6),
      legend.text = element_text(size = 6),
      legend.title = element_text(size = 6),
      plot.caption = element_text(hjust = 0.5, size = 6),
      plot.subtitle = element_text(hjust = 0, size = 6),
      plot.title = element_text(size = 8)
    )
  return(volcano)
}



a <- plotPCs(DGdds, "DG") 
b <- plot.volcano(DGa, "\ns. trained vs s. yoked") + labs(y = "-log10(p-value)")
c <- plot.volcano(DGb, "\nc. trained vs c. yoked")  
d <- plot.volcano(DGc, "\ns. trained vs c. trained")  
e <- plot.volcano(DGd, "\ns. yoked vs. c. yoked")  
f <- plot.volcano(DGe, "\ncontrol vs. trained")  

g <- plotPCs(CA3dds, "CA3")
h <- plot.volcano(CA3a, " ") + labs(y = "-log10(p-value)")
i <- plot.volcano(CA3b, " ")  
j <- plot.volcano(CA3c, " ")  
k <- plot.volcano(CA3d, " ")  
l <- plot.volcano(CA3e, " ")  

m <- plotPCs(CA1dds, "CA1")
n <- plot.volcano(CA1a, " ") + labs(y = "-log10(p-value)")
o <- plot.volcano(CA1b, " ")  
p <- plot.volcano(CA1c, " ")   
q <- plot.volcano(CA1d, " ")  
r <- plot.volcano(CA1e, " ")  


legend <- get_legend(a + theme(legend.position = "bottom", 
                                legend.title = element_blank()) +
                        guides(color = guide_legend(nrow = 2)))

mainplot <- plot_grid(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r, 
                        nrow = 3, rel_widths = c(1,0.8,0.8,0.8,0.8,1),
                        labels = c("a", "b", "c", "d", "e", "f", 
                                    "", "", "", "", "", "", 
                                    "", "", "", "", "", ""),
                        label_size = 12)

## Warning in MASS::cov.trob(data[, vars]): Probable convergence failure

fig4 <- plot_grid(mainplot, legend, ncol = 1, rel_heights = c(1, 0.1))

ggsave(paste0(plotpath, '02.volcano.png'), width = 7, height = 6, plot = fig4, dpi = 300 ) 

ggsave(paste0(plotpath, "02.volcano.tiff"), plot = fig4, 
       width = 7, height = 6, dpi = 300, compression = "lzw")

ggsave(paste0(plotpath, "02.volcano.eps"), plot = fig4,
       width = 7, height = 6, device = cairo_ps, family = "Arial", fallback_resolution = 300)

cairo_pdf(file=paste0(plotpath, '02.volcano.pdf'), family = "Arial", width=7, height=6)
plot(fig4)    
dev.off()

###########


# GO term overlap check 

# LTP_Sanes_Lichtman
# LTP_Sanes_Lichtman = read.csv(paste0(datapath, "/02.sanesLichtman.csv"), header = T) # 237 
LTP_Sanes_Lichtman = read.csv(paste0(datapath, "/02.sanesLichtman2.csv"), header = T) # newly downloaded from rayna's dissociation test data
LTP_Sanes_Lichtman.genelist = LTP_Sanes_Lichtman$Related.Transcripts
LTP_Sanes_Lichtman.genelist <- unlist(strsplit(LTP_Sanes_Lichtman.genelist, " "))
LTP_Sanes_Lichtman.genelist = LTP_Sanes_Lichtman.genelist[LTP_Sanes_Lichtman.genelist != ""]
LTP_Sanes_Lichtman.genelist = LTP_Sanes_Lichtman.genelist[LTP_Sanes_Lichtman.genelist != "NA"] # 249
#  new : [1] "Adra1a" "Adra1b" "Adra1d" "Adcy10" "Adcy2"  "Adcy3"  "Adcy4"  "Adcy5" 
#  [9] "Adcy6"  "Adcy7"  "Adcy8"  "Adcy9"  added 

LTP_Sanes_Lichtman2 = sort(unique(str_to_title(LTP_Sanes_Lichtman.genelist)))


DG_deg_table = DGe[DGe$direction != 'NS',]
DG_deg_list = DG_deg_table$gene
candidategenes = c("Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", 'Gria2', "Pick1", "Nsf", "Numb", "Fmr1","Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci")

overlap_df <- data.frame(matrix(NA, nrow = 4, ncol = 3))

# Response to Stimulus 
GO_0050896 = read.delim(paste0(datapath, "/02.MGI_GO_response_to_stimulus.txt"), header = T, row.names = NULL)
GO_0050896_genes = GO_0050896$MGI.Gene.Marker.ID

overlap_df[1,'X1'] = paste0(candidategenes[candidategenes %in% GO_0050896_genes], collapse = ', ')
overlap_df[1,'X2'] = paste0(DG_deg_list[DG_deg_list %in% GO_0050896_genes], collapse = ', ')
overlap_df[1,'X3'] = paste0(LTP_Sanes_Lichtman2[LTP_Sanes_Lichtman2 %in% GO_0050896_genes], collapse = ', ')


# Translation
GO_0006412 = read.delim(paste0(datapath, "/02.MGI_GO_translation.txt"), header = T, row.names = NULL)
GO_0006412_genes = GO_0006412$MGI.Gene.Marker.ID

overlap_df[2,'X1'] = paste0(candidategenes[candidategenes %in% GO_0006412_genes], collapse = ', ')
overlap_df[2,'X2'] = paste0(DG_deg_list[DG_deg_list %in% GO_0006412_genes], collapse = ', ')
overlap_df[2,'X3'] = paste0(LTP_Sanes_Lichtman2[LTP_Sanes_Lichtman2 %in% GO_0006412_genes], collapse = ', ')


# Synapse Organization
GO_0050808 = read.delim(paste0(datapath, "/02.MGI_GO_synapse_organization.txt"), header = T, row.names = NULL)
GO_0050808_genes = GO_0050808$MGI.Gene.Marker.ID

overlap_df[3,'X1'] = paste0(candidategenes[candidategenes %in% GO_0050808_genes], collapse = ', ')
overlap_df[3,'X2'] = paste0(DG_deg_list[DG_deg_list %in% GO_0050808_genes], collapse = ', ')
overlap_df[3,'X3'] = paste0(LTP_Sanes_Lichtman2[LTP_Sanes_Lichtman2 %in% GO_0050808_genes], collapse = ', ')


# Learning or Memory 
GO_0007611 = read.delim(paste0(datapath, "/02.MGI_GO_learning_or_memory.txt"), header = T, row.names = NULL)
GO_0007611_genes = GO_0007611$MGI.Gene.Marker.ID

overlap_df[4,'X1'] = paste0(candidategenes[candidategenes %in% GO_0007611_genes], collapse = ', ')
overlap_df[4,'X2'] = paste0(DG_deg_list[DG_deg_list %in% GO_0007611_genes], collapse = ', ')
overlap_df[4,'X3'] = paste0(LTP_Sanes_Lichtman2[LTP_Sanes_Lichtman2 %in% GO_0007611_genes], collapse = ', ')

colnames(overlap_df) = c('Candidate Genes','DG DEGs','LTP Genes from Sanes & Lichtman 1999')
rownames(overlap_df) = c('Response to stimulus (GO-0050896)','Translation (GO-0006412)','Synapse organization (GO-0050808)','Learning or memory (GO-0007611)')

write.csv(overlap_df, paste0(datapath, '02.gene_ovlap.csv'), row.names = TRUE)

DG_deg_list[DG_deg_list%in%LTP_Sanes_Lichtman2]
# [1] "Adrb1"  "Bdnf"   "Egr1"   "Homer1" "Ntrk2"  "Stmn4"  "Vamp1" 




