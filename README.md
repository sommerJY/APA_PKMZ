# APA_PKMZ

This project is organized into four main subdirectories: 
- scripts: contains all R and Python based workflows for statistical analyses and data visualization
- UNIXworkflow: contains and explanation and all the UNIX commands used to process the raw sequencing data on the Stampede cluster at the Texas Advanced Computing Facility. Written by Rayna Harris
- figures: all the figures created from the scripts
- data: all the input data and the results 

# Part 1: Behavioral analysis
**R-based**
- scripts/00_behavior_wrangle: behavior data wrangling

**Python-based**
- scripts/01.behavior_analysis: behavior statistics and data visualization

# Part 2: RNA sequencing analysis
**UNIX-based**
- UNIXworkflow/00_rawdata: Download the data to scratch on Stampede with 00_gsaf_download.sh
- UNIXworkflow/01_fastqc: Evaluate the quality of the reads using the program FastQC
- UNIXworkflow/02_filtrimreads: Filter low quality reads and trim adapters using the program cutadapt
- UNIXworkflow/03_fastqc: Evaluate the quality of the processed reads
- UNIXworkflow/04_kallisto: Quantify transcript-level expression using Kallisto

**R-based**
- scripts/00_rnaseq_wrangle: converting the kallisto transcript counts to gene counts
- scripts/02.DEG_analysis: Differential gene expression analysis using DESeq2

# Part 3: Integration of behavioral and RNA sequencing results
**Python-based**
- scripts/03.Behavior_expression: Merging behavior and gene expression analyses
- scripts/04.new_method: Comparing and generating correlation-based candidates
- scripts/06.Robust_check: Testing the robustness of DEGs in 3 independent metrics to rank genes based on their ability to distinguish Trained from Control groups

**R-based**
- scripts/05.WGCNA: WGCNA analysis using pearson and xicor correlation




