#reorganize the edge data
# edge strength 

library(reshape2)
library(ggplot2)
library(patchwork)
library(dplyr)
library(ggpubr)
library(ggsci)
library(scales)
# show_col(pal_nejm("default")(8))
# show_col(pal_npg("nrc", alpha = 0.6)(10))
cols <- pal_nejm("default")(8)
args <- commandArgs(TRUE)
workdir = args[1]
node_file = args[2]
tad_file = args[3]   # filename


ndata <- read.delim(node_file, header = T)
tad <- read.delim(tad_file)
tad <- tad[-1,]
if(grepl("chr", tad$chr1[1])){
    tad$chr1 <- gsub("chr","", tad$chr1)
}
tad$chr1 <- as.numeric(tad$chr1)
tad <- tad[order(tad$chr1, tad$x1),]
tad$index <- 1:nrow(tad)
write.table(ndata[,1:4], file = paste0(workdir,"/inputdata/raw/nodes.txt"), row.names = F, col.names = T, quote = F, sep = "\t")
## annotate tad
ndata$tad_index <- "."
rownames(ndata) <- paste(ndata$chr, ndata$node_idx, sep = "_")
for(i in 1:nrow(tad)){
    chr_ndata <- filter(ndata, chr == tad$chr1[i])
    flag = data.frame(left = chr_ndata$start - tad$x2[i], right = chr_ndata$end - tad$x1[i])
    n <- which(flag$left <= 0 & flag$right >= 0)
    if(length(n) > 0){
        ndata[rownames(chr_ndata)[n], "tad_index"] <- paste(ndata[rownames(chr_ndata)[n], "tad_index"],tad$index[i], ".", sep = "")
    }
}

n_tad = table(ndata$tad_index)
n_tad <- n_tad[which(names(n_tad) != '.')]
#hist(n_tad)
plotdat <- data.frame(n_seg = n_tad)
p = ggplot(plotdat, aes(x = n_seg.Freq, ..scaled..)) + geom_density(alpha = 0.7)
p = p + theme_bw()+ theme_classic() + scale_fill_manual(values = cols[c(7,4)]) + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "top", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 20)) + labs(title = "Number of segments in one TAD") + xlab("Number") + ylab("Density")
p
p1 = p
## 每条染色体中的tad数量
plotdat = data.frame(table(ndata$chr, ndata$tad_index != '.'))
plotdat$Var2 <- as.character(plotdat$Var2)
plotdat <- plotdat[which(plotdat$Var2 == "TRUE"),]
p = ggplot(plotdat, aes(x = Freq, y = Var1, fill = Var1)) + geom_bar(stat = "identity")
p = p + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "none", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 20)) + labs(title = "Number of TADs") + xlab("Number") + ylab("Chromsome")
p
p2 = p
## 每条染色体中的segment数量
plotdat = data.frame(table(ndata$chr))
p = ggplot(plotdat, aes(x = Freq, y = Var1, fill = Var1)) + geom_bar(stat = "identity")
p = p + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "none", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 20)) + labs(title = "Number of Segments") + xlab("Number") + ylab("Chromsome")
p
p3 = p

#(p3 | p2) / (p1 | plot_spacer())
p = p3 | p2 | p1
ggsave(paste0(workdir,"/stat/basic_stat_plot.tiff"), plot = p, width = 15, height = 6)

## CTCF peak
plotdat = ndata[,c("l_ctcf_peak", "r_ctcf_peak")]
plotdat = melt(plotdat)
plotdat$variable <- factor(plotdat$variable, levels = c("l_ctcf_peak", "r_ctcf_peak"))
levels(plotdat$variable)[levels(plotdat$variable) == "r_ctcf_peak"] <- "Right"
levels(plotdat$variable)[levels(plotdat$variable) == "l_ctcf_peak"] <- "Left"

p = ggplot(plotdat, aes(x = value, fill = variable)) + geom_histogram(alpha = 0.8,binwidth = 0.01)
p = p + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "top", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 25)) + labs(title = "CTCF Peak") + xlab("CTCF peak")
p
ggsave(paste0(workdir,"/stat/ctcf_peak_raw.tiff"), plot = p, width = 8, height = 6)
## motif
plotdat = ndata[,c("r_score", "l_score")]
plotdat = melt(plotdat)
plotdat$variable = factor(plotdat$variable, levels = c("l_score","r_score"))
levels(plotdat$variable)[levels(plotdat$variable) == "r_score"] <- "Right"
levels(plotdat$variable)[levels(plotdat$variable) == "l_score"] <- "Left"

p = ggplot(plotdat, aes(x = value, fill = variable)) + geom_histogram(alpha = 0.8)
p = p + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "top", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 25)) + labs(title = "CTCF motif score") + xlab("Score")
p
ggsave(paste0(workdir,"/stat/ctcf_motif_score_raw.tiff"), plot = p, width = 8, height = 6)


#reorganize the node data
maxmin = function(vec_data){
    (vec_data - min(vec_data)) / (max(vec_data) - min(vec_data))
}
atac = maxmin(log10(ndata$ATAC_mean + 1e-5))
h3k27ac = maxmin(log10(ndata$H3K27ac_mean + 1e-5))
h3k4me3 = maxmin(log10(ndata$H3K4me3_mean + 1e-5))
pol2 = maxmin(log10(ndata$Pol2_mean + 1e-5))

chr_len <- c(249250621, 243199373, 198022430, 191154276, 180915260, 171115067,159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566)

v_index <- ndata[, "node_idx"]
v_chr <- ndata[, "chr"]
v_start <- ndata[, "start"] / chr_len[ndata[, "chr"]]
v_end <- ndata[, "end"] / chr_len[ndata[, "chr"]]
v_r_ctcf <- maxmin(log10(ndata[, "r_ctcf_peak"] + 1e-5))    ## 原本为log(ndata[, "r_ctcf_peak"]) 
v_r_ctcf_strand <- ndata[, "r_dir"]
v_r_ctcf_motif <- log2(ndata[, "r_score"] + 0.001)
v_r_ctcf_cohesin <- ndata[, "r_cohesin"]
v_l_ctcf <- maxmin(log10(ndata[, "l_ctcf_peak"] + 1e-5))   ## 原本为log(ndata[, "r_ctcf_peak"])
v_l_ctcf_strand <- ndata[, "l_dir"]
v_l_ctcf_motif <- log2(ndata[, "l_score"] + 0.001)
v_l_ctcf_cohesin <- ndata[, "l_cohesin"]
#h3k4me3 <- ndata[, "H3K4me3_mean"]
#h3k27ac <- ndata[, "H3K27ac_mean"]
#atac <- ndata[, "ATAC_mean"]
#pol2 <- ndata[,"Pol2_mean"]
tad_index <- ndata[, "tad_index"]
v_center <- (v_start+v_end)/2

x <- data.frame(v_index, 
                v_chr, 
                v_start, 
                v_end, 
                v_r_ctcf, 
                v_r_ctcf_strand,
                v_r_ctcf_motif,
                v_r_ctcf_cohesin,
                v_l_ctcf,
                v_l_ctcf_strand,
                v_l_ctcf_motif,
                v_l_ctcf_cohesin,
                h3k4me3,
                h3k27ac,
                atac,
                pol2,
                tad_index,
                v_center)
sapply(unique(x$v_chr), function(i) {
    write.csv(x[which(x[, "v_chr"] == i), ], file = paste0(workdir,"/inputdata/raw/V_chr", i, ".csv"), row.names = F)
})


## ctcf peak
ndata = x
plotdat = ndata[,c("v_r_ctcf", "v_l_ctcf")]
plotdat = melt(plotdat)
plotdat$variable <- factor(plotdat$variable, levels = c("v_l_ctcf", "v_r_ctcf"))
levels(plotdat$variable)[levels(plotdat$variable) == "v_r_ctcf"] <- "Right"
levels(plotdat$variable)[levels(plotdat$variable) == "v_l_ctcf"] <- "Left"

p = ggplot(plotdat, aes(x = value, fill = variable)) + geom_density(alpha = 0.8)
p = p + theme_bw()+ theme_classic() +
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), legend.position = "none") + xlab("log10(CTCF peak)") + facet_wrap(variable~.) + theme(strip.text = element_text(size = 20))
p
p2 = p

ggsave(paste0(workdir,"/stat/ctcf_peak.tiff"), plot = p2, width = 8, height = 6)
## motif
plotdat = ndata[,c("v_r_ctcf_motif", "v_l_ctcf_motif")]
plotdat = melt(plotdat)
plotdat$variable = factor(plotdat$variable, levels = c("v_l_ctcf_motif","v_r_ctcf_motif"))
levels(plotdat$variable)[levels(plotdat$variable) == "v_r_ctcf_motif"] <- "Right"
levels(plotdat$variable)[levels(plotdat$variable) == "v_l_ctcf_motif"] <- "Left"

p = ggplot(plotdat, aes(x = value, fill = variable)) + geom_histogram(alpha = 0.8)
p = p + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 15), legend.position = "top", legend.title = element_blank(), plot.title = element_text(hjust = 0.5, size = 25)) + labs(title = "CTCF motif score") + xlab("Score")
p
p3 = p
ggsave(paste0(workdir,"/stat/ctcf_motif.tiff"), plot = p, width = 8, height = 6)
## atac
p = ggplot(ndata, aes(x = atac)) + geom_histogram(fill = pal_nejm("default", alpha = 0.8)(8)[2]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(ATAC)") + labs(title = "ATAC-Seq")
p
p4 = p
ggsave(paste0(workdir,"/stat/atac_hist.tiff"), plot = p, width = 8, height = 6)

p = ggplot(ndata, aes(x = h3k4me3)) + geom_histogram(fill = pal_nejm("default", alpha = 0.8)(8)[3]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(H3K4me3)") + labs(title = "H3K4me3")
p
p5 = p
ggsave(paste0(workdir,"/stat/h3k4me3_hist.tiff"), plot = p, width = 8, height = 6)

p = ggplot(ndata, aes(x = h3k27ac)) + geom_histogram(fill = pal_nejm("default", alpha = 0.8)(8)[5]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(H3K27ac)") + labs(title = "H3K27ac")
p
p6 = p
ggsave(paste0(workdir,"/stat/h3k27ac_hist.tiff"), plot = p, width = 8, height = 6)

p = ggplot(ndata, aes(x = pol2)) + geom_histogram(fill = pal_nejm("default", alpha = 0.8)(8)[5]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(Pol2)") + labs(title = "Pol2")
p
p7 = p
ggsave(paste0(workdir,"/stat/pol2_hist.tiff"), plot = p, width = 8, height = 6)


## node length
plotdat <- data.frame(node_length = (ndata$v_end - ndata$v_start) * chr_len[ndata$v_chr])
p = ggplot(plotdat, aes(x = log10(node_length))) + geom_histogram(fill = pal_nejm("default", alpha = 0.8)(8)[4]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(length of segment)") + labs(title = "Length of Segment")
p
p8 = p
ggsave(paste0(workdir,"/stat/segment_length_hist.tiff"), plot = p, width = 8, height = 6)

#### segment 物理距离的长度分布
chr_len <- c(249250621, 243199373, 198022430, 191154276, 180915260, 171115067,159138663,
           146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540,
           102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566)
ans <- sapply(1:22, function(x){
    chr = ndata[which(ndata[,"v_chr"] == x), ]
    dist = unlist(sapply(128:nrow(chr),function(i){
        j = i - 127
        chr$v_end[i] - chr$v_start[j]
    }))
    dist * chr_len[x]
})
plotdat = data.frame(seg_dist = unlist(ans))
p = ggplot(plotdat, aes(x = log10(seg_dist))) + geom_histogram(fill = pal_nejm("default", alpha = 0.6)(8)[8]) + theme_bw()+ theme_classic() + 
    theme(panel.grid = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 20),legend.text = element_text(size = 20), plot.title = element_text(hjust = 0.5, size = 25)) + xlab("log10(dist)") + labs(title = "dist of 128 segments covered")
p
p9 = p
ggsave(paste0(workdir,"/stat/segments_dist_hist.tiff"), plot = p, width = 8, height = 6)
c(min(plotdat$seg_dist), max(plotdat$seg_dist))

# patchwork
p = (p2 / p3) | (p4/p5/p6) | (p7/p8/p9)
p

ggsave(paste0(workdir,"/stat/data_distribution.tiff"), width = 17,height = 13)
