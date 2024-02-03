library(reshape2)
library(ggplot2)
library(patchwork)
library(dplyr)
library(ggsci)
library(scales)
library(viridis)
library(parallel)

args <- commandArgs(TRUE)
hic_dir = args[1]
outdir = args[2]
thres = args[3]

# hic_dir = "/public/home/yuanyuan/3Dgenome/data_preprocessing/gm12878_hg19/hic/"
chr_len <- c(249250621, 243199373, 198022430, 191154276, 180915260, 171115067,159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566)
######
mclapply(1:22, function(chrom){
	print(chrom)
	ndata = read.csv(paste0(outdir,"/V_chr",chrom,".csv"), stringsAsFactors = F)
	edata_new <- data.frame(v_index1 = rep(seq(0,max(ndata$v_index)), each = nrow(ndata)),v_index2 = rep(seq(0,max(ndata$v_index)), nrow(ndata)))
	edata_new <- filter(edata_new, v_index2 - v_index1 >= 0)
	## 5kb hic
	hic_raw <- read.delim(paste0(hic_dir,"/chr",chrom,"_5kb.RAWobserved"), header = F)
	norm <- read.delim(paste0(hic_dir,"/chr",chrom,"_5kb.SQRTVCnorm"), header = F)
	indx1 <- hic_raw[,1] / 5000 + 1
	indx2 <- hic_raw[,2] / 5000 + 1
	norm_hic <- hic_raw[,3] / (norm[indx1,1] * norm[indx2,1])
	hic_raw$hic <- norm_hic
	write.csv(hic_raw, file = paste0(hic_dir,"chr", chrom, "_5kb_sqrtv_norm.csv"), row.names = F, col.names = F, quote = F)
	#####
	hic_5kb <- hic_raw
	colnames(hic_5kb) <- c("from","to","hic0","hic")
	hic_id <- unique(c(hic_5kb$from, hic_5kb$to))
	ans <- sapply(1:nrow(ndata),function(x){
	  temp <- hic_id[which(hic_id / chr_len[chrom] >= ndata$v_start[x] & (hic_id + 5000) / chr_len[chrom] <= ndata$v_end[x])]
	  if(length(temp) > 0){
	    data.frame(temp, rep(ndata$v_index[x], length(temp)))
	  }else{
	    return(NULL)
	  }
	})
	hic_id_index <- do.call(rbind, ans)
	colnames(hic_id_index) <- c("index","v_index")
	####
	hic_5kb_new <- hic_5kb
	hic_5kb_new$v_index1 <- hic_id_index$v_index[match(hic_5kb$from, hic_id_index$index)]
	hic_5kb_new$v_index2 <- hic_id_index$v_index[match(hic_5kb$to, hic_id_index$index)]
	hic_5kb_new <- hic_5kb_new %>% filter(!is.na(v_index1), !is.na(v_index2))
	edata_new <- left_join(edata_new, hic_5kb_new, by = c("v_index1","v_index2"))
	edata_new <- filter(edata_new, !is.na(from), !is.na(to))
	edata_new_mean <- edata_new %>% as_tibble() %>% dplyr::group_by(v_index1,v_index2) %>% dplyr::summarize(mean_hic = log(mean(hic, na.rm = T)))
	edata_new_mean <- as.data.frame(edata_new_mean)
	edata_new_mean <- data.frame(edata_new_mean, g1 = ndata[match(edata_new_mean$v_index1, ndata$v_index),c("v_start", "v_end")], g2 = ndata[match(edata_new_mean$v_index2, ndata$v_index),c("v_start", "v_end")])	

	write.csv(edata_new_mean, file = paste0(outdir,"/chr",chrom,"_edges.csv"), row.names = F, quote = F)
}, mc.cores = thres)
####
