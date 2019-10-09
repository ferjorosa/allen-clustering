#
#
# Script to show the difference between classic equal-width discretization and our heuristic with smoothing
# Note: When applying smoothing there are cases where it is not possible to cut the data because all the values are concentrated, for example in 0.
# 

rm(list=ls())

source("f_generate_discretized_data.R")

##############################################################################
#
# Sub-script para buscar genes interesantes. Lo que hace es simplemente ordenarlos
# segun su desviacion estandar. Nos pueden interesar aquellos que tengan un alto valor
#
# Creamos un nuevo dataFrame donde se guarde el nombre de la variable y su sd
search_df = data.frame(sapply(filtered_exon_data, FUN = sd))
colnames(search_df)[1] = "sd"
row_index = 1:nrow(search_df)
search_df = cbind(search_df, row_index)
ordered_search_df = search_df[order(search_df$sd),]

###############################################################################

exon_column = subset(filtered_exon_data, select = c(X23461))
nBreaks = 3
par(mfrow = c(1,2))

##### Classic discretization
dens = density(exon_column[,1])
hist(exon_column[,1], 
     freq = FALSE, 
     main = names(exon_column[1]),
     breaks = seq(min(exon_column[,1]), max(exon_column[,1]), length.out = nBreaks + 1), 
     ylim = c(0, max(dens$y)),
     xlab = "value")
lines(dens, col = "red")

##### Discretization with smoothing
smooth_exon_column = discretize_equal_width(data = exon_column, nBreaks, 1)

cuts = levels(smooth_exon_column[,1])
cuts = paste(cuts, collapse = ";")
cuts = gsub(x=cuts, "\\)",replacement = "")
cuts = gsub(x=cuts, "\\(",replacement = "")
cuts = gsub(x=cuts, "\\]",replacement = "")
cuts = gsub(x=cuts, "\\[",replacement = "")
cuts = strsplit(cuts, ";")[[1]]
cuts = sapply(cuts, FUN = as.numeric)

dens = density(exon_column[,1])
hist(exon_column[,1], 
     freq = FALSE, 
     main = names(exon_column[1]),
     breaks = unique(cuts), 
     ylim = c(0, max(dens$y)),
     xlab = "value")
lines(dens, col = "blue")
