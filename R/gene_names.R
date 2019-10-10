rm(list=ls())

if (!require("foreign")){
  install.packages("foreign", dependencies = TRUE)
  library(foreign)
}

# Cargamos los datos y la tabla de nombres de los genes
lcm_100_data = read.arff("../data/discrete/exon_100_disc.arff")
gene_names = read.csv("human_MTG_2018-06-14_genes-rows.csv")

# Seleccionamos los nombres de las columnas que se corresponden con el ID del gen
lcm_100_gene_ids = gsub("x", "", colnames(lcm_100_data))

# Filtramos la tabla de nombres con los ids de los genes contenidos en los datos
lcm_100_gene_ids = gene_names[gene_names$entrez_id %in% lcm_100_gene_ids,]
