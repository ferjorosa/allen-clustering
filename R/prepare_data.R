############### EXON DATA ################

exon_data = read.csv("../data_mtg/human_MTG_2018-06-14_exon-matrix.csv")

genomes = exon_data[,1]
genomes_string = sapply(X = genomes, FUN = as.character)

instance_names = colnames(exon_data)
instance_names = instance_names[-c(1)]

t_exon_data = transpose(exon_data)
t_exon_data = t_exon_data[-c(1),]

colnames(t_exon_data) = genomes_string

############## INTRON DATA ###############

############## EXTRA ###############
# Podemos despues agregar el nombre de la instancia al dataFrame para que sea facilmente identificable
