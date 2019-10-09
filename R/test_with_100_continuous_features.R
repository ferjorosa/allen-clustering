# Este script fue escrito con el objetivo de generar las matrices de distancias entre instancias para los ejemplos con 100 y 1000 variables

if (!require("rlang")){
  install.packages("rlang",type="win.binary") 
}
if (!require("ClusterR")){
  install.packages("ClusterR", dependencies = TRUE)
  library(ClusterR)
}
if (!require("foreign")){
  install.packages("foreign", dependencies = TRUE)
  library(foreign)
}

# Primero cargamos los datos continuos
exon_data_100 = read.csv("../data_mtg/100/exon_data_100.csv")

# Exportamos en formato ARFF
write.arff(x = exon_data_100, file = "../data_mtg/100/exon_data_100.arff")