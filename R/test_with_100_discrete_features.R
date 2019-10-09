# Este script de codigo fue generado con el objetivo de preparar las matrices de entropias y MIs para el algoritmo de seleccion 
# no supervisada de variables con teoria de informacion

if (!require("infotheo")){
  install.packages("infotheo", dependencies = TRUE)
  library(infotheo)
}
if (!require("foreign")){
  install.packages("foreign", dependencies = TRUE)
  library(foreign)
}

# Primero cargamos los datos discretos
exon_data_discrete = discrete_data_binary

# Despues seleccionamos las 100 primeras variables
exon_data_discrete_subset100 = exon_data_discrete[, 1: 1000]
data = exon_data_discrete_subset100

# Calculamos las entropias y las informaciones mutuas
entropies = rep(-1, ncol(data))
for(i in 1:ncol(data)){
  entropies[i] = infotheo::entropy(data[,i])
}

mutual_informations = infotheo::mutinformation(data)

# Exportamos en formato CSV las matrices de MI y el vector de entropias
write.csv(x = entropies, file = "../data_mtg/discrete/binary/exon_data_binary_1000_entropies.csv",  row.names = FALSE)
write.csv(x = mutual_informations, file = "../data_mtg/discrete/binary/exon_data_binary_1000_mis.csv", row.names = FALSE)

