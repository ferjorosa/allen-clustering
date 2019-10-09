rm(list=ls())

if (!require("foreign")){
  install.packages("foreign", dependencies = TRUE)
  library(foreign)
}

source("f_generate_discretized_data.R")

#### 1 - Cargamos los datos SIN NORMALIZAR

exon_data = read.arff("../data/continuous/exon_data_48278_hclust100.arff")

#### 2 - Filtramos aquellas columnas cuyos valores se encuentren fuertemente centradas alrededor de un valor (por ejemplo el cero)
#### Para ello establecemos que para ser filtrado deben de estar en dicho valor el (100-filter_percentage)% de las instancias

filter_percentage = 0 # 1 - filter_percentage de instancias con el mismo valor
filtered_exon_data = filter_columns_for_disc(exon_data, filter_percentage, debug_output = TRUE)

#### 3 - Discretizamos los datos utilizando nuestra version equal_width que considera el punto cero como un caso especial
#### El numero de breaks es aparte del 0, por lo que si queremos una variable discreta con 3 estados habria que seleccionar 2 breaks
discrete_data = discretize_equal_with_considering_zero(data = filtered_exon_data, numberOfBreaks = 3, debug_output = TRUE)

#### 4 - Cambiamos los nombres de los estados por [0,1] (presencia o no presencia) para que sea mas facil de almacenar
for(i in 1:ncol(discrete_data)){
  levels(discrete_data[,i]) = c(0,"low","medium","high")
}

#### 5 - Guardamos los resultados en formato ARFF
write.arff(x = discrete_data, file = "../data/discrete/exon_data_48278_hclust100_disc.arff", relation = "exon_data_48278_hclust100_disc")

#### EXTRA: Guardamos versiones con 100 y 1000 variables
#write.arff(x = discrete_data[, 1:100], file = "../data_mtg/discrete/exon_data_binary_100.arff", relation = "exon_data_discrete_4_100")
#write.arff(x = discrete_data[, 1:1000], file = "../data_mtg/discrete/exon_data_binary_1000.arff", relation = "exon_data_binary_1000")

##############################################################################################
##############################################################################################
# # 1 - Cargamos los datos sin normalizar
# exon_data = read.arff("../data_mtg/exon_data.arff")
# 
# # 2 - Filtramos aquellas columnas cuyos valores se encuentren fuertemente centradas alrededor de un valor (por ejemplo el cero)
# smoothing_percentage = 1
# 
# filtered_exon_data = filter_columns_for_disc(exon_data, smoothing_percentage)
# 
# filtered_exon_data_100 = filter_columns_for_disc(exon_data, 0)
# 
# # 3 - Discretizamos los datos aplicando smoothing
# discrete_data = discretize_equal_width(data = filtered_exon_data, numberOfBreaks = 3, debug_output = FALSE)
# 
# # 4 - Pasamos a integer las variables para que sea mas facil de almacenar
# discrete_data_integer = data.frame(sapply(discrete_data, FUN = as.integer))
# 
# # 5 - Guardamos los resultados en formato ARFF
# write.arff(x = discrete_data_integer, file = "../data_mtg/exon_data_discrete_int.arff")
# 
# write.arff(x = discrete_data, file = "../data_mtg/exon_data_discrete.arff")