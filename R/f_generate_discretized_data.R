############################################################
### author: Fernando Rodriguez Sanchez (ferjorosa@gmail,com)
############################################################
#
# Este script contiene 3 metodos de discretizacion no supervisados:
# - Equal width
# - Eqaul width con suavizado
# - Equal frequency


# Discretiza un dataFrame mediante equal_width. 
# 
# El objetivo del suavizado es aumentar la probabilidad de los estados extremos. 
# Para ello se omiten en el momento de la discretizacion un porcentaje de los datos 
# de cada lado de la distribucion. Una vez ejecutado, el resultado es un dataFrame 
# discreto cuyos estados de los extremos pueden ser algo mas anchos de lo normal. 
# Esto ayuda a representar mejor aquellas distribuciones con faldas muy pronunciadas.
# 
# Como la discretizacion se hace con cada una de las variables por separado, el suavizado
# tambien se aplica una a una
# 
# data: el dataFrame continuo a discretizar
# numberOfBreaks: el numero de estados en cada una de las variables discretas
# percentage_suavizado (Opcional): el procentaje de instancias que se omite de cada lados de la distribucion
#
discretize_equal_width =  function(data, numberOfBreaks, percentage_suavizado, debug_output) {
  
  if(missing(debug_output)) {
    debug_output = FALSE
  }
  
  if(missing(percentage_suavizado)) {
    percentage_suavizado = 0
  }
  
  percentage_instances = round(((100 - percentage_suavizado)/100) * nrow(data))

  # Creamos un nuevo dataFrame para los datos discretizados
  discretized_data = data.frame()
    
  # Iteramos por cada uno de los atributos
  for(att_col in 1:ncol(data)){
    
    if(debug_output == TRUE){
      print(att_col)
    }
    
    # Generamos el dataSet ordenado que contiene la "distancia" de cada valor a la mediana para el atributo en cuestion
    values = data[,att_col]
    diff = abs(values - mean(values))
    
    # Ordenamos las variables segun su varianza
    df = as.data.frame(diff)
    df$values = values
    ordered_df = df[order(df$diff),]
    
    df = ordered_df
    
    #### Seleccionamos el porcentage de instancias establecido para su discretizacion
    df_percentage = head(df, n = percentage_instances)
    x = df_percentage[,2]
    dens = density(x)
    
    #############################################
    #### Con el nuevo grupo de instancias generamos cortes nuevos para su discretizacion 
    
    cuts = cut(x, breaks = numberOfBreaks, dig.lab = 6)
    t = table(cuts)
    
    # Transformamos los cuts en breakpoints para poder incluir las instancias que hemos filtrado previamente
    
    levels = levels(cuts)
    newBreaks = c()
    for(i in 1:length(levels) - 1){
      interval = unlist(strsplit(levels(cuts)[i], ","))
      
      # interval es siempre de tamaño 2  
      min = gsub(x=interval[1], '\\(',replacement = "")
      min = gsub(x=min, "\\)",replacement = "")
      min = gsub(x=min, "\\[",replacement = "")
      min = gsub(x=min, "\\]",replacement = "")
      
      max = gsub(x=interval[2], "\\(",replacement = "")
      max = gsub(x=max, "\\)",replacement = "")
      max = gsub(x=max, "\\[",replacement = "")
      max = gsub(x=max, "\\]",replacement = "")
      
      newBreaks = c(newBreaks, max)
    }
    newBreaks = as.numeric(newBreaks)
    
    cuts2 = cut(data[,att_col], breaks = c(min(data[,att_col]), newBreaks, max(data[,att_col])), include.lowest = TRUE, dig.lab = 6)
    
    # Cambiamos el estilo de los intervarlos para que pasen a estar separados por ';' en vez de ','
    levels(cuts2) = unlist(lapply(levels(cuts2), FUN = function(y){gsub(x=y, ",",replacement = ";")}))
    if(nrow(discretized_data) == 0){
      discretized_data = data.frame(cuts2)
    } else{
      discretized_data = cbind(discretized_data, cuts2)
    }
    colnames(discretized_data)[att_col] = names(data[att_col])
  }
  
  return(discretized_data)
}

# 
# Filtra aquellas columnas cuyo numero de valores unicos es menor que el numero de instancias seleccionadas por el suavizado. 
# Por ejemplo, si el suavizado es un 1% de 1000 (10 instancias) y solo hay 7 valores unicos, no nos interesa dicha columna, la filtramos
#
filter_columns_for_disc = function(data, percentage_filtrado, debug_output) {
  
  if(missing(percentage_filtrado)) {
    percentage_filtrado = 0
  }
  
  if(missing(debug_output)) {
    debug_output = FALSE
  }
  
  percentage_instances = round(((100 - percentage_filtrado)/100) * nrow(data))
  
  filtered_columns = c()
  
  for(att_col in 1:ncol(data)){
    
    if(debug_output == TRUE){
      print(att_col)
    }
    
    # Generamos la tabla de valores para la columna en cuestion
    t = table(data[, att_col])
    
    # Si se concentra en un unico valor un numero de instancias mayor que (1 - percentage_filtrado), filtramos la columna en cuestion
    if(t[1] >= percentage_instances){
      filtered_columns = c(filtered_columns, colnames(data)[att_col])
    }
  }
  
  print("Las columnas filtradas son:")
  print(filtered_columns)
  
  return(data[, !names(data) %in% filtered_columns])
}

#
# Variante de discretizacion equal_width que considera el numero cero de forma especial, otorgandole un bin especificamente para el.
# 
# El numero de breaks es aparte del cero. Por lo que si queremos 3 estados, habria que poner 2 breaks
#
discretize_equal_with_considering_zero = function(data, numberOfBreaks, debug_output) {
  if(missing(debug_output)) {
    debug_output = FALSE
  }
  
  # Creamos un nuevo dataFrame para los datos discretizados
  discretized_data = data.frame()
  
  # Iteramos por cada uno de los atributos
  for(att_col in 1:ncol(data)){
    
    if(debug_output == TRUE){
      print(att_col)
    }
    
    data_col = data[,att_col]
    
    # Generamos los break points de forma especial. Consideramos un intervalo especifico para el 0 y despues varios segun equal-widht
    breakpoints = c(0,1e-64,seq(max(data_col)/numberOfBreaks, max(data_col), by = max(data_col)/numberOfBreaks))
    cuts = cut(data_col, breaks = breakpoints, include.lowest = TRUE, dig.lab = 6)
    
    # Cambiamos el estilo de los intervarlos para que pasen a estar separados por ';' en vez de ','
    levels(cuts) = unlist(lapply(levels(cuts), FUN = function(y){gsub(x=y, ",",replacement = ";")}))
    if(nrow(discretized_data) == 0){
      discretized_data = data.frame(cuts)
    } else{
      discretized_data = cbind(discretized_data, cuts)
    }
    colnames(discretized_data)[att_col] = names(data[att_col])
  }
  
  return(discretized_data)
}
