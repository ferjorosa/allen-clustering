package experiments.discrete;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.io.data.DataFileLoader;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/** Estima la probabilidad condicionada media de las variables observadas para cada cluster y exportamos un CSV */
public class generate_average_cond_probs {

    public static void main(String[] args) throws Exception {

        /* Cargamos el modelo */
        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("clustVar");
        DiscreteBayesNet lcm = XmlBifReader.processFile(new File("models/discrete/exact/transformed/lcm_100_transformed.xml"), latentVarNames);

        List<String> attributeStates = new ArrayList<>(4);
        attributeStates.add("absent");
        attributeStates.add("low");
        attributeStates.add("medium");
        attributeStates.add("high");

        DiscreteVariable clustVar = lcm.getLatentVariable("clustVar");
        double[][] meanValues = new double[clustVar.getCardinality() + 1][4];

        /* Obtenemos la marginal y estimamos el valor medio de los atributos */
        double[][] marginals = new double[lcm.getManifestVariables().size()][4];
        CliqueTreePropagation ctp = new CliqueTreePropagation(lcm);
        ctp.propagate();
        for(int i = 0; i < lcm.getManifestVariables().size(); i++) {
            DiscreteVariable attribute = lcm.getManifestVariables().get(i);
            Function marginal = ctp.computeBelief(lcm.getNode(attribute.getName()).getVariable());
            marginals[i] = marginal.getCells();
        }

        /* Calculamos el valor medio de los estados de las marginales y los almacenamos */
        double[] marginalMeanValues = new double[4];
        for(int j = 0; j < marginals[0].length; j++) {
            double sum = 0;
            for(int i = 1; i < marginals.length; i++)
                sum += marginals[i][j];
            marginalMeanValues[j] = sum / marginals.length;
        }
        meanValues[0] = marginalMeanValues;

        /* Por cada cluster, hacemos inferencia y obtenemos las CPTs de los atributos para luego calcular su valor medio */
        for(int clust = 0; clust < clustVar.getCardinality(); clust++) {
            double[][] cpts = new double[lcm.getManifestVariables().size()][4];
            for (int i = 0; i < lcm.getManifestVariables().size(); i++) {
                DiscreteVariable attribute = lcm.getManifestVariables().get(i);
                double[] cpt = lcm.getNode(attribute.getName()).getCpt().project(clustVar, clust).getCells();
                cpts[i] = cpt;
            }

            /* Calculamos el valor medio de los estados de las CPTs y los almacenamos */
            double[] clusterMeanValues = new double[4];
            for(int j = 0; j < cpts[0].length; j++) {
                double sum = 0;
                for(int i = 0; i < cpts.length; i++)
                    sum += cpts[i][j];
                clusterMeanValues[j] = sum / cpts.length;
            }
            meanValues[clust+1] = clusterMeanValues;
        }

        /* Exportamos en formato CSV */
        exportCSV(meanValues, attributeStates, "average_cond_probs.csv");
    }

    private static void exportCSV(double[][] matrix, List<String> columnNames, String filePath) throws IOException {

        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));

        /* Escribimos la primera linea que contiene los nombre de las columnas */
        for(int i = 0; i < columnNames.size(); i++)
            if(i != columnNames.size() -1)
                writer.print(columnNames.get(i) + ",");
            else
                writer.println(columnNames.get(i));

        /* Escribimos las matriz */
        for(int i = 0; i <matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                if(j == matrix[i].length - 1)
                    // The last column never ends with a comma
                    writer.println(matrix[i][j]);
                else
                    writer.print(matrix[i][j]+",");
            }
        }
        writer.close();
    }
}
