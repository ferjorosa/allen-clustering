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

public class generate_prob_cluster_lcm {

    public static void main(String[] args) throws Exception {

        /* Cargamos los datos porque nos sirven para iterar con el mismo orden por las variables de los modelos */
        DiscreteData exon_100_transformed = DataFileLoader.loadDiscreteData("data/discrete/exon_100_disc_transformed.arff");
        List<DiscreteVariable> attributes = exon_100_transformed.getVariables();

        /* Cargamos la red */
        List<String> latentVarNames = new ArrayList<>();
        latentVarNames.add("clustVar");
        DiscreteBayesNet bn = XmlBifReader.processFile(new File("models/discrete/exact/transformed/lcm_100_transformed.xml"), latentVarNames);

        List<String> attributeStates = new ArrayList<>(4);
        attributeStates.add("absent");
        attributeStates.add("low");
        attributeStates.add("medium");
        attributeStates.add("high");

        /* Exportamos las marginales */
        DiscreteVariable clustVar = bn.getLatentVariable("clustVar");
        double[][] marginals = new double[attributes.size()][4];
        CliqueTreePropagation ctp = new CliqueTreePropagation(bn);
        ctp.propagate();
        for(int i = 0; i < attributes.size(); i++) {
            DiscreteVariable attribute = attributes.get(i);
            Function marginal = ctp.computeBelief(bn.getNode(attribute.getName()).getVariable());
            marginals[i] = marginal.getCells();
        }
        exportCSV(marginals, attributeStates, "marginals.csv");

        /* Exportamos las CPTs. Iteramos por los atributos y proyectamos su CPT segun el estado de la variable de clustering */
        for(int clust = 0; clust < clustVar.getCardinality(); clust++) {
            double[][] cpts = new double[attributes.size()][4];
            for (int i = 0; i < attributes.size(); i++) {
                DiscreteVariable attribute = attributes.get(i);
                double[] cpt = bn.getNode(attribute.getName()).getCpt().project(clustVar, clust).getCells();
                cpts[i] = cpt;
            }
            exportCSV(cpts, attributeStates, "cpts_"+(clust+1)+".csv");
        }
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
