package experiments.discrete;

import experiments.util.DiscreteClusteringMeasures;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class estimate_most_relevant_per_cluster {

    public static void main(String[] args) throws Exception {
        hellingerMostRelevant(10);
    }

    /** Estimamos las variables mas relevantes para cada uno de los clusters mediante la distancia de Helligner entre la CondProb y la marginal */
    private static void hellingerMostRelevant(int numberOfVars) throws Exception {
        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("clustVar");
        DiscreteBayesNet lcm = XmlBifReader.processFile(new File("models/discrete/exact/transformed/lcm_100_transformed.xml"), latentVarNames);
        List<List<Tuple<DiscreteVariable, Double>>> mostRelevantVarsPerCluster = DiscreteClusteringMeasures.mostRelevantVarsInLcmPerCluster(lcm);
        for(int cluster = 0; cluster < mostRelevantVarsPerCluster.size(); cluster++) {
            List<Tuple<DiscreteVariable, Double>> mostRelevantVarsInCluster = mostRelevantVarsPerCluster.get(cluster);
            System.out.println("\nCluster " + cluster + ":");
            for (int i = 0; i < numberOfVars; i++) {
                Tuple<DiscreteVariable, Double> variablePair = mostRelevantVarsInCluster.get(i);
                //System.out.println(variablePair.getFirst().getName() + " -> " + variablePair.getSecond());
                System.out.print(variablePair.getFirst().getName() + ", ");
            }
        }
    }
}
