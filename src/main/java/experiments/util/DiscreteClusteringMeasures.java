package experiments.util;

import eu.amidst.core.utils.CompoundVector;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.util.Tuple;
import voltric.util.distance.Hellinger;
import voltric.variables.DiscreteVariable;

import java.util.*;

// Esto se puede ampliar para HLCMs facilmente. Considerando multiples LVs e iterando siempre por sus hijos en la particion o fuera de ella
public class DiscreteClusteringMeasures {

    public static List<List<Tuple<DiscreteVariable, Double>>> mostRelevantVarsInLcmPerCluster(DiscreteBayesNet lcm) {
        DiscreteVariable clusterVar = lcm.getLatentVariables().get(0);
        List<List<Tuple<DiscreteVariable, Double>>> mostRelevantVarsPerCluster = new ArrayList<>(clusterVar.getCardinality());
        for(int i = 0; i < clusterVar.getCardinality(); i++) {
            List<Tuple<DiscreteVariable, Double>> mostRelevantVars = mostRelevantVarsInCluster(lcm, i);
            mostRelevantVarsPerCluster.add(mostRelevantVars);
        }
        return mostRelevantVarsPerCluster;
    }

    public static List<Tuple<DiscreteVariable, Double>> mostRelevantVarsInCluster(DiscreteBayesNet lcm, int cluster) {

        if(lcm.getLatentVariables().size() != 1)
            throw new IllegalArgumentException("The BN has to be an LCM");

        DiscreteVariable clusterVar = lcm.getLatentVariables().get(0);
        DiscreteBeliefNode clusterNode = lcm.getNode(clusterVar);
        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(lcm);
        List<Tuple<DiscreteVariable, Double>> mostRelevantVars = new ArrayList<>(lcm.getManifestVariables().size());

        /* Iteramos por los nodos hijos de la variable de cluster */
        Map<DiscreteVariable, Integer> evidence = new HashMap<>();

        for(DiscreteVariable var: lcm.getManifestVariables()) {

            DiscreteBeliefNode nodeVar = lcm.getNode(var);

            if(nodeVar.getParentNodes().contains(clusterNode)) {

                /* Obtain the marginal distribution */
                inferenceEngine.clearEvidence();
                inferenceEngine.propagate();
                Function nodeMarginalDist = inferenceEngine.computeBelief(var);

                /* Estimate the conditional distribution */
                evidence.put(clusterVar, cluster);
                inferenceEngine.setEvidence(evidence);
                inferenceEngine.propagate();
                Function nodeConditionalDist = inferenceEngine.computeBelief(var);

                /* Estimate the Hellinger distance between them */
                double dist = Hellinger.distance(nodeMarginalDist, nodeConditionalDist);
                mostRelevantVars.add(new Tuple<>(var, dist));
            }
        }
        mostRelevantVars.sort(new TupleComparator());
        return mostRelevantVars;
    }

    private static class TupleComparator implements Comparator<Tuple<DiscreteVariable, Double>> {

        @Override
        public int compare(Tuple<DiscreteVariable, Double> o1, Tuple<DiscreteVariable, Double> o2) {
            if(o1.getSecond() > o2.getSecond())
                return -1;
            else if(o1.getSecond() < o2.getSecond())
                return 1;
            else
                return 0;
        }
    }
}
