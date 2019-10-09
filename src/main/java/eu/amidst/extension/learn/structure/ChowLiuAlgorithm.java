package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.mi.MutualInformation;
import voltric.graph.AbstractNode;
import voltric.graph.UndirectedNode;
import voltric.graph.weighted.WeightedUndirectedGraph;
import voltric.util.CollectionUtils;
import voltric.util.Tuple;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/** TODO: Puede saltar una excepcion en el caso de datos hibridos ya que no se permite un padre continuo de una variable discreta */
public class ChowLiuAlgorithm {

    public static DAG estimateTree(double[][] miValues,
                                   Variables variables,
                                   boolean randomRoot) {
        if(miValues.length != variables.getNumberOfVars())
            throw new IllegalArgumentException("The number of variables and the dimension of the matrix must coincide");

        /* Introducimos los valores de la matriz en un Map y lo ordenamos */
        Map<Tuple<Integer, Integer>, Double> miValuesMap = new HashMap<>();

        for(int i = 0; i < miValues.length; i++)
            for(int j = i + 1 ; j < miValues.length; j++)
                miValuesMap.put(new Tuple<>(i,j), miValues[i][j]);

        LinkedHashMap<Tuple<Integer, Integer>, Double> sortedMiValuesMap = CollectionUtils.sortByDescendingValue(miValuesMap);

        /*
        * Generar el Maximum weight spanning tree (MWST). Para ello añadimos los (n-1) arcos que mejor score tienen y que
        * no generan ciclos.
        */
        WeightedUndirectedGraph<Variable> mwst = new WeightedUndirectedGraph<>();
        for(Variable variable: variables)
            mwst.addNode(variable);

        for(int i=0; i < variables.getNumberOfVars() - 1; i++)
            for(Tuple<Integer, Integer> key: sortedMiValuesMap.keySet()) {
                UndirectedNode<Variable> firstNode = mwst.getNode(variables.getListOfVariables().get(key.getFirst()));
                UndirectedNode<Variable> secondNode = mwst.getNode(variables.getListOfVariables().get(key.getSecond()));
                if(!mwst.containsPath(firstNode, secondNode))
                    mwst.addEdge(secondNode, firstNode, sortedMiValuesMap.get(key));
            }

        /* Generamos un DAG a partir del MWST */
        LinkedHashMap<AbstractNode<Variable>, Integer> visitedNodes = new LinkedHashMap<>();
        if(randomRoot) {
            Random random = new Random();
            int randomNodeIndex = random.nextInt(mwst.getNumberOfNodes());
            mwst.dfs(mwst.getNodes().get(randomNodeIndex), visitedNodes);
        } else
            mwst.dfs(mwst.getNodes().get(0), visitedNodes);

        DAG dag = new DAG(variables);
        AbstractNode<Variable> previousNode = null;

        for(AbstractNode<Variable> node: visitedNodes.keySet()) {
            if(previousNode != null) {
                dag.getParentSet(node.getContent()).addParent(previousNode.getContent());
                previousNode = node;
            } else
                previousNode = node;
        }

        return dag;
    }

    public static Result learnModel(DataOnMemory<DataInstance> data,
                                    boolean randomRoot,
                                    int n_neighbors_mi,
                                    DistanceFunction distanceFunction_mi,
                                    boolean gaussianNoise_mi,
                                    long gaussianNoiseSeed_mi,
                                    boolean normalizedMI) {

        /* Aprendemos el arbol (DAG) con las variables del dataSet */

        double[][] miValues = MutualInformation.estimate(data, n_neighbors_mi, distanceFunction_mi, gaussianNoise_mi, gaussianNoiseSeed_mi, normalizedMI);
        Variables variables = new Variables(data.getAttributes());
        DAG chowLiuTree = estimateTree(miValues, variables, randomRoot);

        /* Estimamos sus parametros con el VB y el ELBO */
        return learnParameters(data, chowLiuTree);
    }

    private static Result learnParameters(DataOnMemory<DataInstance> data,
                                          DAG dag) {

        VBEM vbem = new VBEM();
        double score = vbem.learnModel(data, dag);

        return new Result(vbem.getPlateuStructure(), score, dag, "LTM");
    }
}
