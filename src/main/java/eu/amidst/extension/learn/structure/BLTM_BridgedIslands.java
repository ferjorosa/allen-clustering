package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.attribute.grouping.IslandFinder;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.mi.MutualInformation;
import voltric.util.Tuple;

import java.util.*;

// TODO: El debug deberia de mostrar el resultado del IslandFinder, el CL-Tree y el refinement
// TODO: No tiene sentido que tarde lo que tarda, seguramente requiere una revision con respecto a la implementacion
public class BLTM_BridgedIslands {

    private int maxIslandCardinality;

    private int n_neighbors_mi;

    private DistanceFunction distanceFunction_mi;

    private boolean gaussianNoise_mi;

    private long gaussianNoiseSeed;

    private boolean normalizedMI;

    public BLTM_BridgedIslands(int maxIslandCardinality, int n_neighbors_mi, DistanceFunction distanceFunction_mi, boolean gaussianNoise_mi, long gaussianNoiseSeed, boolean normalizedMI) {
        this.maxIslandCardinality = maxIslandCardinality;
        this.n_neighbors_mi = n_neighbors_mi;
        this.distanceFunction_mi = distanceFunction_mi;
        this.gaussianNoise_mi = gaussianNoise_mi;
        this.gaussianNoiseSeed = gaussianNoiseSeed;
        this.normalizedMI = normalizedMI;
    }

    public Result learnModel(DataOnMemory<DataInstance> data, boolean randomChowLiuRoot, Map<String, double[]> priors, boolean debug) {

        /* Estimate sibling clusters */
        double[][] mis = MutualInformation.estimate(data, this.n_neighbors_mi, this.distanceFunction_mi, this.gaussianNoise_mi, this.gaussianNoiseSeed, this.normalizedMI);
        IslandFinder islandFinder = new IslandFinder(this.maxIslandCardinality);
        DAG islandsDAG = islandFinder.generate(data, mis, priors);

        LogUtils.printf("Island Finder result: ", debug);
        LogUtils.printf(islandsDAG.toString(), debug);

        /* Learn the unconnected LTM */
        VBEM vbem = new VBEM();
        vbem.learnModel(data, islandsDAG, priors);

        /* Complete the LVs data and estimate their Mutual Information values to form a tree using the Chow-Liu algorithm */
        DataOnMemory<DataInstance> completeData = DataUtils.completeDiscreteLatent(data, vbem.getLearntBayesianNetwork());
        double[][] latentVarsMIs = MutualInformation.estimate(completeData, this.n_neighbors_mi, this.distanceFunction_mi, this.gaussianNoise_mi, this.gaussianNoiseSeed, this.normalizedMI);
        List<Variable> discreteLatentVars = new ArrayList<>();
        for(Variable variable: islandsDAG.getVariables()) {
            if(!variable.isObservable() && variable.isMultinomial())
                discreteLatentVars.add(variable);
        }
        DAG chowLiuTree = ChowLiuAlgorithm.estimateTree(latentVarsMIs, new Variables(discreteLatentVars), randomChowLiuRoot);

        /* Form the LTM structure by connecting the islands using the Chow-Liu tree */
        for(Variable discreteLatentVar: discreteLatentVars) {
            List<Variable> parentsInChowLiuTree = chowLiuTree.getParentSet(discreteLatentVar).getParents(); // It is a tree, there should only be one
            for(Variable parent: parentsInChowLiuTree)
                islandsDAG.getParentSet(discreteLatentVar).addParent(parent);
        }

        /* Learn the LTM parameters */
        Result learnedLTM = learnLtmParameters(data, islandsDAG, priors);

        LogUtils.printf("\nChow-Liu algorithm connecting the islands", debug);
        LogUtils.printf("Resulting LTM score: " + learnedLTM.getElbo(), debug);

        /*
        * Model refinement:
        * - Increase the cardinality of each LV if beneficial for the model
        * - Relocate observed nodes if they have a higher MI with other LV
        */
        LogUtils.printf("\nModel refinement:", debug);
        Result resultOfRefiningCardinality = refineCardinality(learnedLTM);
        LogUtils.printf("After refine cardinality -> " + resultOfRefiningCardinality.getElbo() + "  (Old = " +learnedLTM.getElbo()+")", debug);

        Result resultOfRelocatingNodes = relocateNodes(resultOfRefiningCardinality, data, priors);
        LogUtils.printf("After relocate nodes -> " + resultOfRelocatingNodes.getElbo() + "  (Old = " +resultOfRefiningCardinality.getElbo()+")", debug);

        return resultOfRelocatingNodes;
    }

    private Result learnLtmParameters(DataOnMemory<DataInstance> data, DAG dag, Map<String, double[]> priors) {

        VBEM vbem = new VBEM();
        vbem.learnModel(data, dag, priors);

        return new Result(vbem.getPlateuStructure(), vbem.getPlateuStructure().getLogProbabilityOfEvidence(), dag, "LTM");
    }

    private Result refineCardinality(Result result) {

        PlateuStructure currentPlateau = result.getPlateuStructure();
        DAG currentDAG = result.getDag();
        double currentScore = result.getElbo();

        Variables copyVariables = currentDAG.getVariables().deepCopy();
        DAG copyDAG = currentDAG.deepCopy(copyVariables);

        /* Iteramos por el conjunto de variables latentes */
        for(Variable variable: copyVariables) {

            if (variable.getAttribute() == null && variable.getNumberOfStates() < this.maxIslandCardinality) {
                /* Incrementamos la cardinalidad de la variable */
                int newCardinality = variable.getNumberOfStates() + 1;
                variable.setNumberOfStates(newCardinality);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality));

                /* Creamos un nuevo Plateau para el aprendizaje donde omitimos copiar la variable en cuestion y sus hijos */
                HashSet<Variable> omittedVariables = new HashSet<>();
                omittedVariables.add(variable);
                omittedVariables.addAll(GraphUtilsAmidst.getChildren(variable, copyDAG));
                PlateuStructure copyPlateauStructure = currentPlateau.deepCopy(copyDAG, omittedVariables);

                /* Aprendemos el modelo de forma local */
                VBEM_Local vbem_local = new VBEM_Local();
                vbem_local.learnModel(copyPlateauStructure, copyDAG, variable);

                /* Comparamos el modelo generado con el mejor modelo actual */
                if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > currentScore) {
                    currentPlateau = vbem_local.getPlateuStructure();
                    currentScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                } else {
                    /* Decrementamos la cardinalidad de la variable */
                    variable.setNumberOfStates(newCardinality - 1);
                    variable.setStateSpaceType(new FiniteStateSpace(newCardinality - 1));
                }
            }
        }
        return new Result(currentPlateau, currentScore, copyDAG, "RefineCardinality");
    }

    private Result relocateNodes(Result result, DataOnMemory<DataInstance> data, Map<String, double[]> priors) {

        /* Obtenemos la posterior predictive distribution */
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        Variables variables = posteriorPredictive.getVariables();
        DAG dag = posteriorPredictive.getDAG();

        /* Completamos los datos */
        DataOnMemory<DataInstance> latentVarData = DataUtils.completeDiscreteLatent(data, posteriorPredictive);

        /* Calculamos la MI de cada variable observada con cada variable latente y la emparejamos con aquella que devuelve mayor MI */
        int N = data.getNumberOfDataInstances();
        int M_i = data.getAttributes().getNumberOfAttributes();
        int M_j = latentVarData.getAttributes().getNumberOfAttributes();
        double[][] dataForMi = new double[N][2];
        Map<Variable, Tuple<Variable, Double>> pairs = new LinkedHashMap<>();
        for (int i = 0; i < M_i; i++) {
            double bestMI = 0;
            for (int j = 0; j < M_j; j++) {
                /* Generate data */
                for(int instIndex = 0; instIndex < N; instIndex++) {
                    dataForMi[instIndex][0] = data.getDataInstance(instIndex).toArray()[i];
                    dataForMi[instIndex][1] = latentVarData.getDataInstance(instIndex).toArray()[j];
                }
                /* Estimate MI */
                Attribute attribute_i = data.getAttributes().getFullListOfAttributes().get(i);
                Attribute attribute_j = latentVarData.getAttributes().getFullListOfAttributes().get(j);
                double mi = MutualInformation.estimate(dataForMi, attribute_i, attribute_j, this.n_neighbors_mi, this.distanceFunction_mi, this.gaussianNoise_mi, this.gaussianNoiseSeed, this.normalizedMI);

                if(mi > bestMI) {
                    bestMI = mi;
                    pairs.put(variables.getVariableByName(attribute_i.getName()), new Tuple<>(variables.getVariableByName(attribute_j.getName()), mi));
                }
            }
        }

        /* Iteramos por los pairs. En caso de que no coincida la variable de mayor MI con su padre latente, la recolocamos con ella */
        for(Variable variable: pairs.keySet()) {
            Variable bestMiLatent = pairs.get(variable).getFirst();
            Variable currentLatentParent = dag.getParentSet(variable).getParents().get(0); // It is a tree. It should only have one parent

            if(!bestMiLatent.equals(currentLatentParent)) {
                dag.getParentSet(variable).addParent(bestMiLatent);
                dag.getParentSet(variable).removeParent(currentLatentParent);
            }
        }

        /* Eliminamos aquellas variables latentes que no tengan hijos o cuyo unico hijo sea una variable latente discreta */
        List<Variable> latentVarsWithoutChildren = new ArrayList<>();
        for(Variable variable: variables) {
            if(!variable.isObservable()) {
                List<Variable> children = GraphUtilsAmidst.getChildren(variable, dag);
                if (children.size() == 0)
                    latentVarsWithoutChildren.add(variable);
                else if(children.size() == 1 && children.get(0).isDiscrete() && !children.get(0).isObservable()) {
                    dag.getParentSet(children.get(0)).removeParent(variable);
                    latentVarsWithoutChildren.add(variable);
                }
            }
        }

        for(Variable var: latentVarsWithoutChildren) {
            variables.remove(var);
            dag.removeVariable(var);
        }

        /* Aprendemos los parametros con VBEM y devolvemos el modelo */
        return learnLtmParameters(data, dag, priors);
    }
}
