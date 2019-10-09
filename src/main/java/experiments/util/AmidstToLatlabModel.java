package experiments.util;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.distribution.ConditionalLinearGaussian;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.models.ParentSet;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.util.GraphUtilsAmidst;
import org.latlab.model.*;
import org.latlab.util.DiscreteVariable;
import org.latlab.util.Function;
import org.latlab.util.JointContinuousVariable;
import org.latlab.util.SingularContinuousVariable;

import java.util.*;
import java.util.stream.Collectors;

public class AmidstToLatlabModel {


    public static void transform(BayesianNetwork amidstGltm) {


        /* Creamos un Map con las variables continuas y sus nodos asociados (varNodesMap) */
        /* Creamos un Map con las variables y su indice en el nodo al que pertenecen (varIndexMap) */

    }

    private static void generateModelWithoutParams(DAG dag) {

        List<Variable> manifestVariables = dag.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() != null).collect(Collectors.toList());
        List<Variable> latentVariables = dag.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() == null).collect(Collectors.toList());

        List<Variable> possibleGaussianRoots = new ArrayList<>(); // Las MVs que pueden ser raices de una CLG
        List<Variable> variablesWithLatentParent = new ArrayList<>(); // Las variables cuyo padre es una LV

        for(Variable var: manifestVariables) {
            ParentSet parentSet = dag.getParentSet(var);
            if(parentSet.getNumberOfParents() == 0) {
                possibleGaussianRoots.add(var);
            } else {
                Variable parent = parentSet.getParents().get(0);
                if(parent.isDiscrete() && !parent.isObservable()) {
                    possibleGaussianRoots.add(var);
                    variablesWithLatentParent.add(var);
                }
            }
        }

        for(Variable var: latentVariables) {
            ParentSet parentSet = dag.getParentSet(var);
            if(parentSet.getNumberOfParents() > 0)
                variablesWithLatentParent.add(var);
        }

        /* Realizamos DFS desde las MVs seleccionadas para obtener sus descendientes que forman la CLG */
        Map<Variable, List<Variable>> mvWithDescendants = new LinkedHashMap<>();
        for(Variable var: possibleGaussianRoots) {
            Map<Variable, Integer> descendants = new LinkedHashMap<>();
            dag.dfs(var, descendants);
            mvWithDescendants.put(var, new ArrayList<>(descendants.keySet()));
        }

        /* Generamos el modelo en formato Latlab */
        Gltm latLabModel = new Gltm(dag.getName());

        /* Añadimos las variables manifest al modelo */
        Map<Variable, BeliefNode> varNodeMap = new HashMap<>();
        for(Variable variable: mvWithDescendants.keySet()){
            if(mvWithDescendants.get(variable).size() > 1){
                // Sort var with descendants
                List<Variable> varWithDescendants = mvWithDescendants.get(variable);
                varWithDescendants.sort(new VariableComparator());
                // Create a singular node for each of them
                List<SingularContinuousVariable> varWithDescendantsNodes = new ArrayList<>(varWithDescendants.size());
                for(Variable descendant: varWithDescendants)
                    varWithDescendantsNodes.add(new SingularContinuousVariable(descendant.getName()));
                // Create a joint node with them
                BeliefNode node = latLabModel.addNode(new JointContinuousVariable(varWithDescendantsNodes));
                varNodeMap.put(variable, node);
            } else {
                BeliefNode node = latLabModel.addNode(new SingularContinuousVariable(variable.getName()));
                varNodeMap.put(variable, node);
            }
        }

        /* Añadimos las variables latentes al modelo */
        for(Variable variable: latentVariables) {
            FiniteStateSpace type = variable.getStateSpaceType();
            BeliefNode node = latLabModel.addNode(new DiscreteVariable(variable.getName(), type.getStatesNames()));
            varNodeMap.put(variable, node);
        }

        /* Añadimos los arcos de las LVs a las LVs o de las LVs a las MVs, ya sea a un JointNode o a un SingularNode*/
        for(Variable variable: variablesWithLatentParent) {
            ParentSet parentSet = dag.getParentSet(variable);
            BeliefNode varNode = varNodeMap.get(variable);
            for(Variable parent: parentSet.getParents()) {
                BeliefNode parentNode = varNodeMap.get(parent);
                latLabModel.addEdge(varNode, parentNode);
            }
        }
    }

    private static void assignParameters(Gltm latlabModel,
                                         BayesianNetwork amidstModel,
                                         Map<String, ContinuousBeliefNode> varNodesMap,
                                         Map<Variable, Integer> varIndexMap) {

        List<Variable> sortedVariables = GraphUtilsAmidst.topologicalSort(amidstModel.getDAG());

        // Iteramos por las variables de la BN y obtenemos su distribucion
        for(Variable variable: amidstModel.getVariables()){

            ConditionalDistribution dist = amidstModel.getConditionalDistribution(variable);

            /* La variable es discreta */
            if(variable.getStateSpaceTypeEnum() == StateSpaceTypeEnum.FINITE_SET) {

                DiscreteBeliefNode latlabNode = (DiscreteBeliefNode) latlabModel.getNode(variable.getName());
                Function cpt = latlabNode.potential();
                int[] indices; // No es mas que un vector de dimension 1 con los estados para computeIndex

                /* No tiene padres */
                if(amidstModel.getDAG().getParentSet(variable).getNumberOfParents() == 0) {
                    indices = new int[1]; // Una sola variable
                    for (int i = 0; i < cpt.getDomainSize(); i++) {
                        indices[0] = i;
                        cpt.getCells()[cpt.computeIndex(indices)] = dist.getParameters()[i];
                    }

                /* Tiene un padre (solo funciona con arboles) */
                } else {
                    int[] _indices = new int[2];
                    int array_index = 0;
                    for(int i=0; i < cpt.getVariables().get(0).getCardinality(); i++) {
                        for (int j = 0; j < cpt.getVariables().get(1).getCardinality(); j++) {
                            _indices[0] = i;
                            _indices[1] = j;
                            int index = cpt.computeIndex(_indices);
                            cpt.getCells()[index] = dist.getParameters()[array_index++];
                        }
                    }
                }

            /* La variable es continua */
            } else if(variable.getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL) {

                ContinuousBeliefNode latlabNode = varNodesMap.get(variable.getName());

                /* No tiene padres */
                if(amidstModel.getDAG().getParentSet(variable).getNumberOfParents() == 0) {

                    /* Es univariante (Univariate Gaussian) */
                    if(latlabNode.getVariable().variables().size() == 1) {
                        DenseDoubleMatrix1D mean = new DenseDoubleMatrix1D(1);
                        mean.set(0, ((Normal) dist).getMean());
                        DenseDoubleMatrix2D variance = new DenseDoubleMatrix2D(1, 1);
                        variance.set(0, 0, ((Normal) dist).getVariance());
                        CGParameter cgParameter = new CGParameter(1.0, mean, variance);
                        latlabNode.potential().set(0, cgParameter);

                    /* Tiene hijos (Multivariate Gaussian [CLG]) */
                    } else {
                        // La variable se corresponde con la raiz de una CLG dist
                        Normal rootDist = (Normal) dist;
                        // Obtenemos sus CLGs que conforman el nodo Joint y por tanto la gaussiana multivariante
                        List<ConditionalLinearGaussian> childrenDists = new ArrayList<>();
                        for(SingularContinuousVariable nodeVar: latlabNode.getVariable().variables()){
                            if(!nodeVar.getName().equals(variable.getName())) {
                                Variable nodeAnidstVar = amidstModel.getVariables().getVariableByName(nodeVar.getName());
                                childrenDists.add(amidstModel.getConditionalDistribution(nodeAnidstVar));
                            }
                        }
                        // Estimamos la distribucion multivariante gaussiana
                        getMultivariateNormal(rootDist, childrenDists, sortedVariables, varIndexMap);
                    }

                /* Si tiene padres discretos (si es un padre gaussiano se ignora) */
                } else {
                    Variable parent = amidstModel.getDAG().getParentSet(variable).getParents().get(0);
                    if(parent.getStateSpaceTypeEnum() == StateSpaceTypeEnum.FINITE_SET) {

                    }
                }
            }

        }


    }

    // Le pasamos un Map porque si bien vamos tenemos el orden topologico de las variables,
    // al asignar sus parametros debemos hacerlo en el orden original
    private static void getMultivariateNormal(Normal rootDist,
                                              List<ConditionalLinearGaussian> childrenDists,
                                              List<Variable> sortedVariables,
                                              Map<Variable, Integer> variableIndexMap) {



    }


    private static void generateMultivariateNormal(List<ConditionalLinearGaussian> dists, Map<Variable, Integer> sortedVariablesIndices) {

        int dimension = dists.size();
        DenseDoubleMatrix1D meanVector = new DenseDoubleMatrix1D(dimension);
        DenseDoubleMatrix2D covarianceMatrix = new DenseDoubleMatrix2D(dimension, dimension);

        for(int i = 0; i < dimension; i++) {
            ConditionalLinearGaussian dist = dists.get(i);
            int variableIndex = sortedVariablesIndices.get(dist.getVariable());
            List<Variable> distParents = dist.getParents();

            /* Mean estimation */
            double mean = 0;
            for(Variable parent: distParents){
                double parentMean = meanVector.get(sortedVariablesIndices.get(parent));
                mean += dist.getCoeffForParent(parent) * parentMean;
            }
            mean += dist.getIntercept();
            meanVector.set(variableIndex, mean);

            /* Variance estimation */
            double variance = 0;
            for(Variable parent: distParents){
                double parentVariance = covarianceMatrix.get(sortedVariablesIndices.get(parent), sortedVariablesIndices.get(parent));
                variance += Math.pow(dist.getCoeffForParent(parent), 2) * parentVariance;
            }
            variance += dist.getVariance();
            covarianceMatrix.set(variableIndex, variableIndex, variance);

            /*
             * Covariance estimation.
             *
             * In order to estimate it, we have to first estimate the covariances of the variables with its parents and
             * then search through the set of variables that are before this variable in the "sortedNodes" list to see
             * if there is a directed path between them. If so, their covariance also needs to be estimated.
             *
             * See page 252 of Koller & Friedman (2009)
             */
            for(Variable parent: distParents){

                // Parent covariance estimation
                int parentIndex = sortedVariablesIndices.get(parent);
                double parentVariance = covarianceMatrix.get(parentIndex, parentIndex);
                double covariance = dist.getCoeffForParent(parent) * parentVariance;
                covarianceMatrix.set(parentIndex, variableIndex, covariance);
                covarianceMatrix.set(variableIndex, parentIndex, covariance);

                // Ancestors covariance estimation
                /*
                int parentIndexInSortedVariables = sortedVariables.indexOf(parent);

                List<Variable> ancestors = sortedVariables.stream()
                        .filter(x-> sortedVariables.indexOf(x) < parentIndexInSortedVariables)
                        .filter(x-> GraphUtilsAmidst.containsPath(x, parent, this.model.getDAG()))
                        .collect(Collectors.toList());

                for(Variable ancestor: ancestors){
                    int ancestorIndex = sortedVariablesIndices.get(ancestor);
                    double partialAncestorCovariance = dist.getCoeffForParent(parent) * covarianceMatrix.get(ancestorIndex, parentIndex);
                    double updatedPartialAncestorCovariance = covarianceMatrix.get(ancestorIndex, variableIndex);
                    updatedPartialAncestorCovariance += partialAncestorCovariance;

                    covarianceMatrix.set(ancestorIndex, variableIndex, updatedPartialAncestorCovariance);
                    covarianceMatrix.set(variableIndex, ancestorIndex, updatedPartialAncestorCovariance);
                }
                */
            }
        }
    }

    static class VariableComparator implements Comparator<Variable> {

        @Override
        public int compare(Variable o1, Variable o2) {
            if (o1.getAttribute().getIndex() < o2.getAttribute().getIndex())
                return -1;
            else if (o1.getAttribute().getIndex() > o2.getAttribute().getIndex())
                return 1;
            else
                return 0;
        }
    }
}
