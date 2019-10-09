package experiments.util;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.distribution.Normal_MultinomialParents;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.ParentSet;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import org.latlab.model.*;
import org.latlab.util.DiscreteVariable;
import org.latlab.util.Function;
import org.latlab.util.SingularContinuousVariable;

import java.util.List;
import java.util.stream.Collectors;

/** NOTA: Solo funciona con arboles sin nodos observados internos */
public class OldAmidstToLatlabModel {

    // TODO: Esta pensado para arboles
    public static Gltm transform(BayesianNetwork amidstLTM) {

        List<Variable> manifestVariables = amidstLTM.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() != null).collect(Collectors.toList());

        List<Variable> latentVariables = amidstLTM.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() == null).collect(Collectors.toList());

        Gltm latLabModel = new Gltm(amidstLTM.getName());

        /* Añadimos las variables manifest al modelo */
        for(Variable variable: manifestVariables)
            latLabModel.addNode(new SingularContinuousVariable(variable.getName()));

        /* Añadimos las variables latentes al modelo */
        for(Variable variable: latentVariables) {
            FiniteStateSpace type = variable.getStateSpaceType();
            latLabModel.addNode(new DiscreteVariable(variable.getName(), type.getStatesNames()));
        }

        /* Añadimos los arcos */
        for(ParentSet parentSet: amidstLTM.getDAG().getParentSets()) {
            List<String> parentNames = parentSet.getParents().stream().map(x -> x.getName()).collect(Collectors.toList());
            String mainVarName = parentSet.getMainVar().getName();

            for(String parentName: parentNames) {
                BeliefNode parentNode = latLabModel.getNode(parentName);
                BeliefNode mainNode = latLabModel.getNode(mainVarName);
                latLabModel.addEdge(mainNode, parentNode);
            }
        }

        /* Modificamos los potenciales segun sus valores en la red de AMIDST*/
        for(ConditionalDistribution dist: amidstLTM.getConditionalDistributions()){

            Variable mainVar = dist.getVariable();

            /* La variable es discreta */
            if(mainVar.getStateSpaceTypeEnum() == StateSpaceTypeEnum.FINITE_SET) {

                DiscreteBeliefNode latlabNode = (DiscreteBeliefNode) latLabModel.getNode(dist.getVariable().getName());
                Function cpt = latlabNode.potential();
                int[] indices; // No es mas que un vector de dimension 1 con los estados para computeIndex

                /* No tiene padres */
                if(dist.getConditioningVariables().size() == 0 ) {
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
            } else if(mainVar.getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL) {

                ContinuousBeliefNode latlabNode = (ContinuousBeliefNode) latLabModel.getNode(mainVar.getName());

                /* No tiene padres*/
                if(dist.getConditioningVariables().size() == 0 ) {

                    DenseDoubleMatrix1D mean = new DenseDoubleMatrix1D(1);
                    mean.set(0, ((Normal) dist).getMean());
                    DenseDoubleMatrix2D variance = new DenseDoubleMatrix2D(1, 1);
                    variance.set(0,0, ((Normal) dist).getVariance());
                    CGParameter cgParameter = new CGParameter(1.0, mean, variance);
                    latlabNode.potential().set(0, cgParameter);

                /* Tiene un padre (solo funciona con arboles) */
                } else {

                    Variable parentVar = amidstLTM.getDAG().getParentSet(mainVar).getParents().get(0);
                    Normal_MultinomialParents mainVarDist = (Normal_MultinomialParents) dist;

                    for(int i = 0; i < parentVar.getNumberOfStates(); i++){
                        Normal normal = mainVarDist.getNormal(i);
                        DenseDoubleMatrix1D mean = new DenseDoubleMatrix1D(1);
                        mean.set(0, normal.getMean());
                        DenseDoubleMatrix2D variance = new DenseDoubleMatrix2D(1, 1);
                        variance.set(0,0, normal.getVariance());
                        CGParameter cgParameter = new CGParameter(1.0, mean, variance);
                        latlabNode.potential().set(i, cgParameter);
                    }

                }
            }
        }

        return latLabModel;
    }

    public static Gltm transform(BayesianNetwork amidstLTM, List<org.latlab.util.Variable> manifestVariables) {

        List<Variable> latentVariables = amidstLTM.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() == null).collect(Collectors.toList());

        Gltm latLabModel = new Gltm(amidstLTM.getName());

        /* Añadimos las variables manifest al modelo */
        for(org.latlab.util.Variable variable: manifestVariables)
            latLabModel.addNode(variable);

        /* Añadimos las variables latentes al modelo */
        for(Variable variable: latentVariables) {
            FiniteStateSpace type = variable.getStateSpaceType();
            latLabModel.addNode(new DiscreteVariable(variable.getName(), type.getStatesNames()));
        }

        /* Añadimos los arcos */
        for(ParentSet parentSet: amidstLTM.getDAG().getParentSets()) {
            List<String> parentNames = parentSet.getParents().stream().map(x -> x.getName()).collect(Collectors.toList());
            String mainVarName = parentSet.getMainVar().getName();

            for(String parentName: parentNames) {
                BeliefNode parentNode = latLabModel.getNode(parentName);
                BeliefNode mainNode = latLabModel.getNode(mainVarName);
                latLabModel.addEdge(mainNode, parentNode);
            }
        }

        /* Modificamos los potenciales segun sus valores en la red de AMIDST*/
        for(ConditionalDistribution dist: amidstLTM.getConditionalDistributions()){

            Variable mainVar = dist.getVariable();

            /* La variable es discreta */
            if(mainVar.getStateSpaceTypeEnum() == StateSpaceTypeEnum.FINITE_SET) {

                DiscreteBeliefNode latlabNode = (DiscreteBeliefNode) latLabModel.getNode(dist.getVariable().getName());
                Function cpt = latlabNode.potential();
                int[] indices; // No es mas que un vector de dimension 1 con los estados para computeIndex

                /* No tiene padres */
                if(dist.getConditioningVariables().size() == 0 ) {
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
            } else if(mainVar.getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL) {

                ContinuousBeliefNode latlabNode = (ContinuousBeliefNode) latLabModel.getNode(mainVar.getName());

                /* No tiene padres*/
                if(dist.getConditioningVariables().size() == 0 ) {

                    DenseDoubleMatrix1D mean = new DenseDoubleMatrix1D(1);
                    mean.set(0, ((Normal) dist).getMean());
                    DenseDoubleMatrix2D variance = new DenseDoubleMatrix2D(1, 1);
                    variance.set(0,0, ((Normal) dist).getVariance());
                    CGParameter cgParameter = new CGParameter(1.0, mean, variance);
                    latlabNode.potential().set(0, cgParameter);

                    /* Tiene un padre (solo funciona con arboles) */
                } else {

                    Variable parentVar = amidstLTM.getDAG().getParentSet(mainVar).getParents().get(0);
                    Normal_MultinomialParents mainVarDist = (Normal_MultinomialParents) dist;

                    for(int i = 0; i < parentVar.getNumberOfStates(); i++){
                        Normal normal = mainVarDist.getNormal(i);
                        DenseDoubleMatrix1D mean = new DenseDoubleMatrix1D(1);
                        mean.set(0, normal.getMean());
                        DenseDoubleMatrix2D variance = new DenseDoubleMatrix2D(1, 1);
                        variance.set(0,0, normal.getVariance());
                        CGParameter cgParameter = new CGParameter(1.0, mean, variance);
                        latlabNode.potential().set(i, cgParameter);
                    }

                }
            }
        }

        return latLabModel;
    }
}
