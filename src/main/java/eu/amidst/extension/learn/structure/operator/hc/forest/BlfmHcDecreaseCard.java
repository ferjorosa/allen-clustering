package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.GraphUtilsAmidst;

import java.util.HashSet;

public class BlfmHcDecreaseCard implements BlfmHcOperator {

    private int minCardinality;

    public BlfmHcDecreaseCard(int minCardinality) {
        this.minCardinality = minCardinality;
    }

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Variable bestVariable = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iteramos por el conjunto de variables latentes */
        for(Variable variable: copyVariables){

            if(variable.getAttribute() == null && variable.getNumberOfStates() > this.minCardinality) {

                /* Decrementamos la cardinalidad de la variable */
                int newCardinality = variable.getNumberOfStates() - 1;
                variable.setNumberOfStates(newCardinality);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality));

                /* Creamos un nuevo Plateau para el aprendizaje donde omitimos copiar la variable en cuestion y sus hijos */
                HashSet<Variable> omittedVariables = new HashSet<>();
                omittedVariables.add(variable);
                omittedVariables.addAll(GraphUtilsAmidst.getChildren(variable, copyDAG));
                PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                /* Aprendemos el modelo de forma local */
                VBEM_Local vbem_local = new VBEM_Local();
                vbem_local.learnModel(copyPlateauStructure, copyDAG, variable);

                /* Comparamos el modelo generado con el mejor modelo actual */
                if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                    bestModel = vbem_local.getPlateuStructure();
                    bestModelScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                    bestVariable = variable;
                }

                /* Incrementamos la cardinalidad de la variable para poder resetear el proceso */
                variable.setNumberOfStates(newCardinality + 1);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality + 1));
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Decrementamos la cardinalidad de la mejor variable para que no salte una IndexOutOfBoundsException en VMP.runInference
            bestVariable.setNumberOfStates(bestVariable.getNumberOfStates() - 1);
            bestVariable.setStateSpaceType(new FiniteStateSpace(bestVariable.getNumberOfStates()));

            if(globalVBEM) {
                VBEM_Global vbem_hc = new VBEM_Global();
                vbem_hc.learnModel(bestModel, copyDAG);

                bestModel = vbem_hc.getPlateuStructure();
                bestModelScore = vbem_hc.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "DecreaseCard");
    }
}
