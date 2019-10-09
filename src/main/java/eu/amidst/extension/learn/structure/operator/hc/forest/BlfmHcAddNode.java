package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.GraphUtilsAmidst;
import org.apache.commons.math3.util.Combinations;
import voltric.util.Tuple;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Este operador selecciona dos nodos observados y crea una nueva variable latente como su padre. Posee un comportamiento
 * diferente dependiendo de si esas variables tienen padre o no (o las 2 tienen el mismo padre o ninguna tiene padre)
 *
 * Caso 1 - Si ambas tienen el mismo padre y dicho padre tiene mas de dos hijos, se crea una nueva latente como padre de las
 * dos variables observadas e hija de la antigua variable latente padre. Por defecto la cardinalidad del hijo es la del padre.
 *
 * Caso 2 - Si ninguna de las dos variables tienen un padre latente, se crea un LCM con ellas. Por defecto la cardinalidad es 2.
 *
 */
public class BlfmHcAddNode implements BlfmHcOperator {

    private int maxNumberOfLatentNodes;

    private int latentVarNameCounter = 0;

    public BlfmHcAddNode() {
        this.maxNumberOfLatentNodes = Integer.MAX_VALUE;
    }

    public BlfmHcAddNode(int maxNumberOfLatentNodes) {
        this.maxNumberOfLatentNodes = maxNumberOfLatentNodes;
    }

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestPair = null;
        Variable bestPairParent = null;
        String newLatentVarName = "";

        /* En caso de que el numero permitido de variables latentes sea ya maximo, devolvemos el modelo actual como resultado */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.getAttribute() == null).count();
        if(numberOfLatentNodes >= maxNumberOfLatentNodes)
            return new Result(bestModel, bestModelScore, dag, "AddNode");

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Caso 1 - Iteramos por todas aquellas variables latentes discretas con mas de dos hijos observados */
        List<Variable> discreteLatentVariables = copyVariables.getListOfVariables()
                .stream()
                .filter(x->x.getAttribute() == null
                        && x.isMultinomial()
                        && GraphUtilsAmidst.getObservedChildren(x, copyDAG).size() > 2)
                .collect(Collectors.toList());

        for(Variable variable: discreteLatentVariables) {

            /* Iteramos por los pares de variables observadas hijas de dicha variable latente */
            List<Variable> observedChildren = GraphUtilsAmidst.getObservedChildren(variable, copyDAG);
            List<Tuple<Variable, Variable>> childrenCombinations = generateVariableCombinations(observedChildren);

            for(Tuple<Variable, Variable> combination: childrenCombinations) {
                /*
                    - Creamos una nueva variable latente de cardinalidad igual a la variable latente padre.
                    - Ponemos como hijos de esta nueva variable latente el par de variables observadas seleccionado (modificando el grafo)
                 */
                Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), variable.getNumberOfStates());
                copyDAG.addVariable(newLatentVar);
                copyDAG.getParentSet(combination.getFirst()).removeParent(variable);
                copyDAG.getParentSet(combination.getSecond()).removeParent(variable);

                copyDAG.getParentSet(combination.getFirst()).addParent(newLatentVar);
                copyDAG.getParentSet(combination.getSecond()).addParent(newLatentVar);
                copyDAG.getParentSet(newLatentVar).addParent(variable);

                /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la nueva variable latente y sus hijos */
                HashSet<Variable> omittedVariables = new HashSet<>();
                omittedVariables.add(newLatentVar);
                omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
                PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                /* Aprendemos el modelo de forma local */
                VBEM_Local vbem_local = new VBEM_Local();
                vbem_local.learnModel(copyPlateauStructure, copyDAG, newLatentVar);

                /* Comparamos el modelo generado con el mejor modelo actual */
                if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                    bestModel = vbem_local.getPlateuStructure();
                    bestModelScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                    bestPair = combination;
                    bestPairParent = variable;
                    newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies between DAG and Plateau at the end
                }

                /* Modificamos el grafo y eliminamos el nuevo nodo latente para poder resetear el proceso */
                copyDAG.getParentSet(combination.getFirst()).removeParent(newLatentVar);
                copyDAG.getParentSet(combination.getSecond()).removeParent(newLatentVar);
                copyDAG.getParentSet(newLatentVar).removeParent(variable);
                copyDAG.removeVariable(newLatentVar);

                copyDAG.getParentSet(combination.getFirst()).addParent(variable);
                copyDAG.getParentSet(combination.getSecond()).addParent(variable);

                copyVariables.remove(newLatentVar);
            }

        }

        /* Caso 2 - Iteramos por los pares de variables observadas que no tengan padre latente */
        List<Variable> observedVariablesWithoutParent = copyVariables.getListOfVariables()
                .stream()
                .filter(x-> x.getAttribute() != null
                        && copyDAG.getParentSet(x).getNumberOfParents() == 0)
                .collect(Collectors.toList());

        List<Tuple<Variable, Variable>> observedVariableCombinations = generateVariableCombinations(observedVariablesWithoutParent);

        for(Tuple<Variable, Variable> combination: observedVariableCombinations) {

            /* Creamos una nueva variable latente de cardinalidad 2 y ponemos las variables de la combinacion como hijas */
            Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), 2);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(combination.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(combination.getSecond()).addParent(newLatentVar);

            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la nueva variable latente y sus hijos */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(newLatentVar);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Aprendemos el modelo de forma local */
            VBEM_Local vbem_local = new VBEM_Local();
            vbem_local.learnModel(copyPlateauStructure, copyDAG, newLatentVar);

            /* Comparamos el modelo generado con el mejor modelo actual */
            if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                bestModel = vbem_local.getPlateuStructure();
                bestModelScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                bestPair = combination;
                bestPairParent = null; // En este caso no hay padre de la nueva latente
                newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies between DAG and Plateau at the end
            }

            /* Modificamos el grafo y eliminamos el nuevo nodo latente para poder resetear el proceso */
            copyDAG.getParentSet(combination.getFirst()).removeParent(newLatentVar);
            copyDAG.getParentSet(combination.getSecond()).removeParent(newLatentVar);
            copyDAG.removeVariable(newLatentVar);
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            /* El mejor par de variables pertenece al caso 1 */
            if(bestPairParent != null) {
                // Modificamos el grafo para que no haya diferencias con la estructura del Plateau
                copyDAG.getParentSet(bestPair.getFirst()).removeParent(bestPairParent);
                copyDAG.getParentSet(bestPair.getSecond()).removeParent(bestPairParent);

                // To avoid name discrepancies between DAG and Plateau, we use the stored name
                Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, bestPairParent.getNumberOfStates());
                copyDAG.addVariable(newLatentVar);
                copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
                copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
                copyDAG.getParentSet(newLatentVar).addParent(bestPairParent);

            /* El mejor par de variables pertenece al caso 2 */
            } else {
                // To avoid name discrepancies between DAG and Plateau, we use the stored name
                Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, 2);
                copyDAG.addVariable(newLatentVar);
                copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
                copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
            }

            if(globalVBEM) {
                VBEM_Global vbem_hc = new VBEM_Global();
                vbem_hc.learnModel(bestModel, copyDAG);

                bestModel = vbem_hc.getPlateuStructure();
                bestModelScore = vbem_hc.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "AddNode");
    }

    private List<Tuple<Variable, Variable>> generateVariableCombinations(List<Variable> variables) {

        List<Tuple<Variable, Variable>> variableCombinations = new ArrayList<>();
        Iterator<int[]> variableIndexCombinations = new Combinations(variables.size(), 2).iterator();

        /* Iteramos por las combinaciones no repetidas de variables observadas y generamos una Tuple con cada una */
        while(variableIndexCombinations.hasNext()) {
            // Indices de los clusters a comparar
            int[] combination = variableIndexCombinations.next();
            // Añadimos la nueva tupla
            variableCombinations.add(new Tuple<>(variables.get(combination[0]), variables.get(combination[1])));
        }

        return variableCombinations;
    }
}
