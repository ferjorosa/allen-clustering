package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.GraphUtilsAmidst;
import voltric.util.Tuple;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * It has the same behaviour as its tree equivalent
 */
public class BlfmHcRemoveNode implements BlfmHcOperator {

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestLatentPair = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Obtenemos el conjunto de nodos latentes */
        List<Variable> latentVariables = copyVariables.getListOfVariables().stream()
                .filter(var->var.getAttribute() == null)
                .collect(Collectors.toList());

        /* Seleccionamos aquellos nodos latentes que tengan un padre latente y generamos una Tupla */
        List<Tuple<Variable, Variable>> latentVarsWithLatentParent = new ArrayList<>();
        for(Variable latentVariable: latentVariables){

            Optional<Variable> latentParent = copyDAG.getParentSet(latentVariable).getParents().stream()
                    .filter(latentVariables::contains)
                    .findFirst();

            if(latentParent.isPresent())
                latentVarsWithLatentParent.add(new Tuple<>(latentVariable, latentParent.get()));
        }

        /* Iteramos por los pares de variables donde la variable a eliminar es la primera y el padre la segunda */
        for(Tuple<Variable, Variable> latentPair: latentVarsWithLatentParent) {

            /* Eliminamos la variable latente y añadimos sus hijos correspondientes al padre */
            Variable latentVariable = latentPair.getFirst();
            Variable latentParent = latentPair.getSecond();
            List<Variable> latentVarChildren = GraphUtilsAmidst.getChildren(latentVariable, copyDAG);

            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentVariable);
                copyDAG.getParentSet(child).addParent(latentParent);
            }

            copyDAG.removeVariable(latentVariable);
            copyVariables.remove(latentVariable);

            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la variable eliminada, sus hijas y su nuevo padre */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(latentVariable);
            omittedVariables.add(latentParent);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(latentParent, copyDAG));
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Aprendemos el modelo de forma local */
            VBEM_Local vbem_local = new VBEM_Local();
            vbem_local.learnModel(copyPlateauStructure, copyDAG, latentParent);

            /* Comparamos el modelo generado con el mejor modelo actual */
            if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                bestModel = vbem_local.getPlateuStructure();
                bestModelScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                bestLatentPair = latentPair;
            }

            /* Modificamos el grafo y volvemos a añadir el nodo con sus hijos para resetear el proceso */
            copyVariables.add(latentVariable);
            copyDAG.addVariable(latentVariable);

            copyDAG.getParentSet(latentVariable).addParent(latentParent);
            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentParent);
                copyDAG.getParentSet(child).addParent(latentVariable);
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Modificamos el grafo para que no haya diferencias con la estructura del Plateau
            Variable latentVariable = bestLatentPair.getFirst();
            Variable latentParent = bestLatentPair.getSecond();
            List<Variable> latentVarChildren = GraphUtilsAmidst.getChildren(latentVariable, copyDAG);

            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentVariable);
                copyDAG.getParentSet(child).addParent(latentParent);
            }

            copyDAG.removeVariable(latentVariable);
            copyVariables.remove(latentVariable);

            if(globalVBEM) {
                VBEM_Global vbem_hc = new VBEM_Global();
                vbem_hc.learnModel(bestModel, copyDAG);

                bestModel = vbem_hc.getPlateuStructure();
                bestModelScore = vbem_hc.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "RemoveNode");
    }
}
