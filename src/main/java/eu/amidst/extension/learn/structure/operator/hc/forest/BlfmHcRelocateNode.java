package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.GraphUtilsAmidst;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * El objetivo de este operador es la recolocacion de nodos pertenecientes a un tree. La recolocacion se produce dentro
 * del tree, se mueve el nodo de un subTree a otro.
 // TODO: He implementado la nueva version del LocalEM, donde se actualiza el nuevo padre y el antiguo*/
public class BlfmHcRelocateNode implements BlfmHcOperator {

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Variable bestChild = null;
        Variable bestParent = null;
        Variable bestNewParent = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Obtenemos el conjunto de nodos latentes discretos que sean raiz */
        List<Variable> rootVariables = copyVariables.getListOfVariables()
                .stream()
                .filter(x->x.getAttribute() == null
                        && x.isMultinomial()
                        && copyDAG.getParentSet(x).getNumberOfParents() == 0)
                .collect(Collectors.toList());

        /*
         * Cada una de estas variables es por tanto la raiz de un arbol. Seleccionamos pues las variables discretas
         * latentes hijas con mas de dos hijos observados
         */
        for(Variable root: rootVariables) {

            LinkedHashMap<Variable, Integer> treeNodes = new LinkedHashMap<>();
            copyDAG.dfs(root, treeNodes);

            List<Variable> latentVariables = treeNodes.keySet().stream()
                    .filter(var->var.getAttribute() == null)
                    .collect(Collectors.toList());

            /* Iteramos por el conjunto de nodos latentes y seleccionamos sus hijos y las otras variables latentes */
            for(Variable latentVariable: latentVariables) {

                List<Variable> otherLatentVariables = latentVariables.stream()
                        .filter(var -> !var.equals(latentVariable)).collect(Collectors.toList());

                List<Variable> observedChildren = GraphUtilsAmidst.getObservedChildren(latentVariable, copyDAG);

                /* Si el numero de hijos de esta variable es mayor que 2 y hay otras latentes, podriamos probar a trasladar uno a uno sus hijos */
                if(observedChildren.size() > 2 && otherLatentVariables.size() > 0) {

                    /* Iteramos por los hijos */
                    for (Variable child : observedChildren) {

                        /* Iteramos por las otras variables latentes y trasladamos la variable hija a su nueva particion */
                        for(Variable otherLatentVariable: otherLatentVariables) {

                            copyDAG.getParentSet(child).removeParent(latentVariable);
                            copyDAG.getParentSet(child).addParent(otherLatentVariable);

                            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos el nuevo padre y sus hijos */
                            HashSet<Variable> omittedVariables = new HashSet<>();
                            omittedVariables.add(otherLatentVariable);
                            omittedVariables.addAll(GraphUtilsAmidst.getChildren(otherLatentVariable, copyDAG));
                            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                            /* Aprendemos el modelo de forma local */
                            List<Variable> latentVariablesToUpdate = new ArrayList<>();
                            latentVariablesToUpdate.add(latentVariable);
                            latentVariablesToUpdate.add(otherLatentVariable);
                            VBEM_Local vbem_local = new VBEM_Local();
                            vbem_local.learnModel(copyPlateauStructure, copyDAG, latentVariablesToUpdate);

                            /* Comparamos el modelo generado con el mejor modelo actual */
                            if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                                bestModel = vbem_local.getPlateuStructure();
                                bestModelScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
                                bestChild = child;
                                bestParent = latentVariable;
                                bestNewParent = otherLatentVariable;
                            }

                            /* Modificamos el grafo y devolvemos la variable a su posicion inicial para resetear el proceso */
                            copyDAG.getParentSet(child).removeParent(otherLatentVariable);
                            copyDAG.getParentSet(child).addParent(latentVariable);
                        }
                    }
                }
            }

        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        if(bestModelScore > -Double.MAX_VALUE) {

            // Modificamos la estructura para que no haya diferencias con el PlateauStructure
            copyDAG.getParentSet(bestChild).removeParent(bestParent);
            copyDAG.getParentSet(bestChild).addParent(bestNewParent);

            if(globalVBEM) {
                VBEM_Global vbem_hc = new VBEM_Global();
                vbem_hc.learnModel(bestModel, copyDAG);

                bestModel = vbem_hc.getPlateuStructure();
                bestModelScore = vbem_hc.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "RelocateNode");
    }
}
