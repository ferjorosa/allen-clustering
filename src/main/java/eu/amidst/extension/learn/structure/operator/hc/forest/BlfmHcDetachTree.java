package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.structure.Result;

import java.util.List;
import java.util.stream.Collectors;

/**
 *
 */
public class BlfmHcDetachTree implements BlfmHcOperator {

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Variable bestChild = null;
        Variable bestParent = null;
        Variable bestNewParent = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Obtenemos el conjunto de nodos latentes */
        List<Variable> latentVariables = copyVariables.getListOfVariables().stream()
                .filter(var->var.getAttribute() == null)
                .collect(Collectors.toList());

        /* Iteramos por todas aquellas que tengan un padre latente discreto */
        for(Variable latentVariable: latentVariables) {

            List<Variable> parents = copyDAG.getParentSet(latentVariable).getParents();



        }

        return null;
    }
}
