package eu.amidst.extension.learn.structure.operator.incremental;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.Triple;
import voltric.util.Tuple;

import java.util.*;

public class BlfmIncAddArc implements BlfmIncOperator {

    private boolean allowObservedToObservedDiscreteArc;

    private boolean allowObservedToObservedArc;

    private boolean allowObservedToLatentArc;

    private VBEMConfig localVBEMConfig;

    public BlfmIncAddArc(boolean allowObservedToObservedArc,
                         boolean allowObservedToLatentArc,
                         boolean allowObservedToObservedDiscreteArc) {
        this(allowObservedToObservedArc, allowObservedToLatentArc, allowObservedToObservedDiscreteArc, new VBEMConfig());
    }

    public BlfmIncAddArc(boolean allowObservedToObservedArc,
                         boolean allowObservedToLatentArc,
                         boolean allowObservedToObservedDiscreteArc,
                         VBEMConfig localVBEMConfig) {
        this.allowObservedToObservedArc = allowObservedToObservedArc;
        this.allowObservedToLatentArc = allowObservedToLatentArc;
        this.allowObservedToObservedDiscreteArc = allowObservedToObservedDiscreteArc;
        this.localVBEMConfig = localVBEMConfig;
    }

    @Override
    public Triple<Variable, Variable, Result> apply(Set<Variable> currentSet, PlateuStructure plateuStructure, DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestPair = null;

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);
        Set<Variable> copyCurrentSet = new LinkedHashSet<>();
        for(Variable var: currentSet)
            copyCurrentSet.add(copyVariables.getVariableByName(var.getName()));

        /*
         * Iterate through all the combinations of variables in the currentSet. We consider all combinations becase we are
         * searching for the best directed arc.
         */
        for(Variable fromVar: copyCurrentSet) {
            for (Variable toVar : copyCurrentSet) {
                // We dont allow arcs between a variable and itself.
                // We dont also allow arcs from continuous to discrete.
                if (!fromVar.equals(toVar) && !(fromVar.isContinuous() && toVar.isDiscrete())) {
                    if ((!fromVar.isObservable() && !toVar.isObservable())                                           // LV -> LV
                            || (!fromVar.isObservable() && toVar.isObservable())                                     // LV -> OV
                            || (fromVar.isObservable() && !toVar.isObservable() && this.allowObservedToLatentArc)    // OV -> LV        [only if allowed]
                            || (fromVar.isObservable() && toVar.isObservable() && this.allowObservedToObservedArc))  // OV -> OV        [only if allowed]
                    {
                        if(fromVar.isObservable() && toVar.isObservable() && fromVar.isDiscrete() && toVar.isDiscrete() && !this.allowObservedToObservedDiscreteArc)
                            continue;

                        copyDAG.getParentSet(toVar).addParent(fromVar);

                        /* Create a new plateau by copying current one and omitting the var receiving the arc) */
                        HashSet<Variable> omittedVariables = new HashSet<>();
                        omittedVariables.add(toVar);
                        PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                        /* Aprendemos el modelo de forma local, actualizando ambas variables (con sus hijos) de forma local */
                        List<Variable> bothVariables = new ArrayList<>(2);
                        bothVariables.add(fromVar);
                        bothVariables.add(toVar);
                        VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                        localVBEM.learnModel(copyPlateauStructure, copyDAG, bothVariables);

                        /* Compare its score with current best model */
                        if (localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                            bestModel = localVBEM.getPlateuStructure();
                            bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                            bestPair = new Tuple<>(fromVar, toVar);
                        }

                        /* Remove the newly created arc to reset the process for the next pair */
                        copyDAG.getParentSet(toVar).removeParent(fromVar);
                    }
                }
            }
        }

        /* Modify the DAG with the best arc */
        if(bestModelScore > -Double.MAX_VALUE) {

            copyDAG.getParentSet(bestPair.getSecond()).addParent(bestPair.getFirst());

            return new Triple<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
        }

        return new Triple<>(null, null, new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
    }

    @Override
    public Triple<Variable, Variable, Result> apply(PriorityQueue<Triple<Variable, Variable, Double>> selectedTriples, PlateuStructure plateuStructure, DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestPair = null;

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iterate through the combinations of selected triples */
        for(Triple<Variable, Variable, Double> triple: selectedTriples){
            // Generate a 2-item list to iterate (in order to avoid repeated code)
            List<Variable> tupleList = new ArrayList<>(2);
            tupleList.add(triple.getFirst());
            tupleList.add(triple.getSecond());

            for(Variable fromVar: tupleList) {
                for (Variable toVar : tupleList) {
                    // We dont allow arcs between a variable and itself.
                    // We dont also allow arcs from continuous to discrete.
                    if (!fromVar.equals(toVar) && !(fromVar.isContinuous() && toVar.isDiscrete())) {
                        if ((!fromVar.isObservable() && !toVar.isObservable())                                           // LV -> LV
                                || (!fromVar.isObservable() && toVar.isObservable())                                     // LV -> OV
                                || (fromVar.isObservable() && !toVar.isObservable() && this.allowObservedToLatentArc)    // OV -> LV        [only if allowed]
                                || (fromVar.isObservable() && toVar.isObservable() && this.allowObservedToObservedArc))  // OV -> OV        [only if allowed]
                        {
                            if(fromVar.isObservable() && toVar.isObservable() && fromVar.isDiscrete() && toVar.isDiscrete() && !this.allowObservedToObservedDiscreteArc)
                                continue;

                            copyDAG.getParentSet(toVar).addParent(fromVar);

                            /* Create a new plateau by copying current one and omitting the var receiving the arc) */
                            HashSet<Variable> omittedVariables = new HashSet<>();
                            omittedVariables.add(toVar);
                            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                            /* Aprendemos el modelo de forma local, actualizando ambas variables (con sus hijos) de forma local */
                            List<Variable> bothVariables = new ArrayList<>(2);
                            bothVariables.add(fromVar);
                            bothVariables.add(toVar);
                            VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                            localVBEM.learnModel(copyPlateauStructure, copyDAG, bothVariables);

                            /* Compare its score with current best model */
                            if (localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                                bestModel = localVBEM.getPlateuStructure();
                                bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                                bestPair = new Tuple<>(fromVar, toVar);
                            }

                            /* Remove the newly created arc to reset the process for the next pair */
                            copyDAG.getParentSet(toVar).removeParent(fromVar);
                        }
                    }
                }
            }
        }

        /* Modify the DAG with the best arc */
        if(bestModelScore > -Double.MAX_VALUE) {
            copyDAG.getParentSet(bestPair.getSecond()).addParent(bestPair.getFirst());
            return new Triple<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
        }

        return new Triple<>(null, null, new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
    }
}
