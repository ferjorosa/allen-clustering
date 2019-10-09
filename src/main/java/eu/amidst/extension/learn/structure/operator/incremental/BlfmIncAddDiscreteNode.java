package eu.amidst.extension.learn.structure.operator.incremental;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.Triple;
import voltric.util.Tuple;

import java.util.*;

public class BlfmIncAddDiscreteNode implements BlfmIncOperator {

    private int newNodeCardinality;

    private int maxNumberOfDiscreteLatentNodes;

    private VBEMConfig localVBEMConfig;

    private int latentVarNameCounter = 0;

    public BlfmIncAddDiscreteNode(int newNodeCardinality,
                                  int maxNumberOfDiscreteLatentNodes) {
        this(newNodeCardinality, maxNumberOfDiscreteLatentNodes, new VBEMConfig());
    }

    public BlfmIncAddDiscreteNode(int newNodeCardinality,
                                  int maxNumberOfDiscreteLatentNodes,
                                  VBEMConfig localVBEMConfig) {
        this.newNodeCardinality = newNodeCardinality;
        this.maxNumberOfDiscreteLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.localVBEMConfig = localVBEMConfig;
    }

    /**
     *
     */
    @Override
    public Triple<Variable, Variable, Result> apply(Set<Variable> currentSet,
                                                    PlateuStructure plateuStructure,
                                                    DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestPair = null;
        String newLatentVarName = "";

        /* Return current model if current number of discrete latent nodes is maximum */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();
        if(numberOfLatentNodes >= this.maxNumberOfDiscreteLatentNodes)
            return new Triple<>(null, null, new Result(bestModel, bestModelScore, dag, "AddDiscreteNode"));

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Transform currentSet into a list using copyVariables */
        List<Variable> currentList = new ArrayList<>();
        for(Variable var: currentSet)
            currentList.add(copyVariables.getVariableByName(var.getName()));

        /* Iterate through the list of variable pairs that belong to currentSet */
        for(int i = 0; i < currentList.size(); i++)
            for(int j = i+1; j < currentList.size(); j++) {
                Variable firstVar = currentList.get(i);
                Variable secondVar = currentList.get(j);

                /* Create a new Latent variable as the pair's new parent */
                Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), this.newNodeCardinality);
                copyDAG.addVariable(newLatentVar);
                copyDAG.getParentSet(firstVar).addParent(newLatentVar);
                copyDAG.getParentSet(secondVar).addParent(newLatentVar);

                /* Create a new plateau by copying current one and omitting the new variable and its children */
                HashSet<Variable> omittedVariables = new HashSet<>();
                omittedVariables.add(newLatentVar);
                omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
                PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                /* Learn the new model with Local VBEM */
                VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                localVBEM.learnModel(copyPlateauStructure, copyDAG, newLatentVar);

                /* Compare its score with current best model */
                if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                    bestModel = localVBEM.getPlateuStructure();
                    bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                    bestPair = new Tuple<>(firstVar, secondVar);
                    newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies with the Plateau
                }

                /* Remove the newly created node to reset the process for the next pair */
                copyDAG.getParentSet(firstVar).removeParent(newLatentVar);
                copyDAG.getParentSet(secondVar).removeParent(newLatentVar);
                copyDAG.removeVariable(newLatentVar);
                copyVariables.remove(newLatentVar);
            }

        /* Modify the DAG with the best latent var */
        if(bestModelScore > -Double.MAX_VALUE) {
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
        }

        return new Triple<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddDiscreteNode"));
    }

    /**
     *
     */
    @Override
    public Triple<Variable, Variable, Result> apply(PriorityQueue<Triple<Variable, Variable, Double>> selectedTriples,
                                                    PlateuStructure plateuStructure,
                                                    DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple<Variable, Variable> bestPair = null;
        String newLatentVarName = "";

        /* Return current model if current number of discrete latent nodes is maximum */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();
        if(numberOfLatentNodes >= this.maxNumberOfDiscreteLatentNodes)
            return new Triple<>(null, null, new Result(bestModel, bestModelScore, dag, "AddDiscreteNode"));

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iterate through the queue of selected triples */
        for(Triple<Variable, Variable, Double> triple: selectedTriples){
            Variable firstVar = copyVariables.getVariableByName(triple.getFirst().getName());
            Variable secondVar = copyVariables.getVariableByName(triple.getSecond().getName());

            /* Create a new Latent variable as the pair's new parent */
            Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(firstVar).addParent(newLatentVar);
            copyDAG.getParentSet(secondVar).addParent(newLatentVar);

            /* Create a new plateau by copying current one and omitting the new variable and its children */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(newLatentVar);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Learn the new model with Local VBEM */
            VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
            localVBEM.learnModel(copyPlateauStructure, copyDAG, newLatentVar);

            /* Compare its score with current best model */
            if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                bestModel = localVBEM.getPlateuStructure();
                bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                bestPair = new Tuple<>(firstVar, secondVar);
                newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies with the Plateau
            }

            /* Remove the newly created node to reset the process for the next pair */
            copyDAG.getParentSet(firstVar).removeParent(newLatentVar);
            copyDAG.getParentSet(secondVar).removeParent(newLatentVar);
            copyDAG.removeVariable(newLatentVar);
            copyVariables.remove(newLatentVar);
        }

        /* Modify the DAG with the best latent var */
        if(bestModelScore > -Double.MAX_VALUE) {
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
        }

        return new Triple<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddDiscreteNode"));
    }
}
