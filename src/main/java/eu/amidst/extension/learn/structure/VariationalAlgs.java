package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.operator.hc.tree.*;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddArc;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddDiscreteNode;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.DistanceFunction;
import voltric.util.Tuple;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class VariationalAlgs {

    public static Tuple<BayesianNetwork, Double> bingG(DataOnMemory<DataInstance> data,
                                                       Map<String, double[]> priors,
                                                       long seed,
                                                       int n_neighbors_mi,
                                                       DistanceFunction distanceFunction_mi,
                                                       boolean gaussianNoise_mi,
                                                       boolean normalizedMI,
                                                       LogUtils.LogLevel logLevel,
                                                       boolean printNetwork) {

        /* */
        System.out.println("\n==========================");
        System.out.println("BLFM - BinG");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        /* Aprendemos el modelo */
        double initTime = System.currentTimeMillis();
        BLFM_BinG blfm_binG = new BLFM_BinG(n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                seed,
                normalizedMI,
                initialVBEMConfig,
                localVBEMConfig,
                finalVBEMConfig);
        Result result = blfm_binG.learnModel(data, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> binA(DataOnMemory<DataInstance> data,
                                                      Map<String, double[]> priors,
                                                      long seed,
                                                      int n_neighbors_mi,
                                                      DistanceFunction distanceFunction_mi,
                                                      boolean gaussianNoise_mi,
                                                      boolean normalizedMI,
                                                      BLFM_BinA.LinkageType linkageType,
                                                      LogUtils.LogLevel logLevel,
                                                      boolean printNetwork) {
        /* */
        System.out.println("\n==========================");
        System.out.println("BLFM - BinA");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        /* Aprendemos el modelo */
        double initTime = System.currentTimeMillis();
        BLFM_BinA blfm_binA = new BLFM_BinA(n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                seed,
                normalizedMI,
                linkageType,
                initialVBEMConfig,
                localVBEMConfig,
                finalVBEMConfig);
        Result result = blfm_binA.learnModel(data, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> incrementalLearnerMax(DataOnMemory<DataInstance> data,
                                                                       Map<String, double[]> priors,
                                                                       long seed,
                                                                       boolean iterationGlobalVBEM,
                                                                       boolean observedInternalNodes,
                                                                       boolean discreteToDiscreteObserved,
                                                                       LogUtils.LogLevel logLevel,
                                                                       boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLFM - Incremental Learner (alpha = Brute, observedInternalNodes = " + observedInternalNodes + ") ");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        double initTime = System.currentTimeMillis();

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes, discreteToDiscreteObserved, localVBEMConfig);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        BLFM_IncLearnerMax incLearnerMax = new BLFM_IncLearnerMax(operators,
                iterationGlobalVBEM,
                initialVBEMConfig,
                localVBEMConfig,
                iterationVBEMConfig,
                finalVBEMConfig);

        Result result = incLearnerMax.learnModel(data, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> incrementalLearner(DataOnMemory<DataInstance> data,
                                                                    Map<String, double[]> priors,
                                                                    long seed,
                                                                    boolean iterationGlobalVBEM,
                                                                    boolean observedInternalNodes,
                                                                    boolean discreteToDiscreteObserved,
                                                                    int alpha,
                                                                    int n_neighbors_mi,
                                                                    DistanceFunction distanceFunction_mi,
                                                                    boolean gaussianNoise_mi,
                                                                    boolean normalizedMI,
                                                                    LogUtils.LogLevel logLevel,
                                                                    boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLFM - Incremental Learner (alpha = "+alpha+", observedInternalNodes = " + observedInternalNodes + ") ");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        double initTime = System.currentTimeMillis();

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes, discreteToDiscreteObserved, localVBEMConfig);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        BLFM_IncLearner incrementalLearner = new BLFM_IncLearner(operators,
                iterationGlobalVBEM,
                n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                seed,
                normalizedMI,
                initialVBEMConfig,
                localVBEMConfig,
                iterationVBEMConfig,
                finalVBEMConfig);

        Result result = incrementalLearner.learnModel(data, alpha, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> hillClimbing(DataOnMemory<DataInstance> data,
                                                              Map<String, double[]> priors,
                                                              DAG initialStructure,
                                                              long seed,
                                                              boolean operatorGlobalVBEM,
                                                              boolean debug,
                                                              boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLTM - Hill climbing");
        System.out.println("==========================");

        InitializationVBEM localOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, localOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM globalOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig globalOperatorOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, globalOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());

        Set<BltmHcOperator> operators = new LinkedHashSet<>();
        operators.add(new BltmHcAddNode(Integer.MAX_VALUE, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        operators.add(new BltmHcRemoveNode(localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        operators.add(new BltmHcRelocateNode(localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        operators.add(new BltmHcIncreaseCard(100, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        operators.add(new BltmHcDecreaseCard(2, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));

        BLTM_HillClimbing hillClimbing = new BLTM_HillClimbing(operators,
                operatorGlobalVBEM,
                initialVBEMConfig,
                iterationVBEMConfig);

        double initTime = System.currentTimeMillis();
        Result result = hillClimbing.learnModel(initialStructure, data, priors, debug);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> east(DataOnMemory<DataInstance> data,
                                                      Map<String, double[]> priors,
                                                      DAG initialStructure,
                                                      long seed,
                                                      boolean debug,
                                                      boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLTM - EAST");
        System.out.println("==========================");

        InitializationVBEM localOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, localOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM globalOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig globalOperatorOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, globalOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM processVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig processVBEMConfig = new VBEMConfig(seed, 0.01, 100, processVBEMinitialization, new BishopPenalizer());

        Set<BltmHcOperator> expansionOperators = new LinkedHashSet<>();
        expansionOperators.add(new BltmHcAddNode(Integer.MAX_VALUE, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        expansionOperators.add(new BltmHcIncreaseCard(10, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        Set<BltmHcOperator> simplificationOperators = new LinkedHashSet<>();
        simplificationOperators.add(new BltmHcRemoveNode(localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        simplificationOperators.add(new BltmHcDecreaseCard(2, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        Set<BltmHcOperator> adjustmentOperators = new LinkedHashSet<>();
        adjustmentOperators.add(new BltmHcRelocateNode(localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        BLTM_EAST east = new BLTM_EAST(expansionOperators,
                simplificationOperators,
                adjustmentOperators,
                true,
                initialVBEMConfig,
                processVBEMConfig);

        double initTime = System.currentTimeMillis();
        Result result = east.learnModel(initialStructure, data, priors, debug);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    // TODO: Arreglar los tiempos de ejecucion y definir la configuracion del VBEM
    public static Tuple<BayesianNetwork, Double> bridgedIslands(DataOnMemory<DataInstance> data,
                                                                Map<String, double[]> priors,
                                                                long seed,
                                                                int n_neighbors_mi,
                                                                DistanceFunction distanceFunction_mi,
                                                                boolean gaussianNoise_mi,
                                                                boolean normalizedMI,
                                                                boolean randomChowLiuRoot,
                                                                boolean debug,
                                                                boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLTM - Bridged Islands");
        System.out.println("==========================");

        BLTM_BridgedIslands bridgedIslands = new BLTM_BridgedIslands(100,
                n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                seed,
                normalizedMI);

        double initTime = System.currentTimeMillis();
        Result result = bridgedIslands.learnModel(data, randomChowLiuRoot, priors, debug);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static Tuple<BayesianNetwork, Double> lcm(DataOnMemory<DataInstance> data,
                                                     Map<String, double[]> priors,
                                                     long seed,
                                                     boolean debug,
                                                     boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("BLCM");
        System.out.println("==========================");

        InitializationVBEM localOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig localOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, localOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM globalOperatorVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig globalOperatorOperatorVBEMConfig = new VBEMConfig(seed, 0.01, 100, globalOperatorVBEMinitialization, new BishopPenalizer());
        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());

        DAG lcm = generateLCM(data, "latentVar", 2);

        Set<BltmHcOperator> operators = new LinkedHashSet<>();
        operators.add(new BltmHcIncreaseCard(10, localOperatorVBEMConfig, globalOperatorOperatorVBEMConfig));
        BLTM_HillClimbing hillClimbing = new BLTM_HillClimbing( operators, true, initialVBEMConfig, iterationVBEMConfig);

        double initTime = System.currentTimeMillis();
        Result result = hillClimbing.learnModel(lcm, data, priors, debug);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double endTime = System.currentTimeMillis();

        System.out.println("\nFinal ELBO Score: " + result.getElbo());
        System.out.println("Learning time: " + (endTime - initTime));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple<>(posteriorPredictive, result.getElbo());
    }

    public static DAG generateLCM(DataOnMemory<DataInstance> data, String latentVarName, int cardinality) {

        /* Creamos un Naive Bayes con padre latente */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable(latentVarName, cardinality);

        DAG dag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                dag.getParentSet(var).addParent(latentVar);

        return dag;
    }

    public static Tuple<BayesianNetwork, Double> learnVBEM(DAG dag, DataOnMemory<DataInstance> data, VBEMConfig config, Map<String, double[]> priors) {

        VBEM vbem = new VBEM(config);

        double elbo = vbem.learnModelWithPriorUpdate(data, dag, priors);
        BayesianNetwork bn = vbem.getLearntBayesianNetwork();

        return new Tuple<>(bn, elbo);
    }
}
