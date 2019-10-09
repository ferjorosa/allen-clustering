package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_IncLearnerMax;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddArc;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddDiscreteNode;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple3;
import eu.amidst.extension.util.Tuple4;
import experiments.util.AmidstToVoltricData;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import org.latlab.data.MixedDataSet;
import org.latlab.util.Variable;
import voltric.data.DiscreteData;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;

public class IncrementalLearnerMax {

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   Map<String, double[]> priors,
                                                                   long seed,
                                                                   boolean iterationGlobalVBEM,
                                                                   boolean observedInternalNodes,
                                                                   boolean discreteToDiscreteObserved,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Incremental Learner (alpha = Max, observedInternalNodes = " + observedInternalNodes + ", discreteObsArcs = "+discreteToDiscreteObserved+") ");
        System.out.println("==========================\n");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

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

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\nELBO Score: " + f.format(result.getElbo()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getElbo() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple3<>(posteriorPredictive, result.getElbo(), learningTimeMs);

    }

    public static List<Tuple4<DiscreteBayesNet, Double, Double, Long>> runDiscrete(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                                   Map<String, double[]> priors,
                                                                                   long seed,
                                                                                   boolean iterationGlobalVBEM,
                                                                                   boolean observedInternalNodes,
                                                                                   boolean discreteToDiscreteObserved,
                                                                                   LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("Incremental Learner (alpha = Max, observedInternalNodes = " + observedInternalNodes + ", discreteObsArcs = "+discreteToDiscreteObserved+") ");
        System.out.println("==========================\n");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            long initTime = System.currentTimeMillis();

            Set<BlfmIncOperator> operators = new LinkedHashSet<>();
            BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig);
            BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes, discreteToDiscreteObserved, localVBEMConfig);
            operators.add(addDiscreteNodeOperator);
            operators.add(addArcOperator);

            BLFM_IncLearnerMax incrementalLearnerMax = new BLFM_IncLearnerMax(operators,
                    iterationGlobalVBEM,
                    initialVBEMConfig,
                    localVBEMConfig,
                    iterationVBEMConfig,
                    finalVBEMConfig);

            Result result = incrementalLearnerMax.learnModel(trainData, priors, LogUtils.LogLevel.NONE);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            long endTime = System.currentTimeMillis();

            DiscreteData voltricTestData = AmidstToVoltricData.transform(testData);
            DiscreteBayesNet voltricResultNet = AmidstToVoltricModel.transform(posteriorPredictive);

            double foldPLL = EstimatePredictiveScore.voltricLL(voltricResultNet, voltricTestData);
            double foldPBIC = EstimatePredictiveScore.voltricBIC(voltricResultNet, voltricTestData);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(voltricResultNet, foldPLL, foldPBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Predictive Log-Likelihood: " + foldPLL, foldLogLevel);
            LogUtils.info("Predictive BIC: " + foldPBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }

    // TODO: Not finished. Hay que terminar la implementacion del paso de AMIDST a LATLAB con nodos joint para calcular el LL y BIC
    @Deprecated
    public static List<Tuple4<BayesianNetwork, Double, Double, Long>> runContinuous(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                                    Map<String, double[]> priors,
                                                                                    long seed,
                                                                                    boolean iterationGlobalVBEM,
                                                                                    boolean observedInternalNodes,
                                                                                    boolean discreteToDiscreteObserved,
                                                                                    LogUtils.LogLevel foldLogLevel,
                                                                                    String settingsLocation) {

        System.out.println("\n==========================");
        System.out.println("Incremental Learner (alpha = Max, observedInternalNodes = " + observedInternalNodes + ") ");
        System.out.println("==========================\n");

        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            long initTime = System.currentTimeMillis();

            Set<BlfmIncOperator> operators = new LinkedHashSet<>();
            BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE);
            BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes, discreteToDiscreteObserved);
            operators.add(addDiscreteNodeOperator);
            operators.add(addArcOperator);

            BLFM_IncLearnerMax incrementalLearnerBrute = new BLFM_IncLearnerMax(operators, iterationGlobalVBEM);

            Result result = incrementalLearnerBrute.learnModel(trainData, priors, LogUtils.LogLevel.NONE);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            long endTime = System.currentTimeMillis();

            long foldTime = (endTime - initTime);
            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }

/*
    public static List<Tuple4<DiscreteBayesNet, Double, Double, Long>> runDiscreteEM(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                                     Map<String, double[]> priors,
                                                                                     long seed,
                                                                                     boolean iterationGlobalVBEM,
                                                                                     boolean observedInternalNodes,
                                                                                     boolean foldDebug) {

        System.out.println("\n==========================");
        System.out.println("Incremental Learner with EM (alpha = Max, observedInternalNodes = " + observedInternalNodes + ") ");
        System.out.println("==========================\n");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            DiscreteData voltricTrainData = AmidstToVoltricData.transform(trainData);
            DiscreteData voltricTestData = AmidstToVoltricData.transform(testData);

            long initTime = System.currentTimeMillis();

            Set<BlfmIncOperator> operators = new LinkedHashSet<>();
            BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig);
            BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes, localVBEMConfig);
            operators.add(addDiscreteNodeOperator);
            operators.add(addArcOperator);

            BLFM_IncLearnerMax incrementalLearnerMax = new BLFM_IncLearnerMax(operators,
                    iterationGlobalVBEM,
                    initialVBEMConfig,
                    iterationVBEMConfig,
                    finalVBEMConfig);

            Result result = incrementalLearnerMax.learnModel(trainData, priors, false);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            DiscreteBayesNet voltricBLTM = AmidstToVoltricModel.transform(posteriorPredictive);
            AbstractEM em = new EM(seed);
            LearningResult<DiscreteBayesNet> voltricResult = em.learnModel(voltricBLTM, voltricTrainData);

            long endTime = System.currentTimeMillis();

            DiscreteBayesNet voltricResultNet = voltricResult.getBayesianNetwork();
            double foldPLL = EstimatePredictiveScore.voltricLL(voltricResultNet, voltricTestData);
            double foldPBIC = EstimatePredictiveScore.voltricBIC(voltricResultNet, voltricTestData);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(voltricResultNet, foldPLL, foldPBIC, foldTime));

            LogUtils.printf("----------------------------------------", foldDebug);
            LogUtils.printf("Fold " + (i+1) , foldDebug);
            LogUtils.printf("Predictive Log-Likelihood: " + foldPLL, foldDebug);
            LogUtils.printf("Predictive BIC: " + foldPBIC, foldDebug);
            LogUtils.printf("Time: " + foldTime + " ms", foldDebug);
        }

        return foldsResults;
    }

    // TODO: Not finished. Hay que terminar la implementacion del paso de AMIDST a LATLAB con nodos joint para calcular el LL y BIC
    @Deprecated
    public static List<Tuple4<Gltm, Double, Double, Long>> runContinuousEM(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                                       Map<String, double[]> priors,
                                                                                       boolean iterationGlobalVBEM,
                                                                                       boolean observedInternalNodes,
                                                                                       boolean foldDebug,
                                                                                       String settingsLocation) throws Exception{

        System.out.println("\n==========================");
        System.out.println("Incremental Learner with EM (alpha = Max, observedInternalNodes = " + observedInternalNodes + ") ");
        System.out.println("==========================\n");

        List<Tuple4<Gltm, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);
            MixedDataSet testDataLatlab = prepareLatLabTestData(testData, trainDataLatlab.variables());

            long initTime = System.currentTimeMillis();

            Set<BlfmIncOperator> operators = new LinkedHashSet<>();
            BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE);
            BlfmIncAddArc addArcOperator = new BlfmIncAddArc(observedInternalNodes, observedInternalNodes);
            operators.add(addDiscreteNodeOperator);
            operators.add(addArcOperator);

            BLFM_IncLearnerMax incrementalLearnerBrute = new BLFM_IncLearnerMax(operators, iterationGlobalVBEM);

            Result result = incrementalLearnerBrute.learnModel(trainData, priors, false);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            Gltm latlabModel = OldAmidstToLatlabModel.transform(posteriorPredictive, trainDataLatlab.getNonClassVariables());
            Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
            Geast geast = settings.createGeast();
            IModelWithScore latlabResult = geast.context().estimationEm().estimate(latlabModel);

            long endTime = System.currentTimeMillis();

            double foldPLL = EstimatePredictiveScore.latLabLL(latlabResult.model(), testDataLatlab);
            double foldPBIC = EstimatePredictiveScore.latLabBIC(latlabResult.model(), testDataLatlab);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(latlabResult.model(), foldPLL, foldPBIC, foldTime));

            LogUtils.printf("----------------------------------------", foldDebug);
            LogUtils.printf("Fold " + (i+1) , foldDebug);
            LogUtils.printf("Predictive Log-Likelihood: " + foldPLL, foldDebug);
            LogUtils.printf("Predictive BIC: " + foldPBIC, foldDebug);
            LogUtils.printf("Time: " + foldTime + " ms", foldDebug);
        }

        return foldsResults;
    }

    */

    /** Both the test and train datasets need to have the same list of variable objects or it wont work (Latlab code) */
    private static MixedDataSet prepareLatLabTestData(DataOnMemory<DataInstance> amidstTestData, List<Variable> trainVariables) {
        MixedDataSet testData = MixedDataSet.createEmpty(trainVariables, amidstTestData.getNumberOfDataInstances());

        for(DataInstance instance: amidstTestData)
            testData.add(1, instance.toArray());

        return testData;
    }

}
