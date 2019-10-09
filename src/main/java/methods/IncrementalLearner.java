package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_IncLearner;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddArc;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddDiscreteNode;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple3;
import eu.amidst.extension.util.Tuple4;
import eu.amidst.extension.util.distance.ChebyshevDistance;
import experiments.util.AmidstToVoltricData;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import voltric.data.DiscreteData;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;

public class IncrementalLearner {

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   Map<String, double[]> priors,
                                                                   long seed,
                                                                   int alpha,
                                                                   boolean iterationGlobalVBEM,
                                                                   boolean observedInternalNodes,
                                                                   boolean discreteToDiscreteObserved,
                                                                   int n_neighbors_mi,
                                                                   boolean gaussianNoise_mi,
                                                                   boolean normalizedMI,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Incremental Learner (alpha = "+alpha+", observedInternalNodes = " + observedInternalNodes + ") ");
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

        BLFM_IncLearner incrementalLearner = new BLFM_IncLearner(operators,
                iterationGlobalVBEM,
                n_neighbors_mi,
                new ChebyshevDistance(),
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
                                                                                   int alpha,
                                                                                   boolean iterationGlobalVBEM,
                                                                                   boolean observedInternalNodes,
                                                                                   boolean discreteToDiscreteObserved,
                                                                                   int n_neighbors_mi,
                                                                                   boolean gaussianNoise_mi,
                                                                                   boolean noramlized_mi,
                                                                                   LogUtils.LogLevel foldLogLevel) {
        System.out.println("\n==========================");
        System.out.println("Incremental Learner (alpha = "+alpha+", observedInternalNodes = " + observedInternalNodes + ") ");
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

            BLFM_IncLearner incLearner = new BLFM_IncLearner(operators,
                    iterationGlobalVBEM,
                    n_neighbors_mi,
                    new ChebyshevDistance(),
                    gaussianNoise_mi,
                    seed,
                    noramlized_mi,
                    initialVBEMConfig,
                    localVBEMConfig,
                    iterationVBEMConfig,
                    finalVBEMConfig);

            Result result = incLearner.learnModel(trainData, n_neighbors_mi, priors, LogUtils.LogLevel.NONE);
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
}
