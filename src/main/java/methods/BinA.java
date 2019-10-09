package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_BinA;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple3;
import eu.amidst.extension.util.Tuple4;
import eu.amidst.extension.util.distance.ChebyshevDistance;
import experiments.util.AmidstToVoltricData;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.EM;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;

// NOTA: no le ponemos priors porque no las tiene en el articulo original
public class BinA {

    public static List<Tuple4<BayesianNetwork, Double, Double, Long>> run(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                          long seed,
                                                                          BLFM_BinA.LinkageType linkageType,
                                                                          LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("Bin-A");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            long initTime = System.currentTimeMillis();

            BLFM_BinA blfm_binA = new BLFM_BinA(3,
                    new ChebyshevDistance(),
                    false,
                    seed,
                    false,
                    linkageType,
                    initialVBEMConfig,
                    localVBEMConfig,
                    finalVBEMConfig);
            Result result = blfm_binA.learnModel(trainData, new LinkedHashMap<>(), LogUtils.LogLevel.NONE);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            long endTime = System.currentTimeMillis();

            double foldPLL = EstimatePredictiveScore.discreteLL(posteriorPredictive, testData);
            double foldPBIC = EstimatePredictiveScore.discreteBIC(posteriorPredictive, testData);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(posteriorPredictive, foldPLL, foldPBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Predictive Log-Likelihood: " + foldPLL, foldLogLevel);
            LogUtils.info("Predictive BIC: " + foldPBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }

    public static List<Tuple4<DiscreteBayesNet, Double, Double, Long>> runDiscreteEM(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                                     long seed,
                                                                                     BLFM_BinA.LinkageType linkageType,
                                                                                     LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("Bin-A");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            DiscreteData voltricTrainData = AmidstToVoltricData.transform(trainData);
            DiscreteData voltricTestData = AmidstToVoltricData.transform(testData);

            long initTime = System.currentTimeMillis();

            BLFM_BinA blfm_binA = new BLFM_BinA(3,
                    new ChebyshevDistance(),
                    false,
                    seed,
                    false,
                    linkageType,
                    initialVBEMConfig,
                    localVBEMConfig,
                    finalVBEMConfig);
            Result result = blfm_binA.learnModel(trainData, new LinkedHashMap<>(), LogUtils.LogLevel.NONE);
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

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Predictive Log-Likelihood: " + foldPLL, foldLogLevel);
            LogUtils.info("Predictive BIC: " + foldPBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   long seed,
                                                                   BLFM_BinA.LinkageType linkageType,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Bin-A");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        BLFM_BinA blfm_binA = new BLFM_BinA(3,
                new ChebyshevDistance(),
                false,
                seed,
                false,
                linkageType,
                initialVBEMConfig,
                localVBEMConfig,
                finalVBEMConfig);
        Result result = blfm_binA.learnModel(data, new LinkedHashMap<>(), logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\n---------------------------------------------");
        System.out.println("\nELBO Score: " + f.format(result.getElbo()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getElbo() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple3<>(posteriorPredictive, result.getElbo(), learningTimeMs);
    }
}
