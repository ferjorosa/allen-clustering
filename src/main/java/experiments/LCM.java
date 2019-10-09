package experiments;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.util.AmidstToVoltricData;
import org.latlab.learner.geast.IModelWithScore;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.EM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.em.initialization.ChickeringHeckerman;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.hillclimbing.global.GlobalHillClimbing;
import voltric.learning.structure.hillclimbing.global.HcOperator;
import voltric.learning.structure.hillclimbing.global.IncreaseLatentCardinality;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.util.Tuple;

import java.util.*;

public class LCM {

    private static int nRestarts = 64;
    private static double threshold = 1e-2;
    private static int nMaxSteps = 500;


    /** Learns an LCM on each fold and returns the CVPLL score and learning time of each one */
    public static List<Tuple4<DiscreteBayesNet, Double, Double, Long>> run(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                           long seed,
                                                                           boolean foldDebug) {

        System.out.println("\n==========================");
        System.out.println("LCM");
        System.out.println("==========================\n");

        EmConfig emConfig = new EmConfig(nRestarts, threshold, nMaxSteps, new ChickeringHeckerman(), true, new HashSet<>());
        EM em = new EM(emConfig, seed);
        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        /* Iterate through the folds and learn an LCM on each one */
        for(int i = 0; i < folds.size(); i++) {

            Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);

            DataOnMemory<DataInstance> trainData = fold.getFirst();
            DataOnMemory<DataInstance> testData = fold.getSecond();
            DiscreteData trainDataVoltric = AmidstToVoltricData.transform(trainData);
            DiscreteData testDataVoltric = AmidstToVoltricData.transform(testData);

            long initTime = System.currentTimeMillis();
            LearningResult<DiscreteBayesNet> result = learnLcmToMaxCardinality(trainDataVoltric, em, seed, threshold, LogUtils.LogLevel.NONE);
            long endTime = System.currentTimeMillis();

            DiscreteBayesNet resultNet = result.getBayesianNetwork();
            double foldPLL = LearningScore.calculateLogLikelihood(testDataVoltric, resultNet);
            double foldPBIC = LearningScore.calculateBIC(testDataVoltric, resultNet);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(resultNet, foldPLL, foldPBIC, foldTime));

            LogUtils.printf("----------------------------------------", foldDebug);
            LogUtils.printf("Fold " + (i+1) , foldDebug);
            LogUtils.printf("Predictive Log-Likelihood: " + foldPLL, foldDebug);
            LogUtils.printf("Predictive BIC: " + foldPBIC, foldDebug);
            LogUtils.printf("Time: " + foldTime + " ms", foldDebug);
        }

        return foldsResults;
    }

    public static Tuple4<DiscreteBayesNet, Double, Double, Long> run(DataOnMemory<DataInstance> data,
                                                                     long seed,
                                                                     LogUtils.LogLevel logLevel) {

        System.out.println("\n==========================");
        System.out.println("LCM");
        System.out.println("==========================\n");

        EmConfig emConfig = new EmConfig(nRestarts, threshold, nMaxSteps, new ChickeringHeckerman(), true, new HashSet<>());
        EM em = new EM(emConfig, seed);

        DiscreteData trainDataVoltric = AmidstToVoltricData.transform(data);

        long initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> result = learnLcmToMaxCardinality(trainDataVoltric, em, seed, threshold, logLevel);
        long endTime = System.currentTimeMillis();

        DiscreteBayesNet resultNet = result.getBayesianNetwork();
        long learnTime = (endTime - initTime);
        double logLikelihood = LearningScore.calculateLogLikelihood(trainDataVoltric, resultNet);
        double bic = LearningScore.calculateBIC(trainDataVoltric, resultNet);
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("Time: " + learnTime + " ms");

        return new Tuple4<>(resultNet, logLikelihood, bic, learnTime);
    }

    public static Tuple4<DiscreteBayesNet, Double, Double, Long> runParallel(DataOnMemory<DataInstance> data,
                                                                             long seed,
                                                                             LogUtils.LogLevel logLevel) {

        System.out.println("\n==========================");
        System.out.println("LCM");
        System.out.println("==========================\n");

        EmConfig emConfig = new EmConfig(nRestarts, threshold, nMaxSteps, new ChickeringHeckerman(), true, new HashSet<>());
        ParallelEM parallelEM = new ParallelEM(emConfig, ScoreType.BIC);

        DiscreteData trainDataVoltric = AmidstToVoltricData.transform(data);

        long initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> result = learnLcmToMaxCardinality(trainDataVoltric, parallelEM, seed, threshold, logLevel);
        long endTime = System.currentTimeMillis();

        DiscreteBayesNet resultNet = result.getBayesianNetwork();
        long learnTime = (endTime - initTime);
        double logLikelihood = LearningScore.calculateLogLikelihood(trainDataVoltric, resultNet);
        double bic = LearningScore.calculateBIC(trainDataVoltric, resultNet);
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("Time: " + learnTime + " ms");

        return new Tuple4<>(resultNet, logLikelihood, bic, learnTime);
    }

    private static LearningResult<DiscreteBayesNet> learnLcmToMaxCardinality(DiscreteData dataSet,
                                                                             AbstractEM em,
                                                                             long seed,
                                                                             double threshold,
                                                                             LogUtils.LogLevel logLevel) {

        LearningResult<DiscreteBayesNet> bestResult = new LearningResult<>(null, -Double.MAX_VALUE, ScoreType.BIC);

        for(int card = 2; card < Integer.MAX_VALUE; card++) {
            long initTime = System.currentTimeMillis();
            HLCM lcm = HlcmCreator.createLCM(dataSet.getVariables(), card, new Random(seed));
            LearningResult<DiscreteBayesNet> result = em.learnModel(lcm, dataSet);
            long endTime = System.currentTimeMillis();

            long learnTime = (endTime - initTime);
            double currentlogLikelihood = LearningScore.calculateLogLikelihood(dataSet, result.getBayesianNetwork());

            LogUtils.info("\nCardinality " + card, logLevel);
            System.out.println("Log-Likelihood: " + currentlogLikelihood);
            System.out.println("BIC: " + result.getScoreValue());
            System.out.println("Time: " + learnTime + " ms");

            if(result.getScoreValue() > bestResult.getScoreValue()) {
                bestResult = result;
            } else {
                System.out.println("SCORE STOPPED IMPROVING");
                return result;
            }
        }

        return bestResult;
    }
}
