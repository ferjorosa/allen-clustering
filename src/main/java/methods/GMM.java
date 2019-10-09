package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.util.AmidstToLatlabData;
import experiments.util.EstimatePredictiveScore;
import org.latlab.data.MixedDataSet;
import org.latlab.learner.geast.Geast;
import org.latlab.learner.geast.IModelWithScore;
import org.latlab.learner.geast.Settings;
import org.latlab.model.Builder;
import org.latlab.model.Gltm;
import org.latlab.util.Variable;
import voltric.util.Tuple;

import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class GMM {

    public static List<Tuple4<Gltm, Double, Double, Long>> run(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                               boolean foldDebug,
                                                               String settingsLocation) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GMM");
        System.out.println("==========================\n");
        List<Tuple4<Gltm, Double, Double, Long>> foldsResults = new ArrayList<>();

        /* Iterate through the folds and learn an LCM on each one */
        for(int i = 0; i < folds.size(); i++) {
            Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);

            DataOnMemory<DataInstance> trainData = fold.getFirst();
            DataOnMemory<DataInstance> testData = fold.getSecond();
            MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);
            MixedDataSet testDataLatlab = prepareTestData(testData, trainDataLatlab.variables());

            Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
            Geast geast = settings.createGeast();

            long initTime = System.currentTimeMillis();
            IModelWithScore result = learntoMaxCardinality(geast, trainDataLatlab.getNonClassVariables(), LogUtils.LogLevel.NONE);
            long endTime = System.currentTimeMillis();

            double foldPLL = EstimatePredictiveScore.latLabLL(result.model(), testDataLatlab);
            double foldPBIC = EstimatePredictiveScore.latLabBIC(result.model(), testDataLatlab);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(result.model(), foldPLL, foldPBIC, foldTime));

            LogUtils.printf("----------------------------------------", foldDebug);
            LogUtils.printf("Fold " + (i+1) , foldDebug);
            LogUtils.printf("Predictive Log-Likelihood: " + foldPLL, foldDebug);
            LogUtils.printf("Predictive BIC: " + foldPBIC, foldDebug);
            LogUtils.printf("Time: " + foldTime + " ms", foldDebug);
        }

        return foldsResults;
    }

    public static Tuple4<IModelWithScore, Double, Double, Long> learnModel(DataOnMemory<DataInstance> trainData,
                                                                String settingsLocation,
                                                                LogUtils.LogLevel logLevel) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GMM");
        System.out.println("==========================\n");

        MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);
        Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
        Geast geast = settings.createGeast();

        long initTime = System.currentTimeMillis();
        IModelWithScore result = learntoMaxCardinality(geast, trainDataLatlab.getNonClassVariables(), logLevel);
        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + result.loglikelihood());
        System.out.println("BIC: " + result.BicScore());
        System.out.println("Learning time (ms): " + learningTimeMs + " ms");
        System.out.println("Learning time (s): " + learningTimeS + " s");

        return new Tuple4<>(result, result.loglikelihood(), result.BicScore(), learningTimeMs);
    }

    private static IModelWithScore learntoMaxCardinality(Geast geast, List<Variable> nonClassVariables, LogUtils.LogLevel logLevel) {

        IModelWithScore bestResult = null;
        double bestBIC = -Double.MAX_VALUE;

        for(int card = 2; card < Integer.MAX_VALUE; card++) {

            long initTime = System.currentTimeMillis();
            Gltm gmm = Builder.buildMixedMixtureModel(
                    new Gltm(),
                    card,
                    nonClassVariables);

            geast.context().parameterGenerator().generate(gmm);
            IModelWithScore result = geast.context().estimationEm().estimate(gmm);
            long endTime = System.currentTimeMillis();
            long learnTime = (endTime - initTime);

            double currentBic = result.BicScore();

            LogUtils.info("\nCardinality " + card, logLevel);
            LogUtils.info("BIC: " +  currentBic, logLevel);
            LogUtils.info("Log-likelihood: " + result.loglikelihood(), logLevel);
            LogUtils.info("Time: " + learnTime + " ms", logLevel);

            if(currentBic > bestBIC) {
                bestResult = result;
                bestBIC = currentBic;
            } else {
                System.out.println("SCORE STOPPED IMPROVING");
                return result;
            }
        }

        return bestResult;
    }

    /** Both the test and train datasets need to have the same list of variable objects or it wont work (Latlab code) */
    private static MixedDataSet prepareTestData(DataOnMemory<DataInstance> amidstTestData, List<Variable> trainVariables) {
        MixedDataSet testData = MixedDataSet.createEmpty(trainVariables, amidstTestData.getNumberOfDataInstances());

        for(DataInstance instance: amidstTestData)
            testData.add(1, instance.toArray());

        return testData;
    }
}
