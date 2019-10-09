package experiments;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.util.AmidstToLatlabData;
import experiments.util.EstimatePredictiveScore;
import org.latlab.data.MixedDataSet;
import org.latlab.learner.geast.GeastWithoutPouch;
import org.latlab.learner.geast.IModelWithScore;
import org.latlab.learner.geast.ParameterGenerator;
import org.latlab.learner.geast.Settings;
import org.latlab.model.Builder;
import org.latlab.model.Gltm;
import org.latlab.util.DiscreteVariable;
import org.latlab.util.Variable;
import voltric.util.Tuple;

import java.util.ArrayList;
import java.util.List;

public class GEAST {

    public static List<Tuple4<Gltm, Double, Double, Long>> run(List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                               String settingsLocation,
                                                               LogUtils.LogLevel logLevel) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GEAST");
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
            GeastWithoutPouch geastWithoutPouch = settings.createGeastWithoutPouch();

            long initTime = System.currentTimeMillis();
            Gltm lcm =  Builder.buildNaiveBayesModel(
                    new Gltm(),
                    new DiscreteVariable(2),
                    trainDataLatlab.getNonClassVariables());
            ParameterGenerator parameterGenerator = new ParameterGenerator(trainDataLatlab);
            parameterGenerator.generate(lcm);
            IModelWithScore result = geastWithoutPouch.learn(lcm);
            long endTime = System.currentTimeMillis();

            double foldPLL = EstimatePredictiveScore.latLabLL(result.model(), testDataLatlab);
            double foldPBIC = EstimatePredictiveScore.latLabBIC(result.model(), testDataLatlab);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(result.model(), foldPLL, foldPBIC, foldTime));

            LogUtils.info("----------------------------------------", logLevel);
            LogUtils.info("Fold " + (i+1) , logLevel);
            LogUtils.info("Predictive Log-Likelihood: " + foldPLL, logLevel);
            LogUtils.info("Predictive BIC: " + foldPBIC, logLevel);
            LogUtils.info("Time: " + foldTime + " ms", logLevel);
        }

        return foldsResults;
    }

    public static Tuple4<IModelWithScore, Double, Double, Long> run(DataOnMemory<DataInstance> trainData,
                                                         String settingsLocation) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GEAST");
        System.out.println("==========================\n");

        MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);

        Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
        GeastWithoutPouch geastWithoutPouch = settings.createGeastWithoutPouch();

        long initTime = System.currentTimeMillis();
        Gltm lcm =  Builder.buildNaiveBayesModel(
                new Gltm(),
                new DiscreteVariable(2),
                trainDataLatlab.getNonClassVariables());
        ParameterGenerator parameterGenerator = new ParameterGenerator(trainDataLatlab);
        parameterGenerator.generate(lcm);
        IModelWithScore result = geastWithoutPouch.learn(lcm);
        long endTime = System.currentTimeMillis();

        long learnTime = (endTime - initTime);
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + result.loglikelihood());
        System.out.println("BIC: " + result.BicScore());
        System.out.println("Time: " + learnTime + " ms");

        return new Tuple4<>(result, result.loglikelihood(), result.BicScore(), learnTime);
    }

    /** Both the test and train datasets need to have the same list of variable objects or it wont work (Latlab code) */
    private static MixedDataSet prepareTestData(DataOnMemory<DataInstance> amidstTestData, List<Variable> trainVariables) {
        MixedDataSet testData = MixedDataSet.createEmpty(trainVariables, amidstTestData.getNumberOfDataInstances());

        for(DataInstance instance: amidstTestData)
            testData.add(1, instance.toArray());

        return testData;
    }
}
