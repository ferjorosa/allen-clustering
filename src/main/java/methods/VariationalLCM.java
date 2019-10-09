package methods;

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
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple3;
import voltric.util.Tuple;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.Map;

public class VariationalLCM {

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   long seed,
                                                                   Map<String, double[]> priors,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Variational LCM");
        System.out.println("==========================");

        InitializationVBEM vbemInitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, vbemInitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        Tuple<BayesianNetwork, Double> result = learnLcmToMaxCardinality(data, vbemConfig, priors, logLevel);

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\n---------------------------------------------");
        System.out.println("\nELBO Score: " + f.format(result.getSecond()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getSecond() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+result.getFirst());

        return new Tuple3<>(result.getFirst(), result.getSecond(), learningTimeMs);
    }

    private static Tuple<BayesianNetwork, Double> learnLcmToMaxCardinality(DataOnMemory<DataInstance> data,
                                                                           VBEMConfig config,
                                                                           Map<String, double[]> priors,
                                                                           LogUtils.LogLevel logLevel) {


        VBEM vbem = new VBEM(config);
        double bestScore = -Double.MAX_VALUE;
        BayesianNetwork bestModel = null;

        for(int card = 2; card < Integer.MAX_VALUE; card++) {
            long initTime = System.currentTimeMillis();
            DAG lcmStructure = generateLcmDAG(data, "ClustVar", card);
            double currentScore = vbem.learnModelWithPriorUpdate(data, lcmStructure, priors);
            BayesianNetwork currentModel = vbem.getLearntBayesianNetwork();
            long endTime = System.currentTimeMillis();

            long learnTime = (endTime - initTime);

            LogUtils.info("\nCardinality " + card, logLevel);
            LogUtils.info("ELBO: " + currentScore, logLevel);
            LogUtils.info("Time: " + learnTime + " ms", logLevel);

            if(currentScore > bestScore) {
                bestModel = currentModel;
                bestScore = currentScore;
            } else {
                System.out.println("SCORE STOPPED IMPROVING");
                return new Tuple<>(bestModel, bestScore);
            }
        }
        return new Tuple<>(bestModel, bestScore);
    }

    private static DAG generateLcmDAG(DataOnMemory<DataInstance> data, String latentVarName, int cardinality) {

        /* Creamos un Naive Bayes con padre latente */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable(latentVarName, cardinality);

        DAG dag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                dag.getParentSet(var).addParent(latentVar);

        return dag;
    }
}
