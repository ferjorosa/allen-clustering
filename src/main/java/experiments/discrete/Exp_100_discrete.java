package experiments.discrete;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.structure.BLFM_BinA;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.Tuple3;
import eu.amidst.extension.util.Tuple4;
import experiments.util.AmidstToVoltricModel;
import experiments.util.DiscreteClusteringMeasures;
import methods.BinA;
import methods.BinG;
import methods.LCM;
import methods.VariationalLCM;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Exp_100_discrete {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        /* Aprendemos los modelos */
        //learnExperiment(seed, LogUtils.LogLevel.INFO);

        /* Cargamos el LCM y estimamos sus variables mas relevantes en orden por cada cluster*/

        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("clustVar");
        DiscreteBayesNet lcm = XmlBifReader.processFile(new File("models/discrete/exact/lcm_100.xml"), latentVarNames);
        List<List<Tuple<DiscreteVariable, Double>>> mostRelevantVarsPerCluster = DiscreteClusteringMeasures.mostRelevantVarsInLcmPerCluster(lcm);
        for(int cluster = 0; cluster < mostRelevantVarsPerCluster.size(); cluster++) {
            List<Tuple<DiscreteVariable, Double>> mostRelevantVarsInCluster = mostRelevantVarsPerCluster.get(cluster);
            System.out.println("\nCluster " + cluster + ":");
            for (int i = 0; i < 10; i++) {
                Tuple<DiscreteVariable, Double> variablePair = mostRelevantVarsInCluster.get(i);
                System.out.println(variablePair.getFirst().getName() + " -> " + variablePair.getSecond());
            }
        }

    }


    private static void learnExperiment(long seed, LogUtils.LogLevel logLevel) throws IOException {

        String filename = "data/discrete/exon_100_disc.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();

        /* Bin-A */
        //Tuple3<BayesianNetwork, Double, Long> binAresult = BinA.learnModel(data, seed, BLFM_BinA.LinkageType.AVERAGE, logLevel, true);
        //DiscreteBayesNet binAmodel = AmidstToVoltricModel.transform(binAresult.getFirst());
        //String output = "models/discrete/binA_100.bif";
        //BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        //writer.write(binAmodel);

        /* Bin-G */
        //Tuple3<BayesianNetwork, Double, Long> binGresult = BinG.learnModel(data, seed, logLevel, true);
        //DiscreteBayesNet binGmodel = AmidstToVoltricModel.transform(binGresult.getFirst());
        //output = "models/discrete/binG_100.bif";
        //writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        //writer.write(binGmodel);

        /* Learn LCM */
        Tuple4<DiscreteBayesNet, Double, Double, Long> lcmResult = LCM.runParallel(data, seed, logLevel);
        String output = "models/lcm_100.bif";
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(lcmResult.getFirst());

        /* Learn Variational LCM */
        //Map<String, double[]> priors = new LinkedHashMap<>();
        //priors = PriorsFromData.generate(data, 1);
        //Tuple3<BayesianNetwork, Double, Long> variationalLCMresult = VariationalLCM.learnModel(data, seed, priors, logLevel, true);
        //DiscreteBayesNet variationalLcmModel = AmidstToVoltricModel.transform(variationalLCMresult.getFirst());
        //output = "models/discrete/varlcm_100.bif";
        //writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        //writer.write(variationalLcmModel);
    }
}
