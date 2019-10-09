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
import methods.BinA;
import methods.BinG;
import methods.VariationalLCM;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.model.DiscreteBayesNet;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class Exp_100_discrete {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        learnExperiment(seed, LogUtils.LogLevel.INFO);
    }

    private static void learnExperiment(long seed, LogUtils.LogLevel logLevel) throws IOException {

        String filename = "data/discrete/exon_100_disc.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();

        /* Bin-A */
        Tuple3<BayesianNetwork, Double, Long> binAresult = BinA.learnModel(data, seed, BLFM_BinA.LinkageType.AVERAGE, logLevel, true);
        DiscreteBayesNet binAmodel = AmidstToVoltricModel.transform(binAresult.getFirst());
        String output = "models/discrete/binA_100.bif";
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(binAmodel);

        /* Bin-G */
        Tuple3<BayesianNetwork, Double, Long> binGresult = BinG.learnModel(data, seed, logLevel, true);
        DiscreteBayesNet binGmodel = AmidstToVoltricModel.transform(binGresult.getFirst());
        output = "models/discrete/binG_100.bif";
        writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(binGmodel);

        /* Learn LCM */
        //Tuple4<DiscreteBayesNet, Double, Double, Long> lcmResult = LCM.runParallel(data, seed, logLevel);
        //String output = "models/lcm_100.bif";
        //BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        //writer.write(lcmResult.getFirst());

        /* Learn Variational LCM */
        Map<String, double[]> priors = new LinkedHashMap<>();
        priors = PriorsFromData.generate(data, 1);
        Tuple3<BayesianNetwork, Double, Long> variationalLCMresult = VariationalLCM.learnModel(data, seed, priors, logLevel, true);
        DiscreteBayesNet variationalLcmModel = AmidstToVoltricModel.transform(variationalLCMresult.getFirst());
        output = "models/discrete/varlcm_100.bif";
        writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(variationalLcmModel);
    }
}
