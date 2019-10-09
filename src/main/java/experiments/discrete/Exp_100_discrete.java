package experiments.discrete;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.LCM;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.model.DiscreteBayesNet;

import java.io.FileOutputStream;
import java.io.IOException;

public class Exp_100_discrete {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        learnExperiment(seed, LogUtils.LogLevel.INFO);
    }

    private static void learnExperiment(long seed, LogUtils.LogLevel logLevel) throws IOException {

        String filename = "data/discrete/exon_100_disc.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();

        /* Learn model */
        Tuple4<DiscreteBayesNet, Double, Double, Long> lcmResult = LCM.runParallel(data, seed, logLevel);

        String output = "models/lcm_100.bif";
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(lcmResult.getFirst());
    }
}
