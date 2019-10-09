package experiments.continuous;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.GLCM;
import experiments.GMM;
import org.latlab.io.bif.BifWriter;
import org.latlab.learner.geast.IModelWithScore;

import java.io.FileOutputStream;
import java.io.IOException;

public class Exp_100_continuous {

    public static void main(String[] args) throws Exception {

        long seed = 0;

        learnExperiment(seed, LogUtils.LogLevel.INFO);
    }

    private static void learnExperiment(long seed, LogUtils.LogLevel logLevel) throws Exception{

        String filename = "data/continuous/exon_100.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();

        /* GLCM */
        Tuple4<IModelWithScore, Double, Double, Long> glcmResult = GLCM.run(data, "geast_settings.xml", logLevel);
        storeResult(glcmResult.getFirst(), "models/glcm_100.bif");

        /* GMM */
        Tuple4<IModelWithScore, Double, Double, Long> gmmResult = GMM.run(data, "geast_settings.xml", logLevel);
        storeResult(gmmResult.getFirst(), "models/gmm_100.bif");

        /* PLTM-EAST */
        //Tuple4<Gltm, Double, Double, Long> pltmResult = PLTM_EAST.run(data, "geast_settings.xml");
    }

    private static void storeResult(IModelWithScore result, String filename) throws IOException {
        BifWriter writer = new BifWriter(new FileOutputStream(filename));
        writer.write(result);
    }
}
