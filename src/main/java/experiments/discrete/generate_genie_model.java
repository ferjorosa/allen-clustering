package experiments.discrete;

import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

/* */
public class generate_genie_model {

    public static void main(String[] args) throws Exception {
        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("variable677");
        latentVarNames.add("variable1188");
        latentVarNames.add("variable1442");
        latentVarNames.add("variable1483");
        latentVarNames.add("variable1484");
        latentVarNames.add("variable1485");
        DiscreteBayesNet bi_model = XmlBifReader.processFile(new File("models/discrete/exact/bi_100.xml"), latentVarNames);
        String output = "models/bi_100_transformed.bif";
        BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
        writer.write(bi_model);
    }
}
