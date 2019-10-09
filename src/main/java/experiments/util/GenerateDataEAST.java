package experiments.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.io.data.DataFileLoader;
import voltric.variables.DiscreteVariable;

import java.io.FileWriter;
import java.io.IOException;
import java.util.stream.Collectors;

/**
 * Simple script to transform ARFF files into DATA files that can be read by the EAST algorithm
 */
public class GenerateDataEAST {

    public static void main(String[] args) throws IOException{

        String dataName = "zoo";

        DiscreteData dataVoltric = DataFileLoader.loadDiscreteData("data/discrete/"+dataName+"/"+dataName+".arff");
        dataVoltric = dataVoltric.project(dataVoltric.getVariables().stream().filter(x->
                !x.getName().equals("class")
                        && !x.getName().equals("type")
                        && !x.getName().equals("animal")
                        && !x.getName().equals("C-class_flares_production_by_this_region")
                        && !x.getName().equals("M-class_flares_production_by_this_region")
                        && !x.getName().equals("X-class_flares_production_by_this_region")
        ).collect(Collectors.toList()));
        GenerateDataEAST.generate(dataVoltric, "data_east/"+dataName+"/"+dataName+".data");

        for(int i = 1; i < 11; i++) {

            DiscreteData foldTrain = DataFileLoader.loadDiscreteData("data/discrete/"+dataName+"/10_folds/"+dataName+"_"+i+"_train.arff");
            foldTrain = foldTrain.project(dataVoltric.getVariables().stream().filter(x->
                    !x.getName().equals("class")
                            && !x.getName().equals("type")
                            && !x.getName().equals("animal")
                            && !x.getName().equals("C-class_flares_production_by_this_region")
                            && !x.getName().equals("M-class_flares_production_by_this_region")
                            && !x.getName().equals("X-class_flares_production_by_this_region")
            ).collect(Collectors.toList()));
            GenerateDataEAST.generate(foldTrain, "data_east/"+dataName+"/10_folds/"+dataName+"_"+i+"_train.data");

            DiscreteData foldTest = DataFileLoader.loadDiscreteData("data/discrete/"+dataName+"/10_folds/"+dataName+"_"+i+"_test.arff");
            foldTest = foldTest.project(dataVoltric.getVariables().stream().filter(x->
                    !x.getName().equals("class")
                            && !x.getName().equals("type")
                            && !x.getName().equals("animal")
                            && !x.getName().equals("C-class_flares_production_by_this_region")
                            && !x.getName().equals("M-class_flares_production_by_this_region")
                            && !x.getName().equals("X-class_flares_production_by_this_region")
            ).collect(Collectors.toList()));
            GenerateDataEAST.generate(foldTest, "data_east/"+dataName+"/10_folds/"+dataName+"_"+i+"_test.data");
        }
    }

    public static void generate(DiscreteData data, String filePathString) throws IOException{

        FileWriter fw = new FileWriter(filePathString);

        // write data name
        fw.write("Name: "+ data.getName());

        // write attributes
        fw.write("\n\n//Variables: name of variable followed by names of states\n");

        for(DiscreteVariable var: data.getVariables()) {
            fw.write("\n" + var.getName()+": ");
            for(String state: var.getStates())
                fw.write(state+" ");
        }

        fw.write("\n\n//Records: Numbers in the last column are frequencies.\n\n");
        // write instances
        for (DiscreteDataInstance instance : data.getInstances())
            writeInstanceToFile(instance, fw, " ");

        fw.close();
    }

    private static void writeInstanceToFile(DiscreteDataInstance instance, FileWriter writer, String separator) throws IOException{
        try {
            String instanceString = instanceToString(instance, separator);
            double weight = instance.getData().getWeight(instance);

            writer.write(instanceString + "   " + weight + "\n");
        } catch (Exception e) {
            throw e;
        }
    }

    private static String instanceToString(DiscreteDataInstance instance, String separator) {
        String s = "";

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i = 0; i < instance.getTextualValues().size() - 1; i++)
            s += instance.getNumericValue(i) + separator;
        // Append the last column of the instance without the separator
        s += instance.getNumericValue(instance.getTextualValues().size() - 1);
        return s;
    }
}
