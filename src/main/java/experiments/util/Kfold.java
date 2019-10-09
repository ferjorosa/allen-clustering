package experiments.util;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataOnMemoryListContainer;
import eu.amidst.core.datastream.filereaders.arffFileReader.ARFFDataWriter;
import eu.amidst.core.io.DataStreamLoader;
import voltric.util.Tuple;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Generates k folds from a dataset in the form of a list of tuples */
public class Kfold {

    public static void main(String[] args) throws Exception {

        /* Cargamos los datos */
        String filename = "data/original/discrete/asia.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();

        //generate(data, 5);
        generateAndExport(data, 5, "asia", "data/10-folds/discrete/asia/");
    }

    /** Generates a list of k train-test folds */
    public static List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> generate (DataOnMemory<DataInstance> data, int k) {

        /* Initialize the list of train-set folds */
        List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> trainTestFolds = new ArrayList<>(k);

        /* First, divide the dataset into k folds */
        int[] indices = new int[k+1];
        int division = data.getNumberOfDataInstances() / k;
        for(int i = 1; i < indices.length -1; i++) {
            int t = indices[i-1];
            indices[i] = t + division;
        }
        indices[k] = data.getNumberOfDataInstances();

        List<List<DataInstance>> folds = new ArrayList<>(k);

        for(int i = 0; i < k; i++){
            List<DataInstance> fold = new ArrayList<>(indices[i+1] - indices[i]);
            folds.add(fold);
            for(int j = indices[i]; j < indices[i+1]; j++){
                fold.add(data.getDataInstance(j));
            }
        }

        /* Then rotate these folds to generate a pair of train and test datasets */
        for(int i = 0; i < k; i++) {
            List<DataInstance> trainInstances = new ArrayList<>();
            List<DataInstance> testInstances = new ArrayList<>();
            for(int j = 0; j < k; j++) {
                if (j == i)
                    testInstances.addAll(folds.get(j));
                else
                    trainInstances.addAll(folds.get(j));
            }
            DataOnMemory<DataInstance> foldTrainData = new DataOnMemoryListContainer<>(data.getAttributes(), trainInstances);
            DataOnMemory<DataInstance> foldTestData = new DataOnMemoryListContainer<>(data.getAttributes(), testInstances);
            trainTestFolds.add(new Tuple<>(foldTrainData, foldTestData));
        }

        return trainTestFolds;
    }

    /** Generates a list of k train-test folds. Then it exports it to the specified path with the dataName + index in ARFF format */
    public static List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> generateAndExport(DataOnMemory<DataInstance> data,
                                                                                                        int k,
                                                                                                        String dataName,
                                                                                                        String path) throws IOException {
        List<Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> trainTestFolds = generate(data, k);

        for(int i = 0; i < trainTestFolds.size(); i++){
            Tuple<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = trainTestFolds.get(i);
            ARFFDataWriter.writeToARFFFile(fold.getFirst(), path + dataName + "_" + (i+1) + "_train.arff");
            ARFFDataWriter.writeToARFFFile(fold.getSecond(), path + dataName + "_" + (i+1) + "_test.arff");
        }

        return trainTestFolds;
    }
}
