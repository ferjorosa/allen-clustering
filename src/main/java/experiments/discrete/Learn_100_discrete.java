package experiments.discrete;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.Tuple4;
import experiments.util.DiscreteClusteringMeasures;
import methods.LCM;
import voltric.clustering.util.GenerateCompleteData;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.util.CollectionUtils;
import voltric.util.Tuple;
import voltric.util.information.mi.MI;
import voltric.variables.DiscreteVariable;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Learn_100_discrete {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        /* Aprendemos los modelos */
        //learnModels(seed, LogUtils.LogLevel.INFO);

        /* Cargamos el LCM y estimamos sus variables mas relevantes en orden por cada cluster (pantalla) */
        //hellingerMostRelevant(10);
        /* Cargamos el LCM y estimamos sus 20 variables mas relevantes segun la informacion mutua con la var de clustering (pantalla)*/
        mutualInfoMostRelevant(20);

        /* Cargamos el LCM y estimamos sus probabilidades condicionadas medias para cada cluster (CSV)*/

    }

    private static void learnModels(long seed, LogUtils.LogLevel logLevel) throws IOException {

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

    /** Estimamos las variables mas relevantes para cada uno de los clusters mediante la distancia de Helligner entre la CondProb y la marginal */
    private static void hellingerMostRelevant(int numberOfVars) throws Exception {
        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("clustVar");
        DiscreteBayesNet lcm = XmlBifReader.processFile(new File("models/discrete/exact/lcm_100.xml"), latentVarNames);
        List<List<Tuple<DiscreteVariable, Double>>> mostRelevantVarsPerCluster = DiscreteClusteringMeasures.mostRelevantVarsInLcmPerCluster(lcm);
        for(int cluster = 0; cluster < mostRelevantVarsPerCluster.size(); cluster++) {
            List<Tuple<DiscreteVariable, Double>> mostRelevantVarsInCluster = mostRelevantVarsPerCluster.get(cluster);
            System.out.println("\nCluster " + cluster + ":");
            for (int i = 0; i < numberOfVars; i++) {
                Tuple<DiscreteVariable, Double> variablePair = mostRelevantVarsInCluster.get(i);
                System.out.println(variablePair.getFirst().getName() + " -> " + variablePair.getSecond());
            }
        }
    }

    /** Estima las variables mas relevantes segun su MI con la variable de clustering y las muestra por pantalla */
    private static void mutualInfoMostRelevant(int numberOfVars) throws Exception {

        /* Cargamos el modelo */
        List<String> latentVarNames = new ArrayList<>(1);
        latentVarNames.add("clustVar");
        DiscreteBayesNet lcm = XmlBifReader.processFile(new File("models/discrete/exact/transformed/lcm_100_transformed.xml"), latentVarNames);

        /* Cargamos los datos */
        String filename = "data/discrete/exon_100_disc_transformed.arff";
        DiscreteData data = DataFileLoader.loadDiscreteData(filename);

        /* Completamos los datos */
        DiscreteData completedData = GenerateCompleteData.generateUnidimensional(data, lcm);

        /* Estimamos la MI entre los atributos y la variable de clustering */
        Map<DiscreteVariable, Double> mis = new LinkedHashMap<>();
        DiscreteVariable clustVar = completedData.getVariable("clustVar").get();
        for(DiscreteVariable attribute: completedData.getVariables()) {
            if(!attribute.getName().equals("clustVar")) {
                List<DiscreteVariable> projectVariables = new ArrayList<>();
                projectVariables.add(clustVar);
                projectVariables.add(attribute);
                DiscreteData projectedData = completedData.project(projectVariables);
                mis.put(attribute, MI.computePairwise(attribute, clustVar, projectedData));
            }
        }

        /* Ordenamos los resultados  de mayor a menor y devolvemos los n primeros */
        mis = CollectionUtils.sortByDescendingValue(mis);
        int n = 0;
        for(DiscreteVariable variable: mis.keySet()) {
            n++;
            if(n <= numberOfVars)
                System.out.println(variable.getName() + " -> " + mis.get(variable));
        }
    }
}
