package experiments.discrete;

import voltric.clustering.util.GenerateCompleteData;
import voltric.data.DiscreteData;
import voltric.io.data.DataFileLoader;
import voltric.io.model.xmlbif.XmlBifReader;
import voltric.model.DiscreteBayesNet;
import voltric.util.CollectionUtils;
import voltric.util.information.mi.MI;
import voltric.variables.DiscreteVariable;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/* Cargamos el LCM y estimamos sus 20 variables mas relevantes segun la informacion mutua con la var de clustering (pantalla)*/
public class estimate_most_relevant {

    public static void main(String[] args) throws Exception {
        mutualInfoMostRelevant(30);
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
