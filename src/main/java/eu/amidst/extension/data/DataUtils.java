package eu.amidst.extension.data;

import eu.amidst.core.datastream.*;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.inference.InferenceEngine;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;
import voltric.util.Tuple;

import java.util.ArrayList;
import java.util.List;

public class DataUtils {

    public static DataOnMemory<DataInstance> project(DataOnMemory<DataInstance> data, List<Attribute> attributesForProjection) {

        if(!data.getAttributes().getFullListOfAttributes().containsAll(attributesForProjection))
            throw new IllegalArgumentException("Not all the attributes are present in the data");

        /* Clonamos los atributos pero les asignamos un nuevo indice*/
        List<Attribute> projectedAttributesList = new ArrayList<>();
        for(int i = 0; i < attributesForProjection.size(); i++){
            Attribute attribute = attributesForProjection.get(i);
            projectedAttributesList.add(new Attribute(i, attribute.getName(), attribute.getStateSpaceType()));
        }

        Attributes projectedAttributes = new Attributes(projectedAttributesList);
        List<DataInstance> dataInstanceList = new ArrayList<>();

        for(DataInstance dataInstance: data) {
            double[] projectedData = new double[projectedAttributes.getNumberOfAttributes()];
            for (int i = 0; i < projectedAttributes.getNumberOfAttributes(); i++) {
                projectedData[i] = dataInstance.getValue(attributesForProjection.get(i));
            }
            dataInstanceList.add(new DataInstanceFromRawData(projectedAttributes, projectedData));
        }

        return new DataOnMemoryListContainer<>(projectedAttributes, dataInstanceList);
    }

    public static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel, List<Variable> discreteLatentVariables) {

        /* Seleccionamos las variables discretas latentes del modelo y creamos un atributo para cada una de ellas */
        List<Attribute> discreteLatentAttributesList = new ArrayList<>(discreteLatentVariables.size());
        int currentIndex = 0;
        for(Variable variable: discreteLatentVariables) {
            Attribute newAttribute = new Attribute(currentIndex, variable.getName(), variable.getStateSpaceType());
            discreteLatentAttributesList.add(newAttribute);
            currentIndex++;
        }

        Attributes discreteLatentAttributes = new Attributes(discreteLatentAttributesList);
        DataOnMemoryListContainer<DataInstance> completeData = new DataOnMemoryListContainer<>(discreteLatentAttributes);

        /* Iteramos por el conjunto de instancias y completamos los datos con un hard assignment de la posterior */
        for(DataInstance instance: data) {
            double[] newInstanceValues = new double[discreteLatentVariables.size()];
            for (int i = 0; i < discreteLatentVariables.size(); i++) {
                Variable latentVariable = discreteLatentVariables.get(i);
                Multinomial posterior = InferenceEngine.getPosterior(latentVariable, latentModel, instance);
                // Hard assignment (complete data with the state of maximum probability)
                int indexMaxProbability = maxProbIndex(posterior.getProbabilities());
                newInstanceValues[i] = indexMaxProbability;
            }
            completeData.add(new DataInstanceFromRawData(discreteLatentAttributes, newInstanceValues));
        }
        return completeData;
    }

    public static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel, Variable discreteLatentVariable) {
        List<Variable> discreteLatentVariables = new ArrayList<>(1);
        discreteLatentVariables.add(discreteLatentVariable);

        return completeDiscreteLatent(data, latentModel, discreteLatentVariables);
    }

    /** Devuelve un nuevo DataSet con los datos completados de las variables latentes del modelo */
    public static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel) {
        List<Variable> discreteLatentVariables = new ArrayList<>();
        for(Variable variable: latentModel.getVariables())
            if(variable.isDiscrete() && !variable.isObservable())
                discreteLatentVariables.add(variable);

        return completeDiscreteLatent(data, latentModel, discreteLatentVariables);
    }

    public static void defineAttributesMaxMinValues(DataOnMemory<DataInstance> data) {
        for(Attribute attribute: data.getAttributes()){
            if(attribute.isContinuous()) {
                /* Project data */
                List<Attribute> attributes = new ArrayList<>();
                attributes.add(attribute);
                DataOnMemory<DataInstance> projectedData = project(data, attributes);
                double[] projectedDataArray = new double[projectedData.getNumberOfDataInstances()];
                for(int i = 0; i < projectedData.getNumberOfDataInstances(); i++)
                    projectedDataArray[i] = projectedData.getDataInstance(i).toArray()[0];

                /* Estimate min and max values */
                Tuple<Double, Double> minMaxValues = estimateMinMaxValues(projectedDataArray);
                RealStateSpace stateSpace = attribute.getStateSpaceType();
                stateSpace.setMinInterval(minMaxValues.getFirst());
                stateSpace.setMaxInterval(minMaxValues.getSecond());
            }
        }
    }

    private static Tuple<Double, Double> estimateMinMaxValues(double[] projectedDataArray) {
        double max = -Double.MAX_VALUE;
        double min = Double.MAX_VALUE;

        for(int i = 0; i < projectedDataArray.length; i++){
            if(projectedDataArray[i] > max)
                max = projectedDataArray[i];
            else if(projectedDataArray[i] < min)
                min = projectedDataArray[i];
        }

        return new Tuple<>(min, max);
    }

    private static int maxProbIndex(double[] probs) {
        int index = 0;
        for(int i = 0; i < probs.length; i++)
            if(probs[index] < probs[i])
                index = i;

        return index;
    }
}
