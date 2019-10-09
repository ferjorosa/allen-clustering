package experiments.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import org.latlab.data.MixedDataSet;
import org.latlab.util.SingularContinuousVariable;
import org.latlab.util.Variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AmidstToLatlabData {

    public static MixedDataSet transform(DataOnMemory<DataInstance> data){
        List<Variable> continuosVariables = new ArrayList<>();

        for(Attribute attribute: data.getAttributes())
            continuosVariables.add(new SingularContinuousVariable(attribute.getName()));

        MixedDataSet latlabData = MixedDataSet.createEmpty(continuosVariables, data.getNumberOfDataInstances());
        for(DataInstance instance: data){
            latlabData.add(1, Arrays.copyOf(instance.toArray(), continuosVariables.size()));
        }

        return latlabData;
    }
}
