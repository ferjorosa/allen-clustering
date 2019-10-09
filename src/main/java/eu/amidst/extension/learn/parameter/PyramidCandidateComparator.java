package eu.amidst.extension.learn.parameter;

import eu.amidst.core.utils.CompoundVector;
import voltric.util.Tuple;

import java.util.Comparator;

public class PyramidCandidateComparator implements Comparator<Tuple<CompoundVector, Double>> {

    @Override
    public int compare(Tuple<CompoundVector, Double> o1, Tuple<CompoundVector, Double> o2) {
        if(o1.getSecond() > o2.getSecond())
            return -1;
        else if(o1.getSecond() < o2.getSecond())
            return 1;
        else
            return 0;
    }
}
