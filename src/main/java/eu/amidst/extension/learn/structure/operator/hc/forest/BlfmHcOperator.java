package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.extension.learn.structure.Result;

public interface BlfmHcOperator {

    Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM);
}
