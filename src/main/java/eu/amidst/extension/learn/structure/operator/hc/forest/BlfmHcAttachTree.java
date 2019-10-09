package eu.amidst.extension.learn.structure.operator.hc.forest;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.extension.learn.structure.Result;

/**
 * Este operador itera por todas aquellas variables latentes raiz y por todas las variables observadas e intenta añadirlas
 * como hijas a un nodo latente.
 * TODO: En proceso, tengo que ver si merece la pena
 */
public class BlfmHcAttachTree implements BlfmHcOperator {

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {



        return null;
    }
}
