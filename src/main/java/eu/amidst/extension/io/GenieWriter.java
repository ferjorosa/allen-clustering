package eu.amidst.extension.io;

import eu.amidst.core.distribution.Normal;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Limitaciones actuales:
 * - El padre de una variable solo puede ser discreto
 * - Solo se permiten arboles/bosques, es decir, solo se considera un padre.
 * - Si hubiera variables ocultas continuas lanzaria excepcion porque no tienen valores maximos/minimos
 *
 * Nota sobre Genie:
 * - El orden de las definiciones debe ser jerarquico, es decir, un padre debe de ser definido antes que sus hijos
 * */
public class GenieWriter {

    private int numberOfDiscretizationIntervals;

    public GenieWriter() {
        this.numberOfDiscretizationIntervals = 10;
    }

    public GenieWriter(int numberOfDiscretizationIntervals) {
        this.numberOfDiscretizationIntervals = numberOfDiscretizationIntervals;
    }

    public void write(BayesianNetwork network, File file) throws ParserConfigurationException, TransformerException {
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.newDocument();

        doc.setXmlStandalone(false);

        Element rootElement = writeRoot(network, doc);
        writeNodes(network, doc, rootElement);
        writeExtensions(network, doc, rootElement);

        DOMSource source = new DOMSource(doc);

        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        Transformer transformer = transformerFactory.newTransformer();
        StreamResult result = new StreamResult(file);
        transformer.setOutputProperty(OutputKeys.ENCODING, "ISO-8859-1");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        transformer.transform(source, result);
    }

    private Element writeRoot(BayesianNetwork network, Document doc){
        Element rootElement = doc.createElement("smile");
        doc.appendChild(rootElement);

        /* Attributes */
        rootElement.setAttribute("version", "1.0");
        rootElement.setAttribute("id", network.getName());
        rootElement.setAttribute("numsamples", "10000");
        rootElement.setAttribute("discsamples", "10000");

        return rootElement;
    }

    private void writeNodes(BayesianNetwork network, Document doc, Element rootElement) {

        Element nodesElement = doc.createElement("nodes");
        rootElement.appendChild(nodesElement);

        /* Seleccionamos las variables que no tienen padre en el grafo */
        List<Variable> rootVariables = new ArrayList<>();
        for(Variable variable: network.getVariables()) {
            if(network.getDAG().getParentSet(variable).getNumberOfParents() == 0)
                rootVariables.add(variable);
        }

        /* Iteramos por las variables raiz de la red */
        for(Variable variable: rootVariables) {
            Map<Variable, Integer> visitedVars = new LinkedHashMap<>();
            network.getDAG().bfs(variable, visitedVars);
            for(Variable visitedVar: visitedVars.keySet())
                writeNode(visitedVar, network, doc, nodesElement);
        }
    }

    private void writeNode(Variable variable, BayesianNetwork network, Document doc, Element nodesElement) {

        if(variable.isDiscrete())
            writeDiscreteNode(variable, network, doc, nodesElement);

        else if(variable.isContinuous())
            writeContinuousNode(variable, network, doc, nodesElement);
    }

    private void writeDiscreteNode(Variable variable, BayesianNetwork network, Document doc, Element nodesElement) {

        /* Creamos el elemento correspondiente a la variable discreta */
        Element cptElement = doc.createElement("cpt");
        cptElement.setAttribute("id", variable.getName());
        nodesElement.appendChild(cptElement);

        /* 1 - Definimos sus estados */
        for (int i = 0; i < variable.getNumberOfStates(); i++) {
            Element stateElement = doc.createElement("state");
            stateElement.setAttribute("id", variable.getStateSpaceType().stringValue(i));
            cptElement.appendChild(stateElement);
        }

        /* 2 - Definimos sus padres */
        List<Variable> parentVariables = network.getDAG().getParentSet(variable).getParents();
        Element parentsElement = doc.createElement("parents");
        for (Variable parent : parentVariables) {
            parentsElement.appendChild(doc.createTextNode(parent.getName()));
            cptElement.appendChild(parentsElement);
        }

        /* 3 - Definimos sus probabilidades */
        double[] probabilities = network.getConditionalDistribution(variable).getParameters();
        StringBuilder probabilitiesString = new StringBuilder();
        for (int i = 0; i < probabilities.length; i++) {
            probabilitiesString.append(probabilities[i]);
            probabilitiesString.append(" ");
        }
        Element probabilitiesElement = doc.createElement("probabilities");
        probabilitiesElement.appendChild(doc.createTextNode(probabilitiesString.toString()));
        cptElement.appendChild(probabilitiesElement);
    }

    private void writeContinuousNode(Variable variable, BayesianNetwork network, Document doc, Element nodesElement) {

        /* Creamos el elemento correspondiente a la variable continua */
        Element equationElement = doc.createElement("equation");
        equationElement.setAttribute("id", variable.getName());
        nodesElement.appendChild(equationElement);

        RealStateSpace stateSpace = variable.getStateSpaceType();

        /* 1 - Definimos sus padres */
        List<Variable> parentVariables = network.getDAG().getParentSet(variable).getParents();
        if(parentVariables.size() > 0 ) {

            Element parentsElement = doc.createElement("parents");
            for (Variable parent : parentVariables) {
                parentsElement.appendChild(doc.createTextNode(parent.getName()));
                equationElement.appendChild(parentsElement);
            }

            // TODO: Right now only the first parent is considered
            Variable firstParent = parentVariables.get(0);

            /* 2 - Definimos sus parametros */
            Element definitionElement = doc.createElement("definition");

            if (stateSpace.getMinInterval() == Double.NEGATIVE_INFINITY)
                throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
            else
                definitionElement.setAttribute("lower", stateSpace.getMinInterval() + "");

            if (stateSpace.getMaxInterval() == Double.POSITIVE_INFINITY)
                throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
            else
                definitionElement.setAttribute("upper", stateSpace.getMaxInterval() + "");

            // Generamos la definicion con cada una de las normales asociadas a las componentes de la mixtura para la variable
            StringBuilder definitionString = new StringBuilder();
            definitionString.append(variable.getName());
            definitionString.append("=Choose(");
            definitionString.append(firstParent.getName());

            HashMapAssignment parentAssignment = new HashMapAssignment();
            for (int i = 0; i < firstParent.getNumberOfStates(); i++) {
                parentAssignment.setValue(firstParent, i);
                Normal normal = (Normal) network.getConditionalDistribution(variable).getUnivariateDistribution(parentAssignment);
                definitionString.append(",Normal(" + normal.getMean() + "," + normal.getVariance() + ")");
            }
            definitionString.append(")");

            definitionElement.appendChild(doc.createTextNode(definitionString.toString()));
            equationElement.appendChild(definitionElement);
        } else {

            /* 2 - Definimos sus parametros */
            Element definitionElement = doc.createElement("definition");

            if (stateSpace.getMinInterval() == Double.NEGATIVE_INFINITY)
                throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
            else
                definitionElement.setAttribute("lower", stateSpace.getMinInterval() + "");

            if (stateSpace.getMaxInterval() == Double.POSITIVE_INFINITY)
                throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
            else
                definitionElement.setAttribute("upper", stateSpace.getMaxInterval() + "");

            StringBuilder definitionString = new StringBuilder();
            definitionString.append(variable.getName());
            double[] meanVariance = network.getConditionalDistribution(variable).getParameters();
            definitionString.append("=Normal(" + meanVariance[0] + "," + meanVariance[1] + ")");

            definitionElement.appendChild(doc.createTextNode(definitionString.toString()));
            equationElement.appendChild(definitionElement);
        }

        /* 3 - Definimos su discretización */
        double min = stateSpace.getMinInterval();
        double max = stateSpace.getMaxInterval();
        double[] discretizationCutPoints = estimateDiscretizationCutPoints(min, max, this.numberOfDiscretizationIntervals);

        Element discretizationElement =  doc.createElement("discretization");
        for(int i = 0; i < discretizationCutPoints.length; i++){
            Element intervalElement = doc.createElement("interval");
            intervalElement.setAttribute("upper", discretizationCutPoints[i] + "");
            discretizationElement.appendChild(intervalElement);
        }

        equationElement.appendChild(discretizationElement);
    }

    private void writeExtensions(BayesianNetwork network, Document doc, Element rootElement) {

        Element extensionsElement = doc.createElement("extensions");
        rootElement.appendChild(extensionsElement);

        Element genieElement = doc.createElement("genie");
        genieElement.setAttribute("version", "1.0");
        genieElement.setAttribute("app", "GeNIe 2.3.3705.0 ACADEMIC");
        genieElement.setAttribute("name", network.getName());
        genieElement.setAttribute("faultnameformat", "nodestate");
        extensionsElement.appendChild(genieElement);

        for (Variable var : network.getDAG().getVariables()) {

            Element nodeElement = doc.createElement("node");
            nodeElement.setAttribute("id", var.getName());
            genieElement.appendChild(nodeElement);

            Element nameElement = doc.createElement("name");
            nameElement.appendChild(doc.createTextNode(var.getName()));
            nodeElement.appendChild(nameElement);

            Element interiorElement = doc.createElement("interior");
            interiorElement.setAttribute("color", "e5f6f7");
            nodeElement.appendChild(interiorElement);

            Element outlineElement = doc.createElement("outline");
            outlineElement.setAttribute("color", "000080");
            nodeElement.appendChild(outlineElement);

            Element fontElement = doc.createElement("font");
            fontElement.setAttribute("color", "000080");
            fontElement.setAttribute("name", "Arial");
            fontElement.setAttribute("size", "10");
            fontElement.setAttribute("bold", "true");
            nodeElement.appendChild(fontElement);

            Element positionElement = doc.createElement("position");
            positionElement.appendChild(doc.createTextNode("100 100 100 100"));
            nodeElement.appendChild(positionElement);

            Element barchartElement = doc.createElement("barchart");
            barchartElement.setAttribute("active", "true");
            barchartElement.setAttribute("width", "160");
            barchartElement.setAttribute("height", "110");
            nodeElement.appendChild(barchartElement);

        }
    }

    private double[] estimateDiscretizationCutPoints(double min, double max, int n) {
        double[] cutPoints = new double[n];
        double dist = (max-min) / (double) n;
        double current = min;
        for(int i = 0; i < n; i++) {
            current = current + dist;
            cutPoints[i] = current;
        }

        return cutPoints;
    }
}
