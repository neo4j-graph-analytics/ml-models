package regression;

import java.util.HashMap;
import java.util.Map;
import java.io.*;
import java.util.List;


import org.neo4j.graphdb.*;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;

import org.apache.commons.math3.stat.regression.SimpleRegression;



/** In this class I will create three implementations
 * of linear regression and one update procedure.
 *
 * 1. Create a user defined function in which the user manually inputs
 * the slope/intercept parameters obtained through a third
 * party data analysis tool, and then calls this function on the graph
 * to predict unknown values.
 *
 * 2. Create a user defined procedure in which the user specifies
 * node/relationship type the properties of that entity that will be the x and y values
 * of the regression. The procedure will then create a linear regression
 * model using all entities with known x and y values and map that model onto all entities
 * with known x but unknown y values, storing the predicted y value as another
 * property. Store serialized model object in a new node.
 *
 * 3. Create a user defined procedure in which the user provides one query that will return entities with properties
 * used to create the model, and another query that will return entities on which the model will predict unknown values.
 * Store serialized model object in a new node.
 *
 * 4. User defined procedure in which the user provides three queries: one provides data to remove from existing model,
 * one provides data to add to existing model, and one provides entities on which the model should provide updated
 * values. Existing model de-serialized, updated, and re-serialized to be stored on same LinReg node.
 */

public class LinearRegression {


    @UserFunction
    @Description("regression.predict(b, m, x) - uses the regression parameters intercept = b, slope = m, " +
            "and returns a predicted y value based on the equation y = m * x + b")
    public Double predict(@Name("intercept") Double intercept, @Name("slope") Double slope, @Name("input")
            Double input) {
        return (slope * input) + intercept;
    }

    // This field declares that we need a GraphDatabaseService
    // as context when any procedure in this class is invoked
    @Context
    public GraphDatabaseService db;

    // This gives us a log instance that outputs messages to the
    // standard log, normally found under `data/log/console.log`
    @Context
    public Log log;

    //single variable linear regression using node or relationship properties
    @Procedure(value = "regression.simpleRegression", mode = Mode.WRITE)
    @Description("create a linear regression model using independent and dependent property data from nodes/relationships that have" +
            " the given label and contain both properties. Then store predicted values under the property name " +
            "'newVarName' for nodes/relationships with the same label and known x but no known y property value. " +
            "Store the linear regression model in a new LinReg node. Use of nodes vs relationships specified with dataSource")
    public void simpleRegression(@Name("label") String label, @Name("independent variable") String indVar,
                                 @Name("dependent variable") String depVar, @Name("new variable name") String newVarName,
                                 @Name("data source") String dataSource) {

        if (!(dataSource.equals("node")||dataSource.equals("relationship"))) {
             throw new RuntimeException("Invalid dataSource (acceptable values are 'node' or 'relationship')");
        }

        Map<String, Object> parameters = new HashMap<>();
        parameters.put("label", label);
        parameters.put("indVar", indVar);
        parameters.put("depVar", depVar);

        //gathers points with known indVar and depVar
        Result resultKnown;

        //gathers points with known indVar but no known depVar
        Result resultUnknown;

        if (dataSource.equals("node")) {
            resultKnown = db.execute("MATCH (node) WHERE $label IN labels(node) AND $indVar IN keys(node) AND $depVar IN keys(node) " +
                    "RETURN DISTINCT node", parameters);
            resultUnknown = db.execute("MATCH (node) WHERE $label IN labels(node) AND $indVar IN keys(node) AND NOT $depVar IN keys(node) RETURN DISTINCT node", parameters);

        } else  {

            resultKnown = db.execute("MATCH () - [r] - () WHERE type(r) = $label AND $indVar IN keys(r) AND $depVar IN keys(r)" +
                            "RETURN DISTINCT r AS relationship", parameters);
            resultUnknown = db.execute("MATCH () - [r] - () WHERE type(r) = $label AND $indVar IN keys(r) AND NOT $depVar IN keys(r)" +
                    "RETURN DISTINCT r as relationship", parameters);
        }

        ResourceIterator<Entity> knownValues = resultKnown.columnAs(dataSource);
        SimpleRegression R = new SimpleRegression();

        //build the model
        while (knownValues.hasNext()) {
            Entity curr = knownValues.next();
            Object x = curr.getProperty(indVar);
            Object y = curr.getProperty(depVar);

            if (x instanceof Number && y instanceof Number) {
                R.addData(((Number) x).doubleValue(), ((Number) y).doubleValue());
            }

        }

        if (R.getN() < 2) {
            throw new RuntimeException("not enough known values to create a model");
        }

        //predict depVar values
        ResourceIterator<Entity> unknownValues = resultUnknown.columnAs(dataSource);
        while (unknownValues.hasNext()) {
            Entity curr = unknownValues.next();
            Object x = curr.getProperty(indVar);

            if (x instanceof Number) {
                curr.setProperty(newVarName, R.predict(((Number) x).doubleValue()));
            }
        }
        parameters.put("int", R.getIntercept());
        parameters.put("slope", R.getSlope());
        parameters.put("R2", R.getRSquare());
        ResourceIterator<Entity> modelNode = db.execute("CREATE (n:LinReg {label:$label, indVar:$indVar, depVar:$depVar, rSquare:$R2, " +
                "intercept:$int, slope:$slope}) RETURN n", parameters).columnAs("n");
        Entity n = modelNode.next();

        //Serialize R and store as property "serializedModel" in the new LinReg node
        try (ByteArrayOutputStream model = new ByteArrayOutputStream();
            ObjectOutput out = new ObjectOutputStream(model)){
            out.writeObject(R);
            byte[] byteModel = model.toByteArray();
            n.setProperty("serializedModel", byteModel);

        } catch (IOException e) {

        }
    }

    //separate function to clean up customRegression and updateRegression. Adds known values to the model R
    private void addValuesToModel(Result knownValues, SimpleRegression R) {
        List<String> columns = knownValues.columns();
        String indVar = columns.get(0);
        String depVar = columns.get(1);
        while(knownValues.hasNext()) {
            Map<String, Object> row = knownValues.next();
            Object x = row.get(indVar); Object y = row.get(depVar);
            if (x instanceof Number && y instanceof Number) {
                R.addData(((Number) x).doubleValue(), ((Number) y).doubleValue());
            }
        }
    }

    //separate function to clean up customRegression and updateRegression. Removes values from the model R
    private void removeValuesFromModel(Result toRemove, SimpleRegression R) {
        String indVar = toRemove.columns().get(0);
        String depVar = toRemove.columns().get(1);
        Map<String, Object> row;
        while (toRemove.hasNext()) {
            row = toRemove.next();
            Object x = row.get(indVar);
            Object y = row.get(depVar);
            if (x instanceof Number && y instanceof Number) {
                R.removeData(((Number) x).doubleValue(), ((Number) y).doubleValue());
            }
        }
    }

    //predicts and stores values using the model R
    private void setPredictedValues(Result r, SimpleRegression R, String newVarName) {
        String entity = r.columns().get(0);
        String indVar = r.columns().get(1);
        Map<String, Object> row;
        while(r.hasNext()) {
            row = r.next();
            Entity e = (Entity) row.get(entity);
            double x = ((Number) row.get(indVar)).doubleValue();
            e.setProperty(newVarName, R.predict(x));

        }
    }

    //Serializes the object into a byte array for storage
    static byte[] convertToBytes(Object object) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(object);
            return bos.toByteArray();
        }
    }

    //de serializes the byte array and returns the stored object
    static Object convertFromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        ObjectInput in = new ObjectInputStream(bis)) {
            return in.readObject();
        }
    }

    /* modelQuery must return a Result with first column indVar and second column depVar. This data will be used
    to create the model. If nonempty, mapQuery must return a two column Result with first column of type Entity (node or relationship)
    and second column if indVars. The predicted depVar value will be stored under the property named newVarName.
    Model will be serialized and stored in a node with modelID property. MAKE SURE YOUR QUERIES DON'T CONTAIN
    DUPLICATE VALUES OR THE MODEL WILL NOT BE CREATED CORRECTLY
     */
    @Procedure(value = "regression.customRegression", mode = Mode.WRITE)
    @Description("Create a linear regression model using the the two data points which result from running the modelQuery." +
            " Then store predicted values on the Entities that result from running the mapQuery.")
    public void customRegression(@Name("model query") String modelQuery, @Name("map query") String mapQuery,
                                 @Name("new variable name") String newVarName, @Name("model ID") long modelID) {

        Result knownValues;
        try
        {
            knownValues = db.execute(modelQuery);
        }
        catch (QueryExecutionException q)
        {
            throw new RuntimeException("model query is invalid");
        }

        SimpleRegression R = new SimpleRegression();

        addValuesToModel(knownValues, R);

        if (R.getN() < 2) {
            throw new RuntimeException("not enough data to create a model");
        }
        //store the model in a new LinReg node
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("int", R.getIntercept()); parameters.put("slope", R.getSlope());
        parameters.put("modelID", modelID); parameters.put("rSquare", R.getRSquare());

        ResourceIterator<Entity> modelNode = db.execute("CREATE (n:LinReg {intercept:$int, slope:$slope, " +
                "ID:$modelID, rSquare:$rSquare}) RETURN n", parameters).columnAs("n");
        Entity n = modelNode.next();

        //Serialize R and store as property "serializedModel" in the new LinReg node
        try {
            byte[] byteModel = convertToBytes(R);
            n.setProperty("serializedModel", byteModel);

        } catch (IOException e) {
            throw new RuntimeException("something went wrong, model can't be linearized so no serialized model was stored");
        }
        //if mapQuery is empty, we are done
        if (mapQuery.equals("")) return;
        //otherwise, we need to map our model onto unknown values
        Result r;
        String columnTitle;
        try
        {
            r = db.execute(mapQuery);
            setPredictedValues(r, R, newVarName);
        }
        catch (Exception q)
        {
            throw new RuntimeException("map query is invalid, no predicted values were stored");
        }





    }

    @Procedure(value = "regression.updateRegression", mode = Mode.WRITE)
    @Description("Update the linear regression model stored in the LinReg node with ID modelID by removing data, adding" +
            " data, and mapping updated predictions as specified by the provided queries.")
    public void updateRegression(@Name("remove query") String removeQuery, @Name("add query") String addQuery, @Name("map query") String mapQuery,
                                 @Name("new variable name") String newVarName, @Name("existing model ID") long modelID) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("ID", modelID);

        SimpleRegression R;
        Entity modelNode;
        //retrieve and deserialize the model
        try {
            ResourceIterator<Entity> n = db.execute("MATCH (n:LinReg {ID:$ID}) RETURN " +
                    "n", parameters).columnAs("n");
            modelNode = n.next();
            byte[] model = (byte[]) modelNode.getProperty("serializedModel");
            R = (SimpleRegression) convertFromBytes(model);

        } catch (Exception e) {
            throw new RuntimeException("no existing model for specified independent and dependent variables and model ID");
        }
        //if there is a nonempty removeQuery we must remove these values from the model
        if (!removeQuery.equals("")) {
            Result toRemove;
            try {
                toRemove = db.execute(removeQuery);
                removeValuesFromModel(toRemove, R);
            } catch (QueryExecutionException e) {
                throw new RuntimeException("invalid removeQuery");
            }
        }
        //if addQuery is nonempty add these values to the model
        if (!addQuery.equals("")) {
            Result toAdd;
            try {
                toAdd = db.execute(addQuery);
            } catch (QueryExecutionException e) {
                throw new RuntimeException("invalid addQuery");
            }
            addValuesToModel(toAdd, R);

        }
        if (R.getN() < 2) {
            throw new RuntimeException("not enough data remaining to create a model, process aborted");
        }
        //if mapquery nonempty, map new model
        if (!mapQuery.equals("")) {
            Result toMap;
            String columnTitle;
            try {
                toMap = db.execute(mapQuery);
            } catch (Exception e) {
                throw new RuntimeException("invalid mapQuery");
            }
            setPredictedValues(toMap, R, newVarName);
        }

        try {
            byte[] byteModel = convertToBytes(R);
            modelNode.setProperty("serializedModel", byteModel);

        } catch (IOException e) {
            throw new RuntimeException("something went wrong, model can't be linearized so new model not stored");
        }
        modelNode.setProperty("intercept", R.getIntercept());
        modelNode.setProperty("slope", R.getSlope());
        modelNode.setProperty("rSquare", R.getRSquare());

    }
}



