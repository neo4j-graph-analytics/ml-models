package regression;
import java.util.concurrent.ConcurrentHashMap;
import java.util.*;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.*;
import org.neo4j.procedure.UserAggregationUpdate;
import java.io.*;

public class LRModel {

    private static ConcurrentHashMap<String, LRModel> models = new ConcurrentHashMap<>();
    private final String name;
    private State state;
    SimpleRegression R;


    public LRModel(String model) {
        if (models.containsKey(model))
            throw new IllegalArgumentException("Model " + model + " already exists, please remove it first");
        this.name = model;
        this.state = State.created;
        this.R = new SimpleRegression();
        models.put(name, this);
    }

    public LRModel(String model, SimpleRegression R, String state) {
        if (models.containsKey(model))
            throw new IllegalArgumentException("Model " + model + " already exists, please remove it first");
        this.name = model;
        switch(state) {
            case "created": this.state = State.created;
            case "ready": this.state = State.ready;
            case "removed": this.state = State.removed;
            case "unknown": this.state = State.unknown;
        }
        this.R = R;
        models.put(name, this);
    }

    public static LRModel from(String name) {
        LRModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid LR-Model " + name);
    }

    public void add(double given, double expected) {
        R.addData(given, expected);
        if (R.getN() > 1) {
            this.state = State.ready;
        }
    }

    public LR.PredictResult predict(double given) {
        if (this.state == State.ready)
            return new LR.PredictResult(R.predict(given));
        throw new IllegalArgumentException("Not enough data in model to predict yet");
    }


    public void removeData(double given, double expected) {
        R.removeData(given, expected);
        if (R.getN() < 2) {
            this.state = State.created;
        }
    }

    public static LR.ModelResult removeModel(String model) {
        LRModel existing = models.remove(model);
        return new LR.ModelResult(model, existing == null ? State.unknown : State.removed, 0);
    }

    public void store(GraphDatabaseService db) {
        try{ byte[] serializedR = LinearRegression.convertToBytes(R);
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("name", name);
            ResourceIterator<Entity> n = db.execute("MERGE (n:LRModel {name:$name}) RETURN n", parameters).columnAs("n");
            Entity modelNode = n.next();
            modelNode.setProperty("state", state.name());
            modelNode.setProperty("serializedModel", serializedR);}
        catch (IOException e) { throw new RuntimeException(name + " cannot be serialized."); }
    }

    /*protected void initTypes(Map<String, String> types, String output) {
        if (!types.containsKey(output)) throw new IllegalArgumentException("Outputs not defined: " + output);
        int i = 0;
        for (Map.Entry<String, String> entry : types.entrySet()) {
            String key = entry.getKey();
            this.types.put(key, DataType.from(entry.getValue()));
            if (!key.equals(output)) this.offsets.put(key, i++);
        }
        this.offsets.put(output, i);
    }*/


    /*enum DataType {
        _class, _float, _order;

        public static DataType from(String type) {
            switch (type.toUpperCase()) {
                case "CLASS":
                    return DataType._class;
                case "FLOAT":
                    return DataType._float;
                case "ORDER":
                    return DataType._order;
                default:
                    throw new IllegalArgumentException("Unknown type: " + type);
            }
        }
    }*/

    public enum State {created, ready, removed, unknown}

    public LR.ModelResult asResult() {
        LR.ModelResult result = new LR.ModelResult(this.name, this.state, R.getN());
        return result;
    }



}
