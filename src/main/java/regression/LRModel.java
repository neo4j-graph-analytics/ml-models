package regression;
import java.util.concurrent.ConcurrentHashMap;
import java.util.*;
import java.lang.Double;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.*;
import org.neo4j.procedure.UserAggregationUpdate;
import org.neo4j.register.Register;

import java.io.*;

public abstract class LRModel {

    private static ConcurrentHashMap<String, LRModel> models = new ConcurrentHashMap<>();
    final String name;
    State state;
    private Framework framework;
    //SimpleRegression R;


    LRModel(String model, String framework) {
        if (models.containsKey(model))
            throw new IllegalArgumentException("Model " + model + " already exists, please remove it first");
        this.name = model;
        this.state = State.created;
        switch(framework) {
            case "simple": this.framework = Framework.simple; break;
            case "miller": this.framework = Framework.miller; break;
            default: throw new IllegalArgumentException("Framework not recognized: " + framework);
        }
        //this.R = new SimpleRegression();
        models.put(name, this);
    }

    static LRModel from(String name) {
        LRModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid LR-Model " + name);
    }
    abstract LR.StatResult stats();

    static LR.ModelResult removeModel(String model) {
        LRModel existing = models.remove(model);
        return new LR.ModelResult(model, existing == null ? State.unknown : State.removed,
                existing == null ? Framework.unknown : existing.framework);
    }
    protected abstract long getN();

    public abstract void add(List<Double> given, double expected);

    public abstract double predict(List<Double> given);

    public abstract Object serialize();

    protected void removeData(double given, double expected) {
        throw new IllegalArgumentException("cannot remove data from framework type: " + framework);
    }

    public abstract Map<String, Double> train();

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

    public enum State {created, training, ready, removed, unknown}

    public enum Framework {simple, miller, unknown}

    LR.ModelResult asResult() {
        return new LR.ModelResult(this.name, this.state, this.framework);
    }

    static LRModel create(String name, double numVars, boolean constant, String framework) {
        switch (framework) {
            case "simple": return new SimpleLRModel(name, constant);
            case "miller": return new MillerLRModel(name, new Double(numVars).intValue(), constant);
            default: throw new IllegalArgumentException("Unknown framework: " + framework);
        }
    }

    static LRModel create(String name, Object data, String framework) {
        if (framework.equals("simple")) return new SimpleLRModel(name, data);
        else throw new IllegalArgumentException("Cannot load existing model for this framework: " + framework);
    }



}
