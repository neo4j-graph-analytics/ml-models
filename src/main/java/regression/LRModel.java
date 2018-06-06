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


    LRModel(String model, Framework framework) {
        if (models.containsKey(model))
            throw new IllegalArgumentException("Model " + model + " already exists, please remove it first");
        this.name = model;
        this.state = State.created;
        this.framework = framework;
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
    abstract long getN();

    abstract void add(List<Double> given, double expected);

    abstract double predict(List<Double> given);

    abstract Object data();

    void removeData(List<Double> given, double expected) {
        throw new IllegalArgumentException("cannot remove data from framework type: " + framework);
    }

    abstract Map<String, Double> train();

    public enum State {created, training, ready, removed, unknown}

    public enum Framework {Simple, Miller, OLS, GLS, unknown}

    LR.ModelResult asResult() {
        return new LR.ModelResult(this.name, this.state, this.framework);
    }

}
