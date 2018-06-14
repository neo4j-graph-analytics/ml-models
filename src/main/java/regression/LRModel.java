package regression;
import java.util.concurrent.ConcurrentHashMap;
import java.util.*;
import java.lang.Double;

public abstract class LRModel {

    private static ConcurrentHashMap<String, LRModel> models = new ConcurrentHashMap<>();
    final String name;
    State state;
    Framework framework;

    static LRModel create(String model, String framework, boolean constant, int numVars) {
        switch(framework) {
            case "Simple": return new SimpleLRModel(model, constant);
            case "Miller": return new MillerLRModel(model, constant, numVars);
            case "OLS": return new OlsLRModel(model, constant, numVars);
            case "GLS": throw new IllegalArgumentException("GLS not yet implemented");
            default: throw new IllegalArgumentException("Invalid model type: " + framework);
        }
    }


    LRModel(String model, Framework framework) {
        if (models.containsKey(model))
            throw new IllegalArgumentException("Model " + model + " already exists, please remove it first");
        this.name = model;
        this.state = State.created;
        this.framework = framework;
        models.put(name, this);
    }

    static LRModel from(String name) {
        LRModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid LR-Model " + name);
    }
    //abstract LR.StatResult stats();

    static LR.ModelResult removeModel(String model) {
        LRModel existing = models.remove(model);
        if (existing == null) return new LR.ModelResult(model, Framework.unknown, false, 0,
                State.unknown, 0);
        else return existing.asResult();

    }
    abstract long getN();

    abstract long getNumVars();

    abstract boolean hasConstant();

    abstract void add(List<Double> given, double expected);

    void addMany(List<List<Double>> given, List<Double> expected) {
        if (given.size() != expected.size()) throw new IllegalArgumentException("Length of given does not match length of expected.");
        for (int i = 0; i < given.size(); i++) add(given.get(i), expected.get(i));
    }

    abstract double predict(List<Double> given);

    static LRModel load(String model, Object data, String framework) {
        switch (framework) {
            case "Simple": return new SimpleLRModel(model, data);
            default: throw new IllegalArgumentException("Cannot load model from data for this framework: " + framework);
        }
    }

    abstract Object data();

    //default error because only simple linear regression allows for removal of data points
    void removeData(List<Double> given, double expected) {
        throw new IllegalArgumentException("Cannot remove data from framework type: " + this.framework);
    }

    abstract LR.ModelResult train();

    protected enum State {created, training, ready, removed, unknown}

    protected enum Framework {Simple, Miller, OLS, GLS, unknown}

    LR.ModelResult asResult() {
        return new LR.ModelResult(this.name, this.framework, this.hasConstant(), this.getNumVars(), this.state, this.getN());
    }

}
