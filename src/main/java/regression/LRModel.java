package regression;
import java.util.concurrent.ConcurrentHashMap;
import java.util.*;
import java.lang.Double;
import org.neo4j.logging.Log;

public abstract class LRModel {

    private static ConcurrentHashMap<String, LRModel> models = new ConcurrentHashMap<>();
    final String name;
    State state;
    Framework framework;
    List<List<Double>> xTrain = new ArrayList<>();
    List<Double> yTrain = new ArrayList<>();
    List<List<Double>> xTest = new ArrayList<>();
    List<Double> yTest = new ArrayList<>();

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
        else return new LR.ModelResult(model, existing.framework, existing.hasConstant(), existing.getNumVars(), State.removed, existing.getN());

    }
    abstract long getN();

    abstract int getNumVars();

    abstract boolean hasConstant();

    void add(List<Double> given, double expected, String type, Log log) {
        switch(type) {
            case "train": addTrain(given, expected, log); break;
            case "test": addTest(given, expected, log); break;
            default: throw new IllegalArgumentException("Cannot add data of unrecognized type: " + type);
        }
    }

    boolean checkData(List<Double> given, double expected) {
        if (given == null || given.size() != getNumVars() || given.contains(null)) return false;
        return true;
    }

    abstract void addTrain(List<Double> given, double expected, Log log);

    abstract void addTest(List<Double> given, double expected, Log log);

    void addMany(List<List<Double>> given, List<Double> expected, String type, Log log) {
        if (given.size() != expected.size())
            throw new IllegalArgumentException("Length of given does not match length of expected.");
        switch(type) {
            case "train":
                for (int i = 0; i < given.size(); i++) addTrain(given.get(i), expected.get(i), log);
                break;
            case "test":
                for (int i = 0; i < given.size(); i++) addTest(given.get(i), expected.get(i), log);
                break;
            default:
                throw new IllegalArgumentException("Cannot add data of unrecognized type: " + type);
        }
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
    void removeData(List<Double> given, double expected, Log log) {
        throw new IllegalArgumentException("Cannot remove data from framework type: " + this.framework);
    }

    abstract LR.ModelResult train();

    abstract void test();

    abstract void copy(String source);

    protected enum State {created, training, testing, ready, removed, unknown}

    protected enum Framework {Simple, Miller, OLS, GLS, unknown}

    LR.ModelResult asResult() {
        return new LR.ModelResult(this.name, this.framework, this.hasConstant(), this.getNumVars(), this.state, this.getN());
    }

    void clear(String type) {
        switch(type) {
            case "all": xTrain.clear(); yTrain.clear(); xTest.clear(); yTest.clear(); break;
            case "test": xTest.clear(); yTest.clear(); break;
            case "train": xTrain.clear(); yTrain.clear(); break;
        }
    }

}
