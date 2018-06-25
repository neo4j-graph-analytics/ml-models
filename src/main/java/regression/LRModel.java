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

    /////////CREATE/////////

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

    //////////TRAIN///////////

    void add(List<Double> given, double expected, String type, Log log) {
        switch(type) {
            case "train": addTrain(given, expected, log); break;
            case "test": addTest(given, expected, log); break;
            default: throw new IllegalArgumentException("Cannot add data of unrecognized type: " + type);
        }
    }
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

    void remove(List<Double> given, double expected, String type, Log log) {
        switch(type) {
            case "train": removeTrain(given, expected, log); break;
            case "test": removeTest(given, expected, log); break;
            default: throw new IllegalArgumentException("Cannot remove data of unrecognized type: " + type);
        }
    }

    void removeMany(List<List<Double>> given, List<Double> expected, String type, Log log) {
        if (given.size() != expected.size())
            throw new IllegalArgumentException("Length of given does not match length of expected.");
        switch(type) {
            case "train":
                for (int i = 0; i < given.size(); i++) removeTrain(given.get(i), expected.get(i), log);
                break;
            case "test":
                for (int i = 0; i < given.size(); i++) removeTest(given.get(i), expected.get(i), log);
                break;
            default:
                throw new IllegalArgumentException("Cannot add data of unrecognized type: " + type);
        }
    }

    abstract void addTrain(List<Double> given, double expected, Log log);

    abstract void removeTrain(List<Double> given, double expected, Log log);

    abstract LRModel copy(String source);

    LRModel clear(String type) {
        switch(type) {
            case "all": return clearAll();
            case "test": return clearTest();
            default: throw new IllegalArgumentException("Cannot clear data of type " + type);
        }
    }
    abstract LRModel clearAll();

    abstract LRModel train();

    //////////TEST///////////

    abstract void addTest(List<Double> given, double expected, Log log);
    abstract void removeTest(List<Double> given, double expected, Log log);
    abstract LRModel test();
    abstract LRModel clearTest();

    //////////PREDICT/////////

    abstract double predict(List<Double> given);

    /////////STORE/DELETE///////////

    abstract Object data();


    static LR.ModelResult removeModel(String model) {
        LRModel existing = models.remove(model);
        if (existing == null) return new LR.ModelResult(model, Framework.unknown, false, 0,
                State.unknown, 0, 0);
        else return new LR.ModelResult(model, existing.framework, existing.hasConstant(), existing.getNumVars(), State.removed, existing.getNTrain(), existing.getNTest());

    }

    static LRModel load(String model, Object data, String framework) {
        switch (framework) {
            case "Simple": return new SimpleLRModel(model, data);
            default: throw new IllegalArgumentException("Cannot load model from data for this framework: " + framework);
        }
    }

    //////////INFO//////////

    static LRModel from(String name) {
        LRModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid LR-Model " + name);
    }

    abstract long getNTrain();

    abstract long getNTest();

    abstract int getNumVars();

    abstract boolean hasConstant();

    LR.ModelResult asResult() {
        return new LR.ModelResult(this.name, this.framework, this.hasConstant(), this.getNumVars(), this.state, this.getNTrain(), this.getNTest());
    }

    protected enum State {created, training, testing, ready, removed, unknown}

    protected enum Framework {Simple, Multiple, OLS, GLS, unknown}

    /////////UTILS////////

    boolean dataInvalid(List<Double> given) {
        return (given == null || given.size() != getNumVars() || given.contains(null));
    }

























}
