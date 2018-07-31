package regression;

import org.apache.mahout.classifier.sgd.*;
import org.apache.mahout.math.Vector;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

abstract class FeatureEncoder {
    //TODO: figure out best default vector size for feature hashing, or how to calculate a good defualt size based on number and types of features
    private static int DEFAULT_VECTOR_SIZE = 100;

    boolean hasIntercept;
    private Map<String, Integer> catToInt = new HashMap<>();
    private Map<Integer, String> intToCat = new HashMap<>();
    int vectorSize;
    Prior prior;

    static FeatureEncoder create(List<String> categories, Map<String, String> types, Map<String, Object> config) {
        Map<String, DataType> processedTypes = new HashMap<>();

        //set default values
        boolean hashing = initTypes(types, processedTypes);
        int hashedSize = DEFAULT_VECTOR_SIZE;
        Prior prior = Prior.unknown;
        boolean hasIntercept = true;

        for (Map.Entry<String, Object> entry : config.entrySet()) {
            Object value = entry.getValue();
            switch (entry.getKey()) {
                case "vectorSize":
                    hashedSize = ((Long) value).intValue();
                    hashing = true;
                    break;
                case "intercept":
                    if (!(boolean)value) hasIntercept = false;
                    break;
                case "prior":
                    prior = Prior.from((String) value);
                    if (config.containsKey("priorParam")) {
                        prior.param = ((Number) config.get("priorParam")).doubleValue();
                        prior.custom = true;
                    }
                    break;
                case "priorParam": break;
                default: throw new IllegalArgumentException("Unknown config entry: " + entry.getKey() + ". Acceptable entries " +
                        "are 'vectorSize', 'prior', and 'intercept'.");
            }
        }

        return hashing ? new HashingFeatureEncoder(categories, processedTypes, hasIntercept, hashedSize, prior) :
                new BasicFeatureEncoder(categories, types.keySet(), hasIntercept, prior);
    }

    FeatureEncoder(List<String> categories, boolean hasIntercept, Prior prior) {
        //process types, make sure all types valid and if any type is categorical set vector size to default for hashing
        initCategories(categories);
        this.hasIntercept = hasIntercept;
        this.prior = prior;
    }

    private PriorFunction getPrior() {
        switch(prior) {
            case L1: return new L1();
            case L2: return prior.custom ? new L2(prior.param) : new L2();
            case elastic: return prior.custom ? new ElasticBandPrior(prior.param) : new ElasticBandPrior();
            case uniform: return new UniformPrior();
            case t:
                if (!prior.custom) throw new IllegalArgumentException("Must specify df with 'priorParam' in config to use T prior function.");
                return new TPrior(prior.param);
            case unknown: return new L2(1);
            default: return new L2(1);
        }
    }

    OnlineLogisticRegression getLearner() {
        return new OnlineLogisticRegression(catToInt.size(), vectorSize, getPrior());
    }

    private void initCategories(List<String> categories) {
        int i = 0;
        for (String cat : categories) {
            catToInt.put(cat, i);
            intToCat.put(i++, cat);
        }
    }
    abstract Vector getVector(Map<String, Object> features);

    int outputStringToInt(String output) {
        return catToInt.get(output);
    }
    String outputIntToString(int o) {
        return intToCat.get(o);
    }
    enum DataType {
        _class, _float;
        public static DataType from(String type) {
            switch(type.toUpperCase()) {
                case "CLASS": return DataType._class;
                case "FLOAT": return DataType._float;
                default: throw new IllegalArgumentException("Unknown data type: " + type);
            }
        }

    }
    enum Prior {
        L1, L2, elastic, uniform, unknown, t;
        double param;
        boolean custom = false;
        public static Prior from (String src) {
            switch (src.toUpperCase()) {
                case "L1": return Prior.L1;
                case "L2": return Prior.L2;
                case "ELASTIC": return Prior.elastic;
                case "UNIFORM": return Prior.uniform;
                case "t": return Prior.t;
                default: throw new IllegalArgumentException("Unknown prior function: " + src);
            }
        }
    }
    private static boolean initTypes(Map<String, String> source, Map<String, DataType> dest) {
        boolean containsClass = false;
        for (Map.Entry<String, String> entry : source.entrySet()) {
            DataType t = DataType.from(entry.getValue());
            if (t.equals(DataType._class)) containsClass = true;
            dest.put(entry.getKey(), t);
        }
        return containsClass;
    }
}