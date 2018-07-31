package regression;
import org.apache.mahout.classifier.sgd.*;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.ep.State;

import javax.xml.crypto.Data;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.List;

public class LogisticModel {

    static ConcurrentHashMap<String, LogisticModel> models = new ConcurrentHashMap<>();
    final String name;
    private FeatureEncoder encoder;
    private OnlineLogisticRegression learn;


    public LogisticModel(String name, List<String> categories, Map<String, String> types, Map<String, Object> config) {
        if (models.containsKey(name)) throw new IllegalArgumentException("Model " + name + " already exists, please remove it first.");
        this.name = name;
        encoder = FeatureEncoder.create(categories, types, config);
        learn = encoder.getLearner();
        models.put(name, this);

    }

    public static LogisticModel from(String name) {
        LogisticModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid logistic model " + name);
    }

    public void add(String output, Map<String, Object> features) {
        Vector v = encoder.getVector(features);
        int o = encoder.outputStringToInt(output);
        learn.train(o, v);
    }

    public String predict(Map<String, Object> features) {
        Vector v = encoder.getVector(features);
        int o = learn.classifyFull(v).maxValueIndex();
        return encoder.outputIntToString(o);
    }

    static void remove(String model) {
        LogisticModel existing = models.remove(model);
        existing.learn.close();
    }

}
