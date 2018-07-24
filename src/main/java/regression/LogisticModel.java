package regression;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.ep.State;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.List;

public class LogisticModel {
    static ConcurrentHashMap<String, LogisticModel> models = new ConcurrentHashMap<>();
    final String name;
    private OnlineLogisticRegression learn;
    boolean hasIntercept;


    //TODO: hash names, intercept boolean

    public LogisticModel(String name, int numCategories, int numFeatures, boolean hasIntercept) {
        if (models.containsKey(name)) throw new IllegalArgumentException("Model " + name + " already exists, please remove it first.");

        this.name = name;
        this.hasIntercept = hasIntercept;
        learn = new OnlineLogisticRegression(numCategories, numFeatures + (hasIntercept ? 1 : 0), new L2(1));
        models.put(name, this);

    }

    public static LogisticModel from(String name) {
        LogisticModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid logistic model " + name);
    }

    public void add(List<Double> inputs, int output, int passes) {
        Vector v = listToVector(inputs, hasIntercept);
        for (int pass = 0; pass < passes; pass++) learn.train(output, v);
    }

    public int predict(List<Double> inputs) {
        Vector v = listToVector(inputs, hasIntercept);
        return learn.classifyFull(v).maxValueIndex();
    }

    /*protected void initTypes(Map<String, String> types, String output) {
        //int i = 0;
        for (Map.Entry<String, String> entry : types.entrySet()) {
            String key = entry.getKey();
            this.types.put(key, DataType.from(entry.getValue()));
            //this.offsets.put(key, i++);
        }
        //this.offsets.put(output, i);
    }*/

    static void remove(String model) {
        LogisticModel existing = models.remove(model);
        existing.learn.close();
    }

    static Vector listToVector(List<Double> args, boolean withIntercept) {
        int i; Vector v;
        if (withIntercept) {
            v = new DenseVector(args.size() + 1);
            v.set(0, 1);
            i = 1;
        } else {
            v = new DenseVector(args.size());
            i = 0;
        }
        for (Double d : args) v.set(i++, d);
        return v;
    }
}
