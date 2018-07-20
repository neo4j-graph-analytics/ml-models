package regression;
import vowpalWabbit.learner.VWLearners;
import vowpalWabbit.learner.VWScalarLearner;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

//THIS IS AN ABANDONED ATTEMPT TO IMPLEMENT LOGISTIC REGRESSION WITH THE VOWPAL WABBIT LIBRARY

public class LogisticModel {
    static {
        System.load("/Users/laurenshin/Library/Application Support/Neo4j Desktop/Application/neo4jDatabases/database-4937f2e8-8328-4dcd-826b-102a79630506/installation-3.3.4/plugins/libvw_jni.dylib");
    }
    static ConcurrentHashMap<String, LogisticModel> models = new ConcurrentHashMap<>();
    final String name;
    final Map<String, DataType> types = new HashMap<>();
    //final Map<String, Integer> offsets = new HashMap<>();
    final String output;
    final Map<Object, String> outputVals = new HashMap<>();
    private VWScalarLearner learn;


    //TODO: save model in file

    public LogisticModel(String name, Map<String, String> types, String output, Object high, Object low) {
        if (models.containsKey(name)) throw new IllegalArgumentException("Model " + name + " already exists, please remove it first.");

        this.name = name;
        this.output = output;
        initTypes(types, output);

        this.outputVals.put(high, "1");
        this.outputVals.put(low, "-1");
        learn = VWLearners.create("--quiet --loss_function logistic");
        models.put(name, this);

    }

    public static LogisticModel from(String name) {
        LogisticModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid logistic model " + name);
    }

    public void add(Map<String, Object> inputs, Object output) {
        String data = outputVals.get(output);
        if (data == null) return; //if output is not one of two expected outputs, skip this training data

        data = data.concat(" |");
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            switch (types.get(entry.getKey())) {
                case _class: data = data.concat(" " + entry.getKey() + "=" + entry.getValue().toString());
                case _float: data = data.concat(" " + entry.getKey() + ":" + entry.getValue().toString());
            }
        }
        learn.learn(data);
    }
    //TODO: helper function to transform input map into string

    public Object predict(Map<String, Object> inputs, double threshold) {
        String data = "|";
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            switch (types.get(entry.getKey())) {
                case _class: data = data.concat(" " + entry.getKey() + "=" + entry.getValue().toString());
                case _float: data = data.concat(" " + entry.getKey() + ":" + entry.getValue().toString());
            }
        }
        double val = learn.predict(data);
        String prediction = (val >= threshold) ? "1" : "-1";
        //TODO: clean this up
        for (Object o : outputVals.keySet()) {
            if (outputVals.get(o).equals(prediction)) return o;
        }
        return null;
    }

    protected void initTypes(Map<String, String> types, String output) {
        //int i = 0;
        for (Map.Entry<String, String> entry : types.entrySet()) {
            String key = entry.getKey();
            this.types.put(key, DataType.from(entry.getValue()));
            //this.offsets.put(key, i++);
        }
        //this.offsets.put(output, i);
    }

    static void remove(String model) {
        LogisticModel existing = models.remove(model);
        if (existing != null) {
            try {existing.learn.close(); } catch (Exception e) {

            }

        }
    }

    enum DataType {
        _class, _float;
        public static DataType from(String type) {
            switch(type.toUpperCase()) {
                case "CLASS": return DataType._class;
                case "FLOAT": return DataType._float;
                default: throw new IllegalArgumentException("Unknown type: " + type);
            }
        }
    }
}
