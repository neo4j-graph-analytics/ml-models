package regression;
import vowpalWabbit.learner.VWLearners;
import vowpalWabbit.learner.VWScalarLearner;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LogisticModel {
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
        String data = "1";
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            switch (types.get(entry.getKey())) {
                case _class: data = data.concat(" " + entry.getKey() + "=" + entry.getValue().toString());
                case _float: data = data.concat(" " + entry.getKey() + ":" + entry.getValue().toString());
            }
        }
        String prediction = (learn.predict(data) >= threshold) ? "1" : "-1";
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
