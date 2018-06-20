package regression;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.procedure.Mode;
import java.io.*;
import java.util.*;
import java.util.stream.Stream;
import org.apache.commons.math3.util.MathArrays;

public class LR {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value = "regression.linear.create", mode = Mode.READ)
    @Description("Initialize a linear regression model with 'name' of type 'framework' and store in static memory. " +
            "Indicate whether to include a constant term. Accepted frameworks are 'Simple' and 'Miller'.")
    public Stream<ModelResult> create(@Name("name") String model, @Name("framework") String framework,
                                      @Name(value="include constant term?", defaultValue="true") boolean constant,
                                      @Name(value="# of independent vars", defaultValue="1") Long numVars) {
        return Stream.of(LRModel.create(model, framework, constant, numVars.intValue()).asResult());
    }

    @Procedure(value = "regression.linear.info", mode = Mode.READ)
    @Description("Return a stream containing the model's name (model), type (framework), whether it containes a constant term " +
            "(hasConstant), number of independent variables (numVars), state (state), number of observations (N), and information " +
            "(info). If the model is in state 'ready' info will contain parameters and statistics about the trained model.")
    public Stream<ModelResult> info(@Name("model") String model) {
        return Stream.of(LRModel.from(model).asResult());
    }

    @Procedure(value = "regression.linear.add", mode = Mode.READ)
    @Description("Void procedure which adds a single observation to the model. Indicate whether data is for training or testing the model.")
    public void add(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") double expected,
                    @Name(value="type", defaultValue="train") String type) {
        LRModel.from(model).add(given, expected, type, log);
    }

    @Procedure(value = "regression.linear.addM", mode = Mode.READ)
    @Description("Void procedure which adds multiple observations (given[i], expected[i]) to 'model'. Indicate whether data is for " +
            "training or testing the model.")
    public void addM(@Name("model") String model, @Name("given") List<List<Double>> given, @Name("expected") List<Double> expected,
                     @Name(value="type", defaultValue="train") String type) {
        LRModel.from(model).addMany(given, expected, type, log);
    }

    @Procedure(value = "regression.linear.remove", mode = Mode.READ)
    @Description("Void procedure which removes a single training observation from 'model'. Throws runtime error if model is not of " +
            "framework 'Simple'.")
    public void remove(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") double expected) {
        LRModel.from(model).removeData(given, expected, log);
    }

    @Procedure(value = "regression.linear.delete", mode = Mode.READ)
    @Description("Deletes 'model' from storage. Returns a stream containing the model's name (model), type (framework), " +
            "whether it containes a constant term (hasConstant), number of independent variables (numVars), state (state), " +
            "number of observations (N), and information (info).")
    public Stream<ModelResult> delete(@Name("model") String model) {
        return Stream.of(LRModel.removeModel(model));
    }

    @UserFunction(value = "regression.linear.predict")
    @Description("Function returns the model's prediction at 'given'. If the model is a type that must be trained and the model" +
            " is not in state 'ready', this function will first train the model.")
    public double predict(@Name("mode") String model, @Name("given") List<Double> given) {
        return LRModel.from(model).predict(given);
    }

    //change this?
    @UserFunction(value = "regression.linear.data")
    @Description("If the model is type 'simple' this function will serialize the model's Java object and returns the " +
            "byte[] serialization. If it is a type of multiple regression the function will return the double[] regression " +
            "parameters of the trained model. If this model is not yet trained, this function will first train the model.")
    public Object data(@Name("model") String model) {
        return LRModel.from(model).data();
    }

    @Procedure(value = "regression.linear.load", mode = Mode.READ)
    @Description("This procedure loads the model stored in data into the procedure's memory under the name 'model'. " +
            "The model must be of type 'Simple' and 'data' must be a byte array. Returns a stream containing the model's " +
            "name (model), type (framework), whether it containes a constant term (hasConstant), number of independent " +
            "variables (numVars), state (state), number of observations (N), and information (info).")
    public Stream<ModelResult> load(@Name("model") String model, @Name("data") Object data, @Name("framework") String framework) {
        return Stream.of(LRModel.load(model, data, framework).asResult());
    }

    @Procedure(value = "regression.linear.train", mode = Mode.READ)
    @Description("Trains the model and returns stream containing the model's name (model), type (framework), whether " +
            "it containes a constant term (hasConstant), number of independent variables (numVars), state (state), " +
            "number of observations (N), and information (info).")
    public Stream<ModelResult> train(@Name("model") String model) {
        return Stream.of(LRModel.from(model).train());
    }

    @Procedure(value = "regression.linear.test", mode = Mode.READ)
    @Description("Tests the fit of the model on test data and returns statistics.")
    public void test(@Name("model") String model) {LRModel.from(model).test();}

    @Procedure(value = "regression.linear.copy", mode = Mode.READ)
    @Description("Copies training and test data from model 'source' into model 'dest'.")
    public Stream<ModelResult> copy(@Name("source") String source, @Name("dest") String dest) {
        LRModel lrModel = LRModel.from(dest);
        lrModel.copy(source);
        return Stream.of(lrModel.asResult());
    }

    @UserFunction(value = "regression.linear.split")
    public List<Long> split(@Name("data") List<Long> data, @Name("fraction") double fraction) {
        int n = data.size();
        int k = (int) Math.floor(n*fraction);

        final int[] index = MathArrays.natural(n);
        MathArrays.shuffle(index);

        List<Long> subset = new ArrayList<>();
        for (int i = 0; i < k; i++) subset.add(data.get(index[i]));

        return subset;
    }

    @Procedure(value = "regression.linear.clear", mode=Mode.READ)
    public Stream<ModelResult> clear(@Name("model") String model, @Name(value="type", defaultValue = "all") String type) {
        LRModel lrModel = LRModel.from(model);
        lrModel.clear(type);
        return Stream.of(lrModel.asResult());
    }



    public static class ModelResult {
        public final String model;
        public final String framework;
        public final boolean hasConstant;
        public final long numVars;

        public final String state;
        public final long N;

        public Map<String,Object> info = new HashMap<>();

        ModelResult(String model, LRModel.Framework framework, boolean hasConstant, long numVars, LRModel.State state,
                    long N) {
            this.model = model;
            this.framework = framework.name();
            this.hasConstant = hasConstant;
            this.numVars = numVars;
            this.state = state.name();
            this.N = N;

        }

        ModelResult withInfo(Object...infos) {
            for (int i = 0; i < infos.length; i+=2) {
                info.put(infos[i].toString(),infos[i+1]);
            }
            return this;
        }
    }

    //Serializes the object into a byte array for storage
    static byte[] convertToBytes(Object object) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(object);
            return bos.toByteArray();
        }
    }

    //de serializes the byte array and returns the stored object
    static Object convertFromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
             ObjectInput in = new ObjectInputStream(bis)) {
            return in.readObject();
        }
    }

    static double[] doubleListToArray(List<Double> list) {
        int len = list.size();
        double[] array = new double[len];
        for (int i = 0; i < len; i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    static int[] intListToArray(List<Integer> list) {
        int len = list.size();
        int[] array = new int[len];
        for (int i = 0; i < len; i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    static List<Double> doubleArrayToList(double[] array) {
        List<Double> list = new ArrayList<>();
        for (double d : array) list.add(d);
        return list;
    }
}