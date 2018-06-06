package regression;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.procedure.Mode;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

public class LR {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value = "regression.linear.Simple.create", mode = Mode.READ)
    @Description("Initialize a simple linear regression model named 'model' and store in static memory.")
    public Stream<ModelResult> createSimple(@Name("model") String model, @Name("include constant term?") boolean constant) {
        return Stream.of(new SimpleLRModel(model, constant).asResult());
    }

    @Procedure(value = "regression.linear.Miller.create", mode = Mode.READ)
    @Description("Initialize an updating OLS multiple linear regression model named 'model' and store it in static memory.")
    public Stream<ModelResult> createMiller(@Name("model") String model, @Name("include constant term?") boolean constant,
                                                @Name("number of variables") double numVars) {
        return Stream.of(new MillerLRModel(model, new Double(numVars).intValue(), constant).asResult());
    }

    @Procedure(value = "regression.linear.OLS.create", mode = Mode.READ)
    @Description("Initialize an ordinary least squares model for multiple linear regression and store it in static memory. " +
            "Use this model if the independent variables are uncorrelated.")
    public Stream<ModelResult> createOLS(@Name("model") String model, @Name("include constant term?") boolean constant,
                                         @Name("number of variables") double numVars) {
        return Stream.of(new OlsLRModel(model, new Double(numVars).intValue(), constant).asResult());
    }

    /*@Procedure(value = "regression.linear.GLS.create", mode = Mode.READ)
    @Description("Initialize a general least squares model for multiple linear regression and store it in static memory. " +
            "This model will calculate the covariance matrix and use it to calculate parameters at the time of training.")
    public Stream<ModelResult> createGLS(@Name("model") String model, @Name("include constant term?") boolean constant,
                                         @Name("number of variables") double numVars) {
        return Stream.of(new GlsLRModel(model, new Double(numVars).intValue(), constant).asResult());
    }*/

    @Procedure(value = "regression.linear.info", mode = Mode.READ)
    @Description("Return a stream containing the model's name (model), state (state), and type (framework).")
    public Stream<ModelResult> info(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.stats", mode = Mode.READ)
    @Description("Return a stream containing the model's number of observations (N), number of variables (numVars), " +
            "and other relevant statistics depending on the model's framework.")
    public Stream<StatResult> stat(@Name("model") String model) {
        return Stream.of(LRModel.from(model).stats());
    }

    /*@Procedure(value = "regression.linear.add", mode = Mode.READ)
    @Description("Void procedure which adds a single data point to 'model'.")
    public void add(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.add(given, expected);
    }*/

    @Procedure(value = "regression.linear.add", mode = Mode.READ)
    @Description("Add a single observation to the model. Void procedure.")
    public void add(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") double expected) {
        LRModel.from(model).add(given, expected);
    }

    /*@Procedure(value = "regression.linear.addM", mode = Mode.READ)
    @Description("Void procedure which adds multiple data points (given[i], expected[i]) to 'model'.")
    public void addM(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") List<Double> expected) {
        LRModel lrModel = LRModel.from(model);
        if (given.size() != expected.size()) throw new IllegalArgumentException("Lengths of the two data lists are unequal.");
        for (int i = 0; i < given.size(); i++) {
            lrModel.add(given.get(i), expected.get(i));
        }
    }*/

    @Procedure(value = "regression.linear.Simple.remove", mode = Mode.READ)
    @Description("Void procedure which removes a single observation from 'model'.")
    public void remove(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") double expected) {
        LRModel.from(model).removeData(given, expected);
    }

    @Procedure(value = "regression.linear.delete", mode = Mode.READ)
    @Description("Deletes 'model' from storage. Returns a stream containing the model's name (model), state (state), and " +
            "type (framework).")
    public Stream<ModelResult> delete(@Name("model") String model) {
        return Stream.of(LRModel.removeModel(model));
    }

    @UserFunction(value = "regression.linear.predict")
    @Description("Function returns the model's prediction at 'given'. If the model is a type that must be trained and the model" +
            " is not yet trained, this function will first train the model.")
    public double predict(@Name("mode") String model, @Name("given") List<Double> given) {
        return LRModel.from(model).predict(given);
    }

    @UserFunction(value = "regression.linear.data")
    @Description("If the model is type 'simple' this function will serialize the model's Java object and returns the " +
            "byte[] serialization. If it is a type of multiple regression the function will return the double[] regression " +
            "parameters of the trained model. If this model is not yet trained, this function will first train the model.")
    public Object data(@Name("model") String model) {
        return LRModel.from(model).data();
    }

    @Procedure(value = "regression.linear.Simple.load", mode = Mode.READ)
    @Description("This procedure loads the model stored in data into the procedure's memory under the name 'model'. " +
            "'data' must be a byte array. Returns a stream containing the model's name (model), state (state), and type (framework).")
    public Stream<ModelResult> load(@Name("model") String model, @Name("data") Object data) {
        SimpleRegression R;
        try { R = (SimpleRegression) convertFromBytes((byte[]) data); }
        catch (Exception e) {
            throw new RuntimeException("invalid data");
        }
        return Stream.of(new SimpleLRModel(model, data).asResult());
    }

    @Procedure(value = "regression.linear.train", mode = Mode.READ)
    @Description("Trains the model and returns parameters.")
    public void train(@Name("model") String model) {
        LRModel.from(model).train();
}

    public static class ModelResult {
        public final String model;
        public final String state;
        public final String framework;
        public final Map<String,Object> info = new HashMap<>();

        ModelResult(String model, LRModel.State state, LRModel.Framework framework) {
            this.model = model;
            this.state = state.name();
            this.framework = framework.name();
        }

        ModelResult withInfo(Object...infos) {
            for (int i = 0; i < infos.length; i+=2) {
                info.put(infos[i].toString(),infos[i+1]);
            }
            return this;
        }
    }

    public static class StatResult {
        public final long N;
        public final long numVars;
        public final Map<String, Object> info = new HashMap<>();

        StatResult(long N, long numVars) {
            this.N = N;
            this.numVars = numVars;
        }
        StatResult withInfo(Object...infos) {
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

    static double[] convertFromList(List<Double> list) {
        int len = list.size();
        double[] array = new double[len];
        for (int i = 0; i < len; i++) {
            array[i] = list.get(i);
        }
        return array;
    }




}