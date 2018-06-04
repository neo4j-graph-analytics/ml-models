package regression;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.graphdb.Entity;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.procedure.Mode;
import org.neo4j.unsafe.impl.batchimport.cache.ByteArray;
import sun.java2d.pipe.SpanShapeRenderer;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

public class LR {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value = "regression.linear.create", mode = Mode.READ)
    @Description("Create a simple linear regression named 'model'. Returns a stream containing its name (model), state (state), and " +
            "number of data points (N).")
    public Stream<ModelResult> create(@Name("model") String model) {
        return Stream.of((new LRModel(model)).asResult());
    }

    @Procedure(value = "regression.linear.info", mode = Mode.READ)
    @Description("Returns a stream containing the model's name (model), state (state), and number of data points (N).")
    public Stream<ModelResult> info(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.stats", mode = Mode.READ)
    @Description("Returns a stream containing the model's intercept (intercept), slope (slope), coefficient of determination " +
            "(rSquare), and significance of the slope (significance).")
    public Stream<StatResult> stat(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return Stream.of(lrModel.stats());
    }

    @Procedure(value = "regression.linear.add", mode = Mode.READ)
    @Description("Void procedure which adds a single data point to 'model'.")
    public void add(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.add(given, expected);
    }

    @Procedure(value = "regression.linear.addM", mode = Mode.READ)
    @Description("Void procedure which adds multiple data points (given[i], expected[i]) to 'model'.")
    public void addM(@Name("model") String model, @Name("given") List<Double> given, @Name("expected") List<Double> expected) {
        LRModel lrModel = LRModel.from(model);
        if (given.size() != expected.size()) throw new IllegalArgumentException("Lengths of the two data lists are unequal.");
        for (int i = 0; i < given.size(); i++) {
            lrModel.add(given.get(i), expected.get(i));
        }
    }

    @Procedure(value = "regression.linear.remove", mode = Mode.READ)
    @Description("Void procedure which removes a single data point from 'model'.")
    public void remove(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.removeData(given, expected);
    }

    @Procedure(value = "regression.linear.delete", mode = Mode.READ)
    @Description("Deletes 'model' from storage. Returns a stream containing the model's name (model), state (state), and " +
            "number of data points (N).")
    public Stream<ModelResult> delete(@Name("model") String model) {
        return Stream.of(LRModel.removeModel(model));
    }

    @UserFunction(value = "regression.linear.predict")
    @Description("Function which returns a single double which is 'model' evaluated at the point 'given'.")
    public double predict(@Name("mode") String model, @Name("given") double given) {
        LRModel lrModel = LRModel.from(model);
        return lrModel.predict(given);
    }

    @UserFunction(value = "regression.linear.serialize")
    @Description("Function which serializes the model's Java object and returns the byte[] serialization.")
    public Object serialize(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return lrModel.serialize();
    }

    @Procedure(value = "regression.linear.load", mode = Mode.READ)
    @Description("Loads the model stored in data into the procedure's memory under the name 'model'. 'data' must be a byte array. " +
            "Returns a stream containing the model's name (model), state (state), and number of data points (N).")
    public Stream<ModelResult> load(@Name("model") String model, @Name("data") Object data) {
        SimpleRegression R;
        try { R = (SimpleRegression) convertFromBytes((byte[]) data); }
        catch (Exception e) {
            throw new RuntimeException("invalid data");
        }
        return Stream.of((new LRModel(model, R)).asResult());
    }

    public static class ModelResult {
        public final String model;
        public final String state;
        public final double N;
        public final Map<String,Object> info = new HashMap<>();

        public ModelResult(String model, LRModel.State state, double N) {
            this.model = model;
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

    public static class StatResult {
        public final double intercept;
        public final double slope;
        public final double rSquare;
        public final double significance;

        public StatResult(double intercept, double slope, double rSquare, double significance) {
            this.intercept = intercept;
            this.slope = slope;
            this.rSquare = rSquare;
            this.significance = significance;
        }
    }

    //Serializes the object into a byte array for storage
    public static byte[] convertToBytes(Object object) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(object);
            return bos.toByteArray();
        }
    }

    //de serializes the byte array and returns the stored object
    public static Object convertFromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
             ObjectInput in = new ObjectInputStream(bis)) {
            return in.readObject();
        }
    }




}