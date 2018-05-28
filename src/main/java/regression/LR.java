package regression;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.graphdb.Entity;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.ResourceIterator;
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

    @Procedure(value = "regression.linear.create", mode = Mode.READ)
    public Stream<ModelResult> create(@Name("model") String model) {
        return Stream.of((new LRModel(model)).asResult());
    }

    @Procedure(value = "regression.linear.addData", mode = Mode.READ)
    public Stream<ModelResult> addData(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.add(given, expected);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.removeData", mode = Mode.READ)
    public Stream<ModelResult> removeData(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.removeData(given, expected);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.removeModel", mode = Mode.READ)
    public Stream<ModelResult> removeModel(@Name("model") String model) {
        return Stream.of(LRModel.removeModel(model));
    }

    @Procedure(value = "regression.linear.predict", mode = Mode.READ)
    public Stream<PredictResult> predict(@Name("mode") String model, @Name("given") double given) {
        LRModel lrModel = LRModel.from(model);
        return Stream.of(lrModel.predict(given));
    }

    @Procedure(value = "regression.linear.storeModel", mode = Mode.WRITE)
    public Stream<ModelResult> storeModel(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        lrModel.store(db);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.createFromStorage", mode = Mode.READ)
    public Stream<ModelResult> createFromStorage(@Name("model") String model) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("name", model);
        Entity modelNode;
        SimpleRegression R;
        try {
            ResourceIterator<Entity> n = db.execute("MATCH (n:LRModel {name:$name}) RETURN " +
                    "n", parameters).columnAs("n");
            modelNode = n.next();
            byte[] m = (byte[]) modelNode.getProperty("serializedModel");
            R = (SimpleRegression) LinearRegression.convertFromBytes(m);
        } catch (Exception e) {
            throw new RuntimeException("no existing model for specified independent and dependent variables and model ID");
        }
        return Stream.of(new LRModel(model, R, (String) modelNode.getProperty("state")).asResult());
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

    public static class PredictResult {
        public final double prediction;
        public PredictResult(double p) {
            this.prediction = p;
        }
    }



}