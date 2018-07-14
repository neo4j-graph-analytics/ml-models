package regression;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Procedure;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import java.util.Map;

public class Logistic {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value="regression.logistic.create")
    public void create(@Name("name") String model, @Name("types") Map<String, String> types, @Name("output") String output,
                       @Name("first") Object first, @Name("second") Object second) {
        new LogisticModel(model, types, output, first, second);
    }

    @Procedure(value="regression.logistic.add")
    public void add(@Name("model") String model, @Name("inputs") Map<String, Object> inputs, @Name("output") Object output) {
        LogisticModel lModel = LogisticModel.from(model);
        lModel.add(inputs, output);
    }

    @UserFunction(value="regression.logistic.predict")
    public Object predict(@Name("model") String model, @Name("inputs") Map<String, Object> inputs, @Name("threshold") double threshold) {
        LogisticModel lModel = LogisticModel.from(model);
        return lModel.predict(inputs, threshold);
    }
}
