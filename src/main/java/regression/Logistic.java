package regression;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Procedure;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import java.util.Map;
import java.util.List;

public class Logistic {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    //TODO: output streams

    @Procedure(value="regression.logistic.create")
    public void create(@Name("name") String model, @Name("catagories") List<String> categories, @Name("featureTypes") Map<String, String> types,
                       @Name(value="config", defaultValue="{}") Map<String, Object> config) {
        new LogisticModel(model, categories, types, config);
    }

    @Procedure(value="regression.logistic.add")
    public void add(@Name("model") String model, @Name("category") String category, @Name("features") Map<String, Object> features) {
        LogisticModel lModel = LogisticModel.from(model);
        lModel.add(category, features);
    }

    @UserFunction(value="regression.logistic.predict")
    public String predict(@Name("model") String model, @Name("features") Map<String, Object> features) {
        LogisticModel lModel = LogisticModel.from(model);
        return lModel.predict(features);
    }

    @Procedure(value="regression.logistic.delete")
    public void delete(@Name("model") String model) {
        LogisticModel.remove(model);
    }

    /*public static class LogisticResult {
        public final String name;
        public final Map<S>
    }*/

}
