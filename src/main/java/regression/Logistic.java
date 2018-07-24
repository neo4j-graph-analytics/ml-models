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

    //TODO: take priorfunction as input

    @Procedure(value="regression.logistic.create")
    public void create(@Name("name") String model, @Name("numCatagories") Long numCatagories, @Name("numFeatures") Long numFeatures,
                       @Name("hasIntercept") boolean hasIntercept) {
        new LogisticModel(model, numCatagories.intValue(), numFeatures.intValue(), hasIntercept);
    }

    //TODO: allow user to specify passes, look into issues with multiple passes with same data
    @Procedure(value="regression.logistic.add")
    public void add(@Name("model") String model, @Name("inputs") List<Double> inputs, @Name("output") Long output) {
        LogisticModel lModel = LogisticModel.from(model);
        lModel.add(inputs, output.intValue(), 1);
    }

    @UserFunction(value="regression.logistic.predict")
    public long predict(@Name("model") String model, @Name("inputs") List<Double> inputs) {
        LogisticModel lModel = LogisticModel.from(model);
        return lModel.predict(inputs);
    }

    @Procedure(value="regression.logistic.delete")
    public void delete(@Name("model") String model) {
        LogisticModel.remove(model);
    }

}
