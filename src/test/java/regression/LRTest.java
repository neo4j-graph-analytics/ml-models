package regression;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.List;
import java.util.Arrays;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.*;

import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.equalTo;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

public class LRTest {

    private GraphDatabaseService db;

    @Before
    public void setUp() throws Exception {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(LR.class);
        procedures.registerFunction(LR.class);
    }

    @After
    public void tearDown() throws Exception {
        db.shutdown();
    }

    private void exhaust(Iterator r) {
        while(r.hasNext()) r.next();
    }

    private void check(Result result, Map<Double, Double> expected) {
        while (result.hasNext()) {
            Map<String, Object> actual = result.next();

            double time = (double) actual.get("time");
            double expectedPrediction = expected.get(time);
            double actualPrediction = (double) actual.get("predictedProgress");

            assertThat( actualPrediction, equalTo( expectedPrediction ) );
        }
    }

    @Test
    public void regression() throws Exception {
        //create known relationships for times 1, 2, 3
        db.execute("CREATE (:Node {id:1}) - [:WORKS_FOR {time:1.0, progress:1.345}] -> " +
                "(:Node {id:2}) - [:WORKS_FOR {time:2.0, progress:2.596}] -> " +
                "(:Node {id:3}) - [:WORKS_FOR {time:3.0, progress:3.259}] -> (:Node {id:4})");

        //create unknown relationships for times 4, 5
        db.execute("CREATE (:Node {id:5}) -[:WORKS_FOR {time:4.0}] -> " +
                "(:Node {id:6}) - [:WORKS_FOR {time:5.0}] -> (:Node {id:7})");

        //initialize the model
        db.execute("CALL regression.linear.create('work and progress')");

        //add known data
        Result r = db.execute("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) CALL " +
                "regression.linear.add('work and progress', r.time, r.progress) RETURN r");
        //these rows are computed lazily so we must iterate through all rows to ensure all data points are added to the model
        exhaust(r);

        //check that the correct info is stored in the model (should contain 3 data points)
        Map<String, Object> info = db.execute("CALL regression.linear.info('work and progress') YIELD model, state, N " +
                "RETURN model, state, N").next();
        assertTrue(info.get("model").equals("work and progress"));
        assertTrue(info.get("state").equals("ready"));
        assertThat(info.get("N"), equalTo(3.0));

        //store predictions
        String storePredictions = "MATCH (:Node)-[r:WORKS_FOR]->(:Node) WHERE exists(r.time) AND NOT exists(r.progress) " +
                "SET r.predictedProgress = regression.linear.predict('work and progress', r.time)";
        db.execute(storePredictions);

        //check that predictions are correct

        SimpleRegression R = new SimpleRegression();
        R.addData(1.0, 1.345);
        R.addData(2.0, 2.596);
        R.addData(3.0, 3.259);
        HashMap<Double, Double> expected = new HashMap<>();
        expected.put(4.0, R.predict(4.0));
        expected.put(5.0, R.predict(5.0));

        String gatherPredictedValues = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND " +
                "exists(r.predictedProgress) RETURN r.time as time, r.predictedProgress as predictedProgress";

        Result result = db.execute(gatherPredictedValues);

        check(result, expected);

        //serialize the model
        Map<String, Object> serial = db.execute("RETURN regression.linear.serialize('work and progress') as data").next();
        Object data = serial.get("data");

        //check that the byte[] model returns same predictions as the model stored in the procedure
        SimpleRegression storedR = (SimpleRegression) LR.convertFromBytes((byte[]) data);
        assertThat(storedR.predict(4.0), equalTo(expected.get(4.0)));
        assertThat(storedR.predict(5.0), equalTo(expected.get(5.0)));

        //delete model then re-create using serialization
        db.execute("CALL regression.linear.delete('work and progress')");
        Map<String, Object> params = new HashMap<>();
        params.put("data", data);
        db.execute("CALL regression.linear.load('work and progress', $data)", params);

        //remove data from relationship between nodes 1 and 2
        r = db.execute("MATCH (:Node {id:1})-[r:WORKS_FOR]->(:Node {id:2}) CALL regression.linear.remove('work " +
                "and progress', r.time, r.progress) return r");
        exhaust(r);

        //create a new relationship between nodes 7 and 8
        db.execute("MATCH (n7:Node {id:7}) MERGE (n7)-[:WORKS_FOR {time:6.0, progress:5.870}]->(:Node {id:8})");

        //add data from new relationship to model
        r = db.execute("MATCH (:Node {id:7})-[r:WORKS_FOR]->(:Node {id:8}) CALL regression.linear.add('work " +
                "and progress', r.time, r.progress) RETURN r.time");
        //again must iterate through rows
        exhaust(r);

        //map new model on all relationships with unknown progress
        db.execute(storePredictions);

        //replicate the creation and updates of the model
        R.removeData(1.0, 1.345);
        R.addData(6.0, 5.870);
        expected.put(4.0, R.predict(4.0));
        expected.put(5.0, R.predict(5.0));

        //make sure predicted values are correct
        result = db.execute(gatherPredictedValues);
        check(result, expected);

        //test addM procedure for adding multiple data points
        List<Double> points = Arrays.asList(7.0, 8.0);
        List<Double> observed = Arrays.asList(6.900, 9.234);
        params.put("points", points);
        params.put("observed", observed);

        db.execute("CALL regression.linear.addM('work and progress', $points, $observed)", params);
        db.execute(storePredictions);
        R.addData(7.0, 6.900);
        R.addData(8.0, 9.234);
        expected.put(4.0, R.predict(4.0));
        expected.put(5.0, R.predict(5.0));
        result = db.execute(gatherPredictedValues);

        check(result, expected);

        db.execute("CALL regression.linear.delete('work and progress')").close();


    }
}
