package regression;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;

import static org.junit.Assert.*;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.*;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;
import org.neo4j.graphdb.QueryExecutionException;

import static org.hamcrest.CoreMatchers.equalTo;
import org.apache.commons.math3.stat.regression.SimpleRegression;


public class LRTest {

    private static GraphDatabaseService db;

    @BeforeClass
    public static void setUp() throws Exception {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(LR.class);
        procedures.registerFunction(LR.class);

        //create known relationships for times 1, 2, 3
        db.execute("CREATE (:Node {id:1}) - [:WORKS_FOR {time:1.0, progress:1.345}] -> " +
                "(:Node {id:2}) - [:WORKS_FOR {time:2.0, progress:2.596}] -> " +
                "(:Node {id:3}) - [:WORKS_FOR {time:3.0, progress:3.259}] -> (:Node {id:4})");

        //create unknown relationships for times 4, 5
        db.execute("CREATE (:Node {id:5}) -[:WORKS_FOR {time:4.0}] -> " +
                "(:Node {id:6}) - [:WORKS_FOR {time:5.0}] -> (:Node {id:7})");
    }

    @AfterClass
    public static void tearDown() throws Exception {
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

    private void create() {
        db.execute("CALL regression.linear.create('work and progress', 'Simple', true, 1)").close();
    }

    private void add() {
        //add known data
        Result r = db.execute("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) CALL " +
                "regression.linear.add('work and progress', [r.time], r.progress) RETURN r");
        //these rows are computed lazily so we must iterate through all rows to ensure all data points are added to the model
        exhaust(r);
    }
    private void remove() {
        Result r = db.execute("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) CALL " +
                "regression.linear.remove('work and progress', [r.time], r.progress) RETURN r");
        exhaust(r);
    }

    private void delete() {
        db.execute("CALL regression.linear.delete('work and progress')").close();
    }

    private void storePredictions() {
        String storePredictions = "MATCH (:Node)-[r:WORKS_FOR]->(:Node) WHERE exists(r.time) AND NOT exists(r.progress) " +
                "SET r.predictedProgress = regression.linear.predict('work and progress', [r.time])";
        db.execute(storePredictions);
    }

    private void checkPredictions() {
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
    }

    @Test
    public void testCreate() throws Exception {
        Map<String, Object> info = db.execute("CALL regression.linear.create('work and progress', 'Simple', true, 1)").next();
        assertTrue(info.get("model").equals("work and progress"));
        assertTrue(info.get("framework").equals("Simple"));
        assertTrue((boolean) info.get("hasConstant"));
        assertEquals(1L, info.get("numVars"));
        assertTrue(info.get("state").equals("created"));
        assertEquals(0L, info.get("N"));
        delete();
    }

    @Test
    public void testCreateDefaults() throws Exception {
        Map<String, Object> info = db.execute("CALL regression.linear.create('work and progress', 'Simple')").next();
        assertTrue(info.get("model").equals("work and progress"));
        assertTrue(info.get("framework").equals("Simple"));
        assertTrue((boolean) info.get("hasConstant"));
        assertEquals(1L, info.get("numVars"));
        assertTrue(info.get("state").equals("created"));
        assertEquals(0L, info.get("N"));
        delete();
    }

    @Test
    public void testCreateErrors() throws Exception {
        try{
            db.execute("CALL regression.linear.create('work and progress', 'complicated', true, 1)").close();
            Assert.fail("Expecting QueryExecutionException because 'complicated' is not a valid model type.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        try {
            create();
            create();
            Assert.fail("Expecting QueryExecutionException because model with same name created twice.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        delete();
        try {
            db.execute("CALL regression.linear.create('work and progress', 'Simple', true, 342)").close();
        } catch (Exception ex) {
            Assert.fail("Create 'Simple' failed because input 342 independent variables, but 'Simple' model should" +
                    " automatically be created with 1 independent variable.");
        }
        delete();
    }

    @Test
    public void testAdd() throws Exception {
        create();
        add();
        Map<String, Object> info = db.execute("CALL regression.linear.info('work and progress') YIELD N RETURN N").next();
        assertEquals(3L, info.get("N"));
        delete();
    }

    @Test
    public void testAddErrors() throws Exception {
        try {
            db.execute("CALL regression.linear.add('work and progress', [1], 2)");
            Assert.fail("Expecting QueryExecutionException because tried to add data to model that doesn't exist.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        delete();
    }

    @Test
    public void testRemove() throws Exception {
        create();
        add();
        remove();
        Map<String, Object> info = db.execute("CALL regression.linear.info('work and progress') YIELD N RETURN N").next();
        assertEquals(0L, info.get("N"));
        delete();
    }

    @Test
    public void testAddMErrors() throws Exception {
        create();
        try {
            db.execute("CALL regression.linear.addM('work and progress', [[1]], [2, 3])");
            Assert.fail("Expecting QueryExecutionException because size of given not equal to size of expected.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        delete();
    }

    @Test
    public void testPredict() throws Exception {
        create();
        add();
        storePredictions();
        checkPredictions();
        delete();
    }

    @Test
    public void testSerialize() throws Exception {
        create();
        add();
        //serialize the model
        Map<String, Object> serial = db.execute("RETURN regression.linear.data('work and progress') as data").next();
        Object data = serial.get("data");
        delete();
        Map<String, Object> params = new HashMap<>();
        params.put("data", data);
        db.execute("CALL regression.linear.load('work and progress', $data, 'Simple')", params);

        storePredictions();
        checkPredictions();
        delete();
    }

    @Test
    public void testSerializeErrors() throws Exception {
        create();
        Map<String, Object> serial = db.execute("RETURN regression.linear.data('work and progress') as data").next();
        Object data = serial.get("data");
        Map<String, Object> params = new HashMap<>();
        params.put("data", data);
        try {
            db.execute("CALL regression.linear.load('work and progress', $data, 'Simple')", params);
            Assert.fail("Expecting QueryExecutionException because model 'work and progress' already exists.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        try {
            db.execute("CALL regression.linear.load('work and progress', $data, 'Miller')", params);
            Assert.fail("Expecting QueryExecutionException because tried to load SimpleRegression into Miller model.");
        } catch (QueryExecutionException ex) {
            //expected
        }
        delete();
    }

    @Test
    public void testTrain() throws Exception {
        //train doesn't really do anything
        create();
        db.execute("CALL regression.linear.train('work and progress')");
        delete();
    }
}
