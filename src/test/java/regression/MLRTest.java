package regression;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.After;
import static org.junit.Assert.*;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.*;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

import static org.hamcrest.CoreMatchers.equalTo;
import org.apache.commons.math3.stat.regression.MillerUpdatingRegression;
import java.io.File;
import java.util.Scanner;

//Right now this just performs simple lr using miller lr but I will improve

public class MLRTest {
    private static GraphDatabaseService db;
    private static String file = "resources/Boston_Housing.csv";

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

    @After
    public void deleteModel() {
        delete();
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
        db.execute("CALL regression.linear.create('work and progress', 'Multiple', true, 1)").close();
    }

    private void add() {
        //add known data
        Result r = db.execute("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) CALL " +
                "regression.linear.add('work and progress', [r.time], r.progress) RETURN r");
        //these rows are computed lazily so we must iterate through all rows to ensure all data points are added to the model
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
        Map<String, Object> info = db.execute("CALL regression.linear.create('work and progress', 'Multiple', true, 1)").next();
        assertTrue(info.get("model").equals("work and progress"));
        assertTrue(info.get("framework").equals("Multiple"));
        assertTrue((boolean) info.get("hasConstant"));
        assertEquals(1L, info.get("numVars"));
        assertTrue(info.get("state").equals("created"));
        assertEquals(0L, info.get("nTrain"));
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
    }

    @Test
    public void testAdd() throws Exception {
        create();
        add();
        Map<String, Object> info = db.execute("CALL regression.linear.info('work and progress') YIELD nTrain RETURN nTrain").next();
        assertEquals(3L, info.get("nTrain"));
    }

    @Test
    public void testAddErrors() throws Exception {
        try {
            db.execute("CALL regression.linear.add('work and progress', [1], 2)");
            Assert.fail("Expecting QueryExecutionException because tried to add data to model that doesn't exist.");
        } catch (QueryExecutionException ex) {
            //expected
        }
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
    }

    @Test
    public void testPredict() throws Exception {
        create();
        add();
        db.execute("CALL regression.linear.train('work and progress')");
        storePredictions();
        checkPredictions();
    }

    @Test
    public void testTrain() throws Exception {
        create();
        add();
        db.execute("CALL regression.linear.train('work and progress')");
    }
    /*
    @Test
    public void regression() throws Exception {
        Result r = db.execute("load csv with headers from 'http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv' " +
                "as row with toFloat(row['CRIM']) as crime, toFloat(row['RM']) as rooms, toFloat(row['MEDV']) as price " +
                "return crime, rooms, price");
        MillerUpdatingRegression R = new MillerUpdatingRegression(2, true);
        double[] x = new double[2];
        while (r.hasNext()) {
            Map<String, Object> row = r.next();
            x[0] = (double) row.get("crime"); x[1] = (double) row.get("rooms");
            R.addObservation(x, (double) row.get("price"));
        }
        db.execute("CALL regression.linear.create('boston housing', 'Miller', true, 2)").close();
        db.execute("load csv with headers from 'http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv' " +
                "as row with toFloat(row['CRIM']) as crime, toFloat(row['RM']) as rooms, toFloat(row['MEDV']) as price " +
                "call regression.linear.add('boston housing', [crime, rooms], price) return crime, rooms, price").close();
        //db.execute("call regression.linear.train('boston housing')");
        db.execute("call regression.linear.info('boston housing')").close();
        db.execute("call regression.linear.delete('boston housing')").close();
        assertTrue(true);


    }*/

}
