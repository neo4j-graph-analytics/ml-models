package regression;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;
import org.neo4j.graphdb.Result;
import vowpalWabbit.learner.VWLearners;
import vowpalWabbit.learner.VWScalarLearner;

public class LogisticTest {
    private static GraphDatabaseService db;

    @BeforeClass
    public static void setUp() throws Exception {
        //System.load("/Users/laurenshin/vowpal_wabbit/java/target/libvw_jni.dylib");
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(Logistic.class);
        procedures.registerFunction(Logistic.class);
    }

    @AfterClass
    public static void tearDown() throws Exception {
        db.shutdown();
    }

    @Test
    public void makeModel() throws Exception {

        /*VWScalarLearner learn = VWLearners.create("--quiet --loss_function logistic --binary");
        for (int j = 0; j < 5e4; ++j) {
        learn.learn("-1 | number=2");
        learn.learn("-1 | number=4");
        learn.learn("-1 | number=6");
        learn.learn("-1 | number=8");
        learn.learn("1 | number=1");
        learn.learn("1 | number=3");
        learn.learn("1 | number=5");
        learn.learn("1 | number=7");}


        double exp = learn.predict("| number=5");
        learn.close();
        */

        db.execute("CALL regression.logistic.create('model', {number:'class'}, 'type', 'odd', 'even')").close();

        db.execute("CALL regression.logistic.add('model', {number:2}, 'even')").close();
        db.execute("CALL regression.logistic.add('model', {number:4}, 'even')").close();
        db.execute("CALL regression.logistic.add('model', {number:6}, 'even')").close();
        db.execute("CALL regression.logistic.add('model', {number:8}, 'even')").close();
        db.execute("CALL regression.logistic.add('model', {number:1}, 'odd')").close();
        db.execute("CALL regression.logistic.add('model', {number:3}, 'odd')").close();
        db.execute("CALL regression.logistic.add('model', {number:5}, 'odd')").close();
        db.execute("CALL regression.logistic.add('model', {number:7}, 'odd')").close();

        String r = (String) db.execute("RETURN regression.logistic.predict('model', {number:2}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '2': Expected: even, Actual: " + r, r.equals("even"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:4}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '4': Expected: even, Actual: " + r, r.equals("even"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:6}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '6': Expected: even, Actual: " + r, r.equals("even"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:8}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '8': Expected: even, Actual: " + r, r.equals("even"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:1}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '1': Expected: odd, Actual: " + r, r.equals("odd"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:3}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '3': Expected: odd, Actual: " + r, r.equals("odd"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:5}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '5': Expected: odd, Actual: " + r, r.equals("odd"));
        r = (String) db.execute("RETURN regression.logistic.predict('model', {number:7}, 0.0) as prediction").next().get("prediction");
        assertTrue("Failure to predict '7': Expected: odd, Actual: " + r, r.equals("odd"));

        db.execute("CALL regression.logistic.delete('model')");
    }

}
