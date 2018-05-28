package regression;

import java.util.HashMap;
import org.junit.Rule;
import org.junit.Test;
import org.neo4j.driver.v1.*;
import org.neo4j.harness.junit.Neo4jRule;

import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.equalTo;

import org.apache.commons.math3.stat.regression.SimpleRegression;

public class LRTest {

    // Start a Neo4j instance
    @Rule
    public Neo4jRule neo4j = new Neo4jRule()
            .withFunction(LR.class)
            .withProcedure(LR.class);


    private static String createKnownRelationships = "CREATE (:Node {id:1}) - [:WORKS_FOR {time:1.0, progress:1.345}] -> " +
            "(:Node {id:2}) - [:WORKS_FOR {time:2.0, progress:2.596}] -> " +
            "(:Node {id:3}) - [:WORKS_FOR {time:3.0, progress:3.259}] -> (:Node {id:4})";

    private static String createUnknownRelationships = "CREATE (:Node {id:5}) -[:WORKS_FOR {time:4.0}] -> " +
            "(:Node {id:6}) - [:WORKS_FOR {time:5.0}] -> (:Node {id:7})";

    private static String gatherPredictedValues = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND " +
            "exists(r.predictedProgress) RETURN r.time as time, r.predictedProgress as predictedProgress";

    @Test
    public void shouldPerformRegression() throws Throwable {
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {

            session.run(createKnownRelationships);
            session.run(createUnknownRelationships);
            //initialize the model
            session.run("CALL regression.linear.create('work and progress')");
            //add known data
            session.run("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) CALL " +
                    "regression.linear.addData('work and progress', r.time, r.progress) YIELD state RETURN r.time, r.progress, state");
            //store predictions
            session.run("MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND NOT exists(r.progress) CALL " +
                    "regression.linear.predict('work and progress', r.time) YIELD prediction SET r.predictedProgress = " +
                    "prediction");

            SimpleRegression R = new SimpleRegression();
            R.addData(1.0, 1.345);
            R.addData(2.0, 2.596);
            R.addData(3.0, 3.259);

            HashMap<Double, Double> expected = new HashMap<>();
            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));

            StatementResult result = session.run(gatherPredictedValues);

            while (result.hasNext()) {
                Record actual = result.next();

                double time = actual.get("time").asDouble();
                double expectedPrediction = expected.get(time);
                double actualPrediction = actual.get("predictedProgress").asDouble();

                assertThat(actualPrediction, equalTo(expectedPrediction));
            }

            session.run("CALL regression.linear.storeModel('work and progress')");
            session.run("CALL regression.linear.removeModel('work and progress')");
            session.run("CALL regression.linear.createFromStorage('work and progress')");


            //remove data from relationship between nodes 1 and 2
            session.run("MATCH (:Node {id:1})-[r:WORKS_FOR]->(:Node {id:2}) CALL regression.linear.removeData('work " +
                    "and progress', r.time, r.progress) YIELD state, N RETURN r.time as time, r.progress as progress, state, N");

            //create a new relationship between nodes 7 and 8
            session.run("MATCH (n7:Node {id:7}) MERGE (n7)-[:WORKS_FOR {time:6.0, progress:5.870}]->(:Node {id:8})");

            //add data from new relationship to model
            session.run("MATCH (:Node {id:7})-[r:WORKS_FOR]->(:Node {id:8}) CALL regression.linear.addData('work " +
                    "and progress', r.time, r.progress) YIELD state, N RETURN r.time, r.progress, state, N");

            //map new model on all relationships with unknown progress
            session.run("MATCH (:Node)-[r:WORKS_FOR]->(:Node) WHERE exists(r.time) AND NOT exists(r.progress) " +
                    "CALL regression.linear.predict('work and progress', " +
                    "r.time) YIELD prediction SET r.predictedProgress = prediction");

            //replicate the creation and updates of the model
            R.removeData(1.0, 1.345);
            R.addData(6.0, 5.870);

            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));

            //make sure predicted values are correct
            result = session.run(gatherPredictedValues);
            while (result.hasNext()) {
                Record actual = result.next();

                double time = actual.get("time").asDouble();
                double expectedPrediction = expected.get(time);
                double actualPrediction = actual.get("predictedProgress").asDouble();

                assertThat( actualPrediction, equalTo( expectedPrediction ) );
            }

        }
    }
}
