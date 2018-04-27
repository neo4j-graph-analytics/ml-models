package example;
import java.util.HashMap;
import org.junit.Rule;
import org.junit.Test;
import org.neo4j.cypher.internal.frontend.v2_3.ast.functions.Has;
import org.neo4j.driver.v1.*;
import org.neo4j.harness.junit.Neo4jRule;
import java.io.*;

import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.equalTo;

import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.util.Optional;

public class LinearRegressionTest {


    // Start a Neo4j instance
    @Rule
    public Neo4jRule neo4j = new Neo4jRule()
            .withFunction(LinearRegression.class)
            .withProcedure(LinearRegression.class);

    @Test
    public void shouldPredictValues() throws Throwable {

        // Create a driver session, and run Cypher query
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {

            Double result = session
                    .run("RETURN example.predict(3.9, 0.453, 2) AS result")
                    .single().get("result").asDouble();

            assertThat(result, equalTo(0.453*2 + 3.9));
        }
    }


    @Test
    public void shouldCreateNodeRegression() throws Throwable {
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {

            session.run("CREATE (:node {time:1.0, progress:1.345}), (:node {time:2.0, progress:2.596}), " +
                    "(:node {time:3.0, progress:3.259}), (:node {time:4.0}), (:node {time:5.0})");
            session.run("CALL example.simpleRegression('node', 'time', 'progress', 'predictedProgress', 'node')");
            StatementResult result = session.run("MATCH (n:node) WHERE exists(n.predictedProgress) RETURN n.time as time, n.predictedProgress as predictedProgress");

            SimpleRegression R = new SimpleRegression();
            R.addData(1.0, 1.345);
            R.addData(2.0, 2.596);
            R.addData(3.0, 3.259);

            HashMap<Double, Double> expected= new HashMap<>();
            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));


            while (result.hasNext()) {
                Record actual = result.next();

                double time = actual.get("time").asDouble();
                double expectedPrediction = expected.get(time);
                double actualPrediction = actual.get("predictedProgress").asDouble();

                assertThat(actualPrediction, equalTo(expectedPrediction));
            }

            Record model = session.run("MATCH (n:LinReg {label:'node', indVar:'time', depVar:'progress'}) " +
                    "WHERE exists(n.serializedModel) RETURN n.intercept as intercept, n.slope as slope, n.serializedModel as serializedModel").single();

            assertEquals(model.get("intercept").asDouble(), R.getIntercept(), 0.00000000000001);
            assertEquals(model.get("slope").asDouble(), R.getSlope(), 0.00000000000001);


            try (ByteArrayInputStream serializedModel = new ByteArrayInputStream(model.get("serializedModel").asByteArray());
                ObjectInput in = new ObjectInputStream(serializedModel)){

                SimpleRegression o = (SimpleRegression) in.readObject();
                assertEquals(model.get("intercept").asDouble(), o.getIntercept(), 0.00000000000001);
                assertEquals(model.get("slope").asDouble(), o.getSlope(), 0.00000000000001);

            } catch (IOException e) {
                fail("error in deserialization");
            }


        }
    }

    private static String createKnownRelationships = "CREATE (:Node {id:1}) - [:WORKS_FOR {time:1.0, progress:1.345}] -> " +
            "(:Node {id:2}) - [:WORKS_FOR {time:2.0, progress:2.596}] -> " +
            "(:Node {id:3}) - [:WORKS_FOR {time:3.0, progress:3.259}] -> (:Node {id:4})";

    private static String createUnknownRelationships = "CREATE (:Node {id:5}) -[:WORKS_FOR {time:4.0}] -> " +
            "(:Node {id:6}) - [:WORKS_FOR {time:5.0}] -> (:Node {id:7})";

    private static String gatherPredictedValues = "MATCH () - [r:WORKS_FOR] - () WHERE exists(r.time) AND " +
            "exists(r.predictedProgress) RETURN r.time as time, r.predictedProgress as predictedProgress";
    @Test
    public void shouldCreateRelRegression() throws Throwable {
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {

            session.run(createKnownRelationships);
            session.run(createUnknownRelationships);
            session.run("CALL example.simpleRegression('WORKS_FOR', 'time', 'progress', 'predictedProgress', 'relationship')");
            StatementResult result = session.run(gatherPredictedValues);

            SimpleRegression R = new SimpleRegression();
            R.addData(1.0, 1.345);
            R.addData(2.0, 2.596);
            R.addData(3.0, 3.259);

            HashMap<Double, Double> expected = new HashMap<>();
            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));

            while (result.hasNext()) {
                Record actual = result.next();

                double time = actual.get("time").asDouble();
                double expectedPrediction = expected.get(time);
                double actualPrediction = actual.get("predictedProgress").asDouble();

                assertThat( actualPrediction, equalTo( expectedPrediction ) );


            }
            Record model = session.run("MATCH (n:LinReg {label:'WORKS_FOR', indVar:'time', depVar:'progress'}) " +
                    "WHERE exists(n.serializedModel) RETURN n.intercept as intercept, n.slope as slope, n.serializedModel as serializedModel").single();

            assertEquals(model.get("intercept").asDouble(), R.getIntercept(), 0.00000000000001);
            assertEquals(model.get("slope").asDouble(), R.getSlope(), 0.00000000000001);

            try (ByteArrayInputStream serializedModel = new ByteArrayInputStream(model.get("serializedModel").asByteArray());
                 ObjectInput in = new ObjectInputStream(serializedModel)){

                SimpleRegression o = (SimpleRegression) in.readObject();
                assertEquals(model.get("intercept").asDouble(), o.getIntercept(), 0.00000000000001);
                assertEquals(model.get("slope").asDouble(), o.getSlope(), 0.00000000000001);

            } catch (IOException e) {
                fail("error in deserialization");
            }

        }
    }

    @Test
    public void shouldCreateCustomRegression() throws Throwable {
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {
            session.run(createKnownRelationships);
            session.run(createUnknownRelationships);


            String modelQuery = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) RETURN r.time as time, r.progress as progress";
            String mapQuery = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND NOT exists(r.progress) RETURN r, r.time as time";
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("modelQuery", modelQuery);
            parameters.put("mapQuery", mapQuery);

            session.run("CALL example.customRegression($modelQuery, $mapQuery, 'predictedProgress', 1)", parameters);

            StatementResult result = session.run(gatherPredictedValues);

            SimpleRegression R = new SimpleRegression();
            R.addData(1.0, 1.345);
            R.addData(2.0, 2.596);
            R.addData(3.0, 3.259);

            HashMap<Double, Double> expected = new HashMap<>();
            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));

            while (result.hasNext()) {
                Record actual = result.next();

                double time = actual.get("time").asDouble();
                double expectedPrediction = expected.get(time);
                double actualPrediction = actual.get("predictedProgress").asDouble();

                assertThat( actualPrediction, equalTo( expectedPrediction ) );


            }
            Record model = session.run("MATCH (n:LinReg {ID: 1}) " +
                    "WHERE exists(n.serializedModel) RETURN n.intercept as intercept, n.slope as slope, n.serializedModel as serializedModel").single();

            assertEquals(model.get("intercept").asDouble(), R.getIntercept(), 0.00000000000001);
            assertEquals(model.get("slope").asDouble(), R.getSlope(), 0.00000000000001);

            try (ByteArrayInputStream serializedModel = new ByteArrayInputStream(model.get("serializedModel").asByteArray());
                 ObjectInput in = new ObjectInputStream(serializedModel)){

                SimpleRegression o = (SimpleRegression) in.readObject();
                assertEquals(model.get("intercept").asDouble(), o.getIntercept(), 0.00000000000001);
                assertEquals(model.get("slope").asDouble(), o.getSlope(), 0.00000000000001);

            } catch (IOException e) {
                fail("error in deserialization");
            }


        }
    }
    /* This tests the three input queries for updateRegression. Clearly you would actually want to run all three of these in one
    call to updateRegression so as to de serialize and serialize the model once rather than 3 times, but in the test I want to make
    sure each functions correctly individually.
     */
    @Test
    public void shouldUpdateModel() throws Throwable {
        try (Driver driver = GraphDatabase.driver(neo4j.boltURI(), Config.build().withoutEncryption().toConfig());
             Session session = driver.session()) {
            session.run(createKnownRelationships);
            session.run(createUnknownRelationships);

            //create the initial model
            String modelQuery = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND exists(r.progress) RETURN r.time as time, r.progress as progress";
            String mapQuery = "MATCH () - [r:WORKS_FOR] -> () WHERE exists(r.time) AND NOT exists(r.progress) RETURN r, r.time as time";
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("modelQuery", modelQuery);
            parameters.put("mapQuery", mapQuery);

            session.run("CALL example.customRegression($modelQuery, $mapQuery, 'predictedProgress', 1)", parameters);

            //remove data from relationship between nodes 1 and 2
            String removeQuery = "MATCH (:Node {id:1})-[r:WORKS_FOR]->(:Node {id:2}) RETURN r.time as time, r.progress as progress";
            parameters.put("removeQuery", removeQuery);
            session.run("CALL example.updateRegression($removeQuery, '', '', 'predictedProgress', 1)", parameters);

            //create a new relationship between nodes 7 and 8
            session.run("MATCH (n7:Node {id:7}) MERGE (n7)-[:WORKS_FOR {time:6.0, progress:5.870}]->(:Node {id:8})", parameters);

            //add data from new relationship to model
            String updateQuery = "MATCH (:Node {id:7})-[r:WORKS_FOR]->(:Node {id:8}) RETURN r.time as time, r.progress as progress";
            parameters.put("updateQuery", updateQuery);
            session.run("CALL example.updateRegression('', $updateQuery, '', 'predictedProgress', 1)", parameters);

            //map new model on all relationships with unknown progress
            session.run("CALL example.updateRegression('', '', $mapQuery, 'predictedProgress', 1)", parameters);

            //replicate the creation and updates of the model
            SimpleRegression R = new SimpleRegression();
            R.addData(1.0, 1.345);
            R.addData(2.0, 2.596);
            R.addData(3.0, 3.259);
            R.removeData(1.0, 1.345);
            R.addData(6.0, 5.870);

            HashMap<Double, Double> expected = new HashMap<>();
            expected.put(4.0, R.predict(4.0));
            expected.put(5.0, R.predict(5.0));


            //make sure predicted values are correct
            StatementResult result = session.run(gatherPredictedValues);
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

