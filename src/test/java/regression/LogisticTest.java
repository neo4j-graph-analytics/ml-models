package regression;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static org.neo4j.helpers.collection.MapUtil.map;
import org.apache.mahout.common.RandomUtils;

public class LogisticTest {
    private static GraphDatabaseService db;
    //TODO: larger data set, correct random function

    @BeforeClass
    public static void setUp() throws Exception {
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

        String csvFile = "/Users/laurenshin/documents/linreg-graph-analytics/src/test/resources/iris-full.csv";
        String line = "";
        String csvSplitBy = ",";

        List<Map<String,Double>> data = new ArrayList<>();
        List<String> target = new ArrayList<>();
        List<Integer> order = new ArrayList<>();

        /*Map<String, Integer> stringToInt = new HashMap<>();
        Map<Integer, String> intToString = new HashMap<>();

        stringToInt.put("Iris-setosa", 0);
        stringToInt.put("Iris-versicolor", 1);
        stringToInt.put("Iris-virginica", 2);
        intToString.put(0, "Iris-setosa");
        intToString.put(1, "Iris-versicolor");
        intToString.put(2, "Iris-virginica");*/

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))){
            br.readLine(); //skip headers
            int i = 0;
            while ((line = br.readLine()) != null) {
                String[] flower = line.split(csvSplitBy);
                Map<String, Double> v = new HashMap<>(4);
                v.put("sepallength", Double.parseDouble(flower[1])); //sepal length
                v.put("sepalwidth", Double.parseDouble(flower[2])); //sepal width
                v.put("petallength", Double.parseDouble(flower[3])); //petal length
                v.put("petalwidth", Double.parseDouble(flower[4])); //petal width
                data.add(v);
                target.add(flower[5]); //class
                order.add(i++);
            }
        } catch (IOException e) {
            e.printStackTrace();
            Assert.fail("unable to read csv file for test data");
        }
        RandomUtils.useTestSeed();
        Random random = RandomUtils.getRandom();
        Collections.shuffle(order, random);
        List<Integer> train = order.subList(0, 100);
        List<Integer> test = order.subList(100, 150);

        db.execute("CALL regression.logistic.create('model', ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], " +
                "{sepallength:'float', sepalwidth:'float', petallength:'float', petalwidth:'float'}, {prior:'L2'})").close();
        for (int pass = 0; pass < 30; pass++) {
            Collections.shuffle(train, random);
            for (int j : train) {
                db.execute("CALL regression.logistic.add('model', {output}, {inputs})", map("inputs", data.get(j), "output", target.get(j)));
            }
        }
        int successes = 0;
        int failures = 0;
        for (int k : test) {
            String t;
            String guess = ((String) db.execute("RETURN regression.logistic.predict('model', {inputs}) as prediction", map("inputs", data.get(k))).next().get("prediction"));
            if (guess.equals(target.get(k))) {
                t = "SUCCESS!";
                successes++;
            } else {
                t = "FAIL!";
                failures++;
            }
            System.out.format("Expected: %s, Actual: %s %s%n", target.get(k), guess, t);
        }
        System.out.format("SUCCESSES: %d%n", successes);
        System.out.format("FAILURES: %d%n", failures);


        db.execute("CALL regression.logistic.delete('model')");
    }

}
