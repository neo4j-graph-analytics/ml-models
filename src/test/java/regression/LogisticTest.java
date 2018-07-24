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

        String csvFile = "/Users/laurenshin/documents/linreg-graph-analytics/src/test/resources/iris-full.csv";
        String line = "";
        String csvSplitBy = ",";

        List<List<Double>> data = new ArrayList<>();
        List<Integer> target = new ArrayList<>();
        List<Integer> order = new ArrayList<>();

        Map<String, Integer> stringToInt = new HashMap<>();
        Map<Integer, String> intToString = new HashMap<>();

        stringToInt.put("Iris-setosa", 0);
        stringToInt.put("Iris-versicolor", 1);
        stringToInt.put("Iris-virginica", 2);
        intToString.put(0, "Iris-setosa");
        intToString.put(1, "Iris-versicolor");
        intToString.put(2, "Iris-virginica");

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))){
            br.readLine(); //skip headers
            int i = 0;
            while ((line = br.readLine()) != null) {
                String[] flower = line.split(csvSplitBy);
                List<Double> v = new ArrayList<>(4);
                v.add(Double.parseDouble(flower[1])); //sepal length
                v.add(Double.parseDouble(flower[2])); //sepal width
                v.add(Double.parseDouble(flower[3])); //petal length
                v.add(Double.parseDouble(flower[4])); //petal width
                data.add(v);
                target.add(stringToInt.get(flower[5])); //class
                order.add(i++);
            }
        } catch (IOException e) {
            e.printStackTrace();
            Assert.fail("unable to read csv file for test data");
        }
        RandomUtils.useTestSeed();
        Random random = RandomUtils.getRandom();
        Collections.shuffle(order, random);
        int cutoff = (int) Math.floor(order.size()*0.75);
        List<Integer> train = order.subList(0, 100);
        List<Integer> test = order.subList(100, 150);

        db.execute("CALL regression.logistic.create('model', 3, 4, true)").close();
        for (int pass = 0; pass < 30; pass++) {
            Collections.shuffle(train, random);
            for (int j : train) {
                db.execute("CALL regression.logistic.add('model', {inputs}, {output})", map("inputs", data.get(j), "output", target.get(j)));
            }
        }
        int successes = 0;
        int failures = 0;
        for (int k : test) {
            String t;
            int guess = ((Long) db.execute("RETURN regression.logistic.predict('model', {inputs}) as prediction", map("inputs", data.get(k))).next().get("prediction")).intValue();
            if (guess == target.get(k)) {
                t = "SUCCESS!";
                successes++;
            } else {
                t = "FAIL!";
                failures++;
            }
            System.out.format("Actual: %s, Guess: %s %s%n", intToString.get(target.get(k)), intToString.get(guess), t);
        }
        System.out.format("SUCCESSES: %d%n", successes);
        System.out.format("FAILURES: %d%n", failures);


        db.execute("CALL regression.logistic.delete('model')");
    }

}
