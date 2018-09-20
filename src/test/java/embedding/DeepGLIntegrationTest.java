package embedding;


import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.internal.kernel.api.exceptions.KernelException;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.Arrays;
import java.util.Map;


public class DeepGLIntegrationTest {

    private static GraphDatabaseAPI db;

    @BeforeClass
    public static void setupGraph() throws KernelException {

        final String cypher =
                "CREATE (a:Node {name:'a'})\n" +
                        "CREATE (b:Node {name:'b'})\n" +
                        "CREATE (c:Node {name:'c'})\n" +
                        "CREATE (d:Node {name:'d'})\n" +
                        "CREATE (e:Node {name:'e'})\n" +
                        "CREATE (f:Node {name:'f'})\n" +
                        "CREATE (g:Node {name:'g'})\n" +
                        "CREATE" +
                        " (a)-[:TYPE]->(b),\n" +
                        " (a)-[:TYPE]->(f),\n" +
                        " (b)-[:TYPE]->(c),\n" +
                        " (c)-[:TYPE]->(d),\n" +
                        " (d)-[:TYPE]->(g),\n" +
                        " (d)-[:TYPE]->(e)";


        db = TestDatabaseCreator.createTestDatabase();

        try (Transaction tx = db.beginTx()) {
            db.execute(cypher);
            tx.success();
        }

        db.getDependencyResolver()
                .resolveDependency(Procedures.class)
                .registerProcedure(DeepGLProc.class);

    }

    @AfterClass
    public static void tearDown() throws Exception {
        if (db != null) db.shutdown();
    }

    @Test
    public void stream() throws Exception {

        Result result = db.execute("CALL algo.deepgl.stream('Node', 'TYPE')");

        while (result.hasNext()) {
            System.out.println("result.next() = " + result.next());
        }
    }

    @Test
    public void removeInnerLoopInPruning() throws Exception {

        Result result = db.execute("CALL embedding.deepgl('Node', 'TYPE', {pruningLambda: 0.8, iterations: 3})");

        while (result.hasNext()) {
            System.out.println("result.next() = " + result.next());
        }
    }

    @Test
    public void write() throws Exception {

        String writeProperty = "'foo'";
        Result result = db.execute("CALL algo.deepgl('Node', 'TYPE', {writeProperty: " + writeProperty + ", nodeFeatures:['prop1']})");

        while (result.hasNext()) {
            System.out.println("summary = " + result.next());
        }

        Result embeddings = db.execute("MATCH (n:Node) RETURN n.foo AS foo");
        while(embeddings.hasNext()) {
            Map<String, Object> row = embeddings.next();
            System.out.println("embeddings = " + Arrays.toString((double[])row.get("foo")));
        }
    }
}

