package regression;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

public class LogisticTest {
    private static GraphDatabaseService db;

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
        db.execute("CALL regression.logistic.create('model', {number:'float'}, 'type', 'even', 'odd')").close();
        db.execute("CALL regression.logistic.add('model', {number:2}, 'even')").close();
        db.execute("CALL regression.logistic.add('model', {number:4}, 'even')").close();
    }


}
