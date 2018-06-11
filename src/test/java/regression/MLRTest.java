package regression;

import org.apache.commons.math3.stat.regression.MillerUpdatingRegression;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;
import org.neo4j.graphdb.Result;

import java.io.File;
import java.util.Collections;
import java.util.Map;


public class MLRTest {
    /*private GraphDatabaseService db;

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
        db.execute("CALL regression.linear.create('boston housing', 'Miller', true, 2)");
        db.execute("load csv with headers from 'http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv' " +
                "as row with toFloat(row['CRIM']) as crime, toFloat(row['RM']) as rooms, toFloat(row['MEDV']) as price " +
                "call regression.linear.add('boston housing', [crime, rooms], price) return crime, rooms, price");
        //db.execute("call regression.linear.train('boston housing')");
        db.execute("call regression.linear.info('boston housing')");
        db.execute("call regression.linear.delete('boston housing')");


    }*/
}
