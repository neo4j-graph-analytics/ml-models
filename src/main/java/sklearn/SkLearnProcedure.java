package sklearn;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import regression.LogisticModel;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Result;
import org.neo4j.helpers.collection.MapUtil;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

public class SkLearnProcedure
{
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value = "sklearn.model.save", mode = Mode.WRITE)

    public void save( @Name("name") String modelName,
                      @Name("module") String module,
                      @Name("class") String klass,
                      @Name("fields") Map<String,Object> fields)
    {

        String saveModelQuery = "MERGE (model:Model {name: {modelName} })\n" +
                                "SET model.module = {module}, model.class = {class}\n" +
                                "WITH model\n" +
                                "UNWIND keys({fields}) AS f\n" +
                                "CREATE (field:Field)\n" +
                                "SET field.key = f, \n" +
                                "    field.value = {fields}[f].value,\n" +
                                "    field.type = {fields}[f].type,\n" +
                                "    field.dataType = {fields}[f].dataType,\n" +
                                "    field.shape = {fields}[f].shape\n" +
                                "MERGE (model)-[:HAS_FIELD]->(field)";
        Map<String, Object> params = MapUtil.map("modelName", modelName, "module", module, "class", klass, "fields", fields);

        db.execute( saveModelQuery, params );

    }

    @Procedure(value = "sklearn.model.load")
    public Stream<SkLearnModel> load( @Name("name") String modelName )
    {
        String loadModuleQuery = "MATCH (model:Model {name: {modelName} })\n" +
                                 "RETURN model, [(model)-[:HAS_FIELD]->(field) | field] AS fields";
        Map<String, Object> params = MapUtil.map("modelName", modelName);

        Result result = db.execute( loadModuleQuery, params );
        return result.stream().map( row -> new SkLearnModel( (Node) row.get( "model" ), (List<Node>) row.get("fields") ) );
    }

    public class SkLearnModel
    {
        public Node model;
        public final List<Node> fields;

        public SkLearnModel( Node model, List<Node> fields )
        {
            this.model = model;
            this.fields = fields;
        }
    }
}
