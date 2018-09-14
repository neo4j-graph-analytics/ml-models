package embedding;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.ProcedureConfiguration;
import org.neo4j.graphalgo.core.heavyweight.HeavyGraph;
import org.neo4j.graphalgo.core.utils.Pools;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.ProgressTimer;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphalgo.core.write.Exporter;
import org.neo4j.graphdb.Direction;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;
import org.neo4j.values.storable.INDArrayPropertyTranslator;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;


public class DeepGLProc {
    public static final String DEFAULT_TARGET_PROPERTY = "embedding";

    @Context
    public GraphDatabaseAPI api;

    @Context
    public Log log;

    @Context
    public KernelTransaction transaction;

    @Procedure(value = "embedding.deepgl", mode = Mode.WRITE)
    public Stream<DeepGLProcResult> deepGL(
            @Name(value = "label", defaultValue = "") String label,
            @Name(value = "relationship", defaultValue = "") String relationship,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config) {

        final ProcedureConfiguration configuration = ProcedureConfiguration.create(config);

        int iterations = configuration.getInt("iterations", 10);
        int diffusions = configuration.getInt("diffusions", 10);
        double pruningLambda = configuration.get("pruningLambda", 0.1);

        final DeepGLProcResult.Builder builder = DeepGLProcResult.builder();


        HeavyGraph graph;
        try (ProgressTimer timer = builder.timeLoad()) {
            graph = (HeavyGraph) new GraphLoader(api, Pools.DEFAULT)
                    .init(log, label, relationship, configuration)
                    .withoutNodeProperties()
                    .withDirection(configuration.getDirection(Direction.BOTH))
                    .withOptionalNodeProperties(extractNodeFeatures(config))
                    .load(configuration.getGraphImpl());
        }

        builder.withNodeCount(graph.nodeCount());

        if (graph.nodeCount() == 0) {
            return Stream.empty();
        }

        final TerminationFlag terminationFlag = TerminationFlag.wrap(transaction);
        DeepGL algo = new DeepGL(graph, Pools.DEFAULT, configuration.getConcurrency(), iterations, pruningLambda, diffusions)
                .withProgressLogger(ProgressLogger.wrap(log, "DeepGL"))
                .withTerminationFlag(terminationFlag);

        builder.timeEval(algo::compute);
        graph.release();

        INDArray embedding = algo.embedding();
        builder.withEmbeddingSize(embedding.columns());
        builder.withFeatures(algo.features());
        builder.withLayers(algo.numberOfLayers());

        if (configuration.isWriteFlag()) {
            builder.timeWrite(() -> {
                final String writeProperty = configuration.getWriteProperty(DEFAULT_TARGET_PROPERTY);

                builder.withWriteProperty(writeProperty);

                Exporter.of(api, graph)
                        .withLog(log)
                        .parallel(Pools.DEFAULT, configuration.getConcurrency(), terminationFlag)
                        .build()
                        .write(writeProperty, embedding, new INDArrayPropertyTranslator());
            });
        }

        return Stream.of(builder.build());
    }

    @Procedure(value = "embedding.deepgl.stream")
    public Stream<DeepGL.Result> deepGLStream(
            @Name(value = "label", defaultValue = "") String label,
            @Name(value = "relationship", defaultValue = "") String relationship,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config) {

        final ProcedureConfiguration configuration = ProcedureConfiguration.create(config);

        int iterations = configuration.getInt("iterations", 10);
        int diffusions = configuration.getInt("diffusions", 10);
        double pruningLambda = configuration.get("pruningLambda", 0.1);

        final HeavyGraph graph = (HeavyGraph) new GraphLoader(api, Pools.DEFAULT)
                .init(log, label, relationship, configuration)
                .withoutNodeProperties()
                .withDirection(configuration.getDirection(Direction.BOTH))
                .withOptionalNodeProperties(extractNodeFeatures(config))
                .load(configuration.getGraphImpl());

        if (graph.nodeCount() == 0) {
            return Stream.empty();
        }

        DeepGL algo = new DeepGL(graph,
                Pools.DEFAULT,
                configuration.getConcurrency(),
                iterations,
                pruningLambda,
                diffusions);
        algo.withProgressLogger(ProgressLogger.wrap(log, "DeepGL"));

        algo.compute();
        graph.release();

        return algo.resultStream();
    }

    private PropertyMapping[] extractNodeFeatures(@Name(value = "config", defaultValue = "{}") Map<String, Object> config) {
        List<String> nodeFeatures = (List<String>) config.getOrDefault("nodeFeatures", Collections.emptyList());

        PropertyMapping[] propertyMappings = new PropertyMapping[nodeFeatures.size()];
        for (int i = 0; i < nodeFeatures.size(); i++) {
            propertyMappings[i] = PropertyMapping.of(nodeFeatures.get(i), nodeFeatures.get(i), 0.0);
        }
        return propertyMappings;
    }
}