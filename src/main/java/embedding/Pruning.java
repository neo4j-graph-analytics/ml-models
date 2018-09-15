package embedding;


import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.IdMap;
import org.neo4j.graphalgo.core.WeightMap;
import org.neo4j.graphalgo.core.heavyweight.AdjacencyMatrix;
import org.neo4j.graphalgo.core.heavyweight.HeavyGraph;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.RawValues;
import org.neo4j.graphalgo.core.utils.dss.DisjointSetStruct;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.impl.DSSResult;
import org.neo4j.graphalgo.impl.GraphUnionFind;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class Pruning {

    private final double lambda;
    private final ProgressLogger progressLogger;

    public Pruning() {
        this(0.7, ProgressLogger.NULL_LOGGER);
    }

    public Pruning(double lambda) {
        this(lambda, ProgressLogger.NULL_LOGGER);
    }

    public Pruning(double lambda, ProgressLogger progressLogger) {

        this.lambda = lambda;
        this.progressLogger = progressLogger;
    }

    public Embedding prune(Embedding prevEmbedding, Embedding embedding) {

        INDArray embeddingToPrune = Nd4j.hstack(prevEmbedding.getNDEmbedding(), embedding.getNDEmbedding());
        Feature[] featuresToPrune = ArrayUtils.addAll(prevEmbedding.getFeatures(), embedding.getFeatures());


        progressLogger.log("Feature Pruning: Creating features graph");
        final Graph graph = loadFeaturesGraph(embeddingToPrune, prevEmbedding.features.length);
        progressLogger.log("Feature Pruning: Created features graph");

        progressLogger.log("Feature Pruning: Finding features to keep");
        int[] featureIdsToKeep = findConnectedComponents(graph)
                .collect(Collectors.groupingBy(item -> item.setId))
                .values()
                .stream()
                .mapToInt(results -> results.stream().mapToInt(value -> (int) value.nodeId).min().getAsInt())
                .toArray();
        progressLogger.log("Feature Pruning: Found features to keep");

        progressLogger.log("Feature Pruning: Pruning embeddings");
        INDArray prunedNDEmbedding = pruneEmbedding(embeddingToPrune, featureIdsToKeep);
        progressLogger.log("Feature Pruning: Pruned embeddings");


        Feature[] prunedFeatures = new Feature[featureIdsToKeep.length];

        for (int index = 0; index < featureIdsToKeep.length; index++) {
            prunedFeatures[index] = featuresToPrune[featureIdsToKeep[index]];
        }


        return new Embedding(prunedFeatures, prunedNDEmbedding);
    }

    private Stream<DisjointSetStruct.Result> findConnectedComponents(Graph graph) {
        GraphUnionFind algo = new GraphUnionFind(graph);
        DisjointSetStruct struct = algo.compute();
        algo.release();
        DSSResult dssResult = new DSSResult(struct);
        return dssResult.resultStream(graph);
    }

    private Graph loadFeaturesGraph(INDArray embedding, int numPrevFeatures) {
        int nodeCount = embedding.columns();

        progressLogger.log("Creating IdMap - " + nodeCount + " nodes");
        IdMap idMap = new IdMap(nodeCount);

        for (int i = 0; i < nodeCount; i++) {
            idMap.add(i);
        }
        idMap.buildMappedIds();
        progressLogger.log("Created IdMap");

        WeightMap relWeights = new WeightMap(nodeCount, 0, -1);
        AllocationTracker allocationTracker = AllocationTracker.create();
        AdjacencyMatrix matrix = new AdjacencyMatrix(idMap.size(), false, allocationTracker);
        progressLogger.log(allocationTracker.getUsageString());

        int comparisons = 0;

        progressLogger.log("Size of combined embedding: " + Arrays.toString(embedding.shape()));
        progressLogger.log("Number of prev features: " + numPrevFeatures);
        progressLogger.log("Creating AdjacencyMatrix");

        int[] degrees = new int[nodeCount];

        for (int i = numPrevFeatures; i < nodeCount; i++) {
            for (int j = 0; j < i; j++) {
                INDArray emb1 = embedding.getColumn(i);
                INDArray emb2 = embedding.getColumn(j);

                double score = score(emb1, emb2);

                if(score > lambda) {
                    degrees[idMap.get(i)]++;
                }
            }
        }
        progressLogger.log("Calculated degree distribution");

        for (int i = 0; i < degrees.length; i++) {
            int degree = degrees[i];
            matrix.armOut(idMap.get(i), degree);

        }
        progressLogger.log(allocationTracker.getUsageString());

        for (int i = numPrevFeatures; i < nodeCount; i++) {
            for (int j = 0; j < i; j++) {
                INDArray emb1 = embedding.getColumn(i);
                INDArray emb2 = embedding.getColumn(j);

                double score = score(emb1, emb2);
                comparisons++;

                if(score > lambda) {
                    matrix.addOutgoing(idMap.get(i), idMap.get(j));
                }
            }
        }
        progressLogger.log(allocationTracker.getUsageString());
        progressLogger.log("Created Adjacency Matrix");
        progressLogger.log("Number of comparisons: " + comparisons);

        return new HeavyGraph(idMap, matrix, relWeights, null);
    }

    private INDArray pruneEmbedding(INDArray origEmbedding, int... featIdsToKeep) {
        INDArray ndPrunedEmbedding = Nd4j.create(origEmbedding.shape());
        Nd4j.copy(origEmbedding, ndPrunedEmbedding);
        return ndPrunedEmbedding.getColumns(featIdsToKeep);
    }


    public static class Feature {
        private final String name;
        private final Feature prev;

        public Feature(String name, Feature prev) {
            this.name = name;
            this.prev = prev;
        }

        public Feature(String name) {
            this.prev = null;
            this.name = name;
        }

        @Override
        public String toString() {
            return prev == null ? name : name + "( " + prev.toString() + ")";
        }

        @Override
        public boolean equals(Object obj) {
            return (obj instanceof Feature) && toString().equals(obj.toString());
        }
    }

    static class Embedding {
        private INDArray ndEmbedding;
        private Feature[] features;

        public Embedding(Feature[] Features, INDArray ndEmbedding) {
            this.features = Features;
            this.ndEmbedding = ndEmbedding;
        }

        public Feature[] getFeatures() {
            return features;
        }

        public INDArray getNDEmbedding() {
            return ndEmbedding;
        }
    }

    double score(INDArray feat1, INDArray feat2) {
        return feat1.eq(feat2).sum(0).getDouble(0,0) / feat1.size(0);
    }


}
