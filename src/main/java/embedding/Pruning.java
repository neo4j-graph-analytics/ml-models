package embedding;


import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
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
        AllocationTracker allocationTracker = AllocationTracker.create();
        int nodeCount = embedding.columns();

        progressLogger.log("Creating IdMap - " + nodeCount + " nodes");
        IdMap idMap = new IdMap(nodeCount);

        for (int i = 0; i < nodeCount; i++) {
            idMap.add(i);
        }
        idMap.buildMappedIds();
        progressLogger.log("Created IdMap");
        progressLogger.log("Allocation: " + allocationTracker.getUsageString());

        WeightMap relWeights = new WeightMap(nodeCount, 0, -1);
        AdjacencyMatrix matrix = new AdjacencyMatrix(idMap.size(), false, allocationTracker);
        progressLogger.log("Allocation: " + allocationTracker.getUsageString());


        progressLogger.log("Size of combined embedding: " + Arrays.toString(embedding.shape()));
        progressLogger.log("Number of prev features: " + numPrevFeatures);
        progressLogger.log("Creating AdjacencyMatrix");

        progressLogger.log("Adding columns to array");

        progressLogger.log("Added columns to array");

//        INDArray reusedArray = Nd4j.zeros(1, embedding.rows());
//        int size = reusedArray.size(0);

//        int[] degrees = new int[nodeCount];
        int comparisons = 0;
//        progressLogger.log("Calculating degree distribution");
//        for (int i = numPrevFeatures; i < nodeCount; i++) {
//            for (int j = 0; j < i; j++) {
//                INDArray emb1 = embedding.getColumn(i);
//                INDArray emb2 = embedding.getColumn(j);
//
////                double score = score(emb1, emb2);
////                double score = score(reusedArray, emb1, emb2, size);
//                double score = score(emb1.dup(), emb2, emb1.size(0));
//                comparisons++;
//
//                if(score > lambda) {
//                    degrees[idMap.get(i)]++;
//                }
//            }
//        }
//        progressLogger.log("Calculated degree distribution (" + comparisons + " comparisons)");
//
//        for (int i = 0; i < degrees.length; i++) {
//            int degree = degrees[i];
//            matrix.armOut(idMap.get(i), degree);
//
//        }
//        progressLogger.log("Allocation: " + allocationTracker.getUsageString());
//
        StopWatch timer = new StopWatch();
        timer.start();
//        progressLogger.log("Populating adjacency matrix 1");
//        for (int i = numPrevFeatures; i < nodeCount; i++) {
//            for (int j = 0; j < i; j++) {
//                INDArray emb1 = embedding.getColumn(i);
//                INDArray emb2 = embedding.getColumn(j);
//
////                double score = score(emb1, emb2);
////                double score = score(reusedArray, emb1, emb2, size);
//                double score = score(emb1.dup(), emb2, emb1.size(0));
//                comparisons++;
//
//                if(score > lambda) {
//                    matrix.addOutgoing(idMap.get(i), idMap.get(j));
//                }
//            }
//        }
//        timer.stop();
//        progressLogger.log("Populated adjacency matrix (orig): "  + timer.getTime() + " ms");
//        progressLogger.log("Number of comparisons: " + comparisons);

        timer.reset();
        timer.start();
        progressLogger.log("Populating adjacency matrix 2");
        final INDArray transpose = embedding.transpose();
        final INDArray zeros = Nd4j.zerosLike(transpose);
        final double newLambda = transpose.columns() * lambda;
        final INDArray temp = Nd4j.zerosLike(transpose);
        for (int i = numPrevFeatures; i < embedding.columns(); i++) {
            Nd4j.copy(transpose, temp);
            final int[] total =
                    temp.subiRowVector(transpose.getRow(i))
                            .eqi(zeros)
                            .sum(1)
                            .toIntVector();
            for (int j = 0; j < i; j++) {
                final int score = total[j];
                if (score > newLambda) {
                    matrix.addOutgoing(idMap.get(i), idMap.get(j));
                }
            }
        }
        transpose.cleanup();
        zeros.cleanup();
        temp.cleanup();
        timer.stop();
        progressLogger.log("Populated adjacency matrix (new): " + timer.getTime() + " ms");
        progressLogger.log("Created Adjacency Matrix");

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
//                return feat1.eqi(feat2).sum(0).divi(feat1.size(0)).getDouble(0);
        return feat1.eq(feat2).sum(0).getDouble(0, 0) / feat1.size(0);
    }

    double score(INDArray feat1, INDArray feat2, int size) {
        return feat1.eqi(feat2).sum(0).getDouble(0, 0) / feat1.size(0);
    }

    double score(INDArray reusedArray, INDArray feat1, INDArray feat2, int size) {
        return reusedArray.assign(feat1).eqi(feat2).sum(0).getDouble(0, 0) / size;
//        return feat1.eq(feat2).sum(0).getDouble(0,0) / feat1.size(0);
    }


}
