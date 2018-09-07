package embedding;

import org.neo4j.graphalgo.api.Degrees;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.utils.ParallelUtil;
import org.neo4j.graphalgo.core.utils.Pools;
import org.neo4j.graphalgo.core.utils.QueueBasedSpliterator;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphdb.Direction;
import org.neo4j.procedure.Name;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.PrimitiveIterator;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class NodeWalker {
    public Stream<int[]> internalRandomWalk(@Name(value = "steps", defaultValue = "80") int steps, org.neo4j.graphalgo.impl.walking.NodeWalker.NextNodeStrategy strategy, TerminationFlag terminationFlag,
                                            int concurrency, int limit, PrimitiveIterator.OfInt idStream) {
        int timeout = 100;
        int queueSize = 1000;
        int batchSize = ParallelUtil.adjustBatchSize(limit, concurrency, 100);
        Collection<Runnable> tasks = new ArrayList<>((limit / batchSize) + 1);

        ArrayBlockingQueue<int[]> queue = new ArrayBlockingQueue<>(queueSize);
        int[] TOMB = new int[0];

        while (idStream.hasNext()) {
            int[] ids = new int[batchSize];
            int i = 0;
            while (i < batchSize && idStream.hasNext()) {
                ids[i++] = idStream.nextInt();
            }
            int size = i;
            tasks.add(() -> {
                for (int j = 0; j < size; j++) {
                    put(queue, doInternalWalk(ids[j], steps, strategy, terminationFlag));
                }
            });
        }
        new Thread(() -> {
            ParallelUtil.runWithConcurrency(concurrency, tasks, terminationFlag, Pools.DEFAULT);
            put(queue, TOMB);
        }).start();

        QueueBasedSpliterator<int[]> spliterator = new QueueBasedSpliterator<>(queue, TOMB, terminationFlag, timeout);
        return StreamSupport.stream(spliterator, false);
    }

    private int[] doInternalWalk(int startNodeId, int steps, org.neo4j.graphalgo.impl.walking.NodeWalker.NextNodeStrategy nextNodeStrategy, TerminationFlag terminationFlag) {
        int[] nodeIds = new int[steps + 1];
        int currentNodeId = startNodeId;
        int previousNodeId = currentNodeId;
        nodeIds[0] = currentNodeId;
        for (int i = 1; i <= steps; i++) {
            int nextNodeId = nextNodeStrategy.getNextNode(currentNodeId, previousNodeId);
            previousNodeId = currentNodeId;
            currentNodeId = nextNodeId;

            if (currentNodeId == -1 || !terminationFlag.running()) {
                // End walk when there is no way out and return empty result
                return Arrays.copyOf(nodeIds, 1);
            }
            nodeIds[i] = currentNodeId;
        }

        return nodeIds;
    }

    private static <T> void put(BlockingQueue<T> queue, T items) {
        try {
            queue.put(items);
        } catch (InterruptedException e) {
            // ignore
        }
    }

    public static class RandomNextNodeStrategy extends org.neo4j.graphalgo.impl.walking.NodeWalker.NextNodeStrategy {

        public RandomNextNodeStrategy(Graph graph, Degrees degrees) {
            super(graph, degrees);
        }

        @Override
        public int getNextNode(int currentNodeId, int previousNodeId) {
            int degree = degrees.degree(currentNodeId, Direction.BOTH);
            if (degree == 0) {
                return -1;
            }
            int randomEdgeIndex = ThreadLocalRandom.current().nextInt(degree);

            return graph.getTarget(currentNodeId, randomEdgeIndex, Direction.BOTH);
        }

    }

}
