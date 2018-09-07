package embedding;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;
import org.deeplearning4j.graph.iterator.parallel.GraphWalkIteratorProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MyRandomWalkGraphIteratorProvider<V> implements GraphWalkIteratorProvider<V> {

    private IGraph<V, ?> graph;
    private int walkLength;
    private Random rng;
    private NoEdgeHandling mode;
    private int numberOfWalks;

    public MyRandomWalkGraphIteratorProvider(IGraph<V, ?> graph, int walkLength, long seed, NoEdgeHandling mode, int numberOfWalks) {
        this.graph = graph;
        this.walkLength = walkLength;
        this.rng = new Random(seed);
        this.mode = mode;
        this.numberOfWalks = numberOfWalks;
    }


    @Override
    public List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators) {
        int nVertices = graph.numVertices();
        numIterators = numberOfWalks;

        int verticesPerIter = nVertices / numIterators;

        List<GraphWalkIterator<V>> list = new ArrayList<>(numIterators);
        int last = 0;
        for (int i = 0; i < numIterators; i++) {
            int from = last;
            int to = Math.min(nVertices, from + verticesPerIter);
            if (i == numIterators - 1)
                to = nVertices;

            GraphWalkIterator<V> iter = new RandomWalkIterator<>(graph, walkLength, rng.nextLong(), mode, from, to);
            list.add(iter);
            last = to;
        }

        return list;
    }
}
