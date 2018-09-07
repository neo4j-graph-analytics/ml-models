package embedding;

import java.util.ArrayList;
import java.util.List;

public class DeepWalkResult {
    public Long nodeId;
    public List<Double> embedding = new ArrayList<>();

    public DeepWalkResult(long nodeId, double[] embedding) {
        this.nodeId = nodeId;
        for (double item : embedding) this.embedding.add(item);
    }
}