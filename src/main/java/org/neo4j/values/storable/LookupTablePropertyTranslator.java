package org.neo4j.values.storable;

import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.neo4j.graphalgo.core.write.PropertyTranslator;

public class LookupTablePropertyTranslator implements PropertyTranslator<InMemoryGraphLookupTable> {
    @Override
    public Value toProperty(int propertyId, InMemoryGraphLookupTable data, long nodeId) {

        INDArray row = data.getVector((int) nodeId);

        double[] rowAsDouble = new double[row.size(1)];
        for (int columnIndex = 0; columnIndex < row.size(1); columnIndex++) {
            rowAsDouble[columnIndex] = row.getDouble(columnIndex);
        }

        return new DoubleArray(rowAsDouble);
    }
}
