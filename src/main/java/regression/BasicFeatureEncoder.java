package regression;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.Map;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class BasicFeatureEncoder extends FeatureEncoder {
    Map<String, Integer> offsets = new HashMap<>();
    BasicFeatureEncoder(List<String> categories, Set<String> features, boolean hasIntercept, Prior prior) {
        super(categories, hasIntercept, prior);
        vectorSize = hasIntercept ? features.size() + 1 : features.size();
        initOffsets(features);
    }

    private void initOffsets(Set<String> features) {
        int i = hasIntercept ? 1 : 0;
        for (String feature : features)  offsets.put(feature, i++);
    }

    //TODO: validate vector first
    Vector getVector(Map<String, Object> features) {
        Vector v = new DenseVector(vectorSize);
        if (hasIntercept) {
            v.set(0, 1);
        }
        for (Map.Entry<String, Object> feature : features.entrySet()) {
            v.set(offsets.get(feature.getKey()), (double) feature.getValue());
        }
        return v;
    }
}
