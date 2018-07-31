package regression;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

public class HashingFeatureEncoder extends FeatureEncoder {
    private Map<String, DataType> types;
    ConstantValueEncoder interceptAdder = new ConstantValueEncoder("intercept");
    StaticWordValueEncoder featureAdder = new StaticWordValueEncoder("feature");

    HashingFeatureEncoder(List<String> categories, Map<String, DataType> processedTypes, boolean hasIntercept, int hashedSize, Prior prior) {
        super(categories, hasIntercept, prior);
        types = processedTypes;
        vectorSize = hashedSize;
    }

    //TODO: validate features
    Vector getVector(Map<String, Object> features) {
        Vector v = new RandomAccessSparseVector(vectorSize);
        if (hasIntercept) interceptAdder.addToVector("1", v);
        for (Map.Entry<String, Object> feature : features.entrySet()) {
            String key = feature.getKey();
            Object value = feature.getValue();
            switch (types.get(key)) {
                case _class: featureAdder.addToVector(key + ":" + (String) value, 1, v); break;
                case _float: featureAdder.addToVector(key, (double) value, v); break;
            }
        }
        return v;
    }
}