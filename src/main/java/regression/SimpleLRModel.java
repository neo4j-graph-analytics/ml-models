/**
 * Created by Lauren on 6/5/18.
 */

package regression;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class SimpleLRModel extends LRModel {
    private SimpleRegression R;

    SimpleLRModel(String name, boolean constant) {
        super(name, "simple");
        R = new SimpleRegression(constant);
    }

    SimpleLRModel(String name, Object data) {
        super(name, "simple");
        try {
            R = (SimpleRegression) LR.convertFromBytes((byte[]) data);
            if (R.getN() == 1) this.state = State.training;
            else if (R.getN() > 1) this.state = State.ready;
        }
        catch (Exception e) { throw new IllegalArgumentException("data is invalid, cannot load model");}
    }

    @Override
    protected long getN() {
        return R.getN();
    }

    @Override
    public void add(List<Double> given, double expected) {
        int givenSize = given.size();
        if (givenSize == 1) {
            R.addData(given.get(0), expected);
        } else {
            double[] x = LR.convertFromList(given);
            R.addObservation(x, expected);
        }
        if (R.getN() == 1) this.state = State.training;
        else if (R.getN() > 1) this.state = State.ready;
    }

    @Override
    protected void removeData(double x, double y) {
        R.removeData(x, y);
        if (R.getN() == 1) this.state = State.training;
        else if (R.getN() > 1) this.state = State.ready;
    }

    @Override
    public double predict(List<Double> given) {
        return R.predict(given.get(0));
    }

    @Override
    public Object serialize() {
        try { return LR.convertToBytes(R); }
        catch (IOException e) { throw new RuntimeException(name + " cannot be serialized."); }
    }

    @Override
    public LR.StatResult stats() {
        return new LR.StatResult(R.getN(), 1).withInfo("intercept", R.getIntercept(),
                "slope", R.getSlope(), "rSquared", R.getRSquare(), "significance", R.getSignificance());
    }

    @Override
    public Map<String, Double> train() {
        Map<String, Double> params = new HashMap<>();
        params.put("slope", R.getSlope());
        params.put("intercept", R.getIntercept());
        return params;
    }
}
