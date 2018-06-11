/**
 * Created by Lauren on 6/5/18.
 */

package regression;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SimpleLRModel extends LRModel {
    private SimpleRegression R;

    SimpleLRModel(String name, boolean constant) {
        super(name, Framework.Simple);
        R = new SimpleRegression(constant);
    }

    SimpleLRModel(String name, Object data) {
        super(name, Framework.Simple);
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
    long getNumVars() { return 1; }

    @Override
    boolean hasConstant() {return R.hasIntercept();}

    @Override
    void add(List<Double> given, double expected) {
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
    protected void removeData(List<Double> input, double output) {
        if (input.isEmpty()) {
            throw new IllegalArgumentException("x is empty, cannot remove any data.");
        }
        for (double x: input) R.removeData(x, output);
        if (R.getN() == 1) this.state = State.training;
        else if (R.getN() > 1) this.state = State.ready;
    }

    @Override
    double predict(List<Double> given) {
        return R.predict(given.get(0));
    }

    @Override
    Object data() {
        try { return LR.convertToBytes(R); }
        catch (IOException e) { throw new RuntimeException(name + " cannot be serialized."); }
    }

    /*@Override
    public LR.StatResult stats() {
        return new LR.StatResult(R.getN(), 1).withInfo("intercept", R.getIntercept(),
                "slope", R.getSlope(), "rSquared", R.getRSquare(), "significance", R.getSignificance());
    }*/

    @Override
    LR.ModelResult train() {
        return asResult();
    }

    @Override
    LR.ModelResult asResult() {
        LR.ModelResult r = new LR.ModelResult(name, framework, hasConstant(), getNumVars(), state, getN());
        if (state == State.ready) {
            List<Double> params = new ArrayList<>();
            params.add(R.getIntercept());
            params.add(R.getSlope());
            return r.withInfo("parameters", params, "rSquared", R.getRSquare(), "significance", R.getSignificance(),
                    "slope confidence interval", R.getSlopeConfidenceInterval(), "intercept std error", R.getInterceptStdErr(),
                    "slope std error", R.getSlopeStdErr(), "SSE", R.getSumSquaredErrors(), "MSE", R.getMeanSquareError());
        }
        return r;
    }

}
