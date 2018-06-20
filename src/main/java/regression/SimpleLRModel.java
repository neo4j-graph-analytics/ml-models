/**
 * Created by Lauren on 6/5/18.
 */

package regression;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.regression.RegressionResults;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.neo4j.logging.Log;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.bytedeco.javacpp.annotation.Name;

public class SimpleLRModel extends LRModel {
    private SimpleRegression R;
    double sse;
    double sst;
    double ybar;
    ModelAnalyzer testAnalysis;

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
    int getNumVars() { return 1; }

    @Override
    boolean hasConstant() {return R.hasIntercept();}

    @Override
    void addTest(List<Double> given, double expected, Log log) {
        double fact1 = getN() + 1.0;
        double fact2 = getN()/fact1;
        double dy = expected - ybar;
        ybar += dy/fact1;
        double rdev = expected - R.getIntercept() - given.get(0)*R.getSlope();
        sse += rdev*rdev;
        sst += fact2*dy*dy;
    }

    @Override
    void addTrain(List<Double> given, double expected, Log log) {
        if (!checkData(given, expected)) log.warn("Data point " + given.toString() + ", " + new Double(expected).toString() +
                " is not valid and so was not added to the training data.");
        R.addData(given.get(0), expected);
        state = State.training;
    }

    void clean() {
        double[] yTrainArray = LR.doubleListToArray(yTrain);
        double nineNine = new Percentile().evaluate(yTrainArray);
        double stdDev = new StandardDeviation().evaluate(yTrainArray);
        double max = nineNine + 4*stdDev;
        for (int i = 0; i< xTrain.size(); i++) {
            if (yTrain.get(i) > max) {
                xTrain.remove(i);
                yTrain.remove(i);
            }
        }
    }

    @Override
    void test() {
        double[] params = new double[2];
        params[0] = R.getIntercept();
        params[1] = R.getSlope();
        testAnalysis = new ModelAnalyzer(params, hasConstant(), getNumVars(), sst, sse, getN());
    }



    @Override
    protected void removeData(List<Double> input, double output, Log log) {
        if (!checkData(input, output)) log.warn("Data point " + input.toString() + ", " + new Double(output).toString() +
                " is not valid and so was not added to the training data.");
        state = State.training;
        for (double x: input) R.removeData(x, output);
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
    void copy(String source) {
        LRModel src = LRModel.from(source);
        if (!(src instanceof SimpleLRModel)) throw new IllegalArgumentException(source + " and " + name + " are of incompatible types. Data " +
                "cannot be copied.");
        R.append(((SimpleLRModel) src).R);
        state = State.training;
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
                    "slope std error", R.getSlopeStdErr(), "SSE", R.getSumSquaredErrors(), "MSE", R.getMeanSquareError(),
                    "correlation", R.getR());
        }
        return r;
    }

}
