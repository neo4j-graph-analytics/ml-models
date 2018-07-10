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

public class SimpleLRModel extends LRModel {
    private SimpleRegression R;
    private double sse = 0;
    private double sst = 0;
    private double ybar = 0;
    private ModelAnalyzer tester;
    private long nTest = 0;

    ///////SETUP/////////

    SimpleLRModel(String name, boolean constant) {
        super(name, Framework.Simple);
        R = new SimpleRegression(constant);
        tester = new ModelAnalyzer();
    }

    SimpleLRModel(String name, Object data) {
        super(name, Framework.Simple);
        try {
            R = (SimpleRegression) LR.convertFromBytes((byte[]) data);
            if (R.getN() == 0) this.state = State.created;
            else this.state = State.training;
            tester = new ModelAnalyzer();
        }
        catch (Exception e) { throw new IllegalArgumentException("data is invalid, cannot load model");}
    }

    //////TRAINING///////

    @Override
    void addTrain(List<Double> given, double expected, Log log) {
        if (dataInvalid(given)) log.warn("Data point " + given.toString() + ", " + Double.toString(expected) +
                " is not valid and so was not added to the training data.");
        else {
            R.addData(given.get(0), expected);
            if (state == State.testing || state == State.ready) resetTest();
            state = State.training;
        }
    }

    @Override
    protected void removeTrain(List<Double> input, double output, Log log) {
        if (dataInvalid(input)) log.warn("Data point " + input.toString() + ", " + Double.toString(output) +
                " is not valid and so was not added to the training data.");
        else {
            for (double x : input) R.removeData(x, output);
            if (state == State.testing || state == State.ready) resetTest();
            state = State.training;
        }
    }

    @Override
    SimpleLRModel train() {
        state = State.testing;
        return this;
    }

    @Override
    SimpleLRModel copy(String source) {
        LRModel src = LRModel.from(source);
        if (!(src instanceof SimpleLRModel)) throw new IllegalArgumentException(source + " and " + name + " are of incompatible types. Data " +
                "cannot be copied.");
        R.append(((SimpleLRModel) src).R);
        state = State.training;
        return this;
    }

    private void resetTest() {
        sse = 0;
        sst = 0;
        ybar = 0;
        tester.clear();
        nTest = 0;
    }

    @Override
    SimpleLRModel clearTest() {
        resetTest();
        state = State.training;
        return this;
    }
    @Override
    SimpleLRModel clearAll() {
        R.clear();
        resetTest();
        state = State.created;
        return this;
    }

    ///////TESTING////////

    @Override
    void addTest(List<Double> given, double expected, Log log) {
        if (dataInvalid(given)) log.warn("Data point " + given.toString() + ", " + Double.toString(expected) +
                " is not valid and so was not added to the testing data.");
        else {
            double fact1 = getNTest() + 1.0;
            double fact2 = getNTest() / fact1;
            double dy = expected - ybar;
            ybar += dy / fact1;
            double rdev = expected - R.getIntercept() - given.get(0) * R.getSlope();
            sse += rdev * rdev;
            if (hasConstant()) sst += fact2 * dy * dy;
            else sst += expected * expected;
            nTest++;
            state = State.testing;
        }
    }

    @Override
    void removeTest(List<Double> given, double expected, Log log) {
        if (nTest > 0) {
            if (dataInvalid(given)) log.warn("Data point " + given.toString() + ", " + Double.toString(expected) +
                    " is not valid and so was not removed from the testing data.");
            else {
                double fact1 = nTest - 1;
                double fact2 = nTest / fact1;
                double dy = expected - ybar;
                ybar -= dy / fact1;
                double rdev = expected - R.getIntercept() - given.get(0) * R.getSlope();
                sse -= rdev * rdev;
                sst -= fact2 * dy * dy;
                nTest--;
                state = State.testing;
            }
        }
    }
    @Override
    SimpleLRModel test() {
        state = State.ready;
        double[] params = new double[2];
        params[0] = R.getIntercept();
        params[1] = R.getSlope();
        tester.newTestData(params, hasConstant(), getNumVars() + (R.hasIntercept() ? 1 : 0), sst, sse, getNTest());
        return this;
    }

    //////READY/////

    @Override
    double predict(List<Double> given) {
        return R.predict(given.get(0));
    }

    @Override
    Object data() {
        try { return LR.convertToBytes(R); }
        catch (IOException e) { throw new RuntimeException(name + " cannot be serialized."); }
    }

    @Override
    LR.ModelResult asResult() {
        LR.ModelResult r = new LR.ModelResult(name, framework, hasConstant(), getNumVars(), state, getNTrain(), getNTest());
        if (state != State.created) {
            List<Double> params = new ArrayList<>();
            params.add(R.getIntercept());
            params.add(R.getSlope());
            r.withTrainInfo("parameters", params, "RSquared", R.getRSquare(), "significance", R.getSignificance(),
                    "slope confidence interval", R.getSlopeConfidenceInterval(), "intercept std error", R.getInterceptStdErr(),
                    "slope std error", R.getSlopeStdErr(), "SSE", R.getSumSquaredErrors(), "MSE", R.getMeanSquareError(),
                    "correlation", R.getR(), "SSR", R.getRegressionSumSquares(), "SST", R.getTotalSumSquares());
        }
        if (tester.isReady()) r.withTestInfo(tester.getStatistics());
        return r;
    }

    ////////UTILS///////

    @Override
    protected long getNTrain() {
        return R.getN();
    }

    @Override
    protected long getNTest() { return nTest; }

    @Override
    int getNumVars() { return 1; }

    @Override
    boolean hasConstant() {return R.hasIntercept();}

}
