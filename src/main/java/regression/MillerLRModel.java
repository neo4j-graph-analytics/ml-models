/**
 * Created by Lauren on 6/5/18.
 */

package regression;
import org.apache.commons.math3.stat.regression.MillerUpdatingRegression;

import java.util.List;
import org.apache.commons.math3.stat.regression.RegressionResults;
import org.neo4j.logging.Log;

public class MillerLRModel extends LRModel{
    private MillerUpdatingRegression R;
    private RegressionResults trained;
    private int numVars;
    private double sse=0;
    private double sst=0;
    private double ybar=0;
    private ModelAnalyzer tester;
    private long nTest=0;

    /////////SETUP//////////

    MillerLRModel(String name, boolean constant, int numVars) {
        super(name, Framework.Multiple);
        R = new MillerUpdatingRegression(numVars, constant);
        this.numVars = numVars;
        tester = new ModelAnalyzer();
    }

    //////////TRAINING/////////
    @Override
    void addTrain(List<Double> given, double expected, Log log) {
        if (dataInvalid(given)) log.warn("Data point " + given.toString() + ", " + Double.toString(expected) +
                " is not valid and so was not added to the training data.");
        else {
            double[] givenArr = LR.doubleListToArray(given);
            R.addObservation(givenArr, expected);
            if (state == State.testing || state == State.ready) resetTest();
            state = State.training;
        }
    }

    @Override
    protected void removeTrain(List<Double> input, double output, Log log) {
        throw new IllegalArgumentException("Data cannot be removed from a multiple linear regression.");
    }
    @Override
    MillerLRModel copy(String string) {
        throw new IllegalArgumentException("Cannot copy data from one model to another for multiple linear regression.");
    }
    MillerLRModel clearAll() {
        R.clear();
        resetTest();
        trained = null;
        state = State.created;
        return this;
    }


    @Override
    MillerLRModel train() {
        trained = R.regress();
        state = State.testing;
        return this;
    }
    /////////TESTING///////////

    @Override
    void addTest(List<Double> given, double expected, Log log) {
        if (!(state == State.testing)) {
            clearTest();
            train();
        }
        if (dataInvalid(given)) log.warn("Data point " + given.toString() + ", " + Double.toString(expected) +
                " is not valid and so was not added to the testing data.");
        else {
            double fact1 = nTest + 1.0;
            double fact2 = nTest / fact1;
            double dy = expected - ybar;
            ybar += dy / fact1;
            double[] params = trained.getParameterEstimates();
            double rdev = expected;
            if (hasConstant()) {
                rdev -= params[0];
                for (int i = 1; i < params.length; i++) rdev -= params[i] * given.get(i - 1);
                sst += fact2 * dy * dy;
            } else {
                for (int i = 0; i < params.length; i++) rdev -= params[i] * given.get(i);
                sst += expected * expected;
            }
            sse += rdev * rdev;
            nTest++;
            state = State.testing;
        }
    }

    @Override
    void removeTest(List<Double> given, double expected, Log log) {
        throw new IllegalArgumentException("Cannot remove test data from multiple linear regression.");
    }

    @Override
    MillerLRModel test() {
        state = State.ready;
        tester.newTestData(trained.getParameterEstimates(), hasConstant(), getNumVars(), sst, sse, getNTest());
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
    MillerLRModel clearTest() {
        resetTest();
        state = State.training;
        return this;
    }
    //////////READY////////////
    @Override
    double predict(List<Double> given) {
        if (state == State.created || state == State.training) train();
        if (state == State.ready || state == State.testing) {
            double[] params = trained.getParameterEstimates();
            double result;
            if (R.hasIntercept()) {
                result = params[0];
                for (int i = 0; i < numVars; i++) result += params[i+1]*given.get(i);
            } else {
                result = 0;
                for (int i = 0; i < numVars; i++) result += params[i]*given.get(i);
            }
            return result;
        }
        throw new RuntimeException("Model in state '" + state.name() + "' so cannot make predictions.");
    }

    @Override
    Object data() {
        return trained.getParameterEstimates();
    }

    @Override
    LR.ModelResult asResult() {
        LR.ModelResult r = new LR.ModelResult(name, framework, hasConstant(), getNumVars(), state, getNTrain(), getNTest());
        if (trained != null) r.withTrainInfo("parameters", LR.doubleArrayToList(trained.getParameterEstimates()), "SSE", trained.getErrorSumSquares(),
                "RSquared", trained.getRSquared(), "adjRSquared", trained.getAdjustedRSquared(), "MSE", trained.getMeanSquareError(),
                "SSR", trained.getRegressionSumSquares(), "parameters std error", trained.getStdErrorOfEstimates(),
                "SST", trained.getTotalSumSquares());
        if (tester.isReady()) r.withTestInfo(tester.getStatistics());
        return r;
    }
    ///////////UTILS//////////

    @Override
    protected long getNTrain() {return R.getN();}

    @Override
    protected long getNTest() {
        return nTest;
    }

    @Override
    int getNumVars() { return numVars; }

    @Override
    boolean hasConstant() {return R.hasIntercept();}















}
