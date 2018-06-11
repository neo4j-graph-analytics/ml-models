package regression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Lauren on 6/5/18.
 */
public class OlsLRModel extends LRModel {
    private OLSMultipleLinearRegression R;
    final List<Double> data = new ArrayList<>();
    private int numVars;
    private int numObs;
    private double[] params;

    OlsLRModel(String model, boolean intercept, int numVars) {
        super(model, Framework.OLS);
        R = new OLSMultipleLinearRegression();
        R.setNoIntercept(!intercept);
        numObs = 0;
        this.numVars = numVars;
    }
    @Override
    protected long getN() {
        return numObs;
    }

    @Override
    long getNumVars() { return numVars; }

    @Override
    boolean hasConstant() { return !R.isNoIntercept(); }

    @Override
    void add(List<Double> given, double expected) {
        if (given.size() != numVars) throw new IllegalArgumentException("incorrect number of variables in given.");
        data.add(expected);
        data.addAll(given);
        numObs += 1;
        this.state = State.training;
    }

    @Override
    double predict(List<Double> given) {
        if (given.size() != numVars) throw new IllegalArgumentException("incorrect number of variables in given.");
        if (this.state == State.training) train();
        double result = 0;
        if (R.isNoIntercept()) {
            for (int i = 0; i < numVars; i++) result += params[i] * given.get(i);
        } else {
            result += params[0];
            for (int i = 0; i < numVars; i++) result += params[i + 1] * given.get(i);
        }
        return result;
    }

    @Override
    Object data() {
        if (this.state == State.training) train();
        if (this.state == State.ready) return this.params;
        else throw new RuntimeException(this.name + "is not in a state for serialization.");
    }

    /*@Override
    public LR.StatResult stats() {
        return new LR.StatResult(this.getN(), this.numVars).withInfo("rSquared", R.calculateRSquared());
    }*/

    @Override
    LR.ModelResult train() {
        double[] dataArray = LR.convertFromList(data);
        R.newSampleData(dataArray, numObs, numVars);
        params = R.estimateRegressionParameters();
        this.state = State.ready;
        List<Double> paramList = new ArrayList<>();
        for (int i = 0; i < numVars; i++) {
            paramList.add(params[i]);
        }
        return new LR.ModelResult(name, framework, hasConstant(), getNumVars(), state, getN()).withInfo("parameters", paramList);
    }

    @Override
    LR.ModelResult asResult() {
        LR.ModelResult r = new LR.ModelResult(name, framework, hasConstant(), numVars, state, getN());
        return params != null ? r.withInfo("parameters", LR.doubleArrayToList(params), "rSquared", R.calculateRSquared()) : r;
    }

}
