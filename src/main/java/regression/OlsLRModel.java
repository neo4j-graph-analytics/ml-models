package regression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 * Created by Lauren on 6/5/18.
 */
public class OlsLRModel extends LRModel {
    private OLSMultipleLinearRegression R;
    final List<Double> data = new ArrayList<>();
    private int numVars;
    private int numObs;
    private double[] params;

    OlsLRModel(String model, int numVars, boolean intercept) {
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
    public void add(List<Double> given, double expected) {
        if (given.size() != numVars) throw new IllegalArgumentException("incorrect number of variables in given.");
        data.add(expected);
        data.addAll(given);
        numObs += 1;
        this.state = State.training;
    }

    @Override
    public double predict(List<Double> given) {
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
    public Object data() {
        if (this.state == State.training) train();
        if (this.state == State.ready) return this.params;
        else throw new RuntimeException(this.name + "is not in a state for serialization.");
    }

    @Override
    public LR.StatResult stats() {
        return new LR.StatResult(this.getN(), this.numVars).withInfo("rSquared", R.calculateRSquared());
    }

    @Override
    public Map<String, Double> train() {
        double[] dataArray = LR.convertFromList(data);
        R.newSampleData(dataArray, numObs, numVars);
        params = R.estimateRegressionParameters();
        this.state = State.ready;
        Map<String, Double> paramResult = new HashMap<>();
        for (int i = 0; i < numVars; i++) {
            paramResult.put(Integer.toString(i), params[i]);
        }
        return paramResult;
    }
}
