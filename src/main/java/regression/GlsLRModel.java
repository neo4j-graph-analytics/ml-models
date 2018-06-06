package regression;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.stat.regression.GLSMultipleLinearRegression;
import org.apache.commons.math3.stat.correlation.Covariance;

/**
 * Created by Lauren on 6/5/18.
 */
public class GlsLRModel extends LRModel {
    private GLSMultipleLinearRegression R;
    final List<List<Double>> data = new ArrayList<>();
    final List<Double> response = new ArrayList<>();
    private int numVars;
    private int numObs;
    private double[] params;

    GlsLRModel(String model, int numVars) {
        super(model, "GLS");
        R = new GLSMultipleLinearRegression();
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
        data.add(given);
        response.add(expected);
        numObs += 1;
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
    public Object serialize() {
        if (this.state == State.training) train();
        if (this.state == State.ready) return this.params;
        else throw new RuntimeException(this.name + "is not in a state for serialization.");
    }

    @Override
    public LR.StatResult stats() {
        return new LR.StatResult(this.getN(), this.numVars);
    }

    @Override
    public Map<String, Double> train() {
        double[][] dataArray = new double[this.numObs][this.numVars];
        for (int i = 0; i < numObs; i++) {
            dataArray[i] = LR.convertFromList(data.get(i));
        }
        double[] obs = LR.convertFromList(response);
        double[][] covariance = new Covariance(dataArray).getCovarianceMatrix().getData();
        R.newSampleData(obs, dataArray, covariance);
        params = R.estimateRegressionParameters();
        this.state = State.ready;
        Map<String, Double> paramResult = new HashMap<>();
        for (int i = 0; i < numVars; i++) {
            paramResult.put(Integer.toString(i), params[i]);
        }
        return paramResult;
    }
}
