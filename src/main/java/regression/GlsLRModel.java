package regression;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.stat.regression.GLSMultipleLinearRegression;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.neo4j.logging.Log;

//NOT CURRENTLY WORKING, DEVELOPMENT PAUSED//

/**
 * Created by Lauren on 6/5/18.
 */
public class GlsLRModel extends LRModel {
    private GLSMultipleLinearRegression R;
    private final List<List<Double>> data = new ArrayList<>();
    private final List<Double> response = new ArrayList<>();
    private int numVars;
    private int numObs;
    private double[] params;

    GlsLRModel(String model, int numVars, boolean intercept) {
        super(model, Framework.GLS);
        R = new GLSMultipleLinearRegression();
        R.setNoIntercept(!intercept);
        numObs = 0;
        this.numVars = numVars;
    }
    @Override
    protected long getN() {
        return numObs;
    }

    @Override
    int getNumVars() { return numVars; }

    @Override
    boolean hasConstant() {return !R.isNoIntercept();}

    @Override
    public void addTrain(List<Double> given, double expected, Log log) {
        if (given.size() != numVars) throw new IllegalArgumentException("incorrect number of variables in given.");
        data.add(given);
        response.add(expected);
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

    /*@Override
    public LR.StatResult stats() {
        return new LR.StatResult(this.getN(), this.numVars);
    }*/

    @Override
    public LR.ModelResult train() {
        double[][] dataArray = new double[this.numObs][this.numVars];
        for (int i = 0; i < numObs; i++) {
            for (int j = 0; j < numVars; j++)
                dataArray[i][j] = data.get(i).get(j);
        }
        RealMatrix data = new BlockRealMatrix(dataArray);
        double[] obs = LR.doubleListToArray(response);
        Covariance c = new Covariance(data);
        RealMatrix m = c.getCovarianceMatrix();
        double[][] covariance = m.getData();
        R.newSampleData(obs, dataArray, covariance);
        params = R.estimateRegressionParameters();
        this.state = State.ready;
        List<Double> paramResult = new ArrayList<>();
        for (int i = 0; i < numVars; i++) {
            paramResult.add(params[i]);
        }
        return new LR.ModelResult(name, framework, hasConstant(), getNumVars(), state, getN()).withInfo("parameters", paramResult);
    }

    @Override
    void addTest(List<Double> given, double expected, Log log) {

    }

    @Override
    void test() {

    }

    @Override
    void copy(String string) {

    }
}
