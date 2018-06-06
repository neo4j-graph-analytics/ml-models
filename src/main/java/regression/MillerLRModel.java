/**
 * Created by Lauren on 6/5/18.
 */

package regression;
import org.apache.commons.math3.stat.regression.MillerUpdatingRegression;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import org.apache.commons.math3.stat.regression.RegressionResults;

public class MillerLRModel extends LRModel{
    private MillerUpdatingRegression R;
    private RegressionResults trained;
    private int numVars;
    private double[] params;

    MillerLRModel(String name, int numVars, boolean constant) {
        super(name, Framework.Miller);
        R = new MillerUpdatingRegression(numVars, constant);
        this.numVars = numVars;
    }

    @Override
    protected long getN() {
        return R.getN();
    }

    @Override
    public void add(List<Double> given, double expected) {
        double[] givenArr = LR.convertFromList(given);
        R.addObservation(givenArr, expected);
        this.state = State.training;
    }

    @Override
    public Map<String, Double> train() {
        trained = R.regress();
        Map<String, Double> paramResult = new HashMap<>();
        double[] parameters = trained.getParameterEstimates();
        this.params = parameters;
        for (int i = 0; i < numVars; i++) {
            paramResult.put(Integer.toString(i), parameters[i]);
        }
        this.state = State.ready;
        return paramResult;
    }

    @Override
    public double predict(List<Double> given) {
        if (this.state == State.created) train();
        if (this.state == State.ready) {
            double result = 0;
            for (int i = 0; i < numVars; i++) result += given.get(i)*this.params[i];
            return result;
        }
        throw new RuntimeException("");
    }

    @Override
    public LR.StatResult stats() {
        return new LR.StatResult(R.getN(), numVars).withInfo("SSE", trained.getErrorSumSquares(),
                "rSquared", trained.getRSquared(), "estimates std error", trained.getStdErrorOfEstimates(),
                "hasIntercept", trained.hasIntercept());
    }

    @Override
    public Object data() {
        return trained.getParameterEstimates();
    }
}
