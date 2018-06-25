package regression;
import java.util.HashMap;
import java.util.List;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import java.util.Map;

class ModelAnalyzer {
    private double[] parameters;
    private boolean hasIntercept;
    private long n;
    private int rank;
    private boolean ready;

    Map<String, Double> statistics; //rsquared, adj rsquared, sse, mse, ssr, sst

    ModelAnalyzer() { statistics = new HashMap<>(); ready = false;}

    void newTestData(double[] params, boolean hasIntercept, int rank, double sst, double sse, long nTest) {
        this.parameters = params;
        this.hasIntercept = hasIntercept;
        this.n = nTest;
        this.rank = rank;
        this.statistics = new HashMap<>();

        double ssr = sst - sse;
        statistics.put("RSquared", ssr/sst);
        statistics.put("adjRSquared", 1 - sse*(n - 1)/(sst*(n - rank)));
        statistics.put("SSE", sse);
        statistics.put("MSE", sse/(n - rank));
        statistics.put("SSR", ssr);
        statistics.put("SST", sst);
        ready = true;
    }

    void clear() {
        ready = false;
        statistics.clear();
    }

    boolean isReady() {return ready;}


    Map<String, Double> getStatistics() {
        return statistics;
    }
    double[] getParameters() {return parameters;}
    boolean isHasIntercept() {return hasIntercept;}
    long getN() {return n;}
    int getRank() {return rank;}
    double getRSquared() {return statistics.get("RSquared");}
    double getAdjustedRSquared() {return statistics.get("adjRSquared");}
    double getSumSquaredErrors() {return statistics.get("SSE");}
    double getMeanSquareError() {return statistics.get("MSE");}
    double getRegressionSumSquares() {return statistics.get("SSR");}
    double getTotalSumSquares() {return statistics.get("SST");}


}
