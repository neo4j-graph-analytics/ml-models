package regression;
import java.util.HashMap;
import java.util.Map;

class ModelAnalyzer {
    private double[] parameters;
    private boolean hasIntercept;
    private long n;
    private int rank;
    private boolean ready;

    Map<String, Double> statistics; //rsquared, adj rsquared, sse, mse, sst

    ModelAnalyzer() { statistics = new HashMap<>(); ready = false;}

    void newTestData(double[] params, boolean hasIntercept, int rank, double sst, double sse, long nTest) {
        this.parameters = params;
        this.hasIntercept = hasIntercept;
        this.n = nTest;
        this.rank = rank;
        this.statistics = new HashMap<>();

        double r2 = 1 - sse/sst;
        statistics.put("RSquared", r2);
        if (hasIntercept) statistics.put("adjRSquared", 1 - sse*(n - 1)/(sst*(n - rank)));
        else statistics.put("adjRSquared", 1 - (1 - r2)*(n/(n - rank)));
        statistics.put("SSE", sse);
        statistics.put("MSE", sse/(n - rank));
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
    double getTotalSumSquares() {return statistics.get("SST");}


}
