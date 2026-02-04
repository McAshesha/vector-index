package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import ru.mcashesha.metrics.Metric;

class LloydKMeans implements KMeans<LloydKMeans.Result> {
    private final int clusterCnt;
    private final int maxIterations;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public LloydKMeans(int clusterCnt,
        Metric.Type metricType,
        Metric.Engine metricEngine,
        int maxIterations,
        float tolerance,
        Random random) {
        if (clusterCnt <= 0)
            throw new IllegalArgumentException("clusterCount must be > 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (maxIterations <= 0)
            throw new IllegalArgumentException("maxIterations must be > 0");
        if (tolerance < 0)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.clusterCnt = clusterCnt;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.random = random;
    }

    @Override public Metric.Type getMetricType() {
        return metricType;
    }

    @Override public Metric.Engine getMetricEngine() {
        return metricEngine;
    }

    @Override public Result fit(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");

        int sampleCnt = data.length;
        int dimension = KMeansUtils.validateAndGetDimension(data);

        if (clusterCnt > sampleCnt) {
            throw new IllegalArgumentException(
                "clusterCount (" + clusterCnt + ") must be <= number of samples (" + sampleCnt + ")"
            );
        }

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        Metric.DistanceFunction l2DistFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(metricEngine);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, sampleCnt, dimension, clusterCnt, distFn, metricType, random);

        int[] labels = new int[sampleCnt];
        Arrays.fill(labels, -1);

        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];

        float[] pointErrors = new float[sampleCnt];
        boolean[] taken = new boolean[sampleCnt];

        int performedIterations = 0;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, pointErrors, null, distFn);

            KMeansUtils.recomputeCentroids(data, labels, newCentroids, clusterSizes, clusterCnt, dimension, metricType);

            KMeansUtils.handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors, taken,
                clusterCnt, dimension, metricType, random);

            float maxShift = computeMaxCentroidShift(centroids, newCentroids, l2DistFn);

            float[][] tmp = centroids;
            centroids = newCentroids;
            newCentroids = tmp;

            performedIterations = iteration + 1;

            if (maxShift <= tolerance)
                break;
        }

        Arrays.fill(clusterSizes, 0);
        float finalLoss = KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, null, clusterSizes, distFn);

        return new Result(labels, centroids, performedIterations, finalLoss, clusterSizes);
    }

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");

        int dimension = KMeansUtils.validateAndGetDimension(data);

        float[][] centroids = model.centroids;
        if (centroids == null || centroids.length == 0)
            throw new IllegalArgumentException("model must contain at least one centroid");

        if (centroids.length != clusterCnt) {
            throw new IllegalArgumentException(
                "model cluster count (" + centroids.length +
                    ") does not match this KMeans configuration (" + clusterCnt + ")"
            );
        }

        int centroidDim = centroids[0].length;
        if (centroidDim == 0)
            throw new IllegalArgumentException("centroids must have positive dimension");
        if (centroidDim != dimension)
            throw new IllegalArgumentException("dimension mismatch between data and centroids");

        for (int c = 1; c < centroids.length; c++) {
            if (centroids[c] == null || centroids[c].length != centroidDim)
                throw new IllegalArgumentException("all centroids must be non-null and have the same dimension");
        }

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        int[] labels = new int[data.length];
        KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, null, null, distFn);
        return labels;
    }

    private float computeMaxCentroidShift(float[][] oldCentroids,
        float[][] newCentroids,
        Metric.DistanceFunction l2DistFn) {
        float maxShift = 0;

        for (int c = 0; c < clusterCnt; c++) {
            float shift = l2DistFn.compute(oldCentroids[c], newCentroids[c]);
            if (shift > maxShift)
                maxShift = shift;
        }

        return maxShift;
    }

    static final class Result implements ClusteringResult {
        private final int[] labels;
        private final float[][] centroids;
        private final int iterations;
        private final float loss;
        private final int[] clusterSizes;

        public Result(int[] labels,
            float[][] centroids,
            int iterations,
            float loss,
            int[] clusterSizes) {
            this.labels = labels;
            this.centroids = centroids;
            this.iterations = iterations;
            this.loss = loss;
            this.clusterSizes = clusterSizes;
        }

        @Override public int[] getClusterAssignments() {
            return labels;
        }

        @Override public float[][] getCentroids() {
            return centroids;
        }

        public int getIterations() {
            return iterations;
        }

        @Override public float getLoss() {
            return loss;
        }

        @Override public int[] getClusterSizes() {
            return clusterSizes;
        }
    }
}
