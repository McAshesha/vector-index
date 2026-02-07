package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

class MiniBatchKMeans implements KMeans<MiniBatchKMeans.Result> {

    private final int clusterCnt;
    private final int batchSize;
    private final int maxIterations;
    private final int maxNoImprovementIterations;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public MiniBatchKMeans(
        int clusterCnt,
        int batchSize,
        Metric.Type metricType,
        Metric.Engine metricEngine,
        int maxIterations,
        float tolerance,
        int maxNoImprovementIterations,
        Random random) {
        if (clusterCnt <= 0)
            throw new IllegalArgumentException("clusterCount must be > 0");
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize must be > 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (maxIterations <= 0)
            throw new IllegalArgumentException("maxIterations must be > 0");
        if (maxNoImprovementIterations <= 0)
            throw new IllegalArgumentException("maxNoImprovementIterations must be > 0");
        if (tolerance < 0.0f)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.clusterCnt = clusterCnt;
        this.batchSize = batchSize;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.maxNoImprovementIterations = maxNoImprovementIterations;
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

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, sampleCnt, dimension, clusterCnt, distFn, metricType, random);

        long[] clusterCounts = new long[clusterCnt];

        int performedIterations = 0;
        float lastAverageBatchLoss = Float.POSITIVE_INFINITY;
        int noImprovementIterations = 0;

        int actualBatchSize = Math.min(batchSize, sampleCnt);
        int[] batchIndices = new int[actualBatchSize];
        float[][] batchSums = new float[clusterCnt][dimension];
        int[] batchClusterCounts = new int[clusterCnt];

        // Parallel sample pool initialization for large datasets
        int[] samplePool = new int[sampleCnt];
        if (sampleCnt >= 10000) {
            IntStream.range(0, sampleCnt).parallel().forEach(i -> samplePool[i] = i);
        } else {
            for (int i = 0; i < sampleCnt; i++)
                samplePool[i] = i;
        }

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            for (int i = 0; i < actualBatchSize; i++) {
                int j = i + random.nextInt(sampleCnt - i);
                int tmp = samplePool[i];
                samplePool[i] = samplePool[j];
                samplePool[j] = tmp;
                batchIndices[i] = samplePool[i];
            }

            // Clear batch accumulators - parallel for large cluster counts
            if (clusterCnt >= PARALLEL_UPDATE_THRESHOLD) {
                IntStream.range(0, clusterCnt).parallel().forEach(c -> {
                    if (batchClusterCounts[c] > 0) {
                        Arrays.fill(batchSums[c], 0.0f);
                        batchClusterCounts[c] = 0;
                    }
                });
            } else {
                for (int c = 0; c < clusterCnt; c++) {
                    if (batchClusterCounts[c] > 0) {
                        Arrays.fill(batchSums[c], 0.0f);
                        batchClusterCounts[c] = 0;
                    }
                }
            }

            float batchLossSum = assignMiniBatch(data, centroids, batchIndices, batchSums,
                batchClusterCounts, dimension, distFn);
            float averageBatchLoss = batchLossSum / actualBatchSize;

            updateCentroidsFromMiniBatch(centroids, clusterCounts, batchSums, batchClusterCounts, dimension);

            performedIterations++;

            if (!Float.isFinite(averageBatchLoss))
                break;

            if (Math.abs(lastAverageBatchLoss - averageBatchLoss) <= tolerance) {
                noImprovementIterations++;
                if (noImprovementIterations >= maxNoImprovementIterations)
                    break;
            }
            else
                noImprovementIterations = 0;

            lastAverageBatchLoss = averageBatchLoss;
        }

        int[] labels = new int[sampleCnt];
        float[] pointErrors = new float[sampleCnt];
        boolean[] taken = new boolean[sampleCnt];

        KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, pointErrors, null, distFn);

        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];
        KMeansUtils.recomputeCentroids(data, labels, newCentroids, clusterSizes, clusterCnt, dimension, metricType);

        KMeansUtils.handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors, taken,
            clusterCnt, dimension, metricType, random);

        float[][] finalCentroids = newCentroids;

        Arrays.fill(clusterSizes, 0);
        float finalLoss = KMeansUtils.assignPointsToClusters(data, finalCentroids, clusterCnt, labels, null, clusterSizes, distFn);

        return new Result(labels, finalCentroids, performedIterations, finalLoss, clusterSizes);
    }

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");

        int dimension = KMeansUtils.validateAndGetDimension(data);
        float[][] centroids = model.centroids;

        if (centroids == null || centroids.length == 0)
            throw new IllegalArgumentException("centroids must be non-null and non-empty");

        if (centroids.length != clusterCnt) {
            throw new IllegalArgumentException(
                "model cluster count (" + centroids.length +
                    ") does not match this KMeans configuration (" + clusterCnt + ")"
            );
        }

        int centroidDimension = centroids[0].length;
        if (centroidDimension == 0)
            throw new IllegalArgumentException("centroids must have positive dimension");
        if (centroidDimension != dimension)
            throw new IllegalArgumentException("dimension mismatch between data and centroids");

        for (int c = 1; c < centroids.length; c++) {
            if (centroids[c] == null || centroids[c].length != centroidDimension)
                throw new IllegalArgumentException("all centroids must be non-null and have the same dimension");
        }

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        int[] labels = new int[data.length];
        KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, null, null, distFn);
        return labels;
    }

    private static final int PARALLEL_BATCH_THRESHOLD = 256;

    private float assignMiniBatch(float[][] data,
        float[][] centroids,
        int[] batchIndices,
        float[][] batchSums,
        int[] batchClusterCounts,
        int dimension,
        Metric.DistanceFunction distFn) {

        int batchLen = batchIndices.length;

        if (batchLen >= PARALLEL_BATCH_THRESHOLD) {
            return assignMiniBatchParallel(data, centroids, batchIndices, batchSums, batchClusterCounts, dimension, distFn);
        }

        // Sequential path for small batches
        float batchLoss = 0.0f;

        for (int sampleIdx : batchIndices) {
            float[] point = data[sampleIdx];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            batchLoss += nearestDistance;

            float[] sum = batchSums[nearestClusterIdx];
            for (int d = 0; d < dimension; d++)
                sum[d] += point[d];

            batchClusterCounts[nearestClusterIdx]++;
        }

        return batchLoss;
    }

    private float assignMiniBatchParallel(float[][] data,
        float[][] centroids,
        int[] batchIndices,
        float[][] batchSums,
        int[] batchClusterCounts,
        int dimension,
        Metric.DistanceFunction distFn) {

        int batchLen = batchIndices.length;
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), batchLen);
        int chunkSize = (batchLen + numThreads - 1) / numThreads;

        // Thread-local accumulators
        float[][][] localSums = new float[numThreads][clusterCnt][dimension];
        int[][] localCounts = new int[numThreads][clusterCnt];
        float[] localLoss = new float[numThreads];

        // Parallel assignment with thread-local accumulation
        IntStream.range(0, numThreads).parallel().forEach(threadIdx -> {
            int start = threadIdx * chunkSize;
            int end = Math.min(start + chunkSize, batchLen);

            float[][] mySums = localSums[threadIdx];
            int[] myCounts = localCounts[threadIdx];
            float myLoss = 0f;

            for (int b = start; b < end; b++) {
                int sampleIdx = batchIndices[b];
                float[] point = data[sampleIdx];

                int nearestClusterIdx = 0;
                float nearestDistance = distFn.compute(point, centroids[0]);

                for (int c = 1; c < clusterCnt; c++) {
                    float distance = distFn.compute(point, centroids[c]);
                    if (distance < nearestDistance) {
                        nearestDistance = distance;
                        nearestClusterIdx = c;
                    }
                }

                myLoss += nearestDistance;
                myCounts[nearestClusterIdx]++;

                float[] sum = mySums[nearestClusterIdx];
                for (int d = 0; d < dimension; d++)
                    sum[d] += point[d];
            }

            localLoss[threadIdx] = myLoss;
        });

        // Merge results
        float totalLoss = 0f;
        for (int t = 0; t < numThreads; t++) {
            totalLoss += localLoss[t];

            for (int c = 0; c < clusterCnt; c++) {
                batchClusterCounts[c] += localCounts[t][c];

                float[] threadSum = localSums[t][c];
                float[] targetSum = batchSums[c];
                for (int d = 0; d < dimension; d++)
                    targetSum[d] += threadSum[d];
            }
        }

        return totalLoss;
    }

    private static final int PARALLEL_UPDATE_THRESHOLD = 64;

    private void updateCentroidsFromMiniBatch(float[][] centroids,
        long[] clusterCounts,
        float[][] batchSums,
        int[] batchClusterCounts,
        int dimension) {

        if (clusterCnt >= PARALLEL_UPDATE_THRESHOLD) {
            // Parallel centroid update
            IntStream.range(0, clusterCnt).parallel().forEach(c -> {
                int batchCnt = batchClusterCounts[c];
                if (batchCnt == 0)
                    return;

                long oldCnt = clusterCounts[c];
                long newCnt = oldCnt + batchCnt;

                clusterCounts[c] = newCnt;

                double invNewCnt = 1.0 / newCnt;
                float[] centroid = centroids[c];
                float[] sum = batchSums[c];

                for (int d = 0; d < dimension; d++)
                    centroid[d] = (float) (((double) centroid[d] * oldCnt + sum[d]) * invNewCnt);
            });
        } else {
            // Sequential path
            for (int c = 0; c < clusterCnt; c++) {
                int batchCnt = batchClusterCounts[c];
                if (batchCnt == 0)
                    continue;

                long oldCnt = clusterCounts[c];
                long newCnt = oldCnt + batchCnt;

                clusterCounts[c] = newCnt;

                double invNewCnt = 1.0 / newCnt;
                float[] centroid = centroids[c];
                float[] sum = batchSums[c];

                for (int d = 0; d < dimension; d++)
                    centroid[d] = (float) (((double) centroid[d] * oldCnt + sum[d]) * invNewCnt);
            }
        }

        if (metricType == Metric.Type.COSINE_DISTANCE)
            KMeansUtils.normalizeCentroids(centroids);
    }

    static class Result implements KMeans.ClusteringResult {
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
