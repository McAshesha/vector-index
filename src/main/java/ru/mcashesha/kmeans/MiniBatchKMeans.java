package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

/**
 * Mini-Batch KMeans clustering algorithm (Sculley, 2010).
 *
 * <p>Instead of using the full dataset in each iteration (as in Lloyd's algorithm),
 * Mini-Batch KMeans samples a random subset (mini-batch) and performs a weighted
 * centroid update. This dramatically reduces per-iteration cost while converging
 * to a solution comparable to full-pass KMeans.</p>
 *
 * <h3>Algorithm outline:</h3>
 * <ol>
 *   <li>Initialize centroids via KMeans++.</li>
 *   <li>For each iteration:
 *     <ol type="a">
 *       <li>Sample a random mini-batch using Fisher-Yates partial shuffle.</li>
 *       <li>Assign batch points to nearest centroids; accumulate per-cluster sums.</li>
 *       <li>Update centroids with a weighted average:
 *           {@code centroid = (centroid * oldCount + batchSum) / newCount}.</li>
 *       <li>Check for early stopping: if the average batch loss does not improve
 *           beyond the tolerance for {@code maxNoImprovementIterations} consecutive
 *           iterations, stop.</li>
 *     </ol>
 *   </li>
 *   <li>Perform a final full-pass reassignment and centroid recomputation to produce
 *       clean results (mini-batch updates can leave centroids slightly noisy).</li>
 * </ol>
 *
 * <p>The weighted centroid update uses cumulative counts across all iterations,
 * so early batches have more influence and later batches act as fine-tuning.
 * This is the key insight from the Sculley (2010) paper.</p>
 *
 * @see KMeansUtils#initializeCentroidsKMeansPlusPlus
 */
class MiniBatchKMeans implements KMeans<MiniBatchKMeans.Result> {

    /** Number of clusters (k) to produce. */
    private final int clusterCnt;
    /** Number of randomly sampled data points per mini-batch iteration. */
    private final int batchSize;
    /** Maximum number of mini-batch iterations. */
    private final int maxIterations;
    /** Early-stop patience: stop after this many consecutive no-improvement iterations. */
    private final int maxNoImprovementIterations;
    /** Minimum change in average batch loss to count as improvement. */
    private final float tolerance;
    /** The distance metric to use for clustering (e.g., L2, dot product, cosine). */
    private final Metric.Type metricType;
    /** The computation engine for distance calculations (e.g., Scalar, VectorAPI, SimSIMD). */
    private final Metric.Engine metricEngine;
    /** Random number generator for batch sampling and initialization. */
    private final Random random;

    /**
     * Constructs a MiniBatchKMeans instance with the specified configuration.
     *
     * @param clusterCnt                  the number of clusters; must be positive
     * @param batchSize                   the mini-batch size; must be positive
     * @param metricType                  the distance metric; must not be null
     * @param metricEngine                the computation engine; must not be null
     * @param maxIterations               the maximum number of iterations; must be positive
     * @param tolerance                   the convergence tolerance; must be non-negative
     * @param maxNoImprovementIterations  the early-stopping patience; must be positive
     * @param random                      the RNG; must not be null
     * @throws IllegalArgumentException if any argument violates its constraints
     */
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

    /**
     * Trains the Mini-Batch KMeans model on the provided dataset.
     *
     * <p>The training proceeds in three phases:</p>
     * <ol>
     *   <li><b>Mini-batch iterations:</b> repeatedly sample batches, assign to nearest
     *       centroids, and update centroids with weighted averages.</li>
     *   <li><b>Final full-pass reassignment:</b> assign all data points using the
     *       converged centroids, then recompute centroids as exact means.</li>
     *   <li><b>Empty cluster handling:</b> redistribute points to fill any empty clusters
     *       that may have emerged.</li>
     * </ol>
     *
     * @param data the input dataset; must be non-null, non-empty, with consistent dimensions
     * @return the clustering result containing labels, centroids, iteration count, loss,
     *         and cluster sizes
     * @throws IllegalArgumentException if data is invalid or k exceeds the number of samples
     */
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

        // Step 1: KMeans++ initialization
        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, sampleCnt, dimension, clusterCnt, distFn, metricType, random);

        // Cumulative per-cluster sample counts across all iterations.
        // Used in the weighted centroid update formula: centroid = (centroid * oldCnt + batchSum) / newCnt
        long[] clusterCounts = new long[clusterCnt];

        int performedIterations = 0;
        float lastAverageBatchLoss = Float.POSITIVE_INFINITY;
        int noImprovementIterations = 0;

        // Clamp batch size to dataset size
        int actualBatchSize = Math.min(batchSize, sampleCnt);
        int[] batchIndices = new int[actualBatchSize];

        // Per-batch accumulators for centroid sums and counts
        float[][] batchSums = new float[clusterCnt][dimension];
        int[] batchClusterCounts = new int[clusterCnt];

        // Initialize sample pool for Fisher-Yates partial shuffle.
        // The pool contains all indices [0, sampleCnt) and is partially shuffled
        // each iteration to select the mini-batch without replacement.
        int[] samplePool = new int[sampleCnt];
        if (sampleCnt >= 10000) {
            IntStream.range(0, sampleCnt).parallel().forEach(i -> samplePool[i] = i);
        } else {
            for (int i = 0; i < sampleCnt; i++)
                samplePool[i] = i;
        }

        // Step 2: Mini-batch iteration loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Fisher-Yates partial shuffle: swap each of the first actualBatchSize
            // positions with a random position from the remaining pool.
            // This efficiently selects a random subset without replacement.
            for (int i = 0; i < actualBatchSize; i++) {
                int j = i + random.nextInt(sampleCnt - i);
                int tmp = samplePool[i];
                samplePool[i] = samplePool[j];
                samplePool[j] = tmp;
                batchIndices[i] = samplePool[i];
            }

            // Clear batch accumulators from the previous iteration.
            // Only reset clusters that actually had points (sparse clearing).
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

            // Assign batch points to nearest centroids and accumulate per-cluster sums
            float batchLossSum = assignMiniBatch(data, centroids, batchIndices, batchSums,
                batchClusterCounts, dimension, distFn);
            float averageBatchLoss = batchLossSum / actualBatchSize;

            // Weighted centroid update using cumulative counts
            updateCentroidsFromMiniBatch(centroids, clusterCounts, batchSums, batchClusterCounts, dimension);

            performedIterations++;

            // Bail out on numerical instability (NaN/Inf)
            if (!Float.isFinite(averageBatchLoss))
                break;

            // Early stopping: track consecutive iterations without meaningful improvement
            if (Math.abs(lastAverageBatchLoss - averageBatchLoss) <= tolerance) {
                noImprovementIterations++;
                if (noImprovementIterations >= maxNoImprovementIterations)
                    break;
            }
            else
                noImprovementIterations = 0;

            lastAverageBatchLoss = averageBatchLoss;
        }

        // Step 3: Final full-pass reassignment and exact centroid recomputation.
        // Mini-batch updates can leave centroids slightly noisy; a full pass cleans them up.
        int[] labels = new int[sampleCnt];
        float[] pointErrors = new float[sampleCnt];
        boolean[] taken = new boolean[sampleCnt];

        KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, pointErrors, null, distFn);

        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];
        KMeansUtils.recomputeCentroids(data, labels, newCentroids, clusterSizes, clusterCnt, dimension, metricType);

        // Handle any clusters that became empty during the final recomputation
        KMeansUtils.handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors, taken,
            clusterCnt, dimension, metricType, random);

        float[][] finalCentroids = newCentroids;

        // Final assignment pass for definitive labels, loss, and cluster sizes
        Arrays.fill(clusterSizes, 0);
        float finalLoss = KMeansUtils.assignPointsToClusters(data, finalCentroids, clusterCnt, labels, null, clusterSizes, distFn);

        return new Result(labels, finalCentroids, performedIterations, finalLoss, clusterSizes);
    }

    /**
     * Assigns new data points to the nearest centroids from a previously fitted model.
     *
     * @param data  the data points to classify; must have the same dimensionality as the training data
     * @param model the result of a prior {@link #fit(float[][])} call
     * @return an array of cluster indices, one per input data point
     * @throws IllegalArgumentException if data or model is invalid, or dimensions do not match
     */
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

    /**
     * Minimum mini-batch size to trigger parallel assignment.
     * Below this threshold, sequential processing avoids fork-join overhead.
     */
    private static final int PARALLEL_BATCH_THRESHOLD = 256;

    /**
     * Assigns mini-batch points to nearest centroids and accumulates per-cluster sums.
     *
     * <p>Each batch point is assigned to its nearest centroid via brute-force scan.
     * The point's coordinates are accumulated into {@code batchSums} for the assigned
     * cluster, and the per-cluster count is incremented.</p>
     *
     * <p>Dispatches to the parallel implementation for batches with 256 or more points.</p>
     *
     * @param data               the full dataset
     * @param centroids          the current centroid positions
     * @param batchIndices       indices of the sampled mini-batch points
     * @param batchSums          output: per-cluster coordinate sums from this batch
     * @param batchClusterCounts output: per-cluster point counts from this batch
     * @param dimension          the dimensionality of data points
     * @param distFn             the distance function
     * @return the total batch loss (sum of distances from each batch point to its nearest centroid)
     */
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

            // Find nearest centroid via linear scan
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

            // Accumulate the point's coordinates into the assigned cluster's sum
            float[] sum = batchSums[nearestClusterIdx];
            for (int d = 0; d < dimension; d++)
                sum[d] += point[d];

            batchClusterCounts[nearestClusterIdx]++;
        }

        return batchLoss;
    }

    /**
     * Parallel implementation of mini-batch assignment using thread-local accumulators.
     *
     * <p>The batch is partitioned into chunks (one per available processor). Each thread
     * maintains its own sum and count arrays to avoid synchronization. After parallel
     * execution, thread-local results are merged into the shared output arrays.</p>
     *
     * @param data               the full dataset
     * @param centroids          the current centroid positions
     * @param batchIndices       indices of the sampled mini-batch points
     * @param batchSums          output: per-cluster coordinate sums from this batch
     * @param batchClusterCounts output: per-cluster point counts from this batch
     * @param dimension          the dimensionality of data points
     * @param distFn             the distance function
     * @return the total batch loss
     */
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

        // Thread-local accumulators: each thread writes to its own arrays
        float[][][] localSums = new float[numThreads][clusterCnt][dimension];
        int[][] localCounts = new int[numThreads][clusterCnt];
        float[] localLoss = new float[numThreads];

        // Parallel assignment: each thread processes its chunk independently
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

        // Merge thread-local results into shared output arrays
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

    /**
     * Minimum number of clusters to parallelize the centroid update step.
     * Also used as the threshold for parallel batch accumulator clearing.
     */
    private static final int PARALLEL_UPDATE_THRESHOLD = 64;

    /**
     * Updates centroids using the weighted average formula from the mini-batch results.
     *
     * <p>For each cluster that received at least one batch point, the centroid is updated as:
     * <pre>
     *   newCount = oldCount + batchCount
     *   centroid[d] = (centroid[d] * oldCount + batchSum[d]) / newCount
     * </pre>
     * This gives earlier iterations more weight (since oldCount grows over time),
     * providing a natural annealing effect where later batches make smaller adjustments.</p>
     *
     * <p>If cosine distance is used, centroids are re-normalized to unit length after the update.</p>
     *
     * @param centroids          the current centroid positions (modified in place)
     * @param clusterCounts      cumulative per-cluster sample counts across all iterations
     * @param batchSums          per-cluster coordinate sums from the current batch
     * @param batchClusterCounts per-cluster point counts from the current batch
     * @param dimension          the dimensionality of data points
     */
    private void updateCentroidsFromMiniBatch(float[][] centroids,
        long[] clusterCounts,
        float[][] batchSums,
        int[] batchClusterCounts,
        int dimension) {

        if (clusterCnt >= PARALLEL_UPDATE_THRESHOLD) {
            // Parallel centroid update: each cluster is independent
            IntStream.range(0, clusterCnt).parallel().forEach(c -> {
                int batchCnt = batchClusterCounts[c];
                if (batchCnt == 0)
                    return;

                long oldCnt = clusterCounts[c];
                long newCnt = oldCnt + batchCnt;

                clusterCounts[c] = newCnt;

                // Use double precision for the weighted average to reduce numerical error
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

                // Use double precision for the weighted average to reduce numerical error
                double invNewCnt = 1.0 / newCnt;
                float[] centroid = centroids[c];
                float[] sum = batchSums[c];

                for (int d = 0; d < dimension; d++)
                    centroid[d] = (float) (((double) centroid[d] * oldCnt + sum[d]) * invNewCnt);
            }
        }

        // For cosine distance, project centroids back onto the unit sphere
        if (metricType == Metric.Type.COSINE_DISTANCE)
            KMeansUtils.normalizeCentroids(centroids);
    }

    /**
     * Immutable result of Mini-Batch KMeans clustering.
     *
     * <p>Contains the final cluster assignments (from the full-pass reassignment),
     * cleaned-up centroid positions, the number of mini-batch iterations performed,
     * the total loss, and per-cluster membership counts.</p>
     */
    static class Result implements KMeans.ClusteringResult {
        /** Per-point cluster assignments (0-based cluster indices). */
        private final int[] labels;
        /** Final centroid positions after full-pass recomputation: {@code float[k][dimension]}. */
        private final float[][] centroids;
        /** Number of mini-batch iterations performed. */
        private final int iterations;
        /** Total loss: sum of distances from each point to its assigned centroid. */
        private final float loss;
        /** Number of points assigned to each cluster. */
        private final int[] clusterSizes;

        /**
         * Constructs a clustering result.
         *
         * @param labels       per-point cluster assignments
         * @param centroids    final centroid positions
         * @param iterations   number of mini-batch iterations performed
         * @param loss         total clustering loss
         * @param clusterSizes per-cluster membership counts
         */
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

        /** {@inheritDoc} */
        @Override public int[] getClusterAssignments() {
            return labels;
        }

        /** {@inheritDoc} */
        @Override public float[][] getCentroids() {
            return centroids;
        }

        /**
         * Returns the number of mini-batch iterations performed before convergence,
         * early stopping, or reaching the iteration limit.
         *
         * @return the iteration count
         */
        public int getIterations() {
            return iterations;
        }

        /** {@inheritDoc} */
        @Override public float getLoss() {
            return loss;
        }

        /** {@inheritDoc} */
        @Override public int[] getClusterSizes() {
            return clusterSizes;
        }
    }

}
