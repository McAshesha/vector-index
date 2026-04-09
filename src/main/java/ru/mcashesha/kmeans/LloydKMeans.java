package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

/**
 * Classic Lloyd's algorithm for KMeans clustering (full-pass iterative).
 *
 * <p>Lloyd's algorithm alternates between two steps until convergence or the
 * iteration limit is reached:</p>
 * <ol>
 *   <li><b>Assignment step:</b> each data point is assigned to the nearest centroid.</li>
 *   <li><b>Update step:</b> each centroid is recomputed as the mean of its assigned points.</li>
 * </ol>
 *
 * <p>Convergence is determined by the maximum centroid shift (L2 squared distance
 * between old and new centroid positions). When the maximum shift falls below the
 * configured tolerance, the algorithm terminates early.</p>
 *
 * <p>Distance-based pruning is automatically enabled when the number of clusters
 * is large ({@code k >= 64}) and the dataset has at least 1000 points. This optimization
 * uses precomputed centroid-to-centroid distances to skip likely non-nearest clusters
 * during intermediate assignment steps, reducing the number of distance computations.
 * The final assignment pass always uses unpruned brute-force search for correctness.</p>
 *
 * <p>Centroids are initialized using the KMeans++ algorithm for improved convergence
 * properties. Empty clusters are handled by redistributing the highest-error point
 * from the largest cluster.</p>
 *
 * @see KMeansUtils#initializeCentroidsKMeansPlusPlus
 * @see KMeansUtils#assignPointsToClustersWithPruning  Distance-based pruning for intermediate iterations
 */
class LloydKMeans implements KMeans<LloydKMeans.Result> {
    private static final Logger LOGGER = Logger.getLogger(LloydKMeans.class.getName());

    /** Number of clusters (k) to produce. */
    private final int clusterCnt;
    /** Maximum number of assign-update iterations. */
    private final int maxIterations;
    /** Convergence threshold: maximum centroid shift (L2 squared distance) below which iterations stop. */
    private final float tolerance;
    /** The distance metric to use for clustering (e.g., L2, dot product, cosine). */
    private final Metric.Type metricType;
    /** The computation engine for distance calculations (e.g., Scalar, VectorAPI, SimSIMD). */
    private final Metric.Engine metricEngine;
    /** Random number generator for KMeans++ initialization. */
    private final Random random;

    /**
     * Constructs a LloydKMeans instance with the specified configuration.
     *
     * @param clusterCnt    the number of clusters; must be positive
     * @param metricType    the distance metric; must not be null
     * @param metricEngine  the computation engine; must not be null
     * @param maxIterations the maximum number of iterations; must be positive
     * @param tolerance     the convergence tolerance; must be non-negative
     * @param random        the RNG for initialization; must not be null
     * @throws IllegalArgumentException if any argument violates its constraints
     */
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

    /**
     * Minimum number of clusters required to enable triangle inequality pruning.
     * Below this threshold, the overhead of precomputing centroid distances
     * outweighs the savings from skipping distance computations.
     */
    private static final int PRUNING_THRESHOLD = 64;  // Use pruning when k >= this

    /**
     * Convenience overload for clustering a subset of data identified by indices.
     * Creates a lightweight view (array of references) internally.
     *
     * @param data    the full dataset
     * @param indices indices into {@code data} identifying the subset to cluster
     * @return the clustering result (labels are relative to the subset, not the full dataset)
     */
    public Result fit(float[][] data, int[] indices) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (indices == null || indices.length == 0)
            throw new IllegalArgumentException("indices must be non-null and non-empty");

        float[][] subset = new float[indices.length][];
        for (int i = 0; i < indices.length; i++)
            subset[i] = data[indices[i]];

        return fit(subset);
    }

    /**
     * Trains the Lloyd's KMeans model on the provided dataset.
     *
     * <p>Algorithm flow:</p>
     * <ol>
     *   <li>Initialize centroids via KMeans++.</li>
     *   <li>Iterate: assign points to nearest centroids, recompute centroids,
     *       handle empty clusters, check convergence.</li>
     *   <li>Perform a final full assignment pass to compute definitive labels and loss.</li>
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

        if (metricType == Metric.Type.DOT_PRODUCT) {
            LOGGER.warning("DOT_PRODUCT metric assumes normalized input vectors. "
                + "Consider COSINE_DISTANCE for unnormalized data.");
        }

        // Resolve the configured distance function and an L2 function for convergence checks
        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        Metric.DistanceFunction l2DistFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(metricEngine);

        // For cosine distance, measure convergence by angular change (cosine distance between
        // old and new centroid) rather than L2 shift, since centroids are on the unit sphere.
        Metric.DistanceFunction convergenceDistFn = metricType == Metric.Type.COSINE_DISTANCE
            ? Metric.Type.COSINE_DISTANCE.resolveFunction(metricEngine) : l2DistFn;

        // Step 1: KMeans++ initialization
        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, sampleCnt, dimension, clusterCnt, distFn, metricType, random);

        int[] labels = new int[sampleCnt];
        Arrays.fill(labels, -1);

        // Double-buffered centroids: centroids and newCentroids are swapped each iteration
        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];

        // Per-point error tracking for empty cluster redistribution
        float[] pointErrors = new float[sampleCnt];
        boolean[] taken = new boolean[sampleCnt];

        // Enable distance-based pruning for large cluster counts with sufficient data.
        // This reduces O(n*k) distance computations by skipping likely non-nearest centroids.
        boolean usePruning = clusterCnt >= PRUNING_THRESHOLD && sampleCnt >= 1000;
        float[][] centroidDistances = null;

        // Pre-allocate thread-local buffers for parallel centroid recomputation.
        // This avoids allocating O(threads * k * d) memory on every iteration,
        // which for 16 threads x 1000 clusters x 768 dims would be ~47MB per iteration.
        int numThreads = Runtime.getRuntime().availableProcessors();
        float[][][] preallocSums = new float[numThreads][clusterCnt][dimension];
        int[][] preallocCounts = new int[numThreads][clusterCnt];

        int performedIterations = 0;

        // Step 2: Iterative assign-update loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            if (usePruning) {
                // Recompute centroid-to-centroid distances each iteration (centroids move)
                centroidDistances = KMeansUtils.precomputeCentroidDistances(centroids, distFn);
                KMeansUtils.assignPointsToClustersWithPruning(
                    data, centroids, centroidDistances, clusterCnt, labels, pointErrors, null, distFn);
            } else {
                KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, pointErrors, null, distFn);
            }

            // Recompute centroids as the mean of each cluster's assigned points
            KMeansUtils.recomputeCentroids(data, labels, newCentroids, clusterSizes, clusterCnt, dimension, metricType,
                preallocSums, preallocCounts);

            // Redistribute points to fill any clusters that became empty
            KMeansUtils.handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors, taken,
                clusterCnt, dimension, metricType, random);

            // Convergence check: maximum centroid shift across all centroids
            float maxShift = computeMaxCentroidShift(centroids, newCentroids, convergenceDistFn);

            // Swap centroid buffers to avoid allocation
            float[][] tmp = centroids;
            centroids = newCentroids;
            newCentroids = tmp;

            performedIterations = iteration + 1;

            if (maxShift <= tolerance)
                break;
        }

        // Step 3: Final full assignment pass for definitive labels, loss, and cluster sizes
        Arrays.fill(clusterSizes, 0);
        float finalLoss = KMeansUtils.assignPointsToClusters(data, centroids, clusterCnt, labels, null, clusterSizes, distFn);

        return new Result(labels, centroids, performedIterations, finalLoss, clusterSizes);
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

    /**
     * Minimum number of clusters to parallelize the centroid shift computation.
     * Below this threshold, the sequential loop is faster than fork-join overhead.
     */
    private static final int PARALLEL_SHIFT_THRESHOLD = 64;

    /**
     * Computes the maximum distance between corresponding old and new centroids.
     *
     * <p>This value measures how much the centroids moved in the current iteration
     * and is used as the convergence criterion. When the maximum shift drops below
     * the tolerance, the algorithm has converged.</p>
     *
     * <p>For L2/dot-product metrics, this uses L2 squared distance. For cosine distance,
     * this uses cosine distance between old and new centroid, since centroids are on
     * the unit sphere and angular change is a more meaningful convergence measure.</p>
     *
     * @param oldCentroids the centroid positions before the update step
     * @param newCentroids the centroid positions after the update step
     * @param distFn       the distance function for measuring centroid shift
     * @return the maximum distance across all centroid pairs
     */
    private float computeMaxCentroidShift(float[][] oldCentroids,
        float[][] newCentroids,
        Metric.DistanceFunction distFn) {

        if (clusterCnt >= PARALLEL_SHIFT_THRESHOLD) {
            // Parallel max reduction over all centroid shifts
            return (float) IntStream.range(0, clusterCnt).parallel()
                .mapToDouble(c -> distFn.compute(oldCentroids[c], newCentroids[c]))
                .max()
                .orElse(0);
        }

        // Sequential path
        float maxShift = 0;

        for (int c = 0; c < clusterCnt; c++) {
            float shift = distFn.compute(oldCentroids[c], newCentroids[c]);
            if (shift > maxShift)
                maxShift = shift;
        }

        return maxShift;
    }

    /**
     * Immutable result of Lloyd's KMeans clustering.
     *
     * <p>Contains the final cluster assignments, centroid positions, the number of
     * iterations performed, the total loss, and per-cluster membership counts.</p>
     */
    static final class Result implements ClusteringResult {
        /** Per-point cluster assignments (0-based cluster indices). */
        private final int[] labels;
        /** Final centroid positions: {@code float[k][dimension]}. */
        private final float[][] centroids;
        /** Number of assign-update iterations actually performed. */
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
         * @param iterations   number of iterations performed
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
         * Returns the number of assign-update iterations performed before convergence
         * or reaching the iteration limit.
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
