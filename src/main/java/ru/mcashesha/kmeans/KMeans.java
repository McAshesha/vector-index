package ru.mcashesha.kmeans;

import java.util.Random;
import ru.mcashesha.metrics.Metric;

/**
 * Top-level interface for all KMeans clustering algorithm variants.
 *
 * <p>Provides a unified API for training ({@link #fit(float[][])}) and
 * prediction ({@link #predict(float[][], ClusteringResult)}) across different
 * KMeans implementations: Lloyd's full-pass algorithm, stochastic Mini-Batch,
 * and recursive Hierarchical clustering.</p>
 *
 * <p>Instances are created exclusively through the fluent {@link Builder},
 * obtained via {@link #newBuilder(Type, Metric.Type, Metric.Engine)}.</p>
 *
 * @param <R> the concrete {@link ClusteringResult} type returned by each implementation
 */
public interface KMeans<R extends KMeans.ClusteringResult> {

    /**
     * Creates a new {@link Builder} for constructing a KMeans instance.
     *
     * @param type         the KMeans algorithm variant to use
     * @param metricType   the distance metric (L2, dot product, cosine)
     * @param metricEngine the computation engine (Scalar, VectorAPI, SimSIMD)
     * @return a new builder with sensible defaults
     */
    static Builder newBuilder(Type type,
        Metric.Type metricType,
        Metric.Engine metricEngine) {
        return new Builder(type, metricType, metricEngine);
    }

    /**
     * Trains the KMeans model on the provided dataset.
     *
     * @param data a non-empty array of feature vectors, all with identical dimensionality
     * @return a {@link ClusteringResult} containing centroids, assignments, loss, and cluster sizes
     * @throws IllegalArgumentException if data is null, empty, or dimensionally inconsistent
     */
    R fit(float[][] data);

    /**
     * Assigns new data points to the nearest clusters from a previously fitted model.
     *
     * @param data  a non-empty array of feature vectors to classify
     * @param model the result of a prior {@link #fit(float[][])} call
     * @return an array of cluster indices, one per input data point
     * @throws IllegalArgumentException if data or model is invalid, or dimensions do not match
     */
    int[] predict(float[][] data, R model);

    /**
     * Returns the distance metric type used by this KMeans instance.
     *
     * @return the configured {@link Metric.Type}
     */
    Metric.Type getMetricType();

    /**
     * Returns the distance computation engine used by this KMeans instance.
     *
     * @return the configured {@link Metric.Engine}
     */
    Metric.Engine getMetricEngine();

    /**
     * Enumeration of available KMeans algorithm variants.
     */
    enum Type {
        /** Recursive tree-based hierarchical clustering using Lloyd at each level. */
        HIERARCHICAL,
        /** Classic full-pass iterative KMeans (Lloyd's algorithm). */
        LLOYD,
        /** Stochastic mini-batch KMeans for faster convergence on large datasets. */
        MINI_BATCH
    }

    /**
     * Common result interface returned by all KMeans implementations.
     *
     * <p>Provides access to the final centroids, per-point cluster assignments,
     * the total loss (sum of distances from each point to its nearest centroid),
     * and per-cluster membership counts.</p>
     */
    interface ClusteringResult {
        /**
         * Returns the computed cluster centroids.
         *
         * @return a {@code float[k][dimension]} array of centroid vectors
         */
        float[][] getCentroids();

        /**
         * Returns per-point cluster assignments.
         *
         * @return an array of length {@code n} where each element is the cluster index
         *         (0-based) that the corresponding data point belongs to
         */
        int[] getClusterAssignments();

        /**
         * Returns the total clustering loss.
         *
         * <p>The loss is the sum of distances from each data point to its assigned
         * centroid, computed using the configured distance metric. Lower is better.</p>
         *
         * @return the total loss value
         */
        float getLoss();

        /**
         * Returns the number of points assigned to each cluster.
         *
         * @return an array of length {@code k} with per-cluster membership counts
         */
        int[] getClusterSizes();
    }

    /**
     * Fluent builder for constructing KMeans instances with configurable parameters.
     *
     * <p>All parameters have sensible defaults. The builder validates inputs and
     * dispatches to the appropriate implementation based on the specified {@link Type}.</p>
     *
     * <p><b>Warning:</b> Using {@link Metric.Type#DOT_PRODUCT} with KMeans is only
     * appropriate for normalized (unit-length) input vectors. For unnormalized data,
     * the optimal centroid under negated dot product tends toward infinity, causing
     * the algorithm to produce meaningless results. Use {@link Metric.Type#COSINE_DISTANCE}
     * instead for unnormalized data -- it internally normalizes centroids and measures
     * angular similarity.</p>
     *
     * <h3>Defaults:</h3>
     * <ul>
     *   <li>{@code clusterCount} = 16</li>
     *   <li>{@code batchSize} = 1024 (Mini-Batch only)</li>
     *   <li>{@code maxIterations} = 300</li>
     *   <li>{@code maxNoImprovementIterations} = 50 (Mini-Batch only)</li>
     *   <li>{@code tolerance} = 1e-4</li>
     *   <li>{@code branchFactor} = 2 (Hierarchical only)</li>
     *   <li>{@code maxDepth} = 6 (Hierarchical only)</li>
     *   <li>{@code minClusterSize} = max(2 * branchFactor, 2) (Hierarchical only)</li>
     *   <li>{@code maxIterationsPerLevel} = 50 (Hierarchical only)</li>
     *   <li>{@code beamWidth} = 1 (Hierarchical only; 1 = greedy, &gt;1 = beam search)</li>
     * </ul>
     */
    final class Builder {
        /** The KMeans algorithm variant to construct. */
        private final Type type;
        /** The distance metric used for clustering (e.g., L2, dot product, cosine). */
        private final Metric.Type metricType;
        /** The computation engine for distance calculations (e.g., Scalar, VectorAPI, SimSIMD). */
        private final Metric.Engine metricEngine;

        // --- Shared parameters (Lloyd / Mini-Batch) ---

        /** Number of clusters to produce. Default: 16. */
        private int clusterCount = 16;
        /** Mini-batch size for stochastic updates (Mini-Batch only). Default: 1024. */
        private int batchSize = 1024;
        /** Maximum number of iterations before stopping. Default: 300. */
        private int maxIterations = 300;
        /** Early-stop after this many consecutive no-improvement iterations (Mini-Batch only). Default: 50. */
        private int maxNoImprovementIterations = 50;
        /** Convergence tolerance; interpretation varies by algorithm. Default: 1e-4. */
        private float tolerance = 1e-4f;

        // --- Hierarchical-specific parameters ---

        /** Number of children at each internal tree node. Default: 2. */
        private int branchFactor = 2;
        /** Maximum tree depth for recursive splitting. Default: 6. */
        private int maxDepth = 6;
        /** Minimum points in a node before it becomes a leaf. Default: 4 (auto-adjusted by branchFactor). */
        private int minClusterSize = 4;
        /** Maximum Lloyd iterations at each recursive level. Default: 50. */
        private int maxIterationsPerLevel = 50;
        /** Tracks whether the user explicitly set minClusterSize, preventing auto-adjustment. */
        private boolean minClusterSizeOverridden;
        /** Beam width for hierarchical KMeans prediction (1 = greedy, >1 = beam search). Default: 1. */
        private int beamWidth = 1;

        /** Random number generator for initialization and sampling. */
        private Random random = new Random();

        /**
         * Constructs a new builder with the given algorithm type and metric configuration.
         *
         * @param type         the KMeans variant; must not be null
         * @param metricType   the distance metric; must not be null
         * @param metricEngine the computation engine; must not be null
         * @throws IllegalArgumentException if any argument is null
         */
        private Builder(Type type,
            Metric.Type metricType,
            Metric.Engine metricEngine) {
            if (type == null)
                throw new IllegalArgumentException("type must be non-null");
            if (metricType == null || metricEngine == null)
                throw new IllegalArgumentException("metricType and metricEngine must be non-null");

            this.type = type;
            this.metricType = metricType;
            this.metricEngine = metricEngine;
        }

        /**
         * Sets the number of clusters (k) to produce.
         *
         * @param clusterCount the desired number of clusters
         * @return this builder for method chaining
         */
        public Builder withClusterCount(int clusterCount) {
            this.clusterCount = clusterCount;
            return this;
        }

        /**
         * Sets the mini-batch size for {@link Type#MINI_BATCH} KMeans.
         *
         * <p>Larger batches produce more stable updates but reduce the speed advantage.
         * Ignored by other algorithm types.</p>
         *
         * @param batchSize the number of samples per mini-batch
         * @return this builder for method chaining
         */
        public Builder withBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * Sets the maximum number of iterations before the algorithm terminates.
         *
         * @param maxIterations the iteration cap
         * @return this builder for method chaining
         */
        public Builder withMaxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        /**
         * Sets the convergence tolerance.
         *
         * <p>For Lloyd's algorithm, this is the maximum centroid shift (L2 squared distance)
         * threshold. For Mini-Batch, this is the minimum change in average batch loss.</p>
         *
         * @param tolerance the convergence threshold; must be non-negative
         * @return this builder for method chaining
         */
        public Builder withTolerance(float tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        /**
         * Sets the early-stopping patience for {@link Type#MINI_BATCH} KMeans.
         *
         * <p>The algorithm stops if the average batch loss does not improve
         * (change exceeds tolerance) for this many consecutive iterations.</p>
         *
         * @param maxNoImprovementIterations the patience window; must be positive
         * @return this builder for method chaining
         */
        public Builder withMaxNoImprovementIterations(int maxNoImprovementIterations) {
            this.maxNoImprovementIterations = maxNoImprovementIterations;
            return this;
        }

        /**
         * Sets the branching factor for {@link Type#HIERARCHICAL} KMeans.
         *
         * <p>Each internal tree node will be split into up to {@code branchFactor} children
         * using Lloyd's algorithm. If {@link #withMinClusterSize(int)} has not been explicitly
         * called, the minimum cluster size is automatically adjusted to
         * {@code max(2 * branchFactor, 2)} to ensure meaningful splits.</p>
         *
         * @param branchFactor the number of children per internal node; must be >= 2
         * @return this builder for method chaining
         */
        public Builder withBranchFactor(int branchFactor) {
            this.branchFactor = branchFactor;
            // Auto-adjust minClusterSize unless the user explicitly overrode it
            if (!minClusterSizeOverridden)
                this.minClusterSize = Math.max(2 * branchFactor, 2);
            return this;
        }

        /**
         * Sets the maximum tree depth for {@link Type#HIERARCHICAL} KMeans.
         *
         * @param maxDepth the maximum recursion depth; must be positive
         * @return this builder for method chaining
         */
        public Builder withMaxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        /**
         * Sets the minimum number of points required for a node to be split further
         * in {@link Type#HIERARCHICAL} KMeans.
         *
         * <p>Calling this method disables the automatic adjustment that normally occurs
         * when {@link #withBranchFactor(int)} is called.</p>
         *
         * @param minClusterSize the minimum leaf size; must be positive
         * @return this builder for method chaining
         */
        public Builder withMinClusterSize(int minClusterSize) {
            this.minClusterSize = minClusterSize;
            this.minClusterSizeOverridden = true;
            return this;
        }

        /**
         * Sets the maximum number of Lloyd iterations at each recursive level
         * in {@link Type#HIERARCHICAL} KMeans.
         *
         * @param maxIterationsPerLevel the per-level iteration cap; must be positive
         * @return this builder for method chaining
         */
        public Builder withMaxIterationsPerLevel(int maxIterationsPerLevel) {
            this.maxIterationsPerLevel = maxIterationsPerLevel;
            return this;
        }

        /**
         * Sets the beam width for prediction in {@link Type#HIERARCHICAL} KMeans.
         *
         * <p>During prediction, the tree is traversed from root to leaf. With the
         * default beam width of 1 (greedy), only the nearest child is followed at
         * each internal node. This is fast but susceptible to boundary effects where
         * the query point is near the decision boundary between children.</p>
         *
         * <p>With a beam width greater than 1, the top {@code beamWidth} candidate
         * nodes are retained at each level, improving recall at the cost of additional
         * distance computations. Ignored by non-hierarchical algorithm types.</p>
         *
         * @param beamWidth the number of candidates to retain at each tree level; must be positive
         * @return this builder for method chaining
         */
        public Builder withBeamWidth(int beamWidth) {
            this.beamWidth = beamWidth;
            return this;
        }

        /**
         * Sets the random number generator for centroid initialization and sampling.
         *
         * <p>Providing a seeded {@link Random} enables reproducible clustering results.</p>
         *
         * @param random the RNG instance; must not be null
         * @return this builder for method chaining
         * @throws IllegalArgumentException if random is null
         */
        public Builder withRandom(Random random) {
            if (random == null)
                throw new IllegalArgumentException("random must be non-null");
            this.random = random;
            return this;
        }

        /**
         * Constructs the KMeans instance according to the configured parameters.
         *
         * <p>Dispatches to the appropriate implementation based on the {@link Type}
         * specified at builder creation time.</p>
         *
         * @return a configured KMeans instance ready for {@link KMeans#fit(float[][])}
         * @throws IllegalStateException if the type is not recognized
         */
        public KMeans<? extends ClusteringResult> build() {
            switch (type) {
                case LLOYD: {
                    return new LloydKMeans(
                        clusterCount,
                        metricType,
                        metricEngine,
                        maxIterations,
                        tolerance,
                        random
                    );
                }
                case MINI_BATCH: {
                    return new MiniBatchKMeans(
                        clusterCount,
                        batchSize,
                        metricType,
                        metricEngine,
                        maxIterations,
                        tolerance,
                        maxNoImprovementIterations,
                        random
                    );
                }
                case HIERARCHICAL: {
                    return new HierarchicalKMeans(
                        branchFactor,
                        maxDepth,
                        minClusterSize,
                        maxIterationsPerLevel,
                        tolerance,
                        random,
                        metricType,
                        metricEngine,
                        beamWidth
                    );
                }
                default: {
                    throw new IllegalStateException("Unsupported KMeans type: " + type);
                }
            }
        }
    }
}
