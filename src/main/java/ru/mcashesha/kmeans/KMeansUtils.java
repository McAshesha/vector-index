package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

/**
 * Shared utility methods for all KMeans algorithm implementations.
 *
 * <p>Provides core building blocks used by {@link LloydKMeans}, {@link MiniBatchKMeans},
 * and {@link HierarchicalKMeans}, including:</p>
 * <ul>
 *   <li>KMeans++ centroid initialization with D-squared weighted sampling</li>
 *   <li>Nearest-centroid assignment (with optional triangle inequality pruning)</li>
 *   <li>Mean-based centroid recomputation</li>
 *   <li>Empty cluster redistribution</li>
 *   <li>L2 normalization for cosine distance</li>
 *   <li>Pairwise distance matrix computation</li>
 * </ul>
 *
 * <p>Most methods automatically switch between sequential and parallel execution
 * based on dataset size relative to {@link #PARALLEL_THRESHOLD}.</p>
 *
 * <p>This is a stateless utility class and cannot be instantiated.</p>
 */
final class KMeansUtils {
    private KMeansUtils() {}

    /**
     * Minimum number of data points required to trigger parallel execution paths.
     * Below this threshold, sequential loops are used to avoid thread-pool overhead.
     */
    private static final int PARALLEL_THRESHOLD = 1000;

    /**
     * Validates that all data points are non-null and share the same positive dimensionality.
     *
     * @param data the input dataset; each element is a feature vector
     * @return the common dimensionality of all data points
     * @throws IllegalArgumentException if any point is null, has zero dimension,
     *                                  or has a dimension inconsistent with the first point
     */
    static int validateAndGetDimension(float[][] data) {
        if (data[0] == null)
            throw new IllegalArgumentException("points must be non-null");

        int dimension = data[0].length;
        if (dimension == 0)
            throw new IllegalArgumentException("points must have positive dimension");

        for (int i = 1; i < data.length; i++) {
            float[] point = data[i];
            if (point == null || point.length != dimension)
                throw new IllegalArgumentException("all points must be non-null and have the same dimension");
        }

        return dimension;
    }

    /**
     * Initializes cluster centroids using the KMeans++ algorithm (Arthur and Vassilvitskii, 2007).
     *
     * <p>KMeans++ selects initial centroids with probability proportional to the distance
     * from the nearest already-chosen centroid. For L2 squared distance this is equivalent
     * to the original D-squared weighting from the paper (since the raw L2 squared value
     * equals the square of the Euclidean distance), providing an O(log k)-competitive
     * approximation guarantee. For other metrics (dot product, cosine), the raw distance
     * value is used as the sampling weight, which is a common generalization.</p>
     *
     * <p>The algorithm proceeds as follows:</p>
     * <ol>
     *   <li>Choose the first centroid uniformly at random from the data.</li>
     *   <li>For each subsequent centroid, compute each point's minimum distance to the
     *       nearest already-chosen centroid.</li>
     *   <li>Sample the next centroid with probability proportional to that distance.</li>
     *   <li>Repeat until all {@code clusterCnt} centroids are chosen.</li>
     * </ol>
     *
     * <p>Distance computations and weight summation are parallelized when
     * {@code sampleCnt >= PARALLEL_THRESHOLD}.</p>
     *
     * @param data       the input dataset
     * @param sampleCnt  the number of data points ({@code data.length})
     * @param dimension  the dimensionality of each data point
     * @param clusterCnt the number of centroids to initialize
     * @param distFn     the distance function to use
     * @param metricType the metric type; centroids are L2-normalized if cosine distance
     * @param random     the RNG for centroid selection
     * @return a {@code float[clusterCnt][dimension]} array of initialized centroids
     */
    static float[][] initializeCentroidsKMeansPlusPlus(
        float[][] data,
        int sampleCnt,
        int dimension,
        int clusterCnt,
        Metric.DistanceFunction distFn,
        Metric.Type metricType,
        Random random) {

        float[][] centroids = new float[clusterCnt][dimension];

        // Step 1: Pick the first centroid uniformly at random
        int firstIdx = random.nextInt(sampleCnt);
        System.arraycopy(data[firstIdx], 0, centroids[0], 0, dimension);

        // minDistances[i] holds the minimum distance from data[i] to any already-chosen centroid
        float[] minDistances = new float[sampleCnt];

        // Compute initial distances from all points to the first centroid
        float[] firstCentroid = centroids[0];
        if (sampleCnt >= PARALLEL_THRESHOLD) {
            IntStream.range(0, sampleCnt).parallel().forEach(i ->
                minDistances[i] = distFn.compute(data[i], firstCentroid));
        } else {
            for (int i = 0; i < sampleCnt; i++)
                minDistances[i] = distFn.compute(data[i], firstCentroid);
        }

        // Step 2-k: Choose remaining centroids with D-squared weighting
        for (int c = 1; c < clusterCnt; c++) {
            // Compute the total weight (sum of minimum distances) for probability normalization.
            // Parallel reduction is used for large datasets.
            float totalWeight;
            if (sampleCnt >= PARALLEL_THRESHOLD) {
                totalWeight = (float) IntStream.range(0, sampleCnt).parallel()
                    .mapToDouble(i -> Math.max(0f, minDistances[i]))
                    .sum();
            } else {
                totalWeight = 0f;
                for (int i = 0; i < sampleCnt; i++)
                    totalWeight += Math.max(0f, minDistances[i]);
            }

            int chosenIdx;

            if (totalWeight == 0f)
                // All points coincide with existing centroids; fall back to uniform random
                chosenIdx = random.nextInt(sampleCnt);
            else {
                // D-squared sampling: walk the cumulative distribution until
                // the random threshold is exceeded. This step is inherently
                // sequential because it depends on running cumulative sums.
                float threshold = random.nextFloat() * totalWeight;
                float cumulative = 0f;
                chosenIdx = sampleCnt - 1;

                for (int i = 0; i < sampleCnt; i++) {
                    cumulative += Math.max(0f, minDistances[i]);
                    if (cumulative >= threshold) {
                        chosenIdx = i;
                        break;
                    }
                }
            }

            System.arraycopy(data[chosenIdx], 0, centroids[c], 0, dimension);

            // Update minDistances: for each point, keep the minimum distance across
            // all centroids chosen so far. Only the new centroid can reduce the distance.
            float[] newCentroid = centroids[c];
            if (sampleCnt >= PARALLEL_THRESHOLD) {
                IntStream.range(0, sampleCnt).parallel().forEach(i -> {
                    float distance = distFn.compute(data[i], newCentroid);
                    if (distance < minDistances[i])
                        minDistances[i] = distance;
                });
            } else {
                for (int i = 0; i < sampleCnt; i++) {
                    float distance = distFn.compute(data[i], newCentroid);
                    if (distance < minDistances[i])
                        minDistances[i] = distance;
                }
            }
        }

        // For cosine distance, centroids must lie on the unit sphere
        if (metricType == Metric.Type.COSINE_DISTANCE)
            normalizeCentroids(centroids);

        return centroids;
    }

    /**
     * Assigns each data point to the nearest centroid using brute-force linear scan.
     *
     * <p>Dispatches to the parallel implementation when the dataset size
     * meets or exceeds {@link #PARALLEL_THRESHOLD}.</p>
     *
     * @param data         the input dataset
     * @param centroids    the current centroid positions
     * @param clusterCnt   the number of clusters
     * @param labels       output array to receive per-point cluster assignments
     * @param pointErrors  optional output array for per-point distances to assigned centroid; may be null
     * @param clusterSizes optional output array for per-cluster membership counts; may be null
     * @param distFn       the distance function to use
     * @return the total loss (sum of all point-to-centroid distances)
     */
    static float assignPointsToClusters(
        float[][] data,
        float[][] centroids,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        if (clusterSizes != null)
            Arrays.fill(clusterSizes, 0);

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            return assignPointsToClustersParallel(data, centroids, clusterCnt, labels, pointErrors, clusterSizes, distFn);
        }

        // Sequential path for small datasets
        // Use double accumulation to prevent precision loss over many summands
        // (float has ~7 significant digits, which degrades after >100K additions).
        double loss = 0.0;

        for (int i = 0; i < sampleCnt; i++) {
            float[] point = data[i];

            // Linear scan over all centroids to find the nearest one
            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;

            if (clusterSizes != null)
                clusterSizes[nearestClusterIdx]++;

            loss += nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        }

        return (float) loss;
    }

    /**
     * Parallel implementation of nearest-centroid assignment.
     *
     * <p>Each point's assignment is computed independently in parallel. Cluster sizes
     * are aggregated sequentially after parallel assignment to avoid atomic contention.
     * Loss is computed via a parallel reduction over the per-point distances.</p>
     *
     * @param data         the input dataset
     * @param centroids    the current centroid positions
     * @param clusterCnt   the number of clusters
     * @param labels       output array to receive per-point cluster assignments
     * @param pointErrors  optional output for per-point distances; may be null
     * @param clusterSizes optional output for per-cluster counts; may be null
     * @param distFn       the distance function
     * @return the total loss
     */
    private static float assignPointsToClustersParallel(
        float[][] data,
        float[][] centroids,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        // Each point is assigned independently; store per-point distances for later reduction
        float[] distances = new float[sampleCnt];

        IntStream.range(0, sampleCnt).parallel().forEach(i -> {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;
            distances[i] = nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        });

        // Compute cluster sizes sequentially to avoid atomic contention on the counts array
        if (clusterSizes != null) {
            for (int i = 0; i < sampleCnt; i++)
                clusterSizes[labels[i]]++;
        }

        // Parallel reduction for total loss
        return (float) IntStream.range(0, sampleCnt).parallel()
            .mapToDouble(i -> distances[i])
            .sum();
    }

    /**
     * Recomputes centroids as the mean of all points assigned to each cluster.
     *
     * <p>For datasets exceeding {@link #PARALLEL_THRESHOLD}, accumulation is parallelized
     * using thread-local buffers to avoid synchronization. The final averaging and optional
     * L2 normalization (for cosine distance) are applied after accumulation.</p>
     *
     * <p>Delegates to the overload accepting pre-allocated buffers, passing {@code null}
     * to trigger fresh allocation.</p>
     *
     * @param data         the input dataset
     * @param labels       per-point cluster assignments from the most recent assignment step
     * @param newCentroids output array to receive the recomputed centroid positions;
     *                     must be pre-allocated as {@code float[clusterCnt][dimension]}
     * @param clusterSizes output array to receive per-cluster membership counts
     * @param clusterCnt   the number of clusters
     * @param dimension    the dimensionality of data points
     * @param metricType   the metric type; centroids are L2-normalized if cosine distance
     */
    static void recomputeCentroids(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension,
        Metric.Type metricType) {

        recomputeCentroids(data, labels, newCentroids, clusterSizes, clusterCnt, dimension, metricType, null, null);
    }

    /**
     * Recomputes centroids as the mean of all points assigned to each cluster, optionally
     * reusing pre-allocated thread-local buffers for the parallel accumulation path.
     *
     * <p>Pre-allocated buffers avoid per-iteration allocation of O(threads * k * d) memory.
     * When {@code preallocLocalSums} and {@code preallocLocalCounts} are non-null, they are
     * zeroed and reused instead of allocating new arrays. This eliminates tens of megabytes
     * of garbage per iteration for large cluster counts and high dimensionality.</p>
     *
     * <p>For datasets exceeding {@link #PARALLEL_THRESHOLD}, accumulation is parallelized
     * using thread-local buffers to avoid synchronization. The final averaging and optional
     * L2 normalization (for cosine distance) are applied after accumulation.</p>
     *
     * @param data                the input dataset
     * @param labels              per-point cluster assignments from the most recent assignment step
     * @param newCentroids        output array to receive the recomputed centroid positions;
     *                            must be pre-allocated as {@code float[clusterCnt][dimension]}
     * @param clusterSizes        output array to receive per-cluster membership counts
     * @param clusterCnt          the number of clusters
     * @param dimension           the dimensionality of data points
     * @param metricType          the metric type; centroids are L2-normalized if cosine distance
     * @param preallocLocalSums   pre-allocated buffer {@code float[numThreads][clusterCnt][dimension]}
     *                            for parallel accumulation, or {@code null} to allocate fresh arrays
     * @param preallocLocalCounts pre-allocated buffer {@code int[numThreads][clusterCnt]}
     *                            for parallel accumulation, or {@code null} to allocate fresh arrays
     */
    static void recomputeCentroids(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension,
        Metric.Type metricType,
        float[][][] preallocLocalSums,
        int[][] preallocLocalCounts) {

        int sampleCnt = data.length;

        // Zero out accumulators before summing
        for (int c = 0; c < clusterCnt; c++) {
            Arrays.fill(newCentroids[c], 0);
            clusterSizes[c] = 0;
        }

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            recomputeCentroidsParallel(data, labels, newCentroids, clusterSizes, clusterCnt, dimension,
                preallocLocalSums, preallocLocalCounts);
        } else {
            // Sequential accumulation: sum all points belonging to each cluster
            for (int i = 0; i < sampleCnt; i++) {
                int clusterIdx = labels[i];
                clusterSizes[clusterIdx]++;

                float[] centroidSum = newCentroids[clusterIdx];
                float[] point = data[i];

                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += point[d];
            }
        }

        // Divide accumulated sums by cluster size to obtain the mean (centroid)
        for (int c = 0; c < clusterCnt; c++) {
            int size = clusterSizes[c];
            if (size > 0) {
                float invSize = 1.0f / size;
                float[] centroid = newCentroids[c];
                for (int d = 0; d < dimension; d++)
                    centroid[d] *= invSize;
            }
        }

        // For cosine distance, project centroids onto the unit sphere
        if (metricType == Metric.Type.COSINE_DISTANCE)
            normalizeCentroids(newCentroids);
    }

    /**
     * Parallel centroid recomputation using thread-local accumulators.
     *
     * <p>The data is partitioned into chunks (one per available processor). Each thread
     * accumulates sums and counts into its own local buffer, eliminating the need for
     * locks or atomics. Results are merged across threads per-cluster in parallel.</p>
     *
     * <p>When pre-allocated buffers are provided (non-null), they are zeroed and reused,
     * avoiding O(threads * k * d) allocation on every call. This is critical for iterative
     * algorithms like Lloyd's KMeans where this method is called once per iteration.</p>
     *
     * @param data                the input dataset
     * @param labels              per-point cluster assignments
     * @param newCentroids        output array for centroid sums (to be averaged by the caller)
     * @param clusterSizes        output array for per-cluster counts
     * @param clusterCnt          the number of clusters
     * @param dimension           the dimensionality of data points
     * @param preallocLocalSums   pre-allocated {@code float[numThreads][clusterCnt][dimension]},
     *                            or {@code null} to allocate fresh arrays
     * @param preallocLocalCounts pre-allocated {@code int[numThreads][clusterCnt]},
     *                            or {@code null} to allocate fresh arrays
     */
    private static void recomputeCentroidsParallel(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension,
        float[][][] preallocLocalSums,
        int[][] preallocLocalCounts) {

        int sampleCnt = data.length;
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunkSize = (sampleCnt + numThreads - 1) / numThreads;

        // Use pre-allocated buffers if provided; otherwise allocate fresh arrays.
        // Pre-allocated buffers avoid per-iteration allocation of O(threads * k * d) memory.
        float[][][] localSums;
        int[][] localCounts;

        if (preallocLocalSums != null && preallocLocalCounts != null) {
            localSums = preallocLocalSums;
            localCounts = preallocLocalCounts;
            // Zero out the pre-allocated buffers before accumulation
            for (int t = 0; t < numThreads; t++) {
                for (int c = 0; c < clusterCnt; c++)
                    Arrays.fill(localSums[t][c], 0);
                Arrays.fill(localCounts[t], 0);
            }
        } else {
            localSums = new float[numThreads][clusterCnt][dimension];
            localCounts = new int[numThreads][clusterCnt];
        }

        // Each thread accumulates its data chunk into its own local buffers
        IntStream.range(0, numThreads).parallel().forEach(threadIdx -> {
            int start = threadIdx * chunkSize;
            int end = Math.min(start + chunkSize, sampleCnt);

            float[][] mySums = localSums[threadIdx];
            int[] myCounts = localCounts[threadIdx];

            for (int i = start; i < end; i++) {
                int clusterIdx = labels[i];
                myCounts[clusterIdx]++;

                float[] centroidSum = mySums[clusterIdx];
                float[] point = data[i];

                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += point[d];
            }
        });

        // Merge thread-local results into the shared output arrays (parallelized per cluster)
        IntStream.range(0, clusterCnt).parallel().forEach(c -> {
            float[] centroidSum = newCentroids[c];
            int totalCount = 0;

            for (int t = 0; t < numThreads; t++) {
                totalCount += localCounts[t][c];
                float[] threadSum = localSums[t][c];
                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += threadSum[d];
            }

            clusterSizes[c] = totalCount;
        });
    }

    /**
     * Redistributes data points to fill empty clusters after centroid recomputation.
     *
     * <p>For each empty cluster, the algorithm attempts to "steal" a point from the
     * largest non-empty cluster by choosing the point with the highest assignment error
     * (distance to its current centroid). This prevents degenerate solutions where
     * clusters vanish during iteration.</p>
     *
     * <p>The donor cluster's centroid is analytically updated by removing the stolen
     * point's contribution, avoiding a full recomputation pass.</p>
     *
     * <p>Fallback strategy (in order):</p>
     * <ol>
     *   <li>Pick the highest-error point from the largest cluster.</li>
     *   <li>Pick the globally highest-error point from any cluster.</li>
     *   <li>Pick a random point (last resort).</li>
     * </ol>
     *
     * @param data         the input dataset
     * @param newCentroids the current centroid positions (modified in place for donor clusters)
     * @param clusterSizes the current per-cluster membership counts (modified in place)
     * @param labels       per-point cluster assignments (modified in place for reassigned points)
     * @param pointErrors  per-point distances to assigned centroids
     * @param taken        scratch boolean array to track already-reassigned points
     * @param clusterCnt   the number of clusters
     * @param dimension    the dimensionality of data points
     * @param metricType   the metric type; donor centroids are re-normalized if cosine distance
     * @param random       the RNG for fallback random point selection
     */
    static void handleEmptyClusters(
        float[][] data,
        float[][] newCentroids,
        int[] clusterSizes,
        int[] labels,
        float[] pointErrors,
        boolean[] taken,
        int clusterCnt,
        int dimension,
        Metric.Type metricType,
        Random random) {

        int sampleCnt = data.length;
        Arrays.fill(taken, false);

        for (int emptyCluster = 0; emptyCluster < clusterCnt; emptyCluster++) {
            if (clusterSizes[emptyCluster] != 0)
                continue;

            // Strategy 1: pick the highest-error point from the largest cluster
            int chosenIdx = choosePointFromLargestCluster(labels, clusterSizes, pointErrors, taken, clusterCnt);

            // Strategy 2: pick the globally highest-error point from any cluster
            if (chosenIdx == -1)
                chosenIdx = chooseGlobalWorstPoint(labels, pointErrors, taken);

            // Strategy 3: fall back to a random point
            if (chosenIdx == -1)
                chosenIdx = random.nextInt(sampleCnt);

            taken[chosenIdx] = true;

            int oldCluster = labels[chosenIdx];
            int oldSize = clusterSizes[oldCluster];

            // Reassign the chosen point to the empty cluster
            labels[chosenIdx] = emptyCluster;
            clusterSizes[emptyCluster] = 1;

            // Analytically update the donor cluster's centroid by removing the stolen point.
            // new_centroid = (old_centroid * old_size - stolen_point) / (old_size - 1)
            if (oldCluster >= 0 && oldSize > 0) {
                clusterSizes[oldCluster] = oldSize - 1;

                if (clusterSizes[oldCluster] > 0) {
                    float[] donorCentroid = newCentroids[oldCluster];
                    float[] point = data[chosenIdx];
                    float invNewSize = 1.0f / clusterSizes[oldCluster];
                    for (int d = 0; d < dimension; d++)
                        donorCentroid[d] = (donorCentroid[d] * oldSize - point[d]) * invNewSize;
                    if (metricType == Metric.Type.COSINE_DISTANCE)
                        normalizeSingleCentroid(donorCentroid);
                }
            }

            // Set the empty cluster's centroid to the stolen point
            System.arraycopy(data[chosenIdx], 0, newCentroids[emptyCluster], 0, dimension);
        }
    }

    /**
     * Normalizes all centroids to unit length (L2 norm = 1).
     *
     * <p>Required for cosine distance so that inner-product-based distance computations
     * remain valid. Parallelized when the number of centroids is 32 or more.</p>
     *
     * @param centroids the centroid array to normalize in place
     */
    static void normalizeCentroids(float[][] centroids) {
        if (centroids.length >= 32) {
            IntStream.range(0, centroids.length).parallel()
                .forEach(c -> normalizeSingleCentroid(centroids[c]));
        } else {
            for (float[] centroid : centroids)
                normalizeSingleCentroid(centroid);
        }
    }

    /**
     * Normalizes a single vector to unit length (L2 norm = 1) in place.
     *
     * <p>If the vector has zero norm (all zeros), it is left unchanged.</p>
     *
     * @param centroid the vector to normalize
     */
    static void normalizeSingleCentroid(float[] centroid) {
        float norm = 0f;
        for (float v : centroid)
            norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            float invNorm = 1.0f / norm;
            for (int d = 0; d < centroid.length; d++)
                centroid[d] *= invNorm;
        }
    }

    /**
     * Selects the point with the highest assignment error from the largest cluster.
     *
     * <p>This is the primary strategy for filling empty clusters: steal the worst-fit
     * point from the cluster that can most afford to lose a member.</p>
     *
     * @param labels       per-point cluster assignments
     * @param clusterSizes per-cluster membership counts
     * @param pointErrors  per-point distances to assigned centroids; may be null
     * @param taken        boolean mask of already-reassigned points
     * @param clusterCnt   the number of clusters
     * @return the index of the chosen point, or -1 if no suitable point exists
     */
    private static int choosePointFromLargestCluster(
        int[] labels,
        int[] clusterSizes,
        float[] pointErrors,
        boolean[] taken,
        int clusterCnt) {

        // Find the cluster with the most members
        int largestCluster = -1;
        int maxSize = 0;
        for (int c = 0; c < clusterCnt; c++) {
            if (clusterSizes[c] > maxSize) {
                maxSize = clusterSizes[c];
                largestCluster = c;
            }
        }
        // Cannot steal from a cluster with 0 or 1 members
        if (largestCluster < 0 || maxSize <= 1)
            return -1;

        int n = labels.length;
        final int targetCluster = largestCluster;

        if (n >= PARALLEL_THRESHOLD) {
            // Parallel argmax: find the point in the target cluster with the highest error
            return IntStream.range(0, n).parallel()
                .filter(i -> !taken[i] && labels[i] == targetCluster)
                .boxed()
                .max((a, b) -> {
                    float errA = pointErrors != null ? pointErrors[a] : 1.0f;
                    float errB = pointErrors != null ? pointErrors[b] : 1.0f;
                    return Float.compare(errA, errB);
                })
                .orElse(-1);
        }

        // Sequential argmax
        int bestIdx = -1;
        float bestError = -1f;
        for (int i = 0; i < n; i++) {
            if (taken[i])
                continue;
            if (labels[i] == largestCluster) {
                float err = pointErrors != null ? pointErrors[i] : 1.0f;
                if (err > bestError) {
                    bestError = err;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    /**
     * Selects the point with the highest assignment error across all clusters.
     *
     * <p>This is the fallback strategy when no suitable point could be found
     * in the largest cluster (e.g., all its points are already taken).</p>
     *
     * @param labels      per-point cluster assignments
     * @param pointErrors per-point distances to assigned centroids; may be null
     * @param taken       boolean mask of already-reassigned points
     * @return the index of the chosen point, or -1 if no eligible point exists
     */
    private static int chooseGlobalWorstPoint(
        int[] labels,
        float[] pointErrors,
        boolean[] taken) {

        int n = labels.length;

        if (n >= PARALLEL_THRESHOLD) {
            // Parallel argmax across all non-taken, assigned points
            return IntStream.range(0, n).parallel()
                .filter(i -> !taken[i] && labels[i] >= 0)
                .boxed()
                .max((a, b) -> {
                    float errA = pointErrors != null ? pointErrors[a] : 1.0f;
                    float errB = pointErrors != null ? pointErrors[b] : 1.0f;
                    return Float.compare(errA, errB);
                })
                .orElse(-1);
        }

        // Sequential argmax
        int bestIdx = -1;
        float bestError = -1f;

        for (int i = 0; i < n; i++) {
            if (taken[i])
                continue;
            if (labels[i] < 0)
                continue;
            float err = pointErrors != null ? pointErrors[i] : 1.0f;
            if (err > bestError) {
                bestError = err;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * Computes the full pairwise distance matrix between all points.
     *
     * <p>Only the upper triangle is computed; the lower triangle is mirrored
     * (distance is symmetric). The diagonal is zero by definition. Parallelized
     * when the number of points is 100 or more.</p>
     *
     * @param points the set of points (or centroids) to compare
     * @param distFn the distance function to use
     * @return a symmetric {@code float[n][n]} distance matrix where
     *         {@code result[i][j] = distance(points[i], points[j])}
     */
    static float[][] computeDistanceMatrix(float[][] points, Metric.DistanceFunction distFn) {
        int n = points.length;
        float[][] distances = new float[n][n];

        if (n >= 100) {
            // Parallel computation - only upper triangle, then mirror
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int j = i + 1; j < n; j++) {
                    float d = distFn.compute(points[i], points[j]);
                    distances[i][j] = d;
                    distances[j][i] = d;
                }
            });
        } else {
            // Sequential for small matrices
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    float d = distFn.compute(points[i], points[j]);
                    distances[i][j] = d;
                    distances[j][i] = d;
                }
            }
        }

        return distances;
    }

    /**
     * Precomputes pairwise distances between all centroids for use with triangle
     * inequality pruning in {@link #assignPointsToClustersWithPruning}.
     *
     * @param centroids the current centroid positions
     * @param distFn    the distance function to use
     * @return a symmetric distance matrix between all centroids
     */
    static float[][] precomputeCentroidDistances(float[][] centroids, Metric.DistanceFunction distFn) {
        return computeDistanceMatrix(centroids, distFn);
    }

    /**
     * Assigns each data point to the nearest centroid, using distance-based pruning
     * to skip clusters that are likely not closer.
     *
     * <p>The pruning condition {@code d²(a, b) > 4 * d²(x, a)} is derived from the
     * triangle inequality in Elkan's accelerated KMeans, adapted for L2 squared distance:
     * if d²(a,b) > 4·d²(x,a), then d(a,b) > 2·d(x,a), which guarantees centroid b
     * cannot be closer to x than centroid a. This pruning is only valid for L2 squared
     * distance where the triangle inequality holds; it is not used for dot product or
     * cosine distance.</p>
     *
     * <p>This optimization is most effective when the number of clusters is large
     * (k >= 64), as it can eliminate a significant fraction of distance computations.</p>
     *
     * @param data              the input dataset
     * @param centroids         the current centroid positions
     * @param centroidDistances precomputed centroid-to-centroid distance matrix
     * @param clusterCnt        the number of clusters
     * @param labels            output array for per-point cluster assignments
     * @param pointErrors       optional output for per-point distances; may be null
     * @param clusterSizes      optional output for per-cluster counts; may be null
     * @param distFn            the distance function
     * @return the total loss
     */
    static float assignPointsToClustersWithPruning(
        float[][] data,
        float[][] centroids,
        float[][] centroidDistances,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        if (clusterSizes != null)
            Arrays.fill(clusterSizes, 0);

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            return assignPointsToClustersWithPruningParallel(
                data, centroids, centroidDistances, clusterCnt, labels, pointErrors, clusterSizes, distFn);
        }

        // Sequential path with triangle inequality pruning
        // Use double accumulation to prevent precision loss over many summands
        // (float has ~7 significant digits, which degrades after >100K additions).
        double loss = 0.0;

        for (int i = 0; i < sampleCnt; i++) {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                // For L2 squared: d²(a,b) > 4·d²(x,a) ⟹ d(a,b) > 2·d(x,a),
                // so centroid b cannot be closer to x than centroid a by the triangle inequality.
                if (centroidDistances[nearestClusterIdx][c] > 4 * nearestDistance)
                    continue;

                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;

            if (clusterSizes != null)
                clusterSizes[nearestClusterIdx]++;

            loss += nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        }

        return (float) loss;
    }

    /**
     * Parallel implementation of nearest-centroid assignment with triangle inequality pruning.
     *
     * <p>Follows the same pattern as {@link #assignPointsToClustersParallel}: each point is
     * assigned independently in parallel, with sequential cluster-size aggregation and
     * parallel loss reduction.</p>
     *
     * @param data              the input dataset
     * @param centroids         the current centroid positions
     * @param centroidDistances precomputed centroid-to-centroid distance matrix
     * @param clusterCnt        the number of clusters
     * @param labels            output array for per-point cluster assignments
     * @param pointErrors       optional output for per-point distances; may be null
     * @param clusterSizes      optional output for per-cluster counts; may be null
     * @param distFn            the distance function
     * @return the total loss
     */
    private static float assignPointsToClustersWithPruningParallel(
        float[][] data,
        float[][] centroids,
        float[][] centroidDistances,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;
        float[] distances = new float[sampleCnt];

        IntStream.range(0, sampleCnt).parallel().forEach(i -> {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                // For L2 squared: d²(a,b) > 4·d²(x,a) ⟹ d(a,b) > 2·d(x,a),
                // so centroid b cannot be closer to x than centroid a by the triangle inequality.
                if (centroidDistances[nearestClusterIdx][c] > 4 * nearestDistance)
                    continue;

                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;
            distances[i] = nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        });

        // Compute cluster sizes sequentially to avoid atomic contention
        if (clusterSizes != null) {
            for (int i = 0; i < sampleCnt; i++)
                clusterSizes[labels[i]]++;
        }

        // Parallel reduction for total loss
        return (float) IntStream.range(0, sampleCnt).parallel()
            .mapToDouble(i -> distances[i])
            .sum();
    }
}
