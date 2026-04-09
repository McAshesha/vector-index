package ru.mcashesha.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

/**
 * Hierarchical (recursive tree-based) KMeans clustering algorithm.
 *
 * <p>Builds a balanced clustering tree by recursively splitting data using
 * {@link LloydKMeans} at each level. Each internal node partitions its subset
 * of data into {@code branchFactor} children, and recursion continues until
 * the maximum depth is reached or a node has fewer points than
 * {@code minClusterSize}.</p>
 *
 * <h3>Algorithm outline:</h3>
 * <ol>
 *   <li>Compute the centroid of the current data subset.</li>
 *   <li>If the stopping condition is met (max depth or too few points), create a leaf node.</li>
 *   <li>Otherwise, run Lloyd's KMeans with {@code branchFactor} clusters on the subset.</li>
 *   <li>Distribute points to child nodes based on cluster assignments.</li>
 *   <li>Recursively build each child node (parallelized at each level).</li>
 * </ol>
 *
 * <h3>Tree structure:</h3>
 * <ul>
 *   <li><b>Internal nodes:</b> have {@code children} array and no {@code pointIndices}.</li>
 *   <li><b>Leaf nodes:</b> have {@code pointIndices} and no children. Their centroids
 *       form the final flat centroid list.</li>
 * </ul>
 *
 * <h3>Prediction:</h3>
 * <p>For new data points, prediction traverses the tree greedily from root to leaf,
 * picking the nearest child at each internal node. The returned label is the leaf's
 * sequential ID.</p>
 *
 * <h3>Parallelization:</h3>
 * <p>Children at the same level are independent subtrees and are built in parallel
 * when there are at least 2 children and the tree is not near its leaf level.
 * Large-node operations (subset creation, child index distribution, centroid
 * computation) are also parallelized above configurable thresholds.</p>
 *
 * @see LloydKMeans
 * @see Node
 */
class HierarchicalKMeans implements KMeans<HierarchicalKMeans.Result> {

    /** Number of children per internal tree node. */
    private final int branchFactor;
    /** Maximum recursion depth for the clustering tree. */
    private final int maxDepth;
    /** Minimum number of points for a node to be split further. */
    private final int minClusterSize;
    /** Maximum Lloyd iterations at each recursive level. */
    private final int maxIterationsPerLevel;
    /** Convergence tolerance passed to Lloyd's algorithm at each level. */
    private final float tolerance;
    /** The distance metric to use for clustering (e.g., L2, dot product, cosine). */
    private final Metric.Type metricType;
    /** The computation engine for distance calculations (e.g., Scalar, VectorAPI, SimSIMD). */
    private final Metric.Engine metricEngine;
    /** Random number generator for KMeans++ initialization at each level. */
    private final Random random;
    /** Beam width for prediction tree traversal. 1 = greedy (default), >1 = beam search. */
    private final int beamWidth;

    /**
     * Constructs a HierarchicalKMeans instance with the specified configuration.
     *
     * @param branchFactor         the number of children per internal node; must be >= 2
     * @param maxDepth             the maximum tree depth; must be positive
     * @param minClusterSize       the minimum points for a non-leaf node; must be positive
     * @param maxIterationsPerLevel the max Lloyd iterations per level; must be positive
     * @param tolerance            the convergence tolerance; must be non-negative
     * @param random               the RNG; must not be null
     * @param metricType           the distance metric; must not be null
     * @param metricEngine         the computation engine; must not be null
     * @throws IllegalArgumentException if any argument violates its constraints
     */
    public HierarchicalKMeans(int branchFactor,
        int maxDepth,
        int minClusterSize,
        int maxIterationsPerLevel,
        float tolerance,
        Random random,
        Metric.Type metricType,
        Metric.Engine metricEngine) {
        this(branchFactor, maxDepth, minClusterSize, maxIterationsPerLevel,
            tolerance, random, metricType, metricEngine, 1);
    }

    /**
     * Constructs a HierarchicalKMeans instance with the specified configuration
     * including beam search width for prediction.
     *
     * @param branchFactor         the number of children per internal node; must be >= 2
     * @param maxDepth             the maximum tree depth; must be positive
     * @param minClusterSize       the minimum points for a non-leaf node; must be positive
     * @param maxIterationsPerLevel the max Lloyd iterations per level; must be positive
     * @param tolerance            the convergence tolerance; must be non-negative
     * @param random               the RNG; must not be null
     * @param metricType           the distance metric; must not be null
     * @param metricEngine         the computation engine; must not be null
     * @param beamWidth            the beam width for prediction; 1 = greedy, >1 = beam search
     * @throws IllegalArgumentException if any argument violates its constraints
     */
    public HierarchicalKMeans(int branchFactor,
        int maxDepth,
        int minClusterSize,
        int maxIterationsPerLevel,
        float tolerance,
        Random random,
        Metric.Type metricType,
        Metric.Engine metricEngine,
        int beamWidth) {

        if (branchFactor <= 1)
            throw new IllegalArgumentException("branchFactor must be >= 2");
        if (maxDepth <= 0)
            throw new IllegalArgumentException("maxDepth must be > 0");
        if (minClusterSize <= 0)
            throw new IllegalArgumentException("minClusterSize must be > 0");
        if (maxIterationsPerLevel <= 0)
            throw new IllegalArgumentException("maxIterationsPerLevel must be > 0");
        if (tolerance < 0.0f)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");
        if (beamWidth <= 0)
            throw new IllegalArgumentException("beamWidth must be > 0");

        this.branchFactor = branchFactor;
        this.maxDepth = maxDepth;
        this.minClusterSize = minClusterSize;
        this.maxIterationsPerLevel = maxIterationsPerLevel;
        this.tolerance = tolerance;
        this.random = random;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
        this.beamWidth = beamWidth;
    }

    @Override public Metric.Type getMetricType() {
        return metricType;
    }

    @Override public Metric.Engine getMetricEngine() {
        return metricEngine;
    }

    /**
     * Trains the hierarchical KMeans model by recursively building a clustering tree.
     *
     * <p>After the tree is built, leaf nodes are collected in depth-first order.
     * Each leaf is assigned a sequential ID, and every data point receives the
     * leaf ID of its containing leaf node. The loss is the sum of distances from
     * each point to its leaf's centroid.</p>
     *
     * @param data the input dataset; must be non-null, non-empty, with consistent dimensions
     * @return the clustering result containing the tree root, leaf assignments, leaf centroids,
     *         total loss, and cluster sizes
     * @throws IllegalArgumentException if data is invalid
     */
    @Override public Result fit(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");

        int sampleCnt = data.length;
        int dimension = KMeansUtils.validateAndGetDimension(data);

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);

        // Build the initial index array [0, 1, ..., sampleCnt-1]
        int[] allIndices = new int[sampleCnt];
        if (sampleCnt >= PARALLEL_CENTROID_THRESHOLD) {
            IntStream.range(0, sampleCnt).parallel().forEach(i -> allIndices[i] = i);
        } else {
            for (int i = 0; i < sampleCnt; i++)
                allIndices[i] = i;
        }

        // Recursively build the clustering tree starting at the root (level 0)
        Node root = buildNode(data, allIndices, 0, dimension, random);

        // Collect leaf centroids and assign leaf IDs via depth-first traversal
        List<float[]> leafCentroidsList = new ArrayList<>();
        int[] leafAssignments = new int[sampleCnt];
        // Use double accumulation to prevent precision loss over many summands
        DoubleWrapper lossAccumulator = new DoubleWrapper();

        assignLeafIdsAndCollect(root, data, leafCentroidsList, leafAssignments, lossAccumulator, distFn);

        float[][] leafCentroids = leafCentroidsList.toArray(new float[0][]);
        int[] clusterSizes = computeClusterSizes(leafAssignments, leafCentroids.length);

        return new Result(root, leafAssignments, leafCentroids, (float) lossAccumulator.val, clusterSizes);
    }

    /**
     * Minimum dataset size to trigger parallel prediction (tree traversal per point).
     */
    private static final int PARALLEL_PREDICT_THRESHOLD = 1000;

    /**
     * Assigns new data points to leaf clusters by traversing the clustering tree.
     *
     * <p>Each point is independently routed from the root to a leaf by greedily
     * choosing the nearest child at each internal node. Parallelized when the
     * number of data points exceeds {@link #PARALLEL_PREDICT_THRESHOLD}.</p>
     *
     * @param data  the data points to classify; must have the same dimensionality as the tree
     * @param model the result of a prior {@link #fit(float[][])} call
     * @return an array of leaf cluster IDs, one per input data point
     * @throws IllegalArgumentException if data or model is invalid, or dimensions do not match
     */
    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");
        if (model.getRoot() == null)
            throw new IllegalArgumentException("model root must be non-null");

        int dimension = KMeansUtils.validateAndGetDimension(data);
        if (model.getRoot().getCentroid().length != dimension)
            throw new IllegalArgumentException("data dimension must match tree centroid dimension");

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        int[] labels = new int[data.length];

        if (data.length >= PARALLEL_PREDICT_THRESHOLD) {
            // Parallel prediction: each point's tree traversal is independent
            Node root = model.getRoot();
            if (beamWidth > 1) {
                IntStream.range(0, data.length).parallel().forEach(i ->
                    labels[i] = predictSinglePointBeam(data[i], root, distFn, beamWidth));
            } else {
                IntStream.range(0, data.length).parallel().forEach(i ->
                    labels[i] = predictSinglePoint(data[i], root, distFn));
            }
        } else {
            // Sequential path for small datasets
            if (beamWidth > 1) {
                for (int i = 0; i < data.length; i++)
                    labels[i] = predictSinglePointBeam(data[i], model.getRoot(), distFn, beamWidth);
            } else {
                for (int i = 0; i < data.length; i++)
                    labels[i] = predictSinglePoint(data[i], model.getRoot(), distFn);
            }
        }

        return labels;
    }

    /**
     * Traverses the clustering tree to find the leaf node for a single query point.
     *
     * <p>At each internal node, the nearest child (by centroid distance) is selected.
     * Traversal continues iteratively (not recursively) until a leaf node is reached.</p>
     *
     * @param point  the query point
     * @param node   the root node to start traversal from
     * @param distFn the distance function
     * @return the leaf ID of the terminal node
     * @throws IllegalStateException if a leaf node has no assigned leaf ID
     */
    private int predictSinglePoint(float[] point, Node node, Metric.DistanceFunction distFn) {
        // Iterative tree descent: greedily pick the nearest child at each level
        while (!node.isLeaf()) {
            Node[] children = node.getChildren();
            if (children == null || children.length == 0)
                break;

            Node bestChild = children[0];
            float bestDistance = distFn.compute(point, children[0].centroid);

            for (int c = 1; c < children.length; c++) {
                float distance = distFn.compute(point, children[c].centroid);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestChild = children[c];
                }
            }

            node = bestChild;
        }

        if (node.getLeafId() < 0)
            throw new IllegalStateException("Leaf node has no leafId assigned");

        return node.getLeafId();
    }

    /**
     * Traverses the clustering tree using beam search to find the best leaf node
     * for a single query point.
     *
     * <p>Instead of greedily picking the single nearest child at each level (which
     * is susceptible to boundary effects), beam search maintains the top
     * {@code beamWidth} candidate nodes at each tree level. At each step, all
     * children of all current candidates are collected, sorted by distance to the
     * query point, and the top {@code beamWidth} are kept. This continues until
     * all candidates are leaves, at which point the nearest leaf is returned.</p>
     *
     * @param point     the query point
     * @param node      the root node to start traversal from
     * @param distFn    the distance function
     * @param beamWidth the number of candidates to retain at each level
     * @return the leaf ID of the best-matching terminal node
     * @throws IllegalStateException if a leaf node has no assigned leaf ID
     */
    private int predictSinglePointBeam(float[] point, Node node,
        Metric.DistanceFunction distFn, int beamWidth) {
        List<Node> candidates = new ArrayList<>();
        candidates.add(node);

        while (!candidates.isEmpty()) {
            // Expand all candidates: collect their children
            List<Node> nextCandidates = new ArrayList<>();
            boolean allLeaves = true;

            for (Node cand : candidates) {
                if (cand.isLeaf()) {
                    nextCandidates.add(cand); // keep leaves
                } else {
                    allLeaves = false;
                    for (Node child : cand.getChildren())
                        nextCandidates.add(child);
                }
            }

            if (allLeaves)
                break;

            // Sort by distance to point, keep top beamWidth
            nextCandidates.sort((a, b) -> Float.compare(
                distFn.compute(point, a.getCentroid()),
                distFn.compute(point, b.getCentroid())));
            candidates = nextCandidates.subList(0, Math.min(beamWidth, nextCandidates.size()));
        }

        // Among final candidates (all leaves), pick nearest
        Node best = candidates.get(0);
        float bestDist = distFn.compute(point, best.getCentroid());

        for (int i = 1; i < candidates.size(); i++) {
            float d = distFn.compute(point, candidates.get(i).getCentroid());
            if (d < bestDist) {
                bestDist = d;
                best = candidates.get(i);
            }
        }

        return best.getLeafId();
    }

    /**
     * Recursively builds a tree node for the given data subset.
     *
     * <p>If the stopping condition is met (max depth reached or too few points),
     * a leaf node is created. Otherwise, Lloyd's KMeans is run with
     * {@code branchFactor} clusters on the subset, and child nodes are built
     * recursively for each non-empty cluster.</p>
     *
     * <p>Children at the same level are built in parallel when there are at least
     * 2 children and the current level allows further recursion beyond the next level.</p>
     *
     * @param data      the full dataset (children reference subsets via index arrays)
     * @param indices   indices of data points belonging to this node
     * @param level     current depth in the tree (0 = root)
     * @param dimension the dimensionality of data points
     * @param random    the RNG for this subtree; parallel children receive independent instances
     * @return the constructed node (internal or leaf)
     */
    private Node buildNode(float[][] data,
        int[] indices,
        int level,
        int dimension,
        Random random) {
        int sampleCnt = indices.length;

        // Compute the centroid of this node's data subset
        float[] centroid = computeCentroid(data, indices, dimension);

        // Stopping conditions: max depth reached or too few points to split meaningfully
        if (level >= maxDepth - 1 || sampleCnt < minClusterSize)
            return new Node(level, centroid, null, indices);

        // Determine how many clusters to create at this level (cannot exceed available points)
        int locClusterCnt = Math.min(branchFactor, sampleCnt);
        if (locClusterCnt < 2)
            return new Node(level, centroid, null, indices);

        // Run Lloyd's KMeans on this node's subset to partition into branchFactor clusters.
        // The index-based overload creates a lightweight reference view internally.
        LloydKMeans kmeans = new LloydKMeans(
            locClusterCnt,
            metricType,
            metricEngine,
            maxIterationsPerLevel,
            tolerance,
            random
        );

        LloydKMeans.Result kmResult = kmeans.fit(data, indices);
        int[] labels = kmResult.getClusterAssignments();

        // Count points per cluster to identify non-empty clusters
        int[] clusterSizes = new int[locClusterCnt];
        for (int label : labels) {
            if (label < 0 || label >= locClusterCnt)
                throw new IllegalStateException("KMeans produced invalid label: " + label);
            clusterSizes[label]++;
        }

        // Count non-empty clusters; empty clusters are skipped in child construction
        int nonEmptyClusterCnt = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0)
                nonEmptyClusterCnt++;
        }

        // If all points ended up in a single cluster, splitting failed; create a leaf
        if (nonEmptyClusterCnt <= 1)
            return new Node(level, centroid, null, indices);

        // Build a mapping from original cluster IDs to compacted child indices
        // (skipping empty clusters)
        int[] clusterIdToChildIdx = new int[locClusterCnt];
        Arrays.fill(clusterIdToChildIdx, -1);

        int childIdx = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0) {
                clusterIdToChildIdx[c] = childIdx;
                childIdx++;
            }
        }

        // Allocate child index arrays based on the known sizes
        int[][] childIndices = new int[nonEmptyClusterCnt][];

        int[] childSizes = new int[nonEmptyClusterCnt];
        for (int c = 0; c < locClusterCnt; c++) {
            int mappedChildIdx = clusterIdToChildIdx[c];
            if (mappedChildIdx >= 0)
                childSizes[mappedChildIdx] = clusterSizes[c];
        }

        for (int i = 0; i < nonEmptyClusterCnt; i++)
            childIndices[i] = new int[childSizes[i]];

        // Distribute original data indices to the appropriate child arrays.
        // For large nodes, uses AtomicIntegerArray for thread-safe parallel distribution.
        if (sampleCnt >= PARALLEL_CENTROID_THRESHOLD) {
            AtomicIntegerArray atomicOffsets = new AtomicIntegerArray(nonEmptyClusterCnt);
            IntStream.range(0, sampleCnt).parallel().forEach(i -> {
                int originalCluster = labels[i];
                int mappedChild = clusterIdToChildIdx[originalCluster];
                int pos = atomicOffsets.getAndIncrement(mappedChild);
                childIndices[mappedChild][pos] = indices[i];
            });
        } else {
            // Sequential distribution for small nodes
            int[] offsets = new int[nonEmptyClusterCnt];
            for (int i = 0; i < sampleCnt; i++) {
                int originalCluster = labels[i];
                int mappedChild = clusterIdToChildIdx[originalCluster];
                int pos = offsets[mappedChild]++;
                childIndices[mappedChild][pos] = indices[i];
            }
        }

        // Recursively build child nodes.
        // Children at the same level are independent subtrees and can be built in parallel.
        // Each child subtree gets its own Random seeded from the parent's RNG.
        // Seeds are pre-generated sequentially to preserve determinism, then each child
        // uses its independent RNG to avoid contention on java.util.Random's internal AtomicLong.
        Node[] children = new Node[nonEmptyClusterCnt];
        if (nonEmptyClusterCnt >= 2 && level < maxDepth - 2) {
            // Pre-generate seeds sequentially from the parent RNG to preserve determinism
            long[] childSeeds = new long[nonEmptyClusterCnt];
            for (int i = 0; i < nonEmptyClusterCnt; i++)
                childSeeds[i] = random.nextLong();

            // Parallel recursive building for non-trivial subtrees
            int[][] finalChildIndices = childIndices;
            IntStream.range(0, nonEmptyClusterCnt).parallel().forEach(i ->
                children[i] = buildNode(data, finalChildIndices[i], level + 1, dimension,
                    new Random(childSeeds[i])));
        } else {
            // Sequential for small number of children or near-leaf level
            for (int i = 0; i < nonEmptyClusterCnt; i++)
                children[i] = buildNode(data, childIndices[i], level + 1, dimension, random);
        }

        return new Node(level, centroid, children, null);
    }

    /**
     * Minimum number of points to trigger parallel centroid computation
     * and other large-node parallel operations (subset creation, index distribution).
     */
    private static final int PARALLEL_CENTROID_THRESHOLD = 5000;

    /**
     * Computes the centroid (mean) of the data points at the given indices.
     *
     * <p>For large subsets ({@code >= PARALLEL_CENTROID_THRESHOLD}), uses thread-local
     * accumulators to parallelize the summation. For cosine distance, the resulting
     * centroid is normalized to unit length.</p>
     *
     * @param data      the full dataset
     * @param indices   indices of points to include in the centroid computation
     * @param dimension the dimensionality of data points
     * @return the computed centroid vector
     */
    private float[] computeCentroid(float[][] data,
        int[] indices,
        int dimension) {
        float[] centroid = new float[dimension];
        int cnt = indices.length;
        if (cnt == 0)
            return centroid;

        if (cnt >= PARALLEL_CENTROID_THRESHOLD) {
            // Parallel centroid computation using thread-local sum buffers
            int numThreads = Runtime.getRuntime().availableProcessors();
            int chunkSize = (cnt + numThreads - 1) / numThreads;
            float[][] localSums = new float[numThreads][dimension];

            IntStream.range(0, numThreads).parallel().forEach(threadIdx -> {
                int start = threadIdx * chunkSize;
                int end = Math.min(start + chunkSize, cnt);
                float[] mySum = localSums[threadIdx];

                for (int i = start; i < end; i++) {
                    float[] point = data[indices[i]];
                    for (int d = 0; d < dimension; d++)
                        mySum[d] += point[d];
                }
            });

            // Merge thread-local sums
            for (int t = 0; t < numThreads; t++) {
                for (int d = 0; d < dimension; d++)
                    centroid[d] += localSums[t][d];
            }
        } else {
            // Sequential summation
            for (int idx : indices) {
                float[] point = data[idx];
                for (int d = 0; d < dimension; d++)
                    centroid[d] += point[d];
            }
        }

        // Divide by count to obtain the mean
        float invCnt = 1.0f / (float)cnt;
        for (int d = 0; d < dimension; d++)
            centroid[d] *= invCnt;

        // For cosine distance, project onto the unit sphere
        if (metricType == Metric.Type.COSINE_DISTANCE)
            KMeansUtils.normalizeSingleCentroid(centroid);

        return centroid;
    }

    /**
     * Minimum number of points in a leaf to trigger parallel loss computation.
     */
    private static final int PARALLEL_LEAF_LOSS_THRESHOLD = 1000;

    /**
     * Recursively traverses the tree in depth-first order, assigning sequential leaf IDs
     * and collecting leaf centroids into a flat list.
     *
     * <p>For leaf nodes, computes the contribution to total loss (sum of distances
     * from each point to its leaf centroid) and records each point's leaf assignment.
     * Loss computation is parallelized for large leaves.</p>
     *
     * @param node               the current tree node
     * @param data               the full dataset
     * @param leafCentroidsList  accumulator list for leaf centroids (order = leaf ID)
     * @param leafAssignments    output: per-point leaf ID assignments
     * @param lossAccumulator    mutable wrapper for accumulating total loss
     * @param distFn             the distance function
     */
    private void assignLeafIdsAndCollect(Node node,
        float[][] data,
        List<float[]> leafCentroidsList,
        int[] leafAssignments,
        DoubleWrapper lossAccumulator,
        Metric.DistanceFunction distFn) {
        if (node == null)
            return;

        if (node.isLeaf()) {
            // Assign a sequential leaf ID based on the current list size
            int leafId = leafCentroidsList.size();
            node.leafId = leafId;
            leafCentroidsList.add(node.centroid);

            if (node.pointIndices != null) {
                int[] indices = node.pointIndices;
                float[] centroid = node.centroid;

                if (indices.length >= PARALLEL_LEAF_LOSS_THRESHOLD) {
                    // Parallel loss computation for large leaves
                    double loss = IntStream.of(indices).parallel()
                        .mapToDouble(idx -> distFn.compute(data[idx], centroid))
                        .sum();
                    lossAccumulator.val += (float) loss;

                    // Leaf assignment is sequential (fast, no contention)
                    for (int idx : indices)
                        leafAssignments[idx] = leafId;
                } else {
                    // Sequential path for small leaves: compute loss and assign in one pass
                    for (int idx : indices) {
                        leafAssignments[idx] = leafId;
                        lossAccumulator.val += distFn.compute(data[idx], centroid);
                    }
                }
            }
        }
        else {
            // Internal node: recurse into children (depth-first)
            Node[] children = node.getChildren();
            if (children != null) {
                for (Node child : children)
                    assignLeafIdsAndCollect(child, data, leafCentroidsList, leafAssignments,
                        lossAccumulator, distFn);
            }
        }
    }

    /**
     * Computes per-cluster membership counts from the leaf assignment array.
     *
     * @param assignments per-point leaf cluster IDs
     * @param clusterCnt  the total number of leaf clusters
     * @return an array of length {@code clusterCnt} with per-cluster point counts
     * @throws IllegalStateException if any assignment is out of the valid range
     */
    private int[] computeClusterSizes(int[] assignments, int clusterCnt) {
        int[] sizes = new int[clusterCnt];

        for (int label : assignments) {
            if (label < 0 || label >= clusterCnt) {
                throw new IllegalStateException(
                    "invalid cluster label " + label + " (expected 0.." + (clusterCnt - 1) + ")"
                );
            }
            sizes[label]++;
        }

        return sizes;
    }

    /**
     * Mutable float wrapper used to accumulate loss across recursive calls.
     * Avoids boxing overhead that would result from using {@code AtomicReference<Float>}
     * or similar constructs in the sequential depth-first traversal.
     */
    private static final class DoubleWrapper {
        /** The accumulated value. Uses double to prevent precision loss over many summands. */
        double val;
    }

    /**
     * A node in the hierarchical clustering tree.
     *
     * <p>A node is either:</p>
     * <ul>
     *   <li><b>Internal:</b> has a non-null, non-empty {@code children} array and null
     *       {@code pointIndices}. Represents a partitioning level in the tree.</li>
     *   <li><b>Leaf:</b> has null or empty {@code children} and a non-null
     *       {@code pointIndices} array containing the indices of data points assigned
     *       to this cluster. After tree construction, a sequential {@code leafId} is
     *       assigned during the collection pass.</li>
     * </ul>
     *
     * <p>Every node stores a centroid (the mean of its assigned data points), which is
     * used for routing queries during prediction.</p>
     */
    public static final class Node {
        /** Depth of this node in the tree (0 = root). */
        private final int level;
        /** Centroid (mean vector) of all data points routed through this node. */
        private final float[] centroid;
        /** Child nodes (non-null for internal nodes, null for leaves). */
        private final Node[] children;
        /** Indices of data points belonging to this leaf (null for internal nodes). */
        private final int[] pointIndices;
        /**
         * Sequential leaf ID assigned during the collection pass.
         * Defaults to -1 (unassigned) and is set to a non-negative value only for leaves.
         */
        private int leafId = -1;

        /**
         * Constructs a tree node.
         *
         * @param level        the depth in the tree
         * @param centroid     the centroid vector of this node's data subset
         * @param children     child nodes (null for a leaf)
         * @param pointIndices data point indices (null for an internal node)
         */
        Node(int level,
            float[] centroid,
            Node[] children,
            int[] pointIndices) {
            this.level = level;
            this.centroid = centroid;
            this.children = children;
            this.pointIndices = pointIndices;
        }

        /**
         * Returns the depth of this node in the tree.
         *
         * @return the tree level (0 = root)
         */
        public int getLevel() {
            return level;
        }

        /**
         * Returns the centroid vector of this node.
         *
         * @return the centroid (mean of all data points in this node's subtree)
         */
        public float[] getCentroid() {
            return centroid;
        }

        /**
         * Returns the child nodes of this internal node.
         *
         * @return the children array, or null if this is a leaf
         */
        public Node[] getChildren() {
            return children;
        }

        /**
         * Returns the data point indices stored in this leaf node.
         *
         * @return the point indices, or null if this is an internal node
         */
        public int[] getPointIndices() {
            return pointIndices;
        }

        /**
         * Returns the sequential leaf ID assigned during the collection pass.
         *
         * @return the leaf ID (>= 0 for assigned leaves, -1 if not yet assigned)
         */
        public int getLeafId() {
            return leafId;
        }

        /**
         * Determines whether this node is a leaf (has no children).
         *
         * @return {@code true} if this node has no children
         */
        public boolean isLeaf() {
            return children == null || children.length == 0;
        }
    }

    /**
     * Immutable result of hierarchical KMeans clustering.
     *
     * <p>Contains the tree root for prediction, per-point leaf assignments, the flat
     * list of leaf centroids (in leaf-ID order), the total loss, and per-cluster
     * membership counts.</p>
     */
    static class Result implements ClusteringResult {
        /** Root of the hierarchical clustering tree (used for prediction). */
        private final Node root;
        /** Per-point leaf cluster assignments (0-based leaf IDs). */
        private final int[] leafAssignments;
        /** Flat array of leaf centroids in leaf-ID order: {@code float[numLeaves][dimension]}. */
        private final float[][] leafCentroids;
        /** Total loss: sum of distances from each point to its leaf centroid. */
        private final float loss;
        /** Number of points assigned to each leaf cluster. */
        private final int[] clusterSizes;

        /**
         * Constructs a hierarchical clustering result.
         *
         * @param root            the tree root
         * @param leafAssignments per-point leaf cluster IDs
         * @param leafCentroids   leaf centroid vectors in leaf-ID order
         * @param loss            total clustering loss
         * @param clusterSizes    per-leaf membership counts
         */
        Result(Node root,
            int[] leafAssignments,
            float[][] leafCentroids,
            float loss,
            int[] clusterSizes) {
            this.root = root;
            this.leafAssignments = leafAssignments;
            this.leafCentroids = leafCentroids;
            this.loss = loss;
            this.clusterSizes = clusterSizes;
        }

        /**
         * Returns the root of the hierarchical clustering tree.
         *
         * @return the tree root node
         */
        public Node getRoot() {
            return root;
        }

        /** {@inheritDoc} */
        @Override public int[] getClusterAssignments() {
            return leafAssignments;
        }

        /** {@inheritDoc} */
        @Override public float[][] getCentroids() {
            return leafCentroids;
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
