package ru.mcashesha.ivf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

/**
 * Flat (uncompressed) implementation of the {@link IVFIndex} interface for approximate nearest neighbor search.
 *
 * <h3>Storage Layout</h3>
 * <p>Vectors are stored contiguously in a flat array, reordered by cluster assignment after the build phase.
 * Instead of maintaining per-cluster lists, this implementation uses an offset-based scheme:
 * {@code clusterOffsets[c]} marks the start index and {@code clusterOffsets[c+1]} marks the exclusive end
 * of cluster {@code c}'s vectors in the {@code data} and {@code ids} arrays. This layout improves
 * cache locality during search, as all vectors in a cluster occupy a contiguous memory region.</p>
 *
 * <h3>Build Phase</h3>
 * <ol>
 *   <li>Fit the configured KMeans algorithm to partition vectors into clusters.</li>
 *   <li>Compute prefix-sum offsets from cluster sizes.</li>
 *   <li>Reorder vectors and IDs so that each cluster's data is contiguous in memory.</li>
 * </ol>
 *
 * <h3>Search Phase</h3>
 * <ol>
 *   <li>Compute distances from the query to all cluster centroids.</li>
 *   <li>Select the {@code nProbe} nearest centroids using a max-heap (heap-select + heapsort).</li>
 *   <li>Scan all vectors within the selected clusters, maintaining a max-heap of size {@code topK}
 *       for streaming top-K extraction.</li>
 *   <li>Extract results from the heap in ascending distance order.</li>
 * </ol>
 *
 * <h3>Parallelization</h3>
 * <p>The implementation applies parallelism at several stages when the workload exceeds configurable thresholds:</p>
 * <ul>
 *   <li><b>Data reordering</b> (build): parallelized when the dataset has &ge; 10,000 vectors.</li>
 *   <li><b>Centroid distance computation</b> (search): parallelized when there are &ge; 32 clusters.</li>
 *   <li><b>Cluster scanning</b> (search): parallelized when total points to scan &ge; 2,000 and nProbe &ge; 2.
 *       Each cluster builds a local top-K heap, which are then merged into a global top-K heap.</li>
 *   <li><b>Batch search</b>: queries are processed fully in parallel via {@code IntStream.parallel()}.</li>
 * </ul>
 *
 * <h3>Thread Safety</h3>
 * <p>Search operations are thread-safe: each call to {@link #search} or {@link #searchInternal} allocates
 * its own local buffers for centroid distances, cluster ordering, and top-K heaps, so no shared mutable
 * state is accessed during the search phase.</p>
 *
 * @see IVFIndex
 * @see KMeans
 */
public class IVFIndexFlat implements IVFIndex {

    /** The KMeans algorithm instance used during the build phase to cluster vectors. */
    private final KMeans<? extends KMeans.ClusteringResult> kMeans;

    /** Cluster centroids produced by KMeans; centroids[c] is the centroid vector for cluster c. */
    private float[][] centroids;

    /**
     * Prefix-sum array of cluster sizes. {@code clusterOffsets[c]} is the start index of cluster c's
     * vectors in {@code data}/{@code ids}, and {@code clusterOffsets[c+1]} is the exclusive end.
     * Length = number of clusters + 1.
     */
    private int[] clusterOffsets;

    /** Reordered dataset: vectors are grouped contiguously by cluster assignment for cache locality. */
    private float[][] data;

    /** Reordered IDs corresponding to each vector in {@code data}. */
    private int[] ids;

    /** The dimensionality of each vector. */
    private int dimension;

    /** Flag indicating whether the index has been built and is ready for search queries. */
    private boolean built;

    /** Resolved distance function, cached after build to avoid repeated virtual dispatch on the hot path. */
    private Metric.DistanceFunction distFn;

    /**
     * Minimum number of clusters required to parallelize centroid distance computation.
     * Below this threshold, sequential computation is faster due to parallelism overhead.
     */
    private static final int PARALLEL_CENTROID_THRESHOLD = 32;

    /**
     * Minimum total number of points across all probed clusters required to parallelize cluster scanning.
     * Below this threshold, the overhead of creating per-cluster heaps and merging outweighs the benefit.
     */
    private static final int PARALLEL_CLUSTER_SCAN_THRESHOLD = 2000;

    /**
     * Constructs a new flat IVF index backed by the given KMeans algorithm.
     *
     * <p>The KMeans instance determines the clustering strategy (Lloyd, MiniBatch, or Hierarchical),
     * the distance metric type, and the computation engine used during both build and search.</p>
     *
     * @param kMeans the KMeans algorithm to use for clustering; must not be null
     * @throws IllegalArgumentException if kMeans is null
     */
    public IVFIndexFlat(KMeans<? extends KMeans.ClusteringResult> kMeans) {
        if (kMeans == null)
            throw new IllegalArgumentException("kMeans must be non-null");

        this.kMeans = kMeans;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to {@link #build(float[][], int[])} with auto-generated sequential IDs.</p>
     */
    @Override public void build(float[][] vectors) {
        build(vectors, null);
    }

    /**
     * {@inheritDoc}
     *
     * <p>The build process consists of the following steps:</p>
     * <ol>
     *   <li>Validate input vectors for non-null, non-empty, and consistent dimensions.</li>
     *   <li>Assign IDs: use provided IDs or generate sequential 0-based IDs.</li>
     *   <li>Run KMeans clustering on the vectors to obtain centroids and assignments.</li>
     *   <li>Compute {@code clusterOffsets} as a prefix sum of cluster sizes.</li>
     *   <li>Reorder vectors and IDs so that each cluster's data is stored contiguously.
     *       For datasets with &ge; 10,000 vectors, reordering is done in parallel using
     *       atomic counters for thread-safe offset tracking.</li>
     *   <li>Resolve the distance function for the configured metric type and engine.</li>
     * </ol>
     */
    @Override public void build(float[][] vectors, int[] ids) {
        if (vectors == null || vectors.length == 0)
            throw new IllegalArgumentException("vectors must be non-empty");
        if (vectors[0] == null)
            throw new IllegalArgumentException("vectors[0] must be non-null");

        int locDimension = vectors[0].length;
        if (locDimension == 0)
            throw new IllegalArgumentException("vector dimension must be > 0");

        for (int i = 1; i < vectors.length; i++) {
            if (vectors[i] == null || vectors[i].length != locDimension) {
                throw new IllegalArgumentException(
                    "all vectors must be non-null and have the same dimension"
                );
            }
        }

        this.dimension = locDimension;

        // Use provided IDs or generate default sequential IDs (0, 1, 2, ...)
        int[] originalIds;
        if (ids != null) {
            if (ids.length != vectors.length)
                throw new IllegalArgumentException("ids length must match vectors length");
            originalIds = ids;
        }
        else {
            originalIds = new int[vectors.length];
            for (int i = 0; i < originalIds.length; i++)
                originalIds[i] = i;
        }

        // Step 1: Cluster the vectors using the configured KMeans algorithm
        KMeans.ClusteringResult clusteringResult = kMeans.fit(vectors);

        this.centroids = clusteringResult.getCentroids();
        int[] sizes = clusteringResult.getClusterSizes();
        int[] assignments = clusteringResult.getClusterAssignments();

        if (centroids == null || centroids.length == 0)
            throw new IllegalStateException("KMeans returned empty centroids");
        if (assignments == null || assignments.length != vectors.length)
            throw new IllegalStateException("KMeans returned inconsistent assignments");

        int clusterCnt = centroids.length;

        for (float[] centroid : centroids) {
            if (centroid == null || centroid.length != dimension)
                throw new IllegalStateException("centroid dimension mismatch");
        }

        // Verify that the sum of all cluster sizes equals the total number of vectors
        int totalSize = 0;
        for (int c = 0; c < clusterCnt; c++) {
            int sizeForCluster = sizes[c];
            if (sizeForCluster < 0)
                throw new IllegalStateException("KMeans returned negative cluster size for cluster " + c);
            totalSize += sizeForCluster;
        }
        if (totalSize != vectors.length)
            throw new IllegalStateException(
                "Sum of clusterSizes (" + totalSize + ") != number of vectors (" + vectors.length + ')'
            );

        // Step 2: Build the prefix-sum offset array from cluster sizes.
        // clusterOffsets[c] = start index of cluster c, clusterOffsets[c+1] = exclusive end.
        this.clusterOffsets = new int[clusterCnt + 1];
        clusterOffsets[0] = 0;
        for (int c = 0; c < clusterCnt; c++)
            clusterOffsets[c + 1] = clusterOffsets[c] + sizes[c];

        // Step 3: Reorder vectors and IDs so each cluster's data is contiguous in memory.
        // This dramatically improves cache locality during search, since scanning a cluster
        // becomes a sequential memory access pattern.
        float[][] reorderedData = new float[vectors.length][];
        int[] reorderedIds = new int[vectors.length];

        int n = assignments.length;

        if (n >= 10000) {
            // Parallel reordering for large datasets using a two-pass approach.

            // Pass 1: Compute each vector's target position using atomic counters.
            // Each cluster has an atomic counter that tracks the next write position within
            // that cluster's segment. getAndIncrement ensures no two threads write to the same slot.
            AtomicIntegerArray atomicOffsets = new AtomicIntegerArray(clusterCnt);
            int[] targetPos = new int[n];

            IntStream.range(0, n).parallel().forEach(i -> {
                int c = assignments[i];
                if (c >= 0 && c < clusterCnt) {
                    int localOffset = atomicOffsets.getAndIncrement(c);
                    targetPos[i] = clusterOffsets[c] + localOffset;
                } else {
                    targetPos[i] = -1;  // Invalid assignment
                }
            });

            // Pass 2: Parallel scatter -- each vector is written to its computed target position.
            // Since each target position is unique (guaranteed by atomic increment), there are
            // no write conflicts and no synchronization is needed.
            IntStream.range(0, n).parallel().forEach(i -> {
                int pos = targetPos[i];
                if (pos >= 0) {
                    reorderedData[pos] = vectors[i];
                    reorderedIds[pos] = originalIds[i];
                }
            });
        } else {
            // Sequential path for small datasets: use a simple write-position cursor per cluster
            int[] writePos = new int[clusterCnt];
            System.arraycopy(clusterOffsets, 0, writePos, 0, clusterCnt);

            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                if (c < 0 || c >= clusterCnt)
                    continue;
                int pos = writePos[c]++;
                reorderedData[pos] = vectors[i];
                reorderedIds[pos] = originalIds[i];
            }
        }

        this.data = reorderedData;
        this.ids = reorderedIds;

        // Cache the resolved distance function to avoid virtual dispatch on every distance computation.
        // This is important because distance computation is the innermost hot-path operation.
        this.distFn = kMeans.getMetricType().resolveFunction(kMeans.getMetricEngine());

        this.built = true;
    }

    /** {@inheritDoc} */
    @Override public Metric.Type getMetricType() {
        return kMeans.getMetricType();
    }

    /** {@inheritDoc} */
    @Override public Metric.Engine getMetricEngine() {
        return kMeans.getMetricEngine();
    }

    /** {@inheritDoc} */
    @Override public int getCountClusters() {
        return centroids.length;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Validates the query and delegates to {@link #searchInternal(float[], int, int)},
     * which is thread-safe and allocates its own local buffers.</p>
     */
    @Override public List<SearchResult> search(float[] qry, int topK, int nProbe) {
        if (!built)
            throw new IllegalStateException("Index is not built yet");
        if (qry == null || qry.length != dimension)
            throw new IllegalArgumentException("query must be non-null and match index dimension");
        if (topK <= 0)
            throw new IllegalArgumentException("topK must be > 0");

        return searchInternal(qry, topK, nProbe);
    }

    /**
     * Selects the {@code nProbe} nearest clusters to a query using a max-heap-based selection algorithm.
     *
     * <p>Algorithm overview (heap-select + heapsort):</p>
     * <ol>
     *   <li>Build a max-heap of size {@code nProbe} from the first {@code nProbe} clusters.</li>
     *   <li>For each remaining cluster, if its distance is less than the heap's maximum
     *       (the root), replace the root and sift down. This maintains a heap of the
     *       {@code nProbe} smallest distances seen so far.</li>
     *   <li>Extract elements from the heap via repeated remove-max to produce a sorted order
     *       (ascending by distance).</li>
     * </ol>
     *
     * <p>Time complexity: O(k + nProbe * log(nProbe)) amortized, where k is the total number
     * of clusters. In the worst case (all clusters closer than the initial heap), this degrades
     * to O(k * log(nProbe)). Still significantly better than O(nProbe * k) selection sort
     * when nProbe &lt;&lt; k.</p>
     *
     * @param distances    precomputed distances from the query to each centroid
     * @param clusterOrder output array; on return, positions [0..nProbe-1] contain the indices
     *                     of the nProbe nearest clusters, sorted by ascending distance
     * @param clusterCnt   the total number of clusters
     * @param nProbe       the number of nearest clusters to select
     */
    private static void selectNearestClusters(float[] distances, int[] clusterOrder, int clusterCnt, int nProbe) {
        // Initialize clusterOrder with first nProbe clusters
        for (int i = 0; i < nProbe; i++)
            clusterOrder[i] = i;

        // Build a max-heap of size nProbe. The root (index 0) holds the farthest cluster
        // among the current nProbe candidates. This allows O(1) comparison with new candidates.
        for (int i = nProbe / 2 - 1; i >= 0; i--)
            siftDownCluster(distances, clusterOrder, i, nProbe);

        // Scan remaining clusters: replace the root (farthest selected) if a closer cluster is found.
        // After this loop, the heap contains indices of the nProbe closest clusters.
        for (int c = nProbe; c < clusterCnt; c++) {
            if (distances[c] < distances[clusterOrder[0]]) {
                clusterOrder[0] = c;
                siftDownCluster(distances, clusterOrder, 0, nProbe);
            }
        }

        // Heapsort extraction: repeatedly swap the root (max) to the end and reduce heap size,
        // producing ascending distance order in clusterOrder[0..nProbe-1].
        // After this, clusterOrder[0] = nearest centroid, clusterOrder[nProbe-1] = farthest among selected.
        for (int i = nProbe - 1; i > 0; i--) {
            int tmp = clusterOrder[0];
            clusterOrder[0] = clusterOrder[i];
            clusterOrder[i] = tmp;
            siftDownCluster(distances, clusterOrder, 0, i);
        }
    }

    /**
     * Standard binary max-heap sift-down operation for the cluster selection heap.
     *
     * <p>This variant operates on an indirection array ({@code clusterOrder}) with distances
     * looked up from the {@code distances} array. It restores the max-heap property by
     * moving element at position {@code i} downward until it is larger than both children
     * or reaches a leaf.</p>
     *
     * @param distances    distances array indexed by cluster index
     * @param clusterOrder indirection array mapping heap positions to cluster indices
     * @param i            the position to sift down from
     * @param size         the current heap size (only positions [0..size-1] are in the heap)
     */
    private static void siftDownCluster(float[] distances, int[] clusterOrder, int i, int size) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < size && distances[clusterOrder[left]] > distances[clusterOrder[largest]])
                largest = left;
            if (right < size && distances[clusterOrder[right]] > distances[clusterOrder[largest]])
                largest = right;

            if (largest == i)
                break;

            // Swap current node with the largest child and continue sifting from the child's position
            int tmp = clusterOrder[i];
            clusterOrder[i] = clusterOrder[largest];
            clusterOrder[largest] = tmp;
            i = largest;
        }
    }

    /**
     * Builds a max-heap in-place from the given parallel arrays (bottom-up heap construction).
     *
     * <p>This is the standard Floyd's algorithm for building a heap in O(n) time.
     * The three parallel arrays ({@code dist}, {@code ids}, {@code clusterIds}) are kept
     * synchronized: any swap operation moves corresponding elements in all three arrays.</p>
     *
     * @param dist       distances array (heap key -- max at root)
     * @param ids        vector IDs, parallel to dist
     * @param clusterIds cluster assignments, parallel to dist
     * @param size       number of elements in the heap
     */
    private static void buildMaxHeap(float[] dist, int[] ids, int[] clusterIds, int size) {
        // Start from the last non-leaf node and sift down each node
        for (int i = size / 2 - 1; i >= 0; i--)
            siftDown(dist, ids, clusterIds, i, size);
    }

    /**
     * Standard binary max-heap sift-down for the top-K result heap.
     *
     * <p>Operates on three parallel arrays ({@code dist}, {@code ids}, {@code clusterIds})
     * that are kept in sync during swaps. The distance array is the heap key, with the
     * maximum distance at the root (index 0). This allows O(1) comparison with incoming
     * candidates during streaming top-K selection.</p>
     *
     * @param dist       distances array (heap key)
     * @param ids        vector IDs, parallel to dist
     * @param clusterIds cluster assignments, parallel to dist
     * @param i          the position to sift down from
     * @param size       the current heap size
     */
    private static void siftDown(float[] dist, int[] ids, int[] clusterIds, int i, int size) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < size && dist[left] > dist[largest])
                largest = left;
            if (right < size && dist[right] > dist[largest])
                largest = right;

            if (largest == i)
                break;

            // Swap all three parallel arrays to maintain synchronization
            float tmpD = dist[i]; dist[i] = dist[largest]; dist[largest] = tmpD;
            int tmpId = ids[i]; ids[i] = ids[largest]; ids[largest] = tmpId;
            int tmpC = clusterIds[i]; clusterIds[i] = clusterIds[largest]; clusterIds[largest] = tmpC;

            i = largest;
        }
    }

    /** {@inheritDoc} */
    @Override public int getDimension() {
        return dimension;
    }

    /**
     * {@inheritDoc}
     *
     * <p>All queries are processed independently in parallel using {@code IntStream.parallel()}.
     * Each query invokes {@link #searchInternal}, which is thread-safe due to thread-local
     * buffer allocation.</p>
     */
    @Override public List<List<SearchResult>> searchBatch(float[][] queries, int topK, int nProbe) {
        if (!built)
            throw new IllegalStateException("Index is not built yet");
        if (queries == null || queries.length == 0)
            throw new IllegalArgumentException("queries must be non-empty");
        if (topK <= 0)
            throw new IllegalArgumentException("topK must be > 0");

        for (int i = 0; i < queries.length; i++) {
            if (queries[i] == null || queries[i].length != dimension)
                throw new IllegalArgumentException("query[" + i + "] must be non-null and match index dimension");
        }

        int numQueries = queries.length;

        // Parallel search for all queries -- each slot is written by exactly one thread
        @SuppressWarnings("unchecked")
        List<SearchResult>[] results = new List[numQueries];

        IntStream.range(0, numQueries).parallel().forEach(q ->
            results[q] = searchInternal(queries[q], topK, nProbe));

        return Arrays.asList(results);
    }

    /**
     * Core search implementation that is thread-safe (no shared mutable state).
     *
     * <p>This method performs the following steps:</p>
     * <ol>
     *   <li>Compute distances from the query to all cluster centroids (parallelized for &ge; 32 clusters).</li>
     *   <li>Select the {@code nProbe} nearest centroids via {@link #selectNearestClusters}.</li>
     *   <li>Count the total points across selected clusters to decide sequential vs. parallel scanning.</li>
     *   <li>Scan the selected clusters, maintaining a streaming top-K max-heap.</li>
     *   <li>Extract and return results sorted by ascending distance.</li>
     * </ol>
     *
     * @param qry    the query vector
     * @param topK   the number of nearest neighbors to return
     * @param nProbe the number of clusters to probe (clamped to [1, clusterCnt])
     * @return sorted list of up to {@code topK} nearest neighbors
     */
    private List<SearchResult> searchInternal(float[] qry, int topK, int nProbe) {
        int clusterCnt = centroids.length;
        if (clusterCnt == 0)
            return List.of();

        // Clamp nProbe to valid range [1, clusterCnt]
        nProbe = Math.max(1, Math.min(nProbe, clusterCnt));

        // Capture the distance function in a local variable to avoid field access on each iteration
        Metric.DistanceFunction fn = this.distFn;

        // Allocate thread-local buffers for centroid distances and cluster ordering.
        // This avoids shared state and makes the method safe for concurrent invocation.
        float[] localCentroidDistances = new float[clusterCnt];
        int[] localClusterOrder = new int[clusterCnt];

        // Compute distances from the query to each centroid.
        // For large cluster counts, parallelize the distance computation.
        if (clusterCnt >= PARALLEL_CENTROID_THRESHOLD) {
            IntStream.range(0, clusterCnt).parallel().forEach(c ->
                localCentroidDistances[c] = fn.compute(qry, centroids[c]));
        } else {
            for (int c = 0; c < clusterCnt; c++)
                localCentroidDistances[c] = fn.compute(qry, centroids[c]);
        }

        // Select the nProbe nearest clusters, sorted by ascending distance
        for (int c = 0; c < clusterCnt; c++)
            localClusterOrder[c] = c;
        selectNearestClusters(localCentroidDistances, localClusterOrder, clusterCnt, nProbe);

        // Calculate total points across all selected clusters to decide parallelization strategy
        int totalPointsToScan = 0;
        for (int p = 0; p < nProbe; p++) {
            int clusterId = localClusterOrder[p];
            totalPointsToScan += clusterOffsets[clusterId + 1] - clusterOffsets[clusterId];
        }

        // Use parallel scanning when the workload is large enough to justify the overhead.
        // Requires both sufficient total points AND at least 2 clusters to distribute work.
        if (totalPointsToScan >= PARALLEL_CLUSTER_SCAN_THRESHOLD && nProbe >= 2) {
            return searchClustersParallel(qry, topK, nProbe, localClusterOrder, fn);
        }

        // Sequential cluster scanning for small workloads
        return searchClustersSequential(qry, topK, nProbe, localClusterOrder, fn);
    }

    /**
     * Scans the selected clusters sequentially, maintaining a single max-heap for streaming top-K selection.
     *
     * <p>The max-heap has the largest distance at the root. For each candidate vector:</p>
     * <ul>
     *   <li>If the heap has fewer than {@code topK} elements, insert directly.</li>
     *   <li>Once the heap is full, only insert if the candidate is closer than the current
     *       farthest result (the root). In that case, replace the root and sift down.</li>
     * </ul>
     * <p>This "replace-root-and-sift" pattern achieves O(n * log(k)) time for streaming top-K,
     * where n is the number of scanned vectors and k is topK.</p>
     *
     * @param qry              the query vector
     * @param topK             the number of nearest neighbors to return
     * @param nProbe           the number of clusters to scan
     * @param localClusterOrder the indices of the selected clusters, sorted by ascending centroid distance
     * @param fn               the distance function to use
     * @return sorted list of up to {@code topK} nearest neighbors
     */
    private List<SearchResult> searchClustersSequential(float[] qry, int topK, int nProbe,
                                                         int[] localClusterOrder, Metric.DistanceFunction fn) {
        // Parallel arrays for the top-K max-heap
        int[] heapIds = new int[topK];
        float[] heapDistances = new float[topK];
        int[] heapClusterIds = new int[topK];
        int heapSize = 0;

        for (int p = 0; p < nProbe; p++) {
            int clusterId = localClusterOrder[p];
            // Contiguous range [start, end) for this cluster in the reordered data array
            int start = clusterOffsets[clusterId];
            int end = clusterOffsets[clusterId + 1];

            for (int i = start; i < end; i++) {
                float d = fn.compute(qry, data[i]);

                if (heapSize < topK) {
                    // Heap not yet full -- insert unconditionally
                    heapIds[heapSize] = ids[i];
                    heapDistances[heapSize] = d;
                    heapClusterIds[heapSize] = clusterId;
                    heapSize++;
                    // Build the max-heap once the buffer is exactly full
                    if (heapSize == topK)
                        buildMaxHeap(heapDistances, heapIds, heapClusterIds, heapSize);
                }
                else if (d < heapDistances[0]) {
                    // Candidate is closer than the current farthest in the heap.
                    // Replace the root (farthest) and restore the heap property.
                    heapIds[0] = ids[i];
                    heapDistances[0] = d;
                    heapClusterIds[0] = clusterId;
                    siftDown(heapDistances, heapIds, heapClusterIds, 0, heapSize);
                }
            }
        }

        return extractSortedResults(heapIds, heapDistances, heapClusterIds, heapSize, topK);
    }

    /**
     * Scans the selected clusters in parallel, then merges per-cluster top-K heaps into a global result.
     *
     * <p>Each cluster independently builds its own local max-heap of size {@code topK}. After all
     * clusters are scanned, the local heaps are merged sequentially into a single global top-K heap.
     * This approach avoids synchronization during the most expensive phase (scanning vectors)
     * while keeping the cheaper merge phase simple.</p>
     *
     * @param qry              the query vector
     * @param topK             the number of nearest neighbors to return
     * @param nProbe           the number of clusters to scan
     * @param localClusterOrder the indices of the selected clusters
     * @param fn               the distance function to use
     * @return sorted list of up to {@code topK} nearest neighbors
     */
    private List<SearchResult> searchClustersParallel(float[] qry, int topK, int nProbe,
                                                       int[] localClusterOrder, Metric.DistanceFunction fn) {
        // Allocate per-cluster heap buffers. Each cluster gets independent arrays
        // so parallel threads do not contend on shared state.
        int[][] localHeapIds = new int[nProbe][topK];
        float[][] localHeapDistances = new float[nProbe][topK];
        int[][] localHeapClusterIds = new int[nProbe][topK];
        int[] localHeapSizes = new int[nProbe];

        // Parallel per-cluster scanning: each cluster builds its own local top-K heap
        IntStream.range(0, nProbe).parallel().forEach(p -> {
            int clusterId = localClusterOrder[p];
            int start = clusterOffsets[clusterId];
            int end = clusterOffsets[clusterId + 1];

            // Local references to this cluster's heap arrays for reduced indirection
            int[] hIds = localHeapIds[p];
            float[] hDist = localHeapDistances[p];
            int[] hCluster = localHeapClusterIds[p];
            int hSize = 0;

            for (int i = start; i < end; i++) {
                float d = fn.compute(qry, data[i]);

                if (hSize < topK) {
                    hIds[hSize] = ids[i];
                    hDist[hSize] = d;
                    hCluster[hSize] = clusterId;
                    hSize++;
                    if (hSize == topK)
                        buildMaxHeap(hDist, hIds, hCluster, hSize);
                }
                else if (d < hDist[0]) {
                    hIds[0] = ids[i];
                    hDist[0] = d;
                    hCluster[0] = clusterId;
                    siftDown(hDist, hIds, hCluster, 0, hSize);
                }
            }

            localHeapSizes[p] = hSize;
        });

        // Merge phase: combine all per-cluster heaps into a single global top-K heap.
        // This runs sequentially since it is O(nProbe * topK * log(topK)), which is small
        // compared to the parallel scanning phase.
        int[] mergedIds = new int[topK];
        float[] mergedDistances = new float[topK];
        int[] mergedClusterIds = new int[topK];
        int mergedSize = 0;

        for (int p = 0; p < nProbe; p++) {
            int hSize = localHeapSizes[p];
            int[] hIds = localHeapIds[p];
            float[] hDist = localHeapDistances[p];
            int[] hCluster = localHeapClusterIds[p];

            for (int j = 0; j < hSize; j++) {
                if (mergedSize < topK) {
                    mergedIds[mergedSize] = hIds[j];
                    mergedDistances[mergedSize] = hDist[j];
                    mergedClusterIds[mergedSize] = hCluster[j];
                    mergedSize++;
                    if (mergedSize == topK)
                        buildMaxHeap(mergedDistances, mergedIds, mergedClusterIds, mergedSize);
                }
                else if (hDist[j] < mergedDistances[0]) {
                    mergedIds[0] = hIds[j];
                    mergedDistances[0] = hDist[j];
                    mergedClusterIds[0] = hCluster[j];
                    siftDown(mergedDistances, mergedIds, mergedClusterIds, 0, mergedSize);
                }
            }
        }

        return extractSortedResults(mergedIds, mergedDistances, mergedClusterIds, mergedSize, topK);
    }

    /**
     * Extracts results from the top-K max-heap in ascending distance order (nearest first).
     *
     * <p>This works by repeatedly extracting the maximum element (root of the max-heap)
     * and placing it at the end of the output array, then restoring the heap property.
     * Since the max is extracted last-to-first, the resulting array is sorted in ascending order.</p>
     *
     * <p>If the heap has fewer elements than {@code topK} (i.e., fewer vectors were scanned
     * than requested), the heap is first properly heapified before extraction.</p>
     *
     * @param heapIds        vector IDs in the heap
     * @param heapDistances  distances in the heap (heap key)
     * @param heapClusterIds cluster IDs in the heap
     * @param heapSize       current number of elements in the heap
     * @param topK           the requested number of results (heap capacity)
     * @return list of {@link SearchResult} sorted by ascending distance
     */
    private List<SearchResult> extractSortedResults(int[] heapIds, float[] heapDistances,
                                                     int[] heapClusterIds, int heapSize, int topK) {
        // If we have fewer results than topK, we need to heapify what we have
        // (the heap was never fully built during the scan phase)
        if (heapSize > 0 && heapSize < topK)
            buildMaxHeap(heapDistances, heapIds, heapClusterIds, heapSize);

        // Extract elements in reverse order: each extraction places the current max
        // at position i, then reduces the heap. Result: sorted[0] = smallest distance.
        SearchResult[] sorted = new SearchResult[heapSize];
        for (int i = heapSize - 1; i >= 0; i--) {
            sorted[i] = new SearchResult(heapIds[0], heapDistances[0], heapClusterIds[0]);
            heapSize--;
            if (heapSize > 0) {
                // Move the last element to the root and sift down to restore heap property
                heapIds[0] = heapIds[heapSize];
                heapDistances[0] = heapDistances[heapSize];
                heapClusterIds[0] = heapClusterIds[heapSize];
                siftDown(heapDistances, heapIds, heapClusterIds, 0, heapSize);
            }
        }

        return Arrays.asList(sorted);
    }
}
