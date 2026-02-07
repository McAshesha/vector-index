package ru.mcashesha.ivf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

public class IVFIndexFlat implements IVFIndex {
    private final KMeans<? extends KMeans.ClusteringResult> kMeans;

    private float[][] centroids;
    private int[] clusterOffsets;

    private float[][] data;
    private int[] ids;
    private int dimension;
    private boolean built;

    private Metric.DistanceFunction distFn;

    private static final int PARALLEL_CENTROID_THRESHOLD = 32;
    private static final int PARALLEL_CLUSTER_SCAN_THRESHOLD = 2000;

    public IVFIndexFlat(KMeans<? extends KMeans.ClusteringResult> kMeans) {
        if (kMeans == null)
            throw new IllegalArgumentException("kMeans must be non-null");

        this.kMeans = kMeans;
    }

    @Override public void build(float[][] vectors) {
        build(vectors, null);
    }

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

        this.clusterOffsets = new int[clusterCnt + 1];
        clusterOffsets[0] = 0;
        for (int c = 0; c < clusterCnt; c++)
            clusterOffsets[c + 1] = clusterOffsets[c] + sizes[c];

        float[][] reorderedData = new float[vectors.length][];
        int[] reorderedIds = new int[vectors.length];

        int n = assignments.length;

        if (n >= 10000) {
            // Parallel reordering using atomic counters
            // Pass 1: Compute target positions using atomic increments
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

            // Pass 2: Parallel scatter to target positions
            IntStream.range(0, n).parallel().forEach(i -> {
                int pos = targetPos[i];
                if (pos >= 0) {
                    reorderedData[pos] = vectors[i];
                    reorderedIds[pos] = originalIds[i];
                }
            });
        } else {
            // Sequential path for small datasets
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

        this.distFn = kMeans.getMetricType().resolveFunction(kMeans.getMetricEngine());

        this.built = true;
    }

    @Override public Metric.Type getMetricType() {
        return kMeans.getMetricType();
    }

    @Override public Metric.Engine getMetricEngine() {
        return kMeans.getMetricEngine();
    }

    @Override public int getCountClusters() {
        return centroids.length;
    }

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
     * Selects nProbe nearest clusters using a max-heap.
     * Time complexity: O(k + nProbe * log(nProbe)) instead of O(nProbe * k) for selection sort.
     * Result: clusterOrder[0..nProbe-1] contains indices of nProbe nearest clusters, sorted by distance.
     */
    private static void selectNearestClusters(float[] distances, int[] clusterOrder, int clusterCnt, int nProbe) {
        // Initialize clusterOrder with first nProbe clusters
        for (int i = 0; i < nProbe; i++)
            clusterOrder[i] = i;

        // Build max-heap from first nProbe clusters
        for (int i = nProbe / 2 - 1; i >= 0; i--)
            siftDownCluster(distances, clusterOrder, i, nProbe);

        // Process remaining clusters: if closer than max in heap, replace and sift
        for (int c = nProbe; c < clusterCnt; c++) {
            if (distances[c] < distances[clusterOrder[0]]) {
                clusterOrder[0] = c;
                siftDownCluster(distances, clusterOrder, 0, nProbe);
            }
        }

        // Extract from heap to get sorted order (ascending by distance)
        // After this, clusterOrder[0] = nearest, clusterOrder[nProbe-1] = farthest among selected
        for (int i = nProbe - 1; i > 0; i--) {
            int tmp = clusterOrder[0];
            clusterOrder[0] = clusterOrder[i];
            clusterOrder[i] = tmp;
            siftDownCluster(distances, clusterOrder, 0, i);
        }
    }

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

            int tmp = clusterOrder[i];
            clusterOrder[i] = clusterOrder[largest];
            clusterOrder[largest] = tmp;
            i = largest;
        }
    }

    private static void buildMaxHeap(float[] dist, int[] ids, int[] clusterIds, int size) {
        for (int i = size / 2 - 1; i >= 0; i--)
            siftDown(dist, ids, clusterIds, i, size);
    }

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

            float tmpD = dist[i]; dist[i] = dist[largest]; dist[largest] = tmpD;
            int tmpId = ids[i]; ids[i] = ids[largest]; ids[largest] = tmpId;
            int tmpC = clusterIds[i]; clusterIds[i] = clusterIds[largest]; clusterIds[largest] = tmpC;

            i = largest;
        }
    }

    @Override public int getDimension() {
        return dimension;
    }

    /**
     * Batch search for multiple queries in parallel.
     * Each query is processed independently, allowing full parallelization.
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

        // Parallel search for all queries
        @SuppressWarnings("unchecked")
        List<SearchResult>[] results = new List[numQueries];

        IntStream.range(0, numQueries).parallel().forEach(q ->
            results[q] = searchInternal(queries[q], topK, nProbe));

        return Arrays.asList(results);
    }

    /**
     * Internal search method that doesn't use shared buffers (thread-safe).
     */
    private List<SearchResult> searchInternal(float[] qry, int topK, int nProbe) {
        int clusterCnt = centroids.length;
        if (clusterCnt == 0)
            return List.of();

        nProbe = Math.max(1, Math.min(nProbe, clusterCnt));

        Metric.DistanceFunction fn = this.distFn;

        // Thread-local buffers
        float[] localCentroidDistances = new float[clusterCnt];
        int[] localClusterOrder = new int[clusterCnt];

        // Compute centroid distances - parallelize for large cluster counts
        if (clusterCnt >= PARALLEL_CENTROID_THRESHOLD) {
            IntStream.range(0, clusterCnt).parallel().forEach(c ->
                localCentroidDistances[c] = fn.compute(qry, centroids[c]));
        } else {
            for (int c = 0; c < clusterCnt; c++)
                localCentroidDistances[c] = fn.compute(qry, centroids[c]);
        }

        // Select nearest clusters
        for (int c = 0; c < clusterCnt; c++)
            localClusterOrder[c] = c;
        selectNearestClusters(localCentroidDistances, localClusterOrder, clusterCnt, nProbe);

        // Calculate total points to scan for parallel decision
        int totalPointsToScan = 0;
        for (int p = 0; p < nProbe; p++) {
            int clusterId = localClusterOrder[p];
            totalPointsToScan += clusterOffsets[clusterId + 1] - clusterOffsets[clusterId];
        }

        // Use parallel cluster scanning for large workloads
        if (totalPointsToScan >= PARALLEL_CLUSTER_SCAN_THRESHOLD && nProbe >= 2) {
            return searchClustersParallel(qry, topK, nProbe, localClusterOrder, fn);
        }

        // Sequential cluster scanning
        return searchClustersSequential(qry, topK, nProbe, localClusterOrder, fn);
    }

    private List<SearchResult> searchClustersSequential(float[] qry, int topK, int nProbe,
                                                         int[] localClusterOrder, Metric.DistanceFunction fn) {
        int[] heapIds = new int[topK];
        float[] heapDistances = new float[topK];
        int[] heapClusterIds = new int[topK];
        int heapSize = 0;

        for (int p = 0; p < nProbe; p++) {
            int clusterId = localClusterOrder[p];
            int start = clusterOffsets[clusterId];
            int end = clusterOffsets[clusterId + 1];

            for (int i = start; i < end; i++) {
                float d = fn.compute(qry, data[i]);

                if (heapSize < topK) {
                    heapIds[heapSize] = ids[i];
                    heapDistances[heapSize] = d;
                    heapClusterIds[heapSize] = clusterId;
                    heapSize++;
                    if (heapSize == topK)
                        buildMaxHeap(heapDistances, heapIds, heapClusterIds, heapSize);
                }
                else if (d < heapDistances[0]) {
                    heapIds[0] = ids[i];
                    heapDistances[0] = d;
                    heapClusterIds[0] = clusterId;
                    siftDown(heapDistances, heapIds, heapClusterIds, 0, heapSize);
                }
            }
        }

        return extractSortedResults(heapIds, heapDistances, heapClusterIds, heapSize, topK);
    }

    private List<SearchResult> searchClustersParallel(float[] qry, int topK, int nProbe,
                                                       int[] localClusterOrder, Metric.DistanceFunction fn) {
        // Each cluster builds its own local top-K heap
        int[][] localHeapIds = new int[nProbe][topK];
        float[][] localHeapDistances = new float[nProbe][topK];
        int[][] localHeapClusterIds = new int[nProbe][topK];
        int[] localHeapSizes = new int[nProbe];

        // Parallel per-cluster scanning
        IntStream.range(0, nProbe).parallel().forEach(p -> {
            int clusterId = localClusterOrder[p];
            int start = clusterOffsets[clusterId];
            int end = clusterOffsets[clusterId + 1];

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

        // Merge all per-cluster heaps into final top-K
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

    private List<SearchResult> extractSortedResults(int[] heapIds, float[] heapDistances,
                                                     int[] heapClusterIds, int heapSize, int topK) {
        if (heapSize > 0 && heapSize < topK)
            buildMaxHeap(heapDistances, heapIds, heapClusterIds, heapSize);

        SearchResult[] sorted = new SearchResult[heapSize];
        for (int i = heapSize - 1; i >= 0; i--) {
            sorted[i] = new SearchResult(heapIds[0], heapDistances[0], heapClusterIds[0]);
            heapSize--;
            if (heapSize > 0) {
                heapIds[0] = heapIds[heapSize];
                heapDistances[0] = heapDistances[heapSize];
                heapClusterIds[0] = heapClusterIds[heapSize];
                siftDown(heapDistances, heapIds, heapClusterIds, 0, heapSize);
            }
        }

        return Arrays.asList(sorted);
    }
}
