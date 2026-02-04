package ru.mcashesha.ivf;

import java.util.Arrays;
import java.util.List;
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
    private float[] centroidDistances;
    private int[] clusterOrder;

    private int[] heapIdsBuf;
    private float[] heapDistancesBuf;
    private int[] heapClusterIdsBuf;
    private static final int DEFAULT_HEAP_BUF_SIZE = 256;

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
        int[] writePos = new int[clusterCnt];
        System.arraycopy(clusterOffsets, 0, writePos, 0, clusterCnt);

        for (int i = 0; i < assignments.length; i++) {
            int c = assignments[i];
            if (c < 0 || c >= clusterCnt)
                continue;
            int pos = writePos[c]++;
            reorderedData[pos] = vectors[i];
            reorderedIds[pos] = originalIds[i];
        }

        this.data = reorderedData;
        this.ids = reorderedIds;

        this.distFn = kMeans.getMetricType().resolveFunction(kMeans.getMetricEngine());
        this.centroidDistances = new float[clusterCnt];
        this.clusterOrder = new int[clusterCnt];
        for (int c = 0; c < clusterCnt; c++)
            clusterOrder[c] = c;

        this.heapIdsBuf = new int[DEFAULT_HEAP_BUF_SIZE];
        this.heapDistancesBuf = new float[DEFAULT_HEAP_BUF_SIZE];
        this.heapClusterIdsBuf = new int[DEFAULT_HEAP_BUF_SIZE];

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

        int clusterCnt = centroids.length;
        if (clusterCnt == 0)
            return List.of();

        nProbe = Math.max(1, Math.min(nProbe, clusterCnt));

        Metric.DistanceFunction fn = this.distFn;

        for (int c = 0; c < clusterCnt; c++)
            centroidDistances[c] = fn.compute(qry, centroids[c]);

        for (int i = 0; i < nProbe; i++) {
            int minIdx = i;
            for (int j = i + 1; j < clusterCnt; j++) {
                if (centroidDistances[clusterOrder[j]] < centroidDistances[clusterOrder[minIdx]])
                    minIdx = j;
            }
            int tmp = clusterOrder[i];
            clusterOrder[i] = clusterOrder[minIdx];
            clusterOrder[minIdx] = tmp;
        }

        int[] heapIds;
        float[] heapDistances;
        int[] heapClusterIds;
        if (topK <= heapIdsBuf.length) {
            heapIds = heapIdsBuf;
            heapDistances = heapDistancesBuf;
            heapClusterIds = heapClusterIdsBuf;
        } else {
            heapIds = new int[topK];
            heapDistances = new float[topK];
            heapClusterIds = new int[topK];
        }
        int heapSize = 0;

        for (int p = 0; p < nProbe; p++) {
            int clusterId = clusterOrder[p];
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

        for (int c = 0; c < clusterCnt; c++)
            clusterOrder[c] = c;

        return Arrays.asList(sorted);
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
}
