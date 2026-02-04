package ru.mcashesha.ivf;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;

class IVFIndexFlatTest {

    private static final long SEED = 42L;
    private static final int DIMENSION = 8;

    private static float[][] generateClusters(Random rng, int pointsPerCluster,
        int clusterCount, float spread) {
        float[][] data = new float[pointsPerCluster * clusterCount][DIMENSION];
        for (int c = 0; c < clusterCount; c++) {
            float[] center = new float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++)
                center[d] = c * spread;

            for (int p = 0; p < pointsPerCluster; p++) {
                int idx = c * pointsPerCluster + p;
                for (int d = 0; d < DIMENSION; d++)
                    data[idx][d] = center[d] + (rng.nextFloat() - 0.5f) * 0.1f;
            }
        }
        return data;
    }

    private IVFIndex buildIndex(float[][] data, int clusterCount, Metric.Type metricType) {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, metricType, Metric.Engine.SCALAR
        ).withClusterCount(clusterCount).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);
        return index;
    }

    @Test
    void build_setsProperties() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        assertEquals(DIMENSION, index.getDimension());
        assertEquals(3, index.getCountClusters());
        assertEquals(Metric.Type.L2SQ_DISTANCE, index.getMetricType());
        assertEquals(Metric.Engine.SCALAR, index.getMetricEngine());
    }

    @Test
    void search_returnsCorrectTopK() {
        float[][] data = generateClusters(new Random(SEED), 50, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[] query = data[0];
        List<IVFIndex.SearchResult> results = index.search(query, 5, 3);

        assertEquals(5, results.size());
    }

    @Test
    void search_resultsSortedByDistance() {
        float[][] data = generateClusters(new Random(SEED), 50, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[] query = data[0];
        List<IVFIndex.SearchResult> results = index.search(query, 10, 3);

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                "Results not sorted at position " + i + ": " +
                    results.get(i - 1).distance + " > " + results.get(i).distance);
        }
    }

    @Test
    void search_topKLargerThanData() {
        float[][] data = generateClusters(new Random(SEED), 5, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 100, 2);

        assertEquals(10, results.size());
    }

    @Test
    void search_topKLargerThanData_sortedCorrectly() {
        float[][] data = generateClusters(new Random(SEED), 5, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 100, 2);

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                "Results not sorted when heapSize < topK at position " + i + ": " +
                    results.get(i - 1).distance + " > " + results.get(i).distance);
        }
    }

    @Test
    void search_nProbeLargerThanClusters() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 100);

        assertEquals(5, results.size());
    }

    @Test
    void search_nProbeOne_returnsResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 3, 1);

        assertTrue(results.size() >= 1 && results.size() <= 3);
    }

    @Test
    void search_exactMatchIsFirst() {
        float[][] data = generateClusters(new Random(SEED), 50, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[] query = data[25].clone();
        List<IVFIndex.SearchResult> results = index.search(query, 1, 3);

        assertEquals(1, results.size());
        assertEquals(25, results.get(0).id);
        assertEquals(0f, results.get(0).distance, 1e-6f);
    }

    @Test
    void search_uniqueIds() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 20, 3);

        Set<Integer> ids = new HashSet<>();
        for (IVFIndex.SearchResult r : results)
            assertTrue(ids.add(r.id), "Duplicate id: " + r.id);
    }

    @Test
    void search_nonNegativeDistances() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 10, 3);

        for (IVFIndex.SearchResult r : results)
            assertTrue(r.distance >= 0, "Negative distance: " + r.distance);
    }

    @Test
    void build_withCustomIds() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        int[] customIds = new int[data.length];
        for (int i = 0; i < customIds.length; i++)
            customIds[i] = 1000 + i;

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data, customIds);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 2);

        for (IVFIndex.SearchResult r : results)
            assertTrue(r.id >= 1000 && r.id < 1000 + data.length,
                "Expected custom id range, got " + r.id);
    }

    @Test
    void search_beforeBuild_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        float[] query = new float[DIMENSION];

        assertThrows(IllegalStateException.class, () -> index.search(query, 5, 1));
    }

    @Test
    void build_nullVectors_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).build();

        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(null));
    }

    @Test
    void build_emptyVectors_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).build();

        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(new float[0][]));
    }

    @Test
    void search_wrongQueryDimension_throws() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        float[] wrongDim = new float[DIMENSION + 1];

        assertThrows(IllegalArgumentException.class, () -> index.search(wrongDim, 5, 1));
    }

    @ParameterizedTest
    @EnumSource(Metric.Type.class)
    void search_allMetricTypes(Metric.Type metricType) {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, metricType);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 3);

        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 5);
    }

    @Test
    void search_fullScan_matchesBruteForce() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 20, 3, 50f);
        int k = 3;
        IVFIndex index = buildIndex(data, k, Metric.Type.L2SQ_DISTANCE);

        float[] query = data[10];
        List<IVFIndex.SearchResult> results = index.search(query, 5, k);

        float[] bruteForceDistances = new float[data.length];
        for (int i = 0; i < data.length; i++)
            bruteForceDistances[i] = Metric.Type.L2SQ_DISTANCE.distance(
                Metric.Engine.SCALAR, query, data[i]);

        Integer[] sortedIndices = new Integer[data.length];
        for (int i = 0; i < data.length; i++)
            sortedIndices[i] = i;
        java.util.Arrays.sort(sortedIndices, (a, b) ->
            Float.compare(bruteForceDistances[a], bruteForceDistances[b]));

        for (int i = 0; i < results.size(); i++) {
            assertEquals(sortedIndices[i].intValue(), results.get(i).id,
                "Mismatch at position " + i);
        }
    }

    @Test
    void search_allKMeansTypes() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);

        for (KMeans.Type type : KMeans.Type.values()) {
            KMeans.Builder builder = KMeans.newBuilder(type, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withRandom(new Random(SEED));

            switch (type) {
                case LLOYD:
                    builder.withClusterCount(3).withMaxIterations(50);
                    break;
                case MINI_BATCH:
                    builder.withClusterCount(3).withBatchSize(32).withMaxIterations(100);
                    break;
                case HIERARCHICAL:
                    builder.withBranchFactor(2).withMaxDepth(3).withMaxIterationsPerLevel(30);
                    break;
            }

            IVFIndex index = new IVFIndexFlat(builder.build());
            index.build(data);

            List<IVFIndex.SearchResult> results = index.search(data[0], 5,
                index.getCountClusters());

            assertFalse(results.isEmpty(), type + ": no results");
            assertTrue(results.size() <= 5, type + ": too many results");

            for (int i = 1; i < results.size(); i++) {
                assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                    type + ": not sorted at " + i);
            }
        }
    }
}
