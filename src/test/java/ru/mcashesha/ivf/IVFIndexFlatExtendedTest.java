package ru.mcashesha.ivf;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

class IVFIndexFlatExtendedTest {

    private static final long SEED = 42L;
    private static final int DIMENSION = 8;

    @BeforeAll
    static void checkNativeLibAvailable() {
        try {
            Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        } catch (ExceptionInInitializerError | NoClassDefFoundError | UnsatisfiedLinkError e) {
            assumeTrue(false, "Native library not available, skipping tests");
        }
    }

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

    // ==================== Batch search ====================

    @Test
    void searchBatch_returnsResultsForAllQueries() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] queries = {data[0], data[30], data[60]};
        List<List<IVFIndex.SearchResult>> batchResults = index.searchBatch(queries, 5, 3);

        assertEquals(3, batchResults.size());
        for (List<IVFIndex.SearchResult> results : batchResults) {
            assertFalse(results.isEmpty());
            assertTrue(results.size() <= 5);
        }
    }

    @Test
    void searchBatch_eachQuerySorted() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] queries = {data[0], data[30], data[60]};
        List<List<IVFIndex.SearchResult>> batchResults = index.searchBatch(queries, 10, 3);

        for (int q = 0; q < queries.length; q++) {
            List<IVFIndex.SearchResult> results = batchResults.get(q);
            for (int i = 1; i < results.size(); i++) {
                assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                    "Query " + q + " results not sorted at position " + i);
            }
        }
    }

    @Test
    void searchBatch_matchesSingleSearchResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] queries = {data[0], data[45]};
        List<List<IVFIndex.SearchResult>> batchResults = index.searchBatch(queries, 5, 3);

        for (int q = 0; q < queries.length; q++) {
            List<IVFIndex.SearchResult> singleResults = index.search(queries[q], 5, 3);
            List<IVFIndex.SearchResult> batchResult = batchResults.get(q);
            assertEquals(singleResults.size(), batchResult.size(),
                "Query " + q + ": batch and single search should return same count");
            for (int i = 0; i < singleResults.size(); i++) {
                assertEquals(singleResults.get(i).id, batchResult.get(i).id,
                    "Query " + q + " position " + i + ": IDs should match");
            }
        }
    }

    @Test
    void searchBatch_singleQuery() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] queries = {data[10]};
        List<List<IVFIndex.SearchResult>> batchResults = index.searchBatch(queries, 5, 3);

        assertEquals(1, batchResults.size());
        assertFalse(batchResults.get(0).isEmpty());
    }

    @Test
    void searchBatch_beforeBuild_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withRandom(new Random(SEED)).build();
        IVFIndex index = new IVFIndexFlat(kmeans);

        float[][] queries = {new float[DIMENSION]};
        assertThrows(IllegalStateException.class, () -> index.searchBatch(queries, 5, 1));
    }

    @Test
    void searchBatch_nullQueries_throws() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(null, 5, 3));
    }

    @Test
    void searchBatch_emptyQueries_throws() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(new float[0][], 5, 3));
    }

    @Test
    void searchBatch_wrongDimension_throws() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] badQueries = {new float[DIMENSION + 1]};
        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(badQueries, 5, 3));
    }

    @Test
    void searchBatch_zeroTopK_throws() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        float[][] queries = {data[0]};
        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(queries, 0, 3));
    }

    // ==================== Constructor validation ====================

    @Test
    void constructor_nullKMeans_throws() {
        assertThrows(IllegalArgumentException.class, () -> new IVFIndexFlat(null));
    }

    // ==================== Build error handling ====================

    @Test
    void build_idsLengthMismatch_throws() {
        float[][] data = generateClusters(new Random(SEED), 10, 2, 50f);
        int[] badIds = new int[data.length + 5]; // wrong length

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withMaxIterations(50).withRandom(new Random(SEED)).build();
        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(data, badIds));
    }

    @Test
    void build_nullFirstVector_throws() {
        float[][] data = new float[5][];
        data[0] = null;
        for (int i = 1; i < 5; i++) data[i] = new float[DIMENSION];

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).build();
        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(data));
    }

    @Test
    void build_zeroDimensionVector_throws() {
        float[][] data = {new float[0]};

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(1).build();
        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(data));
    }

    @Test
    void build_inconsistentDimensions_throws() {
        float[][] data = {new float[DIMENSION], new float[DIMENSION + 1]};

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(1).build();
        IVFIndex index = new IVFIndexFlat(kmeans);

        assertThrows(IllegalArgumentException.class, () -> index.build(data));
    }

    // ==================== Search error handling ====================

    @Test
    void search_nullQuery_throws() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        assertThrows(IllegalArgumentException.class, () -> index.search(null, 5, 1));
    }

    @Test
    void search_zeroTopK_throws() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        assertThrows(IllegalArgumentException.class, () -> index.search(data[0], 0, 1));
    }

    @Test
    void search_negativeTopK_throws() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        assertThrows(IllegalArgumentException.class, () -> index.search(data[0], -1, 1));
    }

    // ==================== Build with convenience method ====================

    @Test
    void build_withoutIds_usesSequentialIds() {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        IVFIndex index = buildIndex(data, 2, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 40, 2);
        Set<Integer> ids = new HashSet<>();
        for (IVFIndex.SearchResult r : results) ids.add(r.id);

        // Sequential IDs should be in range [0, 40)
        for (int id : ids)
            assertTrue(id >= 0 && id < 40, "ID " + id + " out of range");
    }

    // ==================== Dot product metric ====================

    @Test
    void search_dotProduct_returnsResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.DOT_PRODUCT);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 3);

        assertFalse(results.isEmpty());
        // DOT_PRODUCT distances are negated, so lower (more negative) = more similar
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                "Dot product results not sorted at " + i);
        }
    }

    // ==================== Cosine distance metric ====================

    @Test
    void search_cosineDistance_returnsResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 50f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.COSINE_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 3);

        assertFalse(results.isEmpty());
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                "Cosine results not sorted at " + i);
        }
    }

    // ==================== Build with all metric types and custom IDs ====================

    @ParameterizedTest
    @EnumSource(Metric.Type.class)
    void build_withCustomIds_allMetrics(Metric.Type metricType) {
        float[][] data = generateClusters(new Random(SEED), 20, 2, 50f);
        int[] customIds = new int[data.length];
        for (int i = 0; i < customIds.length; i++)
            customIds[i] = 5000 + i;

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, metricType, Metric.Engine.SCALAR
        ).withClusterCount(2).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data, customIds);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 2);
        assertFalse(results.isEmpty());
        for (IVFIndex.SearchResult r : results)
            assertTrue(r.id >= 5000 && r.id < 5000 + data.length, "Custom ID out of range: " + r.id);
    }

    // ==================== SearchResult properties ====================

    @Test
    void searchResult_hasValidClusterId() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 10, 3);

        for (IVFIndex.SearchResult r : results)
            assertTrue(r.clusterId >= 0 && r.clusterId < 3,
                "Cluster ID " + r.clusterId + " out of range");
    }

    @Test
    void searchResult_exactMatch_zeroDistance() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        for (int queryIdx : new int[]{0, 30, 60}) {
            float[] query = data[queryIdx].clone();
            List<IVFIndex.SearchResult> results = index.search(query, 1, 3);
            assertEquals(queryIdx, results.get(0).id);
            assertEquals(0f, results.get(0).distance, 1e-6f);
        }
    }

    // ==================== MiniBatch and Hierarchical KMeans via IVF ====================

    @Test
    void search_miniBatchKMeans_returnsResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(3).withBatchSize(32).withMaxIterations(100).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, index.getCountClusters());
        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 5);
    }

    @Test
    void search_hierarchicalKMeans_returnsResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(2).withMaxDepth(3).withMaxIterationsPerLevel(30)
            .withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        int totalClusters = index.getCountClusters();
        assertTrue(totalClusters > 0);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, totalClusters);
        assertFalse(results.isEmpty());
    }

    // ==================== VectorAPI engine ====================

    @Test
    void search_vectorAPIEngine_producesResults() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API
        ).withClusterCount(3).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        assertEquals(Metric.Engine.VECTOR_API, index.getMetricEngine());

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 3);
        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 5);
    }

    // ==================== Large nProbe ====================

    @Test
    void search_nProbeZero_clampedToOne() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        // nProbe=0 should be clamped to at least 1
        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 0);
        assertFalse(results.isEmpty());
    }

    @Test
    void search_nProbeNegative_clampedToOne() {
        float[][] data = generateClusters(new Random(SEED), 30, 3, 100f);
        IVFIndex index = buildIndex(data, 3, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, -1);
        assertFalse(results.isEmpty());
    }

    // ==================== Single cluster ====================

    @Test
    void search_singleCluster_returnsAllIfTopKLarge() {
        float[][] data = generateClusters(new Random(SEED), 10, 1, 0f);
        IVFIndex index = buildIndex(data, 1, Metric.Type.L2SQ_DISTANCE);

        assertEquals(1, index.getCountClusters());

        List<IVFIndex.SearchResult> results = index.search(data[0], 100, 1);
        assertEquals(10, results.size());
    }

    // ==================== Many clusters ====================

    @Test
    void search_manySmallClusters() {
        Random rng = new Random(SEED);
        // 5 points per cluster, 10 clusters = 50 vectors
        float[][] data = generateClusters(rng, 5, 10, 100f);
        IVFIndex index = buildIndex(data, 10, Metric.Type.L2SQ_DISTANCE);

        List<IVFIndex.SearchResult> results = index.search(data[0], 5, 10);
        assertEquals(5, results.size());

        // All results from the same cluster neighborhood
        for (int i = 1; i < results.size(); i++)
            assertTrue(results.get(i).distance >= results.get(i - 1).distance);
    }
}
