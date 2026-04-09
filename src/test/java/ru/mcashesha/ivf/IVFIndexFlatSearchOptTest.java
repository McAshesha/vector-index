package ru.mcashesha.ivf;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for search optimizations in {@link IVFIndexFlat}: cluster skipping,
 * empty cluster handling, and correctness with various nProbe/cluster configurations.
 */
class IVFIndexFlatSearchOptTest {

    private static final long SEED = 42L;
    private static final int DIMENSION = 8;

    /**
     * Generates synthetic clustered data with well-separated centers.
     */
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

    /**
     * With cluster skipping enabled, querying with data[i] as the query should
     * return id=i as the top-1 result with distance approximately zero.
     * Validates 100 random queries.
     */
    @Test
    void clusterSkipping_top1ExactMatch_returnsCorrectId() {
        Random rng = new Random(SEED);
        int pointsPerCluster = 50;
        int clusterCount = 10;
        float[][] data = generateClusters(rng, pointsPerCluster, clusterCount, 10.0f);

        KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withClusterCount(clusterCount)
                .withMaxIterations(20)
                .withRandom(new Random(SEED))
                .build();

        IVFIndexFlat index = new IVFIndexFlat(kmeans);
        index.build(data);

        Random queryRng = new Random(123L);
        int totalPoints = data.length;

        for (int q = 0; q < 100; q++) {
            int queryIdx = queryRng.nextInt(totalPoints);
            float[] query = data[queryIdx];

            List<IVFIndex.SearchResult> results = index.search(query, 1, 3);

            assertFalse(results.isEmpty(), "Query " + q + ": results should not be empty");
            IVFIndex.SearchResult top = results.get(0);
            assertEquals(queryIdx, top.id,
                    "Query " + q + ": top-1 result ID should match query index");
            assertTrue(top.distance < 1e-4f,
                    "Query " + q + ": top-1 distance should be ~0 but was " + top.distance);
        }
    }

    /**
     * When nProbe equals the total cluster count (exhaustive scan), cluster skipping
     * should not degrade results. Results must match a brute-force linear scan ordering.
     */
    @Test
    void fullNProbe_equivalentToBruteForce() {
        Random rng = new Random(SEED);
        int pointsPerCluster = 30;
        int clusterCount = 8;
        float[][] data = generateClusters(rng, pointsPerCluster, clusterCount, 10.0f);

        KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withClusterCount(clusterCount)
                .withMaxIterations(20)
                .withRandom(new Random(SEED))
                .build();

        IVFIndexFlat index = new IVFIndexFlat(kmeans);
        index.build(data);

        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        int topK = 5;

        Random queryRng = new Random(99L);
        for (int q = 0; q < 20; q++) {
            float[] query = data[queryRng.nextInt(data.length)];

            // IVF search with all clusters probed
            List<IVFIndex.SearchResult> ivfResults = index.search(query, topK, clusterCount);

            // Brute-force: compute all distances and find top-K
            float[] allDistances = new float[data.length];
            int[] allIds = new int[data.length];
            for (int i = 0; i < data.length; i++) {
                allDistances[i] = distFn.compute(query, data[i]);
                allIds[i] = i;
            }

            // Simple selection sort for top-K
            for (int i = 0; i < topK && i < data.length; i++) {
                int minIdx = i;
                for (int j = i + 1; j < data.length; j++) {
                    if (allDistances[j] < allDistances[minIdx])
                        minIdx = j;
                }
                float tmpD = allDistances[i]; allDistances[i] = allDistances[minIdx]; allDistances[minIdx] = tmpD;
                int tmpId = allIds[i]; allIds[i] = allIds[minIdx]; allIds[minIdx] = tmpId;
            }

            assertEquals(topK, ivfResults.size(),
                    "Query " + q + ": should return exactly topK results");

            for (int i = 0; i < topK; i++) {
                assertEquals(allIds[i], ivfResults.get(i).id,
                        "Query " + q + ", rank " + i + ": IVF result ID should match brute-force");
                assertEquals(allDistances[i], ivfResults.get(i).distance, 1e-5f,
                        "Query " + q + ", rank " + i + ": IVF distance should match brute-force");
            }
        }
    }

    /**
     * Small number of clusters (k=4, nProbe=2) works correctly with cluster skipping.
     */
    @Test
    void smallClusterCount_searchWorks() {
        Random rng = new Random(SEED);
        int pointsPerCluster = 25;
        int clusterCount = 4;
        float[][] data = generateClusters(rng, pointsPerCluster, clusterCount, 10.0f);

        KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withClusterCount(clusterCount)
                .withMaxIterations(20)
                .withRandom(new Random(SEED))
                .build();

        IVFIndexFlat index = new IVFIndexFlat(kmeans);
        index.build(data);

        int topK = 3;
        int nProbe = 2;

        // Query with each data point and verify basic result properties
        for (int i = 0; i < data.length; i++) {
            List<IVFIndex.SearchResult> results = index.search(data[i], topK, nProbe);

            assertFalse(results.isEmpty(), "Point " + i + ": results should not be empty");
            assertTrue(results.size() <= topK,
                    "Point " + i + ": results size should not exceed topK");

            // Results should be sorted by ascending distance
            for (int r = 1; r < results.size(); r++) {
                assertTrue(results.get(r).distance >= results.get(r - 1).distance,
                        "Point " + i + ": results should be sorted by ascending distance");
            }

            // Top-1 should be the point itself (exact match) with distance ~0
            IVFIndex.SearchResult top = results.get(0);
            assertEquals(i, top.id,
                    "Point " + i + ": top-1 ID should match query point index");
            assertTrue(top.distance < 1e-4f,
                    "Point " + i + ": top-1 distance should be ~0 but was " + top.distance);
        }
    }
}
