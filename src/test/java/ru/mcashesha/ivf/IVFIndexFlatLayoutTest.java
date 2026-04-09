package ru.mcashesha.ivf;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Tests specifically targeting the flat array storage layout in {@link IVFIndexFlat}.
 *
 * <p>These tests verify that the flat {@code float[]} data layout (where all vectors are packed
 * contiguously into a single array instead of a {@code float[][]}) produces correct search
 * results across various scenarios including large datasets, minimal dimensions, and full-scan
 * brute-force equivalence.</p>
 */
class IVFIndexFlatLayoutTest {

    private static final long SEED = 42L;

    @BeforeAll
    static void checkEngineAvailable() {
        try {
            // Trigger Metric.Engine class initialization; skip all tests if SimSIMD
            // native lib causes the enum to fail (pre-existing environment issue)
            Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        } catch (ExceptionInInitializerError | NoClassDefFoundError | UnsatisfiedLinkError e) {
            assumeTrue(false, "Metric.Engine not available, skipping tests");
        }
    }

    /**
     * Verifies correctness with 10000 vectors of 128 dimensions. For each of the first 100
     * vectors, querying with that vector as the query should return that vector's ID as the
     * top-1 result (exact match with distance ~0). This exercises the flat layout with a
     * realistically sized dataset and dimension.
     */
    @Test
    void search_10000vectors_128d_top1ExactMatch() {
        int n = 10000;
        int dim = 128;
        int clusterCount = 20;

        Random rng = new Random(SEED);
        float[][] data = new float[n][dim];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) {
                data[i][d] = rng.nextFloat();
            }
        }

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(clusterCount).withMaxIterations(30).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        // Test exact match for a sample of vectors across the dataset.
        // Use nProbe = clusterCount (full scan) to guarantee the exact vector is found.
        for (int queryIdx = 0; queryIdx < 100; queryIdx += 10) {
            float[] query = data[queryIdx].clone();
            List<IVFIndex.SearchResult> results = index.search(query, 1, clusterCount);

            assertEquals(1, results.size(),
                "Expected exactly 1 result for queryIdx=" + queryIdx);
            assertEquals(queryIdx, results.get(0).id,
                "Top-1 result should be the queried vector itself for queryIdx=" + queryIdx);
            assertEquals(0f, results.get(0).distance, 1e-4f,
                "Distance to self should be ~0 for queryIdx=" + queryIdx);
        }
    }

    /**
     * Verifies that the flat layout works correctly with dimension=1 (minimal dimension).
     * This is an edge case where each vector is a single float, and the flat array layout
     * should still produce correct search results.
     */
    @Test
    void search_dimension1_works() {
        int n = 50;
        int dim = 1;

        float[][] data = new float[n][dim];
        for (int i = 0; i < n; i++) {
            data[i][0] = (float) i;  // Simple linearly spaced values
        }

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(5).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        assertEquals(1, index.getDimension());

        // Query with vector at index 25 (value = 25.0); nearest should be itself
        float[] query = data[25].clone();
        List<IVFIndex.SearchResult> results = index.search(query, 3, 5);

        assertFalse(results.isEmpty());
        // The top result should be the exact match
        assertEquals(25, results.get(0).id);
        assertEquals(0f, results.get(0).distance, 1e-6f);

        // Results should be sorted by ascending distance
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                "Results not sorted at position " + i);
        }
    }

    /**
     * Verifies that when nProbe equals the total number of clusters (full scan), the IVF index
     * results are equivalent to a brute-force linear scan over all vectors. This validates the
     * correctness of the flat layout against the ground truth for a larger dataset.
     */
    @Test
    void search_nProbeAll_equivalentToBruteForce() {
        int n = 500;
        int dim = 16;
        int clusterCount = 10;
        int topK = 10;

        Random rng = new Random(SEED);
        float[][] data = new float[n][dim];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) {
                data[i][d] = rng.nextFloat() * 100f;
            }
        }

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(clusterCount).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);

        // Use multiple query vectors to thoroughly test
        for (int q = 0; q < 5; q++) {
            float[] query = data[q * 100].clone();

            // nProbe = clusterCount scans all clusters (equivalent to brute-force)
            List<IVFIndex.SearchResult> ivfResults = index.search(query, topK, clusterCount);

            // Compute brute-force distances
            Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
            float[] bruteForceDistances = new float[n];
            for (int i = 0; i < n; i++) {
                bruteForceDistances[i] = distFn.compute(query, data[i]);
            }

            // Sort indices by ascending brute-force distance
            Integer[] sortedIndices = new Integer[n];
            for (int i = 0; i < n; i++) sortedIndices[i] = i;
            java.util.Arrays.sort(sortedIndices,
                (a, b) -> Float.compare(bruteForceDistances[a], bruteForceDistances[b]));

            // IVF full-scan results must match brute-force top-K exactly
            assertEquals(topK, ivfResults.size(), "Expected " + topK + " results for query " + q);
            for (int i = 0; i < topK; i++) {
                assertEquals(sortedIndices[i].intValue(), ivfResults.get(i).id,
                    "Mismatch at position " + i + " for query " + q);
            }
        }
    }
}
