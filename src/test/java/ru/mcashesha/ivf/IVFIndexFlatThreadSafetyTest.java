package ru.mcashesha.ivf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Thread-safety tests for {@link IVFIndexFlat}.
 *
 * <p>Validates that:</p>
 * <ul>
 *   <li>Mutating the original vector array after {@code build()} does not corrupt the index
 *       (defensive copy verification).</li>
 *   <li>Concurrent search operations from multiple threads return identical, correct results
 *       (thread-safe read-only access).</li>
 * </ul>
 */
class IVFIndexFlatThreadSafetyTest {

    /** Fixed random seed for deterministic data generation and KMeans initialization. */
    private static final long SEED = 42L;

    /** Dimensionality of test vectors. */
    private static final int DIMENSION = 8;

    /**
     * Generates synthetic clustered data with well-separated centers.
     *
     * @param rng              random number generator
     * @param pointsPerCluster number of points in each cluster
     * @param clusterCount     number of clusters to generate
     * @param spread           distance between adjacent cluster centers along each axis
     * @return a flat array of {@code pointsPerCluster * clusterCount} vectors
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
     * Verifies that mutating the original vector array after build() does not affect
     * search results. After build(), vectors[0] is zeroed out; the index must still
     * return correct distances as if the original data were intact.
     */
    @Test
    void search_afterMutatingOriginalVectors_returnsCorrectDistances() {
        float[][] vectors = generateClusters(new Random(SEED), 50, 3, 100f);

        // Save a copy of vectors[0] before build, so we can verify distances against it
        float[] originalVector0 = Arrays.copyOf(vectors[0], vectors[0].length);

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(3).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(vectors);

        // Search before mutation to get reference results
        float[] query = Arrays.copyOf(originalVector0, originalVector0.length);
        List<IVFIndex.SearchResult> resultsBefore = index.search(query, 5, 3);

        // Mutate the original array: zero out vectors[0]
        Arrays.fill(vectors[0], 0.0f);

        // Search after mutation: results must be identical to before
        List<IVFIndex.SearchResult> resultsAfter = index.search(query, 5, 3);

        assertEquals(resultsBefore.size(), resultsAfter.size(),
            "Result count changed after mutating original vectors");

        for (int i = 0; i < resultsBefore.size(); i++) {
            IVFIndex.SearchResult before = resultsBefore.get(i);
            IVFIndex.SearchResult after = resultsAfter.get(i);
            assertEquals(before.id, after.id,
                "Result ID changed at position " + i + " after mutating original vectors");
            assertEquals(before.distance, after.distance, 1e-6f,
                "Result distance changed at position " + i + " after mutating original vectors");
        }

        // Additionally verify that querying with the original vector still finds it as
        // the exact match (distance ~ 0), confirming the index holds its own copy
        List<IVFIndex.SearchResult> exactMatch = index.search(query, 1, 3);
        assertEquals(1, exactMatch.size());
        assertEquals(0, exactMatch.get(0).id, "Expected ID 0 as the top result");
        assertEquals(0.0f, exactMatch.get(0).distance, 1e-6f,
            "Expected near-zero distance for the exact match");
    }

    /**
     * Verifies that concurrent search operations from 8 threads all return identical
     * results. Each thread executes the same query; all results must match exactly.
     */
    @Test
    void search_concurrentFromMultipleThreads_returnsIdenticalResults() throws Exception {
        float[][] vectors = generateClusters(new Random(SEED), 50, 3, 100f);

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(3).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(vectors);

        float[] query = vectors[10].clone();
        int topK = 10;
        int nProbe = 3;
        int threadCount = 8;

        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        try {
            // Submit identical search tasks from all threads
            List<Callable<List<IVFIndex.SearchResult>>> tasks = new ArrayList<>();
            for (int t = 0; t < threadCount; t++) {
                tasks.add(() -> index.search(query, topK, nProbe));
            }

            List<Future<List<IVFIndex.SearchResult>>> futures = executor.invokeAll(tasks);

            // Collect all results
            List<List<IVFIndex.SearchResult>> allResults = new ArrayList<>();
            for (Future<List<IVFIndex.SearchResult>> future : futures) {
                allResults.add(future.get());
            }

            // All results must be identical to the first thread's results
            List<IVFIndex.SearchResult> reference = allResults.get(0);
            assertFalse(reference.isEmpty(), "Reference result should not be empty");

            for (int t = 1; t < threadCount; t++) {
                List<IVFIndex.SearchResult> current = allResults.get(t);
                assertEquals(reference.size(), current.size(),
                    "Thread " + t + " returned different number of results");

                for (int i = 0; i < reference.size(); i++) {
                    IVFIndex.SearchResult ref = reference.get(i);
                    IVFIndex.SearchResult cur = current.get(i);
                    assertEquals(ref.id, cur.id,
                        "Thread " + t + " result ID mismatch at position " + i);
                    assertEquals(ref.distance, cur.distance, 1e-6f,
                        "Thread " + t + " result distance mismatch at position " + i);
                    assertEquals(ref.clusterId, cur.clusterId,
                        "Thread " + t + " result clusterId mismatch at position " + i);
                }
            }
        } finally {
            executor.shutdown();
        }
    }
}
