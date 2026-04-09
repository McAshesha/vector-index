package ru.mcashesha.ivf;

import java.util.ArrayList;
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
 * Tests that thread-local buffer reuse in {@link IVFIndexFlat} does not corrupt
 * search results. Verifies both sequential stability (same results across 1000
 * repeated searches) and concurrent correctness (8 threads searching in parallel).
 */
class IVFIndexFlatBufferReuseTest {

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
     * Builds a default IVF index on clustered data for reuse across tests.
     */
    private IVFIndexFlat buildIndex(float[][] data) {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(4).withMaxIterations(50).withRandom(new Random(SEED)).build();

        IVFIndexFlat index = new IVFIndexFlat(kmeans);
        index.build(data);
        return index;
    }

    /**
     * Verifies that 1000 consecutive searches on the same index and query return
     * identical results every time. This ensures that thread-local buffer reuse
     * does not leave stale data that corrupts subsequent queries.
     */
    @Test
    void search_1000SequentialRepetitions_resultsAreStable() {
        float[][] data = generateClusters(new Random(SEED), 50, 4, 100f);
        IVFIndexFlat index = buildIndex(data);

        float[] query = data[0].clone();
        int topK = 5;
        int nProbe = 4;

        // Obtain the reference result from the first search
        List<IVFIndex.SearchResult> reference = index.search(query, topK, nProbe);
        assertFalse(reference.isEmpty(), "Reference result should not be empty");

        // Repeat 999 more times (1000 total) and verify every result matches the reference
        for (int iteration = 1; iteration < 1000; iteration++) {
            List<IVFIndex.SearchResult> current = index.search(query, topK, nProbe);
            assertEquals(reference.size(), current.size(),
                "Iteration " + iteration + ": result count changed");

            for (int i = 0; i < reference.size(); i++) {
                IVFIndex.SearchResult ref = reference.get(i);
                IVFIndex.SearchResult cur = current.get(i);
                assertEquals(ref.id, cur.id,
                    "Iteration " + iteration + ": ID mismatch at position " + i);
                assertEquals(ref.distance, cur.distance, 1e-6f,
                    "Iteration " + iteration + ": distance mismatch at position " + i);
                assertEquals(ref.clusterId, cur.clusterId,
                    "Iteration " + iteration + ": clusterId mismatch at position " + i);
            }
        }
    }

    /**
     * Verifies that 1000 sequential searches with different queries all produce
     * valid results. This exercises the buffer reuse path with varying topK needs
     * and ensures buffers are correctly reused across different query vectors.
     */
    @Test
    void search_1000SequentialDifferentQueries_allResultsValid() {
        float[][] data = generateClusters(new Random(SEED), 50, 4, 100f);
        IVFIndexFlat index = buildIndex(data);

        int topK = 5;
        int nProbe = 4;

        for (int iteration = 0; iteration < 1000; iteration++) {
            // Use each data point (cycling) as a query
            float[] query = data[iteration % data.length].clone();
            List<IVFIndex.SearchResult> results = index.search(query, topK, nProbe);

            assertFalse(results.isEmpty(),
                "Iteration " + iteration + ": results should not be empty");
            assertTrue(results.size() <= topK,
                "Iteration " + iteration + ": too many results");

            // Results must be sorted by ascending distance
            for (int i = 1; i < results.size(); i++) {
                assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                    "Iteration " + iteration + ": results not sorted at position " + i);
            }

            // All distances must be non-negative
            for (IVFIndex.SearchResult r : results) {
                assertTrue(r.distance >= 0,
                    "Iteration " + iteration + ": negative distance " + r.distance);
            }
        }
    }

    /**
     * Verifies that concurrent search from 8 threads produces correct results when
     * thread-local buffers are being reused. Each thread runs many searches;
     * all results for the same query must match the single-threaded reference.
     */
    @Test
    void search_concurrentFrom8Threads_resultsAreCorrect() throws Exception {
        float[][] data = generateClusters(new Random(SEED), 50, 4, 100f);
        IVFIndexFlat index = buildIndex(data);

        float[] query = data[10].clone();
        int topK = 10;
        int nProbe = 4;
        int threadCount = 8;
        int searchesPerThread = 100;

        // Compute the reference result on the main thread
        List<IVFIndex.SearchResult> reference = index.search(query, topK, nProbe);
        assertFalse(reference.isEmpty(), "Reference result should not be empty");

        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        try {
            List<Callable<List<List<IVFIndex.SearchResult>>>> tasks = new ArrayList<>();

            for (int t = 0; t < threadCount; t++) {
                tasks.add(() -> {
                    List<List<IVFIndex.SearchResult>> threadResults = new ArrayList<>();
                    for (int s = 0; s < searchesPerThread; s++) {
                        threadResults.add(index.search(query, topK, nProbe));
                    }
                    return threadResults;
                });
            }

            List<Future<List<List<IVFIndex.SearchResult>>>> futures = executor.invokeAll(tasks);

            // Verify every result from every thread matches the reference
            for (int t = 0; t < threadCount; t++) {
                List<List<IVFIndex.SearchResult>> threadResults = futures.get(t).get();
                assertEquals(searchesPerThread, threadResults.size(),
                    "Thread " + t + ": unexpected number of search results");

                for (int s = 0; s < searchesPerThread; s++) {
                    List<IVFIndex.SearchResult> current = threadResults.get(s);
                    assertEquals(reference.size(), current.size(),
                        "Thread " + t + " search " + s + ": result count mismatch");

                    for (int i = 0; i < reference.size(); i++) {
                        IVFIndex.SearchResult ref = reference.get(i);
                        IVFIndex.SearchResult cur = current.get(i);
                        assertEquals(ref.id, cur.id,
                            "Thread " + t + " search " + s + ": ID mismatch at position " + i);
                        assertEquals(ref.distance, cur.distance, 1e-6f,
                            "Thread " + t + " search " + s + ": distance mismatch at position " + i);
                        assertEquals(ref.clusterId, cur.clusterId,
                            "Thread " + t + " search " + s + ": clusterId mismatch at position " + i);
                    }
                }
            }
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Verifies concurrent search with different queries from 8 threads.
     * Each thread uses a different query vector to stress buffer reuse with
     * varying access patterns.
     */
    @Test
    void search_concurrentDifferentQueries_allResultsValid() throws Exception {
        float[][] data = generateClusters(new Random(SEED), 50, 4, 100f);
        IVFIndexFlat index = buildIndex(data);

        int topK = 5;
        int nProbe = 4;
        int threadCount = 8;
        int searchesPerThread = 100;

        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        try {
            List<Callable<Void>> tasks = new ArrayList<>();

            for (int t = 0; t < threadCount; t++) {
                final int threadId = t;
                tasks.add(() -> {
                    for (int s = 0; s < searchesPerThread; s++) {
                        // Each thread uses a different subset of queries
                        int queryIdx = (threadId * searchesPerThread + s) % data.length;
                        float[] query = data[queryIdx].clone();

                        List<IVFIndex.SearchResult> results = index.search(query, topK, nProbe);

                        assertFalse(results.isEmpty(),
                            "Thread " + threadId + " search " + s + ": empty results");
                        assertTrue(results.size() <= topK,
                            "Thread " + threadId + " search " + s + ": too many results");

                        // Results must be sorted by ascending distance
                        for (int i = 1; i < results.size(); i++) {
                            assertTrue(results.get(i).distance >= results.get(i - 1).distance,
                                "Thread " + threadId + " search " + s
                                    + ": results not sorted at position " + i);
                        }

                        // All distances must be non-negative
                        for (IVFIndex.SearchResult r : results) {
                            assertTrue(r.distance >= 0,
                                "Thread " + threadId + " search " + s
                                    + ": negative distance " + r.distance);
                        }
                    }
                    return null;
                });
            }

            List<Future<Void>> futures = executor.invokeAll(tasks);

            // Ensure no exceptions were thrown in any thread
            for (int t = 0; t < threadCount; t++) {
                futures.get(t).get();
            }
        } finally {
            executor.shutdown();
        }
    }
}
