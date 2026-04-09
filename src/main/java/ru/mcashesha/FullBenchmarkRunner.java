package ru.mcashesha;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

/**
 * Comprehensive non-JMH benchmark runner.
 * Tests ALL combinations of: 3 algorithms × 3 engines × 3 metrics.
 * Collects: build time, search time (single + batch), memory, GC, cluster count.
 *
 * Output format: TSV lines prefixed with "RESULT|" for easy parsing.
 */
public class FullBenchmarkRunner {

    // ==================== Build Parameters ====================
    // Tuned for >= 95% recall based on README tuning results.

    /** Lloyd KMeans cluster count (k=128 yields 96.2% recall with nProbe=8). */
    private static final int LLOYD_CLUSTER_COUNT = 128;

    /** Maximum number of full-pass iterations for Lloyd KMeans. */
    private static final int LLOYD_MAX_ITERATIONS = 100;

    /** MiniBatch KMeans cluster count (k=64 yields 97.8% recall with nProbe=8). */
    private static final int MINIBATCH_CLUSTER_COUNT = 64;

    /** Number of samples drawn per mini-batch iteration. */
    private static final int MINIBATCH_BATCH_SIZE = 512;

    /** Maximum number of mini-batch iterations before termination. */
    private static final int MINIBATCH_MAX_ITERATIONS = 300;

    /** Early stopping threshold: stop if no improvement for this many consecutive iterations. */
    private static final int MINIBATCH_MAX_NO_IMPROVEMENT = 30;

    /** Hierarchical KMeans branching factor (children per internal node). */
    private static final int HIERARCHICAL_BRANCH_FACTOR = 8;

    /** Maximum depth of the hierarchical clustering tree. */
    private static final int HIERARCHICAL_MAX_DEPTH = 3;

    /** Minimum number of vectors in a leaf cluster before further splitting stops. */
    private static final int HIERARCHICAL_MIN_CLUSTER_SIZE = 100;

    /** Maximum Lloyd iterations at each level of the hierarchical tree. */
    private static final int HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL = 30;

    // ==================== Search Parameters ====================

    /** Number of nearest neighbors to retrieve per query. */
    private static final int TOP_K = 100;

    /** Number of clusters to probe during search (controls recall vs. speed trade-off). */
    private static final int NPROBE = 8;

    /** Number of warmup iterations before measured search runs (JIT stabilization). */
    private static final int SEARCH_WARMUP = 50;

    /** Number of measured single-query search iterations for latency statistics. */
    private static final int SEARCH_ITERATIONS = 200;

    /** Number of queries per batch in the batch search benchmark. */
    private static final int BATCH_SIZE = 32;

    /** JMX bean for querying heap memory usage before and after operations. */
    private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();

    /** JMX beans for all garbage collectors, used to track GC count and cumulative time. */
    private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

    /**
     * Entry point for the full benchmark suite.
     *
     * <p>Loads embedding data from a CSV file (default: {@code embeddings.csv}, overridable
     * via the first command-line argument), pre-generates 1000 noisy query vectors from the
     * real data, then iterates over all 27 combinations of algorithm, metric type, and engine.
     * Each combination runs {@link #runSingleBenchmark} and emits a TSV result line.
     *
     * @param args optional; {@code args[0]} specifies the path to the embeddings CSV file
     * @throws IOException if the embeddings file cannot be read
     */
    public static void main(String[] args) throws IOException {
        String dataPath = args.length > 0 ? args[0] : "embeddings.csv";
        System.out.println("Loading data from: " + dataPath);
        float[][] data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(dataPath));
        System.out.printf("Loaded %d vectors, %d dimensions%n", data.length, data[0].length);

        // Pre-generate 1000 query vectors by copying real data points and adding small
        // Gaussian-like noise (uniform in [-0.005, +0.005] per dimension). This produces
        // realistic queries that are close to actual data rather than purely random vectors.
        Random rng = new Random(123456);
        int numQueries = 1000;
        int dimension = data[0].length;
        float[][] queries = new float[numQueries][dimension];
        for (int i = 0; i < numQueries; i++) {
            int idx = rng.nextInt(data.length);
            System.arraycopy(data[idx], 0, queries[i], 0, dimension);
            for (int d = 0; d < dimension; d++)
                queries[i][d] += (rng.nextFloat() - 0.5f) * 0.01f;
        }

        // Define the three dimensions of the parameter space to enumerate
        KMeans.Type[] algorithms = { KMeans.Type.MINI_BATCH, KMeans.Type.HIERARCHICAL, KMeans.Type.LLOYD };
        Metric.Type[] metricTypes = { Metric.Type.L2SQ_DISTANCE, Metric.Type.DOT_PRODUCT, Metric.Type.COSINE_DISTANCE };
        Metric.Engine[] engines = { Metric.Engine.SCALAR, Metric.Engine.VECTOR_API, Metric.Engine.SIMSIMD };

        // Print the TSV header line (prefixed with "RESULT|" like all data lines)
        System.out.println();
        System.out.println("RESULT|Algorithm|MetricType|Engine|Clusters|BuildTimeMs|BuildHeapDeltaMB|BuildGcCount|BuildGcTimeMs" +
            "|SearchAvgUs|SearchP50Us|SearchP99Us|SearchMinUs|SearchMaxUs" +
            "|BatchSearchAvgUs|BatchSearchP50Us|BatchSearchP99Us" +
            "|SearchAllocKB|SearchGcCount|IndexMemoryMB");

        int total = algorithms.length * metricTypes.length * engines.length;
        int done = 0;

        // Iterate over all 27 combinations (3 algorithms x 3 metrics x 3 engines)
        for (KMeans.Type algo : algorithms) {
            for (Metric.Type metricType : metricTypes) {
                for (Metric.Engine engine : engines) {
                    done++;
                    System.out.printf("%n=== [%d/%d] %s / %s / %s ===%n", done, total, algo, metricType, engine);

                    try {
                        runSingleBenchmark(data, queries, algo, metricType, engine);
                    } catch (Exception e) {
                        // On failure, emit an ERROR result line so the TSV output remains parseable
                        System.out.printf("ERROR: %s%n", e.getMessage());
                        System.out.printf("RESULT|%s|%s|%s|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR%n",
                            algo, metricType, engine);
                    }
                }
            }
        }

        System.out.println("\n=== ALL BENCHMARKS COMPLETE ===");
    }

    /**
     * Runs a complete benchmark for a single (algorithm, metric, engine) combination.
     *
     * <p>This method performs the following steps:
     * <ol>
     *   <li>Forces GC and records baseline heap/GC metrics</li>
     *   <li>Builds the IVF index (KMeans fit + data reordering) and records build time</li>
     *   <li>Estimates index memory footprint via heap delta</li>
     *   <li>Runs single-query search warmup, then {@value #SEARCH_ITERATIONS} measured iterations
     *       to collect latency distribution (avg, p50, p99, min, max)</li>
     *   <li>Runs batch search warmup, then measured iterations to collect per-query latency</li>
     *   <li>Emits a single "RESULT|..." TSV line with all collected metrics</li>
     * </ol>
     *
     * @param data      the full dataset of embedding vectors to index
     * @param queries   pre-generated query vectors for search benchmarking
     * @param algo      the KMeans clustering algorithm to use
     * @param metricType the distance metric type (L2, dot product, or cosine)
     * @param engine    the distance computation engine (Scalar, Vector API, or SimSIMD)
     */
    private static void runSingleBenchmark(float[][] data, float[][] queries,
                                            KMeans.Type algo, Metric.Type metricType, Metric.Engine engine) {
        // Force GC before build to establish a clean heap baseline
        forceGc();

        long heapBefore = memoryBean.getHeapMemoryUsage().getUsed();
        long gcCountBefore = getTotalGcCount();
        long gcTimeBefore = getTotalGcTime();

        // Build index: includes KMeans clustering (fit) and data reordering by cluster
        long buildStart = System.nanoTime();
        KMeans<? extends KMeans.ClusteringResult> kmeans = createKMeans(algo, metricType, engine);
        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);
        long buildTimeMs = (System.nanoTime() - buildStart) / 1_000_000;

        // Force GC again to get a stable post-build heap measurement
        forceGc();

        long heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
        long gcCountAfter = getTotalGcCount();
        long gcTimeAfter = getTotalGcTime();

        double buildHeapDeltaMB = (heapAfter - heapBefore) / (1024.0 * 1024.0);
        long buildGcCount = gcCountAfter - gcCountBefore;
        long buildGcTimeMs = gcTimeAfter - gcTimeBefore;
        int clusters = index.getCountClusters();

        // Estimate net index memory: heap used now minus the pre-build baseline.
        // This is approximate since GC and other allocations may skew the measurement.
        forceGc();
        long indexMemoryBytes = memoryBean.getHeapMemoryUsage().getUsed() - heapBefore;
        double indexMemoryMB = Math.max(0, indexMemoryBytes) / (1024.0 * 1024.0);

        System.out.printf("  Build: %d ms, clusters=%d, heapDelta=%.1f MB, GC=%d (%.0f ms)%n",
            buildTimeMs, clusters, buildHeapDeltaMB, buildGcCount, (double) buildGcTimeMs);

        // === Single Search Benchmark ===

        // Warmup phase: run SEARCH_WARMUP queries to trigger JIT compilation
        // and stabilize hot paths before taking measurements
        int queryIdx = 0;
        for (int i = 0; i < SEARCH_WARMUP; i++) {
            index.search(queries[queryIdx], TOP_K, NPROBE);
            queryIdx = (queryIdx + 1) % queries.length;
        }

        // Measured runs: record per-query latency in nanoseconds
        long[] searchTimesNs = new long[SEARCH_ITERATIONS];
        forceGc();
        long searchGcCountBefore = getTotalGcCount();
        long searchHeapBefore = memoryBean.getHeapMemoryUsage().getUsed();

        for (int i = 0; i < SEARCH_ITERATIONS; i++) {
            long start = System.nanoTime();
            index.search(queries[queryIdx], TOP_K, NPROBE);
            searchTimesNs[i] = System.nanoTime() - start;
            // Cycle through queries to avoid caching effects on a single query
            queryIdx = (queryIdx + 1) % queries.length;
        }

        long searchHeapAfter = memoryBean.getHeapMemoryUsage().getUsed();
        long searchGcCountAfter = getTotalGcCount();
        long searchGcCount = searchGcCountAfter - searchGcCountBefore;
        // Approximate total allocation during search (may be negative if GC ran)
        double searchAllocKB = Math.max(0, searchHeapAfter - searchHeapBefore) / 1024.0;

        // Sort latencies to compute percentile statistics
        java.util.Arrays.sort(searchTimesNs);
        double searchAvgUs = avg(searchTimesNs) / 1000.0;
        double searchP50Us = searchTimesNs[SEARCH_ITERATIONS / 2] / 1000.0;
        double searchP99Us = searchTimesNs[(int) (SEARCH_ITERATIONS * 0.99)] / 1000.0;
        double searchMinUs = searchTimesNs[0] / 1000.0;
        double searchMaxUs = searchTimesNs[SEARCH_ITERATIONS - 1] / 1000.0;

        System.out.printf("  Search: avg=%.0f us, p50=%.0f us, p99=%.0f us, min=%.0f us, max=%.0f us%n",
            searchAvgUs, searchP50Us, searchP99Us, searchMinUs, searchMaxUs);

        // === Batch Search Benchmark ===

        // Warmup: run enough batch iterations to cover at least SEARCH_WARMUP queries total
        queryIdx = 0;
        for (int i = 0; i < SEARCH_WARMUP / BATCH_SIZE + 1; i++) {
            float[][] batch = new float[BATCH_SIZE][];
            for (int b = 0; b < BATCH_SIZE; b++) {
                batch[b] = queries[queryIdx];
                queryIdx = (queryIdx + 1) % queries.length;
            }
            index.searchBatch(batch, TOP_K, NPROBE);
        }

        // Measured batch runs: each iteration processes BATCH_SIZE queries
        int batchIterations = SEARCH_ITERATIONS / BATCH_SIZE;
        long[] batchTimesNs = new long[batchIterations];

        for (int i = 0; i < batchIterations; i++) {
            float[][] batch = new float[BATCH_SIZE][];
            for (int b = 0; b < BATCH_SIZE; b++) {
                batch[b] = queries[queryIdx];
                queryIdx = (queryIdx + 1) % queries.length;
            }
            long start = System.nanoTime();
            index.searchBatch(batch, TOP_K, NPROBE);
            batchTimesNs[i] = System.nanoTime() - start;
        }

        java.util.Arrays.sort(batchTimesNs);
        // Normalize batch times to per-query latency by dividing by BATCH_SIZE
        double batchAvgUs = avg(batchTimesNs) / 1000.0 / BATCH_SIZE;
        double batchP50Us = batchTimesNs[batchIterations / 2] / 1000.0 / BATCH_SIZE;
        double batchP99Us = batchTimesNs[(int) (batchIterations * 0.99)] / 1000.0 / BATCH_SIZE;

        System.out.printf("  Batch:  avg=%.0f us/q, p50=%.0f us/q, p99=%.0f us/q%n",
            batchAvgUs, batchP50Us, batchP99Us);

        // Emit the machine-parseable TSV result line with all metrics
        System.out.printf("RESULT|%s|%s|%s|%d|%d|%.1f|%d|%d|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.1f|%d|%.1f%n",
            algo, metricType, engine,
            clusters, buildTimeMs, buildHeapDeltaMB, buildGcCount, buildGcTimeMs,
            searchAvgUs, searchP50Us, searchP99Us, searchMinUs, searchMaxUs,
            batchAvgUs, batchP50Us, batchP99Us,
            searchAllocKB, searchGcCount, indexMemoryMB);
    }

    /**
     * Creates a configured {@link KMeans} instance for the given algorithm type, distance
     * metric, and computation engine.
     *
     * <p>Each algorithm type is configured with its own set of optimal parameters
     * (cluster count, batch size, tree structure, etc.) defined as class-level constants.
     * All configurations use a fixed random seed (42) for reproducibility.
     *
     * @param type       the KMeans algorithm variant (Lloyd, MiniBatch, or Hierarchical)
     * @param metricType the distance metric to use for centroid-to-vector comparisons
     * @param engine     the distance computation engine (Scalar, Vector API, or SimSIMD)
     * @return a fully configured KMeans instance ready for {@code fit()} calls
     * @throws IllegalStateException if an unsupported KMeans type is provided
     */
    private static KMeans<? extends KMeans.ClusteringResult> createKMeans(
            KMeans.Type type, Metric.Type metricType, Metric.Engine engine) {
        KMeans.Builder builder = KMeans.newBuilder(type, metricType, engine)
            .withRandom(new Random(42));

        switch (type) {
            case LLOYD:
                builder
                    .withClusterCount(LLOYD_CLUSTER_COUNT)
                    .withMaxIterations(LLOYD_MAX_ITERATIONS)
                    .withTolerance(1e-4f);
                break;
            case MINI_BATCH:
                builder
                    .withClusterCount(MINIBATCH_CLUSTER_COUNT)
                    .withBatchSize(MINIBATCH_BATCH_SIZE)
                    .withMaxIterations(MINIBATCH_MAX_ITERATIONS)
                    .withMaxNoImprovementIterations(MINIBATCH_MAX_NO_IMPROVEMENT)
                    .withTolerance(1e-4f);
                break;
            case HIERARCHICAL:
                builder
                    .withBranchFactor(HIERARCHICAL_BRANCH_FACTOR)
                    .withMaxDepth(HIERARCHICAL_MAX_DEPTH)
                    .withMinClusterSize(HIERARCHICAL_MIN_CLUSTER_SIZE)
                    .withMaxIterationsPerLevel(HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL)
                    .withTolerance(1e-4f);
                break;
            default:
                throw new IllegalStateException("Unsupported type: " + type);
        }

        return builder.build();
    }

    /**
     * Forces garbage collection with two GC passes separated by short sleeps.
     *
     * <p>The double-GC pattern with intervening pauses gives the JVM time to finalize
     * objects and reclaim memory more thoroughly, producing more stable heap measurements.
     * Note that {@link System#gc()} is advisory and the JVM may ignore it.
     */
    private static void forceGc() {
        System.gc();
        try { Thread.sleep(200); } catch (InterruptedException ignored) {}
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}
    }

    /**
     * Returns the total number of garbage collection events across all collectors.
     *
     * @return cumulative GC invocation count summed over all {@link GarbageCollectorMXBean} instances
     */
    private static long getTotalGcCount() {
        long total = 0;
        for (GarbageCollectorMXBean gc : gcBeans)
            total += gc.getCollectionCount();
        return total;
    }

    /**
     * Returns the total cumulative garbage collection time across all collectors.
     *
     * @return cumulative GC time in milliseconds summed over all {@link GarbageCollectorMXBean} instances
     */
    private static long getTotalGcTime() {
        long total = 0;
        for (GarbageCollectorMXBean gc : gcBeans)
            total += gc.getCollectionTime();
        return total;
    }

    /**
     * Computes the arithmetic mean of a long array.
     *
     * @param values the array of values (typically nanosecond latencies)
     * @return the arithmetic mean as a double
     */
    private static double avg(long[] values) {
        double sum = 0;
        for (long v : values) sum += v;
        return sum / values.length;
    }
}
