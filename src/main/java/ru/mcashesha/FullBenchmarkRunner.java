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

    // Build parameters (optimal for >= 95% recall from README)
    private static final int LLOYD_CLUSTER_COUNT = 128;
    private static final int LLOYD_MAX_ITERATIONS = 100;

    private static final int MINIBATCH_CLUSTER_COUNT = 64;
    private static final int MINIBATCH_BATCH_SIZE = 512;
    private static final int MINIBATCH_MAX_ITERATIONS = 300;
    private static final int MINIBATCH_MAX_NO_IMPROVEMENT = 30;

    private static final int HIERARCHICAL_BRANCH_FACTOR = 8;
    private static final int HIERARCHICAL_MAX_DEPTH = 3;
    private static final int HIERARCHICAL_MIN_CLUSTER_SIZE = 100;
    private static final int HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL = 30;

    // Search parameters
    private static final int TOP_K = 100;
    private static final int NPROBE = 8;
    private static final int SEARCH_WARMUP = 50;
    private static final int SEARCH_ITERATIONS = 200;
    private static final int BATCH_SIZE = 32;

    private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

    public static void main(String[] args) throws IOException {
        String dataPath = args.length > 0 ? args[0] : "embeddings.csv";
        System.out.println("Loading data from: " + dataPath);
        float[][] data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(dataPath));
        System.out.printf("Loaded %d vectors, %d dimensions%n", data.length, data[0].length);

        // Pre-generate queries from real data with small noise
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

        KMeans.Type[] algorithms = { KMeans.Type.MINI_BATCH, KMeans.Type.HIERARCHICAL, KMeans.Type.LLOYD };
        Metric.Type[] metricTypes = { Metric.Type.L2SQ_DISTANCE, Metric.Type.DOT_PRODUCT, Metric.Type.COSINE_DISTANCE };
        Metric.Engine[] engines = { Metric.Engine.SCALAR, Metric.Engine.VECTOR_API, Metric.Engine.SIMSIMD };

        // Header
        System.out.println();
        System.out.println("RESULT|Algorithm|MetricType|Engine|Clusters|BuildTimeMs|BuildHeapDeltaMB|BuildGcCount|BuildGcTimeMs" +
            "|SearchAvgUs|SearchP50Us|SearchP99Us|SearchMinUs|SearchMaxUs" +
            "|BatchSearchAvgUs|BatchSearchP50Us|BatchSearchP99Us" +
            "|SearchAllocKB|SearchGcCount|IndexMemoryMB");

        int total = algorithms.length * metricTypes.length * engines.length;
        int done = 0;

        for (KMeans.Type algo : algorithms) {
            for (Metric.Type metricType : metricTypes) {
                for (Metric.Engine engine : engines) {
                    done++;
                    System.out.printf("%n=== [%d/%d] %s / %s / %s ===%n", done, total, algo, metricType, engine);

                    try {
                        runSingleBenchmark(data, queries, algo, metricType, engine);
                    } catch (Exception e) {
                        System.out.printf("ERROR: %s%n", e.getMessage());
                        System.out.printf("RESULT|%s|%s|%s|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR|ERROR%n",
                            algo, metricType, engine);
                    }
                }
            }
        }

        System.out.println("\n=== ALL BENCHMARKS COMPLETE ===");
    }

    private static void runSingleBenchmark(float[][] data, float[][] queries,
                                            KMeans.Type algo, Metric.Type metricType, Metric.Engine engine) {
        // Force GC before build
        forceGc();

        long heapBefore = memoryBean.getHeapMemoryUsage().getUsed();
        long gcCountBefore = getTotalGcCount();
        long gcTimeBefore = getTotalGcTime();

        // Build index
        long buildStart = System.nanoTime();
        KMeans<? extends KMeans.ClusteringResult> kmeans = createKMeans(algo, metricType, engine);
        IVFIndex index = new IVFIndexFlat(kmeans);
        index.build(data);
        long buildTimeMs = (System.nanoTime() - buildStart) / 1_000_000;

        forceGc();

        long heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
        long gcCountAfter = getTotalGcCount();
        long gcTimeAfter = getTotalGcTime();

        double buildHeapDeltaMB = (heapAfter - heapBefore) / (1024.0 * 1024.0);
        long buildGcCount = gcCountAfter - gcCountBefore;
        long buildGcTimeMs = gcTimeAfter - gcTimeBefore;
        int clusters = index.getCountClusters();

        // Estimate index memory
        forceGc();
        long indexMemoryBytes = memoryBean.getHeapMemoryUsage().getUsed() - heapBefore;
        double indexMemoryMB = Math.max(0, indexMemoryBytes) / (1024.0 * 1024.0);

        System.out.printf("  Build: %d ms, clusters=%d, heapDelta=%.1f MB, GC=%d (%.0f ms)%n",
            buildTimeMs, clusters, buildHeapDeltaMB, buildGcCount, (double) buildGcTimeMs);

        // === Single Search Benchmark ===
        // Warmup
        int queryIdx = 0;
        for (int i = 0; i < SEARCH_WARMUP; i++) {
            index.search(queries[queryIdx], TOP_K, NPROBE);
            queryIdx = (queryIdx + 1) % queries.length;
        }

        // Measured runs
        long[] searchTimesNs = new long[SEARCH_ITERATIONS];
        forceGc();
        long searchGcCountBefore = getTotalGcCount();
        long searchHeapBefore = memoryBean.getHeapMemoryUsage().getUsed();

        for (int i = 0; i < SEARCH_ITERATIONS; i++) {
            long start = System.nanoTime();
            index.search(queries[queryIdx], TOP_K, NPROBE);
            searchTimesNs[i] = System.nanoTime() - start;
            queryIdx = (queryIdx + 1) % queries.length;
        }

        long searchHeapAfter = memoryBean.getHeapMemoryUsage().getUsed();
        long searchGcCountAfter = getTotalGcCount();
        long searchGcCount = searchGcCountAfter - searchGcCountBefore;
        double searchAllocKB = Math.max(0, searchHeapAfter - searchHeapBefore) / 1024.0;

        java.util.Arrays.sort(searchTimesNs);
        double searchAvgUs = avg(searchTimesNs) / 1000.0;
        double searchP50Us = searchTimesNs[SEARCH_ITERATIONS / 2] / 1000.0;
        double searchP99Us = searchTimesNs[(int) (SEARCH_ITERATIONS * 0.99)] / 1000.0;
        double searchMinUs = searchTimesNs[0] / 1000.0;
        double searchMaxUs = searchTimesNs[SEARCH_ITERATIONS - 1] / 1000.0;

        System.out.printf("  Search: avg=%.0f us, p50=%.0f us, p99=%.0f us, min=%.0f us, max=%.0f us%n",
            searchAvgUs, searchP50Us, searchP99Us, searchMinUs, searchMaxUs);

        // === Batch Search Benchmark ===
        // Warmup
        queryIdx = 0;
        for (int i = 0; i < SEARCH_WARMUP / BATCH_SIZE + 1; i++) {
            float[][] batch = new float[BATCH_SIZE][];
            for (int b = 0; b < BATCH_SIZE; b++) {
                batch[b] = queries[queryIdx];
                queryIdx = (queryIdx + 1) % queries.length;
            }
            index.searchBatch(batch, TOP_K, NPROBE);
        }

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
        // Per-query time from batch
        double batchAvgUs = avg(batchTimesNs) / 1000.0 / BATCH_SIZE;
        double batchP50Us = batchTimesNs[batchIterations / 2] / 1000.0 / BATCH_SIZE;
        double batchP99Us = batchTimesNs[(int) (batchIterations * 0.99)] / 1000.0 / BATCH_SIZE;

        System.out.printf("  Batch:  avg=%.0f us/q, p50=%.0f us/q, p99=%.0f us/q%n",
            batchAvgUs, batchP50Us, batchP99Us);

        // Output result line
        System.out.printf("RESULT|%s|%s|%s|%d|%d|%.1f|%d|%d|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.0f|%.1f|%d|%.1f%n",
            algo, metricType, engine,
            clusters, buildTimeMs, buildHeapDeltaMB, buildGcCount, buildGcTimeMs,
            searchAvgUs, searchP50Us, searchP99Us, searchMinUs, searchMaxUs,
            batchAvgUs, batchP50Us, batchP99Us,
            searchAllocKB, searchGcCount, indexMemoryMB);
    }

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

    private static void forceGc() {
        System.gc();
        try { Thread.sleep(200); } catch (InterruptedException ignored) {}
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}
    }

    private static long getTotalGcCount() {
        long total = 0;
        for (GarbageCollectorMXBean gc : gcBeans)
            total += gc.getCollectionCount();
        return total;
    }

    private static long getTotalGcTime() {
        long total = 0;
        for (GarbageCollectorMXBean gc : gcBeans)
            total += gc.getCollectionTime();
        return total;
    }

    private static double avg(long[] values) {
        double sum = 0;
        for (long v : values) sum += v;
        return sum / values.length;
    }
}
