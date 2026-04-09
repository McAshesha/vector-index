package ru.mcashesha;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static java.util.concurrent.TimeUnit.SECONDS;

/**
 * Build benchmarks for IVF index with optimal parameters for >= 95% recall.
 *
 * Parameters from README tuning results:
 * - Lloyd: k=128, maxIterations=100 → 96.2% recall with nProbe=8
 * - MiniBatch: k=64, batch=512 → 97.8% recall with nProbe=8
 * - Hierarchical: bf=8, depth=3, minClusterSize=100 → 97.2% recall with nProbe=8
 *
 * Run with GC profiling:
 *   java -jar benchmarks.jar IVFIndexBuildBenchmarks -prof gc
 *
 * Run with memory allocation profiling:
 *   java -jar benchmarks.jar IVFIndexBuildBenchmarks -prof gc:churn
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2, time = 10, timeUnit = SECONDS)
@Measurement(iterations = 3, time = 30, timeUnit = SECONDS)
@Fork(value = 1, jvmArgsAppend = {
    "-Xms2g", "-Xmx4g",
    "-XX:+UseG1GC",
    "-XX:+PrintGCDetails",
    "-XX:+PrintGCDateStamps",
    "--add-modules=jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED"
})
@Threads(1)
@State(Scope.Thread)
public class IVFIndexBuildBenchmarks {

    // ==================== Lloyd KMeans Parameters ====================
    // Lloyd: k=128 → 96.2% recall with nProbe=8

    /** Number of clusters for Lloyd KMeans. Higher k improves recall but increases build time. */
    private static final int LLOYD_CLUSTER_COUNT = 128;

    /** Maximum iterations for Lloyd KMeans convergence. */
    private static final int LLOYD_MAX_ITERATIONS = 100;

    // ==================== MiniBatch KMeans Parameters ====================
    // MiniBatch: k=64, batch=512 → 97.8% recall with nProbe=8

    /** Number of clusters for MiniBatch KMeans. */
    private static final int MINIBATCH_CLUSTER_COUNT = 64;

    /** Number of samples per mini-batch update step. */
    private static final int MINIBATCH_BATCH_SIZE = 512;

    /** Maximum number of mini-batch iterations before termination. */
    private static final int MINIBATCH_MAX_ITERATIONS = 300;

    /** Early stopping: halt if no improvement for this many consecutive iterations. */
    private static final int MINIBATCH_MAX_NO_IMPROVEMENT = 30;

    // ==================== Hierarchical KMeans Parameters ====================
    // Hierarchical: bf=8, depth=3, minClusterSize=100 → 97.2% recall with nProbe=8

    /** Children per internal node in the hierarchical clustering tree. */
    private static final int HIERARCHICAL_BRANCH_FACTOR = 8;

    /** Maximum depth of the hierarchical tree (total leaf clusters up to bf^depth). */
    private static final int HIERARCHICAL_MAX_DEPTH = 3;

    /** Minimum vectors in a leaf cluster; prevents over-fragmentation. */
    private static final int HIERARCHICAL_MIN_CLUSTER_SIZE = 100;

    /** Maximum Lloyd iterations at each level of the hierarchical tree. */
    private static final int HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL = 30;

    /**
     * Creates a fully configured {@link KMeans} instance for the given algorithm type.
     *
     * <p>All configurations use a fixed random seed (42) for reproducibility across
     * benchmark runs. The convergence tolerance is set to {@code 1e-4f} for all types.
     *
     * @param type         the KMeans algorithm variant (Lloyd, MiniBatch, or Hierarchical)
     * @param metricType   the distance metric for centroid-to-vector comparisons
     * @param metricEngine the distance computation engine (Scalar, Vector API, or SimSIMD)
     * @return a configured KMeans instance ready for {@code fit()} calls
     * @throws IllegalStateException if an unsupported KMeans type is provided
     */
    private static KMeans<? extends KMeans.ClusteringResult> createKMeans(
        KMeans.Type type,
        Metric.Type metricType,
        Metric.Engine metricEngine
    ) {
        KMeans.Builder builder = KMeans.newBuilder(type, metricType, metricEngine)
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
                throw new IllegalStateException("Unsupported KMeans type: " + type);
        }

        return builder.build();
    }

    /**
     * Benchmarks end-to-end index build time using Lloyd KMeans clustering.
     *
     * <p>Measures the full build pipeline: KMeans++ initialization, iterative centroid
     * refinement (up to {@value #LLOYD_MAX_ITERATIONS} iterations), cluster assignment,
     * and contiguous data reordering by cluster for cache-friendly search access.
     *
     * <p>The returned index is used as a JMH return value to prevent dead-code elimination.
     *
     * @param state    shared benchmark state containing loaded embeddings and metric configuration
     * @param counters per-invocation memory and GC tracking counters
     * @return the built IVF index (consumed by JMH to prevent DCE)
     */
    @Benchmark
    public IVFIndex buildLloydIndex(BuildState state, MemoryCounters counters) {
        counters.beforeBuild();

        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.LLOYD, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);

        counters.afterBuild();
        counters.clusterCount = idx.getCountClusters();
        return idx;
    }

    /**
     * Benchmarks end-to-end index build time using MiniBatch KMeans clustering.
     *
     * <p>Measures the full build pipeline with stochastic mini-batch updates: KMeans++
     * initialization, iterative weighted centroid updates using random {@value #MINIBATCH_BATCH_SIZE}-sample
     * batches, early stopping on convergence plateau, final full reassignment pass,
     * and contiguous data reordering.
     *
     * @param state    shared benchmark state containing loaded embeddings and metric configuration
     * @param counters per-invocation memory and GC tracking counters
     * @return the built IVF index (consumed by JMH to prevent DCE)
     */
    @Benchmark
    public IVFIndex buildMiniBatchIndex(BuildState state, MemoryCounters counters) {
        counters.beforeBuild();

        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.MINI_BATCH, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);

        counters.afterBuild();
        counters.clusterCount = idx.getCountClusters();
        return idx;
    }

    /**
     * Benchmarks end-to-end index build time using Hierarchical KMeans clustering.
     *
     * <p>Measures the full build pipeline with recursive tree construction: at each level,
     * Lloyd KMeans splits data into {@value #HIERARCHICAL_BRANCH_FACTOR} sub-clusters,
     * recursing up to depth {@value #HIERARCHICAL_MAX_DEPTH}. Leaf clusters smaller than
     * {@value #HIERARCHICAL_MIN_CLUSTER_SIZE} are not split further. The final step
     * reorders data contiguously by leaf cluster.
     *
     * @param state    shared benchmark state containing loaded embeddings and metric configuration
     * @param counters per-invocation memory and GC tracking counters
     * @return the built IVF index (consumed by JMH to prevent DCE)
     */
    @Benchmark
    public IVFIndex buildHierarchicalIndex(BuildState state, MemoryCounters counters) {
        counters.beforeBuild();

        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.HIERARCHICAL, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);

        counters.afterBuild();
        counters.clusterCount = idx.getCountClusters();
        return idx;
    }

    /**
     * JMH shared state that loads embedding data once per trial and holds
     * the parameterized metric configuration.
     *
     * <p>Scoped to {@link Scope#Benchmark} so that all threads (though this benchmark
     * is single-threaded) share the same loaded dataset, avoiding redundant I/O.
     * The embeddings are loaded during the {@link Level#Trial} setup phase, which
     * runs once before all warmup and measurement iterations.
     */
    @State(Scope.Benchmark)
    public static class BuildState {

        /** Path to the embeddings CSV file. JMH parameter, default {@code "embeddings.csv"}. */
        @Param("embeddings.csv")
        public String embeddingsPath;

        /**
         * Distance metric type name. JMH parameter that is resolved to {@link Metric.Type}
         * during setup. Tests all three: L2 squared, dot product, and cosine distance.
         */
        @Param({"L2SQ_DISTANCE", "DOT_PRODUCT", "COSINE_DISTANCE"})
        public String metricTypeName;

        /**
         * Distance computation engine name. JMH parameter that is resolved to {@link Metric.Engine}
         * during setup. Tests Scalar (pure Java), Vector API (SIMD), and SimSIMD (native JNI).
         */
        @Param({"SCALAR", "VECTOR_API", "SIMSIMD"})
        public String metricEngineName;

        /** The loaded embedding vectors; each row is a single vector of {@link #dimension} floats. */
        float[][] data;

        /** Resolved distance metric type enum from {@link #metricTypeName}. */
        Metric.Type metricType;

        /** Resolved distance computation engine enum from {@link #metricEngineName}. */
        Metric.Engine metricEngine;

        /** Estimated memory consumption of the raw data array in bytes. */
        long dataMemoryBytes;

        /** Total number of embedding vectors loaded from the CSV file. */
        int vectorCount;

        /** Dimensionality of each embedding vector (e.g., 512 for 512-d embeddings). */
        int dimension;

        /**
         * Loads embedding data from CSV, resolves metric parameters, and estimates memory usage.
         *
         * <p>Called once per JMH trial. The memory estimate accounts for float data
         * ({@code n * d * 4} bytes) plus per-array object overhead (~16 bytes per row).
         *
         * @throws IOException if the embeddings file cannot be read or parsed
         */
        @Setup(Level.Trial)
        public void setup() throws IOException {
            this.data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(embeddingsPath));
            this.metricType = Metric.Type.valueOf(metricTypeName);
            this.metricEngine = Metric.Engine.valueOf(metricEngineName);

            this.vectorCount = data.length;
            this.dimension = data[0].length;
            // Estimate data memory: n vectors × d dimensions × 4 bytes + array overhead
            this.dataMemoryBytes = (long) vectorCount * dimension * 4L + (long) vectorCount * 16L;

            System.out.printf("[Setup] Loaded %d vectors, %d dimensions, ~%.1f MB%n",
                vectorCount, dimension, dataMemoryBytes / (1024.0 * 1024.0));
        }
    }

    /**
     * Auxiliary counters for memory and GC tracking during index build benchmarks.
     *
     * <p>These metrics are reported alongside the primary timing results by JMH's
     * {@link AuxCounters} mechanism. Each public field becomes a column in the JMH
     * output, providing visibility into heap pressure and GC overhead per build.
     *
     * <p>Scoped to {@link Scope#Thread} so each benchmark thread tracks its own counters.
     * The {@link #reset()} method runs before each iteration to force GC and establish
     * a clean baseline. The {@link #beforeBuild()} and {@link #afterBuild()} methods
     * are called within each benchmark method to bracket the measured operation.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class MemoryCounters {

        /** JMX bean for querying current heap memory usage. */
        private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();

        /** JMX beans for all garbage collectors in the JVM. */
        private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        // ==================== Memory Metrics (reported in MB) ====================

        /** Heap used before the build starts, in megabytes. */
        public double heapUsedBeforeMB;

        /** Heap used after the build completes, in megabytes. */
        public double heapUsedAfterMB;

        /** Net heap change during the build (after - before), in megabytes. May be negative if GC ran. */
        public double heapDeltaMB;

        /** Maximum heap size available to the JVM, in megabytes (from -Xmx). */
        public double heapMaxMB;

        // ==================== GC Metrics ====================

        /** Cumulative GC invocation count before the build. */
        public long gcCountBefore;

        /** Cumulative GC time in milliseconds before the build. */
        public long gcTimeMsBefore;

        /** Cumulative GC invocation count after the build. */
        public long gcCountAfter;

        /** Cumulative GC time in milliseconds after the build. */
        public long gcTimeMsAfter;

        /** Number of GC events that occurred during the build. */
        public long gcCountDelta;

        /** Total GC pause time during the build, in milliseconds. */
        public long gcTimeMsDelta;

        // ==================== Index Metrics ====================

        /** Number of clusters produced by the KMeans algorithm. Set after build completion. */
        public int clusterCount;

        /** Snapshot of heap used at the start of the build, for computing deltas. */
        private long startHeapUsed;

        /** Snapshot of cumulative GC count at the start of the build. */
        private long startGcCount;

        /** Snapshot of cumulative GC time at the start of the build. */
        private long startGcTime;

        /**
         * Resets counters before each JMH iteration by forcing a GC pass.
         *
         * <p>The brief sleep gives the GC time to complete, producing a more stable
         * heap baseline for the subsequent measurement.
         */
        @Setup(Level.Iteration)
        public void reset() {
            // Force GC before measurement to get cleaner baseline
            System.gc();
            try { Thread.sleep(100); } catch (InterruptedException ignored) {}
        }

        /**
         * Records pre-build memory and GC snapshots.
         *
         * <p>Called at the beginning of each {@code @Benchmark} method, immediately
         * before the index build operation. Captures current heap usage, max heap,
         * and cumulative GC statistics as a baseline for delta computation.
         */
        public void beforeBuild() {
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            startHeapUsed = heapUsage.getUsed();
            heapUsedBeforeMB = startHeapUsed / (1024.0 * 1024.0);
            heapMaxMB = heapUsage.getMax() / (1024.0 * 1024.0);

            // Sum GC counts and times across all collectors (e.g., young gen + old gen)
            startGcCount = 0;
            startGcTime = 0;
            for (GarbageCollectorMXBean gc : gcBeans) {
                startGcCount += gc.getCollectionCount();
                startGcTime += gc.getCollectionTime();
            }
            gcCountBefore = startGcCount;
            gcTimeMsBefore = startGcTime;
        }

        /**
         * Records post-build memory and GC snapshots and computes deltas.
         *
         * <p>Called at the end of each {@code @Benchmark} method, immediately after
         * the index build completes. Computes heap delta and GC overhead during
         * the build, which are then reported as auxiliary JMH counters.
         */
        public void afterBuild() {
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            long endHeapUsed = heapUsage.getUsed();
            heapUsedAfterMB = endHeapUsed / (1024.0 * 1024.0);
            heapDeltaMB = (endHeapUsed - startHeapUsed) / (1024.0 * 1024.0);

            // Sum post-build GC counts and times across all collectors
            long endGcCount = 0;
            long endGcTime = 0;
            for (GarbageCollectorMXBean gc : gcBeans) {
                endGcCount += gc.getCollectionCount();
                endGcTime += gc.getCollectionTime();
            }
            gcCountAfter = endGcCount;
            gcTimeMsAfter = endGcTime;
            gcCountDelta = endGcCount - startGcCount;
            gcTimeMsDelta = endGcTime - startGcTime;
        }
    }
}
