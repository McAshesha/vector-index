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

    // Optimal parameters from README for >= 95% recall

    // Lloyd: k=128 → 96.2% recall with nProbe=8
    private static final int LLOYD_CLUSTER_COUNT = 128;
    private static final int LLOYD_MAX_ITERATIONS = 100;

    // MiniBatch: k=64, batch=512 → 97.8% recall with nProbe=8
    private static final int MINIBATCH_CLUSTER_COUNT = 64;
    private static final int MINIBATCH_BATCH_SIZE = 512;
    private static final int MINIBATCH_MAX_ITERATIONS = 300;
    private static final int MINIBATCH_MAX_NO_IMPROVEMENT = 30;

    // Hierarchical: bf=8, depth=3, minClusterSize=100 → 97.2% recall with nProbe=8
    private static final int HIERARCHICAL_BRANCH_FACTOR = 8;
    private static final int HIERARCHICAL_MAX_DEPTH = 3;
    private static final int HIERARCHICAL_MIN_CLUSTER_SIZE = 100;
    private static final int HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL = 30;

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

    @State(Scope.Benchmark)
    public static class BuildState {

        @Param("embeddings.csv")
        public String embeddingsPath;

        @Param({"L2SQ_DISTANCE", "DOT_PRODUCT", "COSINE_DISTANCE"})
        public String metricTypeName;

        @Param({"SCALAR", "VECTOR_API", "SIMSIMD"})
        public String metricEngineName;

        float[][] data;
        Metric.Type metricType;
        Metric.Engine metricEngine;

        long dataMemoryBytes;
        int vectorCount;
        int dimension;

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
     * Auxiliary counters for memory and GC tracking.
     * These metrics are reported alongside the benchmark results.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class MemoryCounters {

        private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        // Memory metrics (in MB)
        public double heapUsedBeforeMB;
        public double heapUsedAfterMB;
        public double heapDeltaMB;
        public double heapMaxMB;

        // GC metrics
        public long gcCountBefore;
        public long gcTimeMsBefore;
        public long gcCountAfter;
        public long gcTimeMsAfter;
        public long gcCountDelta;
        public long gcTimeMsDelta;

        // Index metrics
        public int clusterCount;

        private long startHeapUsed;
        private long startGcCount;
        private long startGcTime;

        @Setup(Level.Iteration)
        public void reset() {
            // Force GC before measurement to get cleaner baseline
            System.gc();
            try { Thread.sleep(100); } catch (InterruptedException ignored) {}
        }

        public void beforeBuild() {
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            startHeapUsed = heapUsage.getUsed();
            heapUsedBeforeMB = startHeapUsed / (1024.0 * 1024.0);
            heapMaxMB = heapUsage.getMax() / (1024.0 * 1024.0);

            startGcCount = 0;
            startGcTime = 0;
            for (GarbageCollectorMXBean gc : gcBeans) {
                startGcCount += gc.getCollectionCount();
                startGcTime += gc.getCollectionTime();
            }
            gcCountBefore = startGcCount;
            gcTimeMsBefore = startGcTime;
        }

        public void afterBuild() {
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            long endHeapUsed = heapUsage.getUsed();
            heapUsedAfterMB = endHeapUsed / (1024.0 * 1024.0);
            heapDeltaMB = (endHeapUsed - startHeapUsed) / (1024.0 * 1024.0);

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
