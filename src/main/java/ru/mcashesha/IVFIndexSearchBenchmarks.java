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
import org.openjdk.jmh.infra.Blackhole;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static java.util.concurrent.TimeUnit.SECONDS;

/**
 * Search benchmarks for IVF index with optimal parameters for >= 95% recall.
 *
 * Parameters from README tuning results:
 * - Lloyd: k=128, nProbe=8 → 96.2% recall, ~3.7ms search
 * - MiniBatch: k=64, nProbe=8 → 97.8% recall, ~7.0ms search
 * - Hierarchical: bf=8, d=3, nProbe=8 → 97.2% recall, ~7.7ms search
 *
 * Run with GC profiling:
 *   java -jar benchmarks.jar IVFIndexSearchBenchmarks -prof gc
 *
 * Run with allocation profiling:
 *   java -jar benchmarks.jar IVFIndexSearchBenchmarks -prof gc:churn
 *
 * Run with stack profiling:
 *   java -jar benchmarks.jar IVFIndexSearchBenchmarks -prof stack
 */
@BenchmarkMode(Mode.SampleTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 5, timeUnit = SECONDS)
@Measurement(iterations = 5, time = 10, timeUnit = SECONDS)
@Fork(value = 2, jvmArgsAppend = {
    "-Xms2g", "-Xmx4g",
    "-XX:+UseG1GC",
    "--add-modules=jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED"
})
@Threads(1)
@State(Scope.Thread)
public class IVFIndexSearchBenchmarks {

    // Search parameters
    private static final int TOP_K = 100;

    // Optimal nProbe for >= 95% recall (from README)
    private static final int NPROBE_LLOYD = 8;        // k=128, 96.2% recall
    private static final int NPROBE_MINI_BATCH = 8;   // k=64, 97.8% recall
    private static final int NPROBE_HIERARCHICAL = 8; // bf=8, d=3, 97.2% recall

    // Build parameters (same as IVFIndexBuildBenchmarks)
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
    public void searchLloyd(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.lloydIndex.search(query, TOP_K, NPROBE_LLOYD);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    @Benchmark
    public void searchMiniBatch(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.miniBatchIndex.search(query, TOP_K, NPROBE_MINI_BATCH);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    @Benchmark
    public void searchHierarchical(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.hierarchicalIndex.search(query, TOP_K, NPROBE_HIERARCHICAL);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    /**
     * Batch search benchmarks - process multiple queries in parallel
     */
    @Benchmark
    public void searchBatchLloyd(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.lloydIndex.searchBatch(queries, TOP_K, NPROBE_LLOYD);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    @Benchmark
    public void searchBatchMiniBatch(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.miniBatchIndex.searchBatch(queries, TOP_K, NPROBE_MINI_BATCH);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    @Benchmark
    public void searchBatchHierarchical(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.hierarchicalIndex.searchBatch(queries, TOP_K, NPROBE_HIERARCHICAL);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    @State(Scope.Benchmark)
    public static class SearchState {

        private static final int QUERY_BATCH_SIZE = 32;

        @Param("embeddings.csv")
        public String embeddingsPath;

        @Param({"L2SQ_DISTANCE", "DOT_PRODUCT", "COSINE_DISTANCE"})
        public String metricTypeName;

        @Param({"SCALAR", "VECTOR_API", "SIMSIMD"})
        public String metricEngineName;

        float[][] data;
        Metric.Type metricType;
        Metric.Engine metricEngine;

        IVFIndex lloydIndex;
        IVFIndex miniBatchIndex;
        IVFIndex hierarchicalIndex;

        // Pre-generated queries from actual data (more realistic than random)
        private float[][] preGeneratedQueries;
        private int queryIndex;

        // Index memory stats
        long lloydIndexMemoryMB;
        long miniBatchIndexMemoryMB;
        long hierarchicalIndexMemoryMB;

        @Setup(Level.Trial)
        public void setup() throws IOException {
            System.out.println("[Setup] Loading data...");
            this.data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(embeddingsPath));
            this.metricType = Metric.Type.valueOf(metricTypeName);
            this.metricEngine = Metric.Engine.valueOf(metricEngineName);

            int dimension = data[0].length;
            System.out.printf("[Setup] Loaded %d vectors, %d dimensions%n", data.length, dimension);

            // Pre-generate queries from random samples of actual data
            Random rng = new Random(123456);
            int numQueries = 1000;
            preGeneratedQueries = new float[numQueries][dimension];
            for (int i = 0; i < numQueries; i++) {
                int idx = rng.nextInt(data.length);
                System.arraycopy(data[idx], 0, preGeneratedQueries[i], 0, dimension);
                // Add small noise to make queries slightly different
                for (int d = 0; d < dimension; d++) {
                    preGeneratedQueries[i][d] += (rng.nextFloat() - 0.5f) * 0.01f;
                }
            }
            queryIndex = 0;

            // Build indexes with memory tracking
            System.gc();
            long beforeLloyd = getUsedMemory();
            System.out.println("[Setup] Building Lloyd index...");
            this.lloydIndex = buildIndex(KMeans.Type.LLOYD);
            System.gc();
            lloydIndexMemoryMB = (getUsedMemory() - beforeLloyd) / (1024 * 1024);
            System.out.printf("[Setup] Lloyd: %d clusters, ~%d MB%n",
                lloydIndex.getCountClusters(), lloydIndexMemoryMB);

            long beforeMiniBatch = getUsedMemory();
            System.out.println("[Setup] Building MiniBatch index...");
            this.miniBatchIndex = buildIndex(KMeans.Type.MINI_BATCH);
            System.gc();
            miniBatchIndexMemoryMB = (getUsedMemory() - beforeMiniBatch) / (1024 * 1024);
            System.out.printf("[Setup] MiniBatch: %d clusters, ~%d MB%n",
                miniBatchIndex.getCountClusters(), miniBatchIndexMemoryMB);

            long beforeHierarchical = getUsedMemory();
            System.out.println("[Setup] Building Hierarchical index...");
            this.hierarchicalIndex = buildIndex(KMeans.Type.HIERARCHICAL);
            System.gc();
            hierarchicalIndexMemoryMB = (getUsedMemory() - beforeHierarchical) / (1024 * 1024);
            System.out.printf("[Setup] Hierarchical: %d clusters, ~%d MB%n",
                hierarchicalIndex.getCountClusters(), hierarchicalIndexMemoryMB);

            System.out.println("[Setup] Ready for benchmarks");
        }

        private static long getUsedMemory() {
            return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
        }

        private IVFIndex buildIndex(KMeans.Type type) {
            KMeans<? extends KMeans.ClusteringResult> kMeans =
                createKMeans(type, metricType, metricEngine);
            IVFIndex index = new IVFIndexFlat(kMeans);
            index.build(data);
            return index;
        }

        /**
         * Get next query vector (cycles through pre-generated queries)
         */
        float[] nextQuery() {
            float[] query = preGeneratedQueries[queryIndex];
            queryIndex = (queryIndex + 1) % preGeneratedQueries.length;
            return query;
        }

        /**
         * Get batch of query vectors for batch search benchmarks
         */
        float[][] nextQueryBatch() {
            float[][] batch = new float[QUERY_BATCH_SIZE][];
            for (int i = 0; i < QUERY_BATCH_SIZE; i++) {
                batch[i] = nextQuery();
            }
            return batch;
        }
    }

    /**
     * Auxiliary counters for search performance metrics.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class SearchCounters {

        private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        // Search result metrics
        public int resultCount;
        public int queryCount;

        // Memory metrics (bytes allocated during search)
        public long heapBefore;
        public long heapAfter;
        public long allocatedBytes;

        // GC metrics
        public long gcCountBefore;
        public long gcCountAfter;
        public long gcDuringSearch;

        @Setup(Level.Iteration)
        public void reset() {
            resultCount = 0;
            queryCount = 0;
            allocatedBytes = 0;
            gcDuringSearch = 0;
        }

        public void beforeSearch() {
            heapBefore = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountBefore = getTotalGcCount();
        }

        public void afterSearch(int results) {
            heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountAfter = getTotalGcCount();

            resultCount = results;
            queryCount = 1;
            allocatedBytes = Math.max(0, heapAfter - heapBefore);
            gcDuringSearch = gcCountAfter - gcCountBefore;
        }

        public void afterBatchSearch(int batchResults, int queries) {
            heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountAfter = getTotalGcCount();

            resultCount = batchResults;
            queryCount = queries;
            allocatedBytes = Math.max(0, heapAfter - heapBefore);
            gcDuringSearch = gcCountAfter - gcCountBefore;
        }

        private long getTotalGcCount() {
            long total = 0;
            for (GarbageCollectorMXBean gc : gcBeans) {
                total += gc.getCollectionCount();
            }
            return total;
        }
    }
}
