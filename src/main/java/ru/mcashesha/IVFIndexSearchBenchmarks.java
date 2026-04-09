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

    // ==================== Search Parameters ====================

    /** Number of nearest neighbors to retrieve per query. */
    private static final int TOP_K = 100;

    // Optimal nProbe values for each algorithm at >= 95% recall (from README tuning).
    // All set to 8, which provides the best recall-vs-latency trade-off.

    /** nProbe for Lloyd index (k=128, 96.2% recall). */
    private static final int NPROBE_LLOYD = 8;        // k=128, 96.2% recall

    /** nProbe for MiniBatch index (k=64, 97.8% recall). */
    private static final int NPROBE_MINI_BATCH = 8;   // k=64, 97.8% recall

    /** nProbe for Hierarchical index (bf=8, d=3, 97.2% recall). */
    private static final int NPROBE_HIERARCHICAL = 8; // bf=8, d=3, 97.2% recall

    // ==================== Build Parameters ====================
    // Identical to IVFIndexBuildBenchmarks to ensure consistent index structure.

    /** Lloyd KMeans cluster count. */
    private static final int LLOYD_CLUSTER_COUNT = 128;

    /** Maximum iterations for Lloyd KMeans. */
    private static final int LLOYD_MAX_ITERATIONS = 100;

    /** MiniBatch KMeans cluster count. */
    private static final int MINIBATCH_CLUSTER_COUNT = 64;

    /** Samples per mini-batch update step. */
    private static final int MINIBATCH_BATCH_SIZE = 512;

    /** Maximum mini-batch iterations. */
    private static final int MINIBATCH_MAX_ITERATIONS = 300;

    /** Early stopping threshold for MiniBatch KMeans. */
    private static final int MINIBATCH_MAX_NO_IMPROVEMENT = 30;

    /** Children per node in the hierarchical clustering tree. */
    private static final int HIERARCHICAL_BRANCH_FACTOR = 8;

    /** Maximum tree depth for Hierarchical KMeans. */
    private static final int HIERARCHICAL_MAX_DEPTH = 3;

    /** Minimum leaf cluster size for Hierarchical KMeans. */
    private static final int HIERARCHICAL_MIN_CLUSTER_SIZE = 100;

    /** Maximum Lloyd iterations per level in the hierarchical tree. */
    private static final int HIERARCHICAL_MAX_ITERATIONS_PER_LEVEL = 30;

    /**
     * Creates a fully configured {@link KMeans} instance for the given algorithm type.
     *
     * <p>Uses a fixed random seed (42) for reproducible index construction across
     * benchmark forks. The convergence tolerance is set to {@code 1e-4f} for all types.
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
     * Benchmarks single-query search latency on a Lloyd-built IVF index.
     *
     * <p>Executes a top-{@value #TOP_K} nearest neighbor search probing
     * {@value #NPROBE_LLOYD} clusters. The query vector cycles through 1000
     * pre-generated noisy variants of real data to avoid caching bias.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchLloyd(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.lloydIndex.search(query, TOP_K, NPROBE_LLOYD);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    /**
     * Benchmarks single-query search latency on a MiniBatch-built IVF index.
     *
     * <p>Executes a top-{@value #TOP_K} nearest neighbor search probing
     * {@value #NPROBE_MINI_BATCH} clusters.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchMiniBatch(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.miniBatchIndex.search(query, TOP_K, NPROBE_MINI_BATCH);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    /**
     * Benchmarks single-query search latency on a Hierarchical-built IVF index.
     *
     * <p>Executes a top-{@value #TOP_K} nearest neighbor search probing
     * {@value #NPROBE_HIERARCHICAL} clusters.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchHierarchical(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[] query = state.nextQuery();
        List<IVFIndex.SearchResult> results = state.hierarchicalIndex.search(query, TOP_K, NPROBE_HIERARCHICAL);

        counters.afterSearch(results.size());
        bh.consume(results);
    }

    /**
     * Benchmarks batch search latency on a Lloyd-built IVF index.
     *
     * <p>Processes {@link SearchState#QUERY_BATCH_SIZE} queries in a single
     * {@code searchBatch()} call, which may leverage internal parallelization.
     * This measures throughput-oriented workloads where multiple queries arrive together.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters (tracks batch size)
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchBatchLloyd(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.lloydIndex.searchBatch(queries, TOP_K, NPROBE_LLOYD);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    /**
     * Benchmarks batch search latency on a MiniBatch-built IVF index.
     *
     * <p>Processes {@link SearchState#QUERY_BATCH_SIZE} queries in a single
     * {@code searchBatch()} call.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters (tracks batch size)
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchBatchMiniBatch(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.miniBatchIndex.searchBatch(queries, TOP_K, NPROBE_MINI_BATCH);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    /**
     * Benchmarks batch search latency on a Hierarchical-built IVF index.
     *
     * <p>Processes {@link SearchState#QUERY_BATCH_SIZE} queries in a single
     * {@code searchBatch()} call.
     *
     * @param state    shared state with pre-built indexes and pre-generated queries
     * @param counters per-invocation search performance counters (tracks batch size)
     * @param bh       JMH blackhole to prevent dead-code elimination of results
     */
    @Benchmark
    public void searchBatchHierarchical(SearchState state, SearchCounters counters, Blackhole bh) {
        counters.beforeSearch();

        float[][] queries = state.nextQueryBatch();
        List<List<IVFIndex.SearchResult>> results = state.hierarchicalIndex.searchBatch(queries, TOP_K, NPROBE_HIERARCHICAL);

        counters.afterBatchSearch(results.size(), queries.length);
        bh.consume(results);
    }

    /**
     * JMH shared state that pre-builds all three IVF index variants and pre-generates
     * a pool of realistic query vectors.
     *
     * <p>Scoped to {@link Scope#Benchmark} so the expensive index build and data loading
     * happen only once per trial, shared across all benchmark threads and methods.
     * The three indexes (Lloyd, MiniBatch, Hierarchical) are built during setup with
     * the same metric type and engine, allowing direct comparison of clustering algorithms.
     *
     * <p>Query vectors are derived from 1000 randomly sampled real data points with small
     * additive noise (uniform in [-0.005, +0.005] per dimension), producing queries that
     * are near actual data points for realistic recall measurement.
     */
    @State(Scope.Benchmark)
    public static class SearchState {

        /** Number of queries per batch in batch search benchmarks. */
        private static final int QUERY_BATCH_SIZE = 32;

        /** Path to the embeddings CSV file. JMH parameter, default {@code "embeddings.csv"}. */
        @Param("embeddings.csv")
        public String embeddingsPath;

        /**
         * Distance metric type name. JMH parameter resolved to {@link Metric.Type} during setup.
         */
        @Param({"L2SQ_DISTANCE", "DOT_PRODUCT", "COSINE_DISTANCE"})
        public String metricTypeName;

        /**
         * Distance computation engine name. JMH parameter resolved to {@link Metric.Engine} during setup.
         */
        @Param({"SCALAR", "VECTOR_API", "SIMSIMD"})
        public String metricEngineName;

        /** The loaded embedding vectors used for index construction. */
        float[][] data;

        /** Resolved distance metric type enum. */
        Metric.Type metricType;

        /** Resolved distance computation engine enum. */
        Metric.Engine metricEngine;

        /** Pre-built IVF index using Lloyd KMeans clustering. */
        IVFIndex lloydIndex;

        /** Pre-built IVF index using MiniBatch KMeans clustering. */
        IVFIndex miniBatchIndex;

        /** Pre-built IVF index using Hierarchical KMeans clustering. */
        IVFIndex hierarchicalIndex;

        /**
         * Pool of 1000 pre-generated query vectors derived from real data with small noise.
         * Queries cycle round-robin to ensure diverse access patterns across benchmark iterations.
         */
        private float[][] preGeneratedQueries;

        /** Current position in the round-robin query cycle. */
        private int queryIndex;

        /** Approximate heap memory consumed by the Lloyd index, in megabytes. */
        long lloydIndexMemoryMB;

        /** Approximate heap memory consumed by the MiniBatch index, in megabytes. */
        long miniBatchIndexMemoryMB;

        /** Approximate heap memory consumed by the Hierarchical index, in megabytes. */
        long hierarchicalIndexMemoryMB;

        /**
         * Loads embedding data, pre-generates queries, and builds all three index variants.
         *
         * <p>Called once per JMH trial. The setup sequence is:
         * <ol>
         *   <li>Load embeddings from CSV</li>
         *   <li>Generate 1000 noisy query vectors from randomly sampled real data</li>
         *   <li>Build Lloyd index with memory tracking</li>
         *   <li>Build MiniBatch index with memory tracking</li>
         *   <li>Build Hierarchical index with memory tracking</li>
         * </ol>
         *
         * <p>Memory tracking uses heap delta between GC passes. The estimates are approximate
         * since concurrent allocations and GC behavior can affect the measurements.
         *
         * @throws IOException if the embeddings file cannot be read or parsed
         */
        @Setup(Level.Trial)
        public void setup() throws IOException {
            System.out.println("[Setup] Loading data...");
            this.data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(embeddingsPath));
            this.metricType = Metric.Type.valueOf(metricTypeName);
            this.metricEngine = Metric.Engine.valueOf(metricEngineName);

            int dimension = data[0].length;
            System.out.printf("[Setup] Loaded %d vectors, %d dimensions%n", data.length, dimension);

            // Pre-generate queries: copy a real data point, then perturb each dimension
            // by a small uniform noise in [-0.005, +0.005]. This simulates "near-duplicate"
            // queries that exercise realistic recall/distance computation paths.
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

            // Build all three indexes sequentially, tracking approximate memory per index.
            // GC is forced between builds to isolate memory measurements.
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

        /**
         * Returns the current heap memory used, in bytes.
         *
         * @return heap memory usage from the JMX {@link MemoryMXBean}
         */
        private static long getUsedMemory() {
            return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
        }

        /**
         * Builds an IVF index using the specified KMeans algorithm type with the
         * metric type and engine from the current benchmark parameters.
         *
         * @param type the KMeans algorithm variant to use for clustering
         * @return the fully built IVF index, ready for search queries
         */
        private IVFIndex buildIndex(KMeans.Type type) {
            KMeans<? extends KMeans.ClusteringResult> kMeans =
                createKMeans(type, metricType, metricEngine);
            IVFIndex index = new IVFIndexFlat(kMeans);
            index.build(data);
            return index;
        }

        /**
         * Returns the next query vector from the pre-generated pool, cycling round-robin.
         *
         * <p>The round-robin pattern ensures that benchmark iterations access diverse
         * query vectors, preventing JIT or cache optimizations from skewing results
         * toward a single query's access pattern.
         *
         * @return a pre-generated query vector with small noise added to real data
         */
        float[] nextQuery() {
            float[] query = preGeneratedQueries[queryIndex];
            queryIndex = (queryIndex + 1) % preGeneratedQueries.length;
            return query;
        }

        /**
         * Returns a batch of {@value #QUERY_BATCH_SIZE} query vectors for batch search benchmarks.
         *
         * <p>Each vector in the batch is drawn sequentially from the round-robin query pool
         * via {@link #nextQuery()}, advancing the shared index accordingly.
         *
         * @return an array of {@value #QUERY_BATCH_SIZE} query vectors
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
     * Auxiliary counters for tracking search performance metrics per invocation.
     *
     * <p>Reported alongside JMH's primary timing results via the {@link AuxCounters}
     * mechanism. Tracks result counts, query counts, heap allocation during search,
     * and GC events. Each public field becomes a column in the JMH output.
     *
     * <p>Scoped to {@link Scope#Thread} for thread-safe per-invocation tracking.
     * The {@link #beforeSearch()} method captures pre-search snapshots, and
     * {@link #afterSearch(int)} / {@link #afterBatchSearch(int, int)} compute deltas.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class SearchCounters {

        /** JMX bean for querying heap memory usage. */
        private static final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();

        /** JMX beans for all garbage collectors. */
        private static final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        // ==================== Search Result Metrics ====================

        /** Number of results returned by the last search invocation. */
        public int resultCount;

        /** Number of queries processed in the last invocation (1 for single, batch size for batch). */
        public int queryCount;

        // ==================== Memory Metrics ====================

        /** Heap usage in bytes before the search invocation. */
        public long heapBefore;

        /** Heap usage in bytes after the search invocation. */
        public long heapAfter;

        /** Approximate bytes allocated during the search (clamped to >= 0). */
        public long allocatedBytes;

        // ==================== GC Metrics ====================

        /** Cumulative GC invocation count before the search. */
        public long gcCountBefore;

        /** Cumulative GC invocation count after the search. */
        public long gcCountAfter;

        /** Number of GC events that occurred during the search invocation. */
        public long gcDuringSearch;

        /**
         * Resets all counters to zero before each JMH iteration.
         *
         * <p>This ensures that counters from a previous iteration do not bleed into
         * the current measurement window.
         */
        @Setup(Level.Iteration)
        public void reset() {
            resultCount = 0;
            queryCount = 0;
            allocatedBytes = 0;
            gcDuringSearch = 0;
        }

        /**
         * Captures pre-search heap and GC snapshots.
         *
         * <p>Called at the beginning of each benchmark method, immediately before
         * the search operation. These snapshots serve as the baseline for computing
         * allocation and GC deltas in {@link #afterSearch(int)}.
         */
        public void beforeSearch() {
            heapBefore = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountBefore = getTotalGcCount();
        }

        /**
         * Records post-search metrics for a single-query search invocation.
         *
         * <p>Computes heap allocation delta (clamped to zero if GC reduced heap)
         * and counts any GC events that occurred during the search.
         *
         * @param results the number of search results returned
         */
        public void afterSearch(int results) {
            heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountAfter = getTotalGcCount();

            resultCount = results;
            queryCount = 1;
            // Clamp to zero: heap may decrease if GC ran during the search
            allocatedBytes = Math.max(0, heapAfter - heapBefore);
            gcDuringSearch = gcCountAfter - gcCountBefore;
        }

        /**
         * Records post-search metrics for a batch search invocation.
         *
         * <p>Similar to {@link #afterSearch(int)} but tracks the number of queries
         * in the batch separately from the number of result lists returned.
         *
         * @param batchResults the number of result lists returned (one per query in the batch)
         * @param queries      the number of queries in the batch
         */
        public void afterBatchSearch(int batchResults, int queries) {
            heapAfter = memoryBean.getHeapMemoryUsage().getUsed();
            gcCountAfter = getTotalGcCount();

            resultCount = batchResults;
            queryCount = queries;
            allocatedBytes = Math.max(0, heapAfter - heapBefore);
            gcDuringSearch = gcCountAfter - gcCountBefore;
        }

        /**
         * Returns the total number of GC events across all collectors.
         *
         * @return cumulative GC invocation count summed over all {@link GarbageCollectorMXBean} instances
         */
        private long getTotalGcCount() {
            long total = 0;
            for (GarbageCollectorMXBean gc : gcBeans) {
                total += gc.getCollectionCount();
            }
            return total;
        }
    }
}
