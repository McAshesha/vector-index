package ru.mcashesha;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

import static java.util.concurrent.TimeUnit.SECONDS;

@BenchmarkMode(Mode.SampleTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 10, timeUnit = SECONDS)
@Measurement(iterations = 10, time = 30, timeUnit = SECONDS)
@Fork(3)
@Threads(1)
@State(Scope.Thread)
public class IVFIndexSearchBenchmarks {

    private static final int TOP_K = 100;
    private static final int NPROBE_LLOYD = 16;
    private static final int NPROBE_MINI_BATCH = 16;
    private static final int NPROBE_HIERARCHICAL = 8;

    private static KMeans<? extends KMeans.ClusteringResult> createKMeans(
        KMeans.Type type,
        Metric.Type metricType,
        Metric.Engine metricEngine
    ) {
        KMeans.Builder builder = KMeans.newBuilder(type, metricType, metricEngine);

        switch (type) {
            case HIERARCHICAL:
                builder
                    .withBranchFactor(2)
                    .withMaxDepth(6)
                    .withMinClusterSize(12)
                    .withMaxIterationsPerLevel(50)
                    .withTolerance(1e-3f);
                break;
            case MINI_BATCH:
                builder
                    .withBatchSize(512)
                    .withMaxNoImprovementIterations(100)
                    .withMaxIterations(800)
                    .withClusterCount(64)
                    .withTolerance(1e-3f);
                break;
            case LLOYD:
                builder
                    .withMaxIterations(100)
                    .withClusterCount(64)
                    .withTolerance(1e-3f);
                break;
            default:
                throw new IllegalStateException("Unsupported KMeans type: " + type);
        }

        return builder.build();
    }

    @Benchmark
    public List<IVFIndex.SearchResult> searchLloyd(IVFIndexSearchBenchmarks.SearchState state) {
        float[] query = state.nextRandomQuery(state.lloydIndex.getDimension());
        return state.lloydIndex.search(query, TOP_K, NPROBE_LLOYD);
    }

    @Benchmark
    public List<IVFIndex.SearchResult> searchMiniBatch(IVFIndexSearchBenchmarks.SearchState state) {
        float[] query = state.nextRandomQuery(state.miniBatchIndex.getDimension());
        return state.miniBatchIndex.search(query, TOP_K, NPROBE_MINI_BATCH);
    }

    @Benchmark
    public List<IVFIndex.SearchResult> searchHierarchical(IVFIndexSearchBenchmarks.SearchState state) {
        float[] query = state.nextRandomQuery(state.hierarchicalIndex.getDimension());
        return state.hierarchicalIndex.search(query, TOP_K, NPROBE_HIERARCHICAL);
    }

    @State(Scope.Benchmark)
    public static class SearchState {

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

        Random queryRandom;

        @Setup(Level.Trial)
        public void setup() throws IOException {
            this.data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(embeddingsPath));
            this.metricType = Metric.Type.valueOf(metricTypeName);
            this.metricEngine = Metric.Engine.valueOf(metricEngineName);

            this.queryRandom = new Random(123456);

            this.lloydIndex = buildIndex(KMeans.Type.LLOYD);
            this.miniBatchIndex = buildIndex(KMeans.Type.MINI_BATCH);
            this.hierarchicalIndex = buildIndex(KMeans.Type.HIERARCHICAL);
        }

        private IVFIndex buildIndex(KMeans.Type type) {
            KMeans<? extends KMeans.ClusteringResult> kMeans =
                createKMeans(type, metricType, metricEngine);
            IVFIndex index = new IVFIndexFlat(kMeans);
            index.build(data);
            return index;
        }

        float[] nextRandomQuery(int dimension) {
            float[] vector = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                float v = queryRandom.nextFloat();
                vector[i] = queryRandom.nextBoolean() ? v : -v;
            }
            return vector;
        }
    }
}
