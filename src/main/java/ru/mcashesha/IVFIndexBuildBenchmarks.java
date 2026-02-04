package ru.mcashesha;

import java.io.IOException;
import java.nio.file.Paths;
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

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 1, time = 30, timeUnit = SECONDS)
@Measurement(iterations = 5, time = 120, timeUnit = SECONDS)
@Fork(1)
@Threads(1)
@State(Scope.Thread)
public class IVFIndexBuildBenchmarks {

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
    public IVFIndex buildLloydIndex(IVFIndexBuildBenchmarks.BuildState state) {
        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.LLOYD, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);
        return idx;
    }

    @Benchmark
    public IVFIndex buildMiniBatchIndex(IVFIndexBuildBenchmarks.BuildState state) {
        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.MINI_BATCH, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);
        return idx;
    }

    @Benchmark
    public IVFIndex buildHierarchicalIndex(IVFIndexBuildBenchmarks.BuildState state) {
        KMeans<? extends KMeans.ClusteringResult> kMeans =
            createKMeans(KMeans.Type.HIERARCHICAL, state.metricType, state.metricEngine);
        IVFIndex idx = new IVFIndexFlat(kMeans);
        idx.build(state.data);
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

        @Setup(Level.Trial)
        public void setup() throws IOException {
            this.data = EmbeddingCsvLoader.loadEmbeddings(Paths.get(embeddingsPath));
            this.metricType = Metric.Type.valueOf(metricTypeName);
            this.metricEngine = Metric.Engine.valueOf(metricEngineName);
        }
    }
}
