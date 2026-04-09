package ru.mcashesha;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

/**
 * Quick demo and smoke-test runner for the IVF vector index.
 *
 * <p>This class provides a minimal end-to-end demonstration that loads embedding data
 * from a CSV file, builds a single IVF index using Lloyd KMeans with the Java Vector API
 * engine, and performs one search query. It prints wall-clock build time and search time
 * to standard output.
 *
 * <p>Three KMeans configurations are defined (Hierarchical, MiniBatch, Lloyd) to allow
 * quick experimentation by swapping which one is passed to the index constructor.
 * By default, only the Lloyd configuration is used for the actual build.
 *
 * <p><b>Usage:</b>
 * <pre>
 * java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
 *   -Djava.library.path=src/main/native/build \
 *   -cp target/benchmarks.jar ru.mcashesha.VectorIndexBenchmark
 * </pre>
 *
 * <p>Requires an {@code embeddings.csv} file in the project root directory.
 * The CSV format is: {@code id, metadata, "[v0, v1, ..., v511]"}.
 *
 * @see IVFIndexFlat
 * @see KMeans
 */
public class VectorIndexBenchmark {

    /** Shared random number generator used for KMeans initialization and query generation. */
    private static final Random RANDOM = new Random();

    /**
     * Entry point for the demo runner.
     *
     * <p>Loads embeddings from {@code embeddings.csv}, configures three KMeans variants
     * (only Lloyd is used), builds the IVF index, then performs a single search with a
     * random query vector. Prints build and search durations in milliseconds.
     *
     * @param args command-line arguments (unused)
     * @throws IOException if the embeddings CSV file cannot be read
     */
    public static void main(String[] args) throws IOException {
        // Load all embedding vectors from the CSV file in the project root
        float[][] data = EmbeddingCsvLoader.loadEmbeddings(
            Paths.get("embeddings.csv")
        );

        // Configure Hierarchical KMeans: binary tree (bf=2) with depth 6,
        // yielding up to 2^6 = 64 leaf clusters. Suitable for nProbe=8 at search time.
        KMeans<? extends KMeans.ClusteringResult> hierarchicalKMeans =
            KMeans.newBuilder(KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withBranchFactor(2)
                .withMaxDepth(6)
                .withMinClusterSize(12)
                .withMaxIterationsPerLevel(50)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=8

        // Configure MiniBatch KMeans: stochastic updates with 512-sample batches,
        // 64 clusters, and early stopping after 100 iterations with no improvement.
        KMeans<? extends KMeans.ClusteringResult> batchKMeans =
            KMeans.newBuilder(KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withBatchSize(512)
                .withMaxNoImprovementIterations(100)
                .withMaxIterations(800)
                .withClusterCount(64)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=16

        // Configure Lloyd KMeans: classic full-pass iterative algorithm,
        // 64 clusters with convergence tolerance 1e-3 and up to 100 iterations.
        KMeans<? extends KMeans.ClusteringResult> lloydKMeans =
            KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withMaxIterations(100)
                .withClusterCount(64)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=16

        // Build the IVF index using Lloyd KMeans (swap with hierarchicalKMeans or
        // batchKMeans above to try different clustering strategies)
        IVFIndex idx = new IVFIndexFlat(lloydKMeans);

        long millis = System.currentTimeMillis();
        idx.build(data);
        long duration = System.currentTimeMillis() - millis;

        System.out.println("Время билда индекса - " + duration + " мс");

        // Generate a random query vector matching the index dimensionality
        float[] qry = getRandomVector(idx.getDimension());

        // Perform a single search: retrieve top-100 nearest neighbors, probing 16 clusters
        millis = System.currentTimeMillis();
        idx.search(qry, 100, 16);
        duration = System.currentTimeMillis() - millis;

        System.out.println("Время поиска по индексу - " + duration + " мс");

    }

    /**
     * Generates a random float vector with values uniformly distributed in [-1, 1].
     *
     * <p>Each component is independently sampled: a random float in [0, 1) is generated,
     * then its sign is randomly flipped with 50% probability.
     *
     * @param dimension the number of dimensions (components) for the vector
     * @return a new float array of the specified dimension with random values
     */
    private static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
            vector[i] = RANDOM.nextFloat() * (RANDOM.nextBoolean() ? 1 : -1);
        return vector;
    }

}
