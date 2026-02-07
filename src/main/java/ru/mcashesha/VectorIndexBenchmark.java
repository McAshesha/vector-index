package ru.mcashesha;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

public class VectorIndexBenchmark {
    private static final Random RANDOM = new Random();

    public static void main(String[] args) throws IOException {
        float[][] data = EmbeddingCsvLoader.loadEmbeddings(
            Paths.get("embeddings.csv")
        );

        KMeans<? extends KMeans.ClusteringResult> hierarchicalKMeans =
            KMeans.newBuilder(KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withBranchFactor(2)
                .withMaxDepth(6)
                .withMinClusterSize(12)
                .withMaxIterationsPerLevel(50)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=8
        KMeans<? extends KMeans.ClusteringResult> batchKMeans =
            KMeans.newBuilder(KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withBatchSize(512)
                .withMaxNoImprovementIterations(100)
                .withMaxIterations(800)
                .withClusterCount(64)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=16
        KMeans<? extends KMeans.ClusteringResult> lloydKMeans =
            KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
                .withMaxIterations(100)
                .withClusterCount(64)
                .withTolerance(1e-3f)
                .withRandom(RANDOM)
                .build(); // search nprobe=16

        IVFIndex idx = new IVFIndexFlat(lloydKMeans);

        long millis = System.currentTimeMillis();
        idx.build(data);
        long duration = System.currentTimeMillis() - millis;

        System.out.println("Время билда индекса - " + duration + " мс");

        float[] qry = getRandomVector(idx.getDimension());

        millis = System.currentTimeMillis();
        idx.search(qry, 100, 16);
        duration = System.currentTimeMillis() - millis;

        System.out.println("Время поиска по индексу - " + duration + " мс");

    }

    private static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
            vector[i] = RANDOM.nextFloat() * (RANDOM.nextBoolean() ? 1 : -1);
        return vector;
    }

}
