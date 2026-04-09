package ru.mcashesha.kmeans;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

class KMeansBuilderTest {

    private static final long SEED = 42L;
    private static boolean engineAvailable;

    @BeforeAll
    static void checkEngineAvailable() {
        try {
            Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
            engineAvailable = true;
        } catch (ExceptionInInitializerError | NoClassDefFoundError | UnsatisfiedLinkError e) {
            engineAvailable = false;
            assumeTrue(false, "Native library not available, skipping tests");
        }
    }

    // ==================== Builder null validation ====================

    @Test
    void builder_nullType_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            KMeans.newBuilder(null, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void builder_nullMetricType_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            KMeans.newBuilder(KMeans.Type.LLOYD, null, Metric.Engine.SCALAR));
    }

    @Test
    void builder_nullMetricEngine_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, null));
    }

    @Test
    void builder_nullRandom_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withRandom(null));
    }

    // ==================== Builder builds correct types ====================

    @Test
    void builder_lloydType_buildsCorrectImplementation() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(4).withRandom(new Random(SEED)).build();

        assertNotNull(kmeans);
        assertEquals(Metric.Type.L2SQ_DISTANCE, kmeans.getMetricType());
        assertEquals(Metric.Engine.SCALAR, kmeans.getMetricEngine());
    }

    @Test
    void builder_miniBatchType_buildsCorrectImplementation() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.DOT_PRODUCT, Metric.Engine.VECTOR_API
        ).withClusterCount(4).withBatchSize(32).withRandom(new Random(SEED)).build();

        assertNotNull(kmeans);
        assertEquals(Metric.Type.DOT_PRODUCT, kmeans.getMetricType());
        assertEquals(Metric.Engine.VECTOR_API, kmeans.getMetricEngine());
    }

    @Test
    void builder_hierarchicalType_buildsCorrectImplementation() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.COSINE_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(3).withMaxDepth(4).withRandom(new Random(SEED)).build();

        assertNotNull(kmeans);
        assertEquals(Metric.Type.COSINE_DISTANCE, kmeans.getMetricType());
        assertEquals(Metric.Engine.SCALAR, kmeans.getMetricEngine());
    }

    // ==================== Builder defaults produce working KMeans ====================

    @ParameterizedTest
    @EnumSource(KMeans.Type.class)
    void builder_defaultParams_producesWorkingKMeans(KMeans.Type type) {
        float[][] data = new float[50][8];
        Random rng = new Random(SEED);
        for (float[] row : data)
            for (int d = 0; d < 8; d++)
                row[d] = rng.nextFloat();

        KMeans.Builder builder = KMeans.newBuilder(type, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
            .withRandom(new Random(SEED));

        // For hierarchical, branchFactor defaults to 2 which is valid
        KMeans<? extends KMeans.ClusteringResult> kmeans = builder.build();
        KMeans.ClusteringResult result = kmeans.fit(data);

        assertNotNull(result);
        assertNotNull(result.getCentroids());
        assertTrue(result.getCentroids().length > 0);
        assertEquals(50, result.getClusterAssignments().length);
    }

    // ==================== Builder chaining ====================

    @Test
    void builder_chainingReturnsBuilder() {
        KMeans.Builder builder = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);

        // Verify fluent chaining returns the same builder reference
        KMeans.Builder same = builder
            .withClusterCount(5)
            .withMaxIterations(100)
            .withTolerance(1e-5f)
            .withRandom(new Random(SEED));

        assertNotNull(same);
        KMeans<? extends KMeans.ClusteringResult> kmeans = same.build();
        assertNotNull(kmeans);
    }

    @Test
    void builder_miniBatchChaining() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        )
            .withClusterCount(3)
            .withBatchSize(64)
            .withMaxIterations(200)
            .withTolerance(1e-3f)
            .withMaxNoImprovementIterations(30)
            .withRandom(new Random(SEED))
            .build();

        assertNotNull(kmeans);
    }

    @Test
    void builder_hierarchicalChaining() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        )
            .withBranchFactor(4)
            .withMaxDepth(5)
            .withMinClusterSize(10)
            .withMaxIterationsPerLevel(50)
            .withTolerance(1e-4f)
            .withRandom(new Random(SEED))
            .build();

        assertNotNull(kmeans);
    }

    // ==================== branchFactor auto-adjusts minClusterSize ====================

    @Test
    void builder_branchFactor_autoAdjustsMinClusterSize() {
        // branchFactor=8 => minClusterSize should auto-adjust to max(2*8, 2) = 16
        // We can verify this by seeing that small data becomes a single leaf
        float[][] data = new float[15][4];
        Random rng = new Random(SEED);
        for (float[] row : data)
            for (int d = 0; d < 4; d++)
                row[d] = rng.nextFloat();

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(8).withMaxDepth(4).withRandom(new Random(SEED)).build().fit(data);

        // With 15 points and auto-adjusted minClusterSize=16, no splitting should occur
        assertEquals(1, result.getCentroids().length);
    }

    @Test
    void builder_explicitMinClusterSize_overridesAuto() {
        // Explicitly set minClusterSize=2, which should allow splitting
        float[][] data = new float[20][4];
        Random rng = new Random(SEED);
        for (float[] row : data)
            for (int d = 0; d < 4; d++)
                row[d] = rng.nextFloat() * 100;

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(8).withMinClusterSize(2).withMaxDepth(4)
            .withRandom(new Random(SEED)).build().fit(data);

        assertTrue(result.getCentroids().length >= 2, "With minClusterSize=2, splitting should occur");
    }

    // ==================== LloydKMeans constructor validation ====================

    @Test
    void lloydKMeans_zeroClusterCount_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(0, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, new Random()));
    }

    @Test
    void lloydKMeans_negativeClusterCount_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(-1, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, new Random()));
    }

    @Test
    void lloydKMeans_nullMetricType_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(3, null, Metric.Engine.SCALAR, 100, 1e-4f, new Random()));
    }

    @Test
    void lloydKMeans_nullEngine_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(3, Metric.Type.L2SQ_DISTANCE, null, 100, 1e-4f, new Random()));
    }

    @Test
    void lloydKMeans_zeroMaxIterations_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(3, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 0, 1e-4f, new Random()));
    }

    @Test
    void lloydKMeans_negativeTolerance_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(3, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, -1f, new Random()));
    }

    @Test
    void lloydKMeans_nullRandom_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new LloydKMeans(3, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, null));
    }

    // ==================== MiniBatchKMeans constructor validation ====================

    @Test
    void miniBatchKMeans_zeroClusterCount_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(0, 32, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, 50, new Random()));
    }

    @Test
    void miniBatchKMeans_zeroBatchSize_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 0, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, 50, new Random()));
    }

    @Test
    void miniBatchKMeans_nullMetricType_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 32, null, Metric.Engine.SCALAR, 100, 1e-4f, 50, new Random()));
    }

    @Test
    void miniBatchKMeans_zeroMaxIterations_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 32, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 0, 1e-4f, 50, new Random()));
    }

    @Test
    void miniBatchKMeans_zeroMaxNoImprovement_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 32, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, 0, new Random()));
    }

    @Test
    void miniBatchKMeans_negativeTolerance_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 32, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, -1f, 50, new Random()));
    }

    @Test
    void miniBatchKMeans_nullRandom_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new MiniBatchKMeans(3, 32, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 100, 1e-4f, 50, null));
    }

    // ==================== HierarchicalKMeans constructor validation ====================

    @Test
    void hierarchicalKMeans_branchFactorOne_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(1, 4, 4, 30, 1e-4f, new Random(),
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_zeroMaxDepth_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 0, 4, 30, 1e-4f, new Random(),
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_zeroMinClusterSize_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 0, 30, 1e-4f, new Random(),
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_zeroMaxIterationsPerLevel_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 4, 0, 1e-4f, new Random(),
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_negativeTolerance_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 4, 30, -1f, new Random(),
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_nullRandom_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 4, 30, 1e-4f, null,
                Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_nullMetricType_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 4, 30, 1e-4f, new Random(),
                null, Metric.Engine.SCALAR));
    }

    @Test
    void hierarchicalKMeans_nullEngine_throws() {
        assertThrows(IllegalArgumentException.class, () ->
            new HierarchicalKMeans(2, 4, 4, 30, 1e-4f, new Random(),
                Metric.Type.L2SQ_DISTANCE, null));
    }

    // ==================== Predict validation ====================

    @Test
    void lloydPredict_nullData_throws() {
        LloydKMeans kmeans = new LloydKMeans(2, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 50, 1e-4f, new Random(SEED));
        float[][] trainData = generateSimpleData();
        LloydKMeans.Result model = kmeans.fit(trainData);

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(null, model));
    }

    @Test
    void lloydPredict_emptyData_throws() {
        LloydKMeans kmeans = new LloydKMeans(2, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 50, 1e-4f, new Random(SEED));
        float[][] trainData = generateSimpleData();
        LloydKMeans.Result model = kmeans.fit(trainData);

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(new float[0][], model));
    }

    @Test
    void lloydPredict_nullModel_throws() {
        LloydKMeans kmeans = new LloydKMeans(2, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 50, 1e-4f, new Random(SEED));
        float[][] data = generateSimpleData();

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(data, null));
    }

    @Test
    void lloydPredict_dimensionMismatch_throws() {
        LloydKMeans kmeans = new LloydKMeans(2, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 50, 1e-4f, new Random(SEED));
        float[][] trainData = generateSimpleData(); // 4-dimensional
        LloydKMeans.Result model = kmeans.fit(trainData);

        float[][] wrongDimData = {{1, 2}}; // 2-dimensional
        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(wrongDimData, model));
    }

    @Test
    void miniBatchPredict_nullData_throws() {
        MiniBatchKMeans kmeans = new MiniBatchKMeans(2, 32, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 100, 1e-4f, 50, new Random(SEED));
        float[][] trainData = generateSimpleData();
        MiniBatchKMeans.Result model = kmeans.fit(trainData);

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(null, model));
    }

    @Test
    void miniBatchPredict_nullModel_throws() {
        MiniBatchKMeans kmeans = new MiniBatchKMeans(2, 32, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 100, 1e-4f, 50, new Random(SEED));
        float[][] data = generateSimpleData();

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(data, null));
    }

    @Test
    void hierarchicalPredict_nullData_throws() {
        HierarchicalKMeans kmeans = new HierarchicalKMeans(2, 3, 4, 30, 1e-4f,
            new Random(SEED), Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);
        float[][] trainData = generateSimpleData();
        HierarchicalKMeans.Result model = kmeans.fit(trainData);

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(null, model));
    }

    @Test
    void hierarchicalPredict_nullModel_throws() {
        HierarchicalKMeans kmeans = new HierarchicalKMeans(2, 3, 4, 30, 1e-4f,
            new Random(SEED), Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);
        float[][] data = generateSimpleData();

        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(data, null));
    }

    @Test
    void hierarchicalPredict_dimensionMismatch_throws() {
        HierarchicalKMeans kmeans = new HierarchicalKMeans(2, 3, 4, 30, 1e-4f,
            new Random(SEED), Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);
        float[][] trainData = generateSimpleData(); // 4-dimensional
        HierarchicalKMeans.Result model = kmeans.fit(trainData);

        float[][] wrongDimData = {{1, 2}}; // 2-dimensional
        assertThrows(IllegalArgumentException.class, () -> kmeans.predict(wrongDimData, model));
    }

    private static float[][] generateSimpleData() {
        Random rng = new Random(SEED);
        float[][] data = new float[20][4];
        for (int i = 0; i < 10; i++)
            for (int d = 0; d < 4; d++)
                data[i][d] = rng.nextFloat();
        for (int i = 10; i < 20; i++)
            for (int d = 0; d < 4; d++)
                data[i][d] = 100 + rng.nextFloat();
        return data;
    }
}
