package ru.mcashesha.kmeans;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KMeansTest {

    private static final long SEED = 42L;
    private static final int DIMENSION = 8;

    private static float[][] generateClusters(Random rng, int pointsPerCluster,
        int clusterCount, int dimension, float spread) {
        float[][] data = new float[pointsPerCluster * clusterCount][dimension];
        for (int c = 0; c < clusterCount; c++) {
            float[] center = new float[dimension];
            for (int d = 0; d < dimension; d++)
                center[d] = c * spread;

            for (int p = 0; p < pointsPerCluster; p++) {
                int idx = c * pointsPerCluster + p;
                for (int d = 0; d < dimension; d++)
                    data[idx][d] = center[d] + (rng.nextFloat() - 0.5f) * 0.1f;
            }
        }
        return data;
    }

    // ==================== Lloyd KMeans ====================

    @Test
    void lloydFit_wellSeparatedClusters_assignsCorrectly() {
        Random rng = new Random(SEED);
        int k = 3;
        int pointsPerCluster = 50;
        float[][] data = generateClusters(rng, pointsPerCluster, k, DIMENSION, 100f);

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(k).withMaxIterations(100).withRandom(new Random(SEED)).build();

        KMeans.ClusteringResult result = kmeans.fit(data);

        assertValidResult(result, data.length, k);
        assertHomogeneousClusters(result.getClusterAssignments(), pointsPerCluster, k);
    }

    @Test
    void lloydFit_singleCluster_allPointsSameLabel() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 30, 1, DIMENSION, 0f);

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(1).withRandom(new Random(SEED)).build();

        KMeans.ClusteringResult result = kmeans.fit(data);

        assertValidResult(result, data.length, 1);
        for (int label : result.getClusterAssignments())
            assertEquals(0, label);
    }

    @Test
    void lloydFit_clusterCountEqualsDataSize() {
        float[][] data = new float[5][DIMENSION];
        Random rng = new Random(SEED);
        for (float[] row : data)
            for (int d = 0; d < DIMENSION; d++)
                row[d] = rng.nextFloat();

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(5).withRandom(new Random(SEED)).build();

        KMeans.ClusteringResult result = kmeans.fit(data);

        assertValidResult(result, 5, 5);
    }

    @Test
    void lloydPredict_assignsToNearestCentroid() {
        Random rng = new Random(SEED);
        int k = 3;
        float[][] trainData = generateClusters(rng, 50, k, DIMENSION, 100f);

        LloydKMeans kmeans = new LloydKMeans(k, Metric.Type.L2SQ_DISTANCE,
            Metric.Engine.SCALAR, 100, 1e-4f, new Random(SEED));

        LloydKMeans.Result model = kmeans.fit(trainData);
        float[][] testData = generateClusters(new Random(123), 10, k, DIMENSION, 100f);
        int[] predicted = kmeans.predict(testData, model);

        assertEquals(testData.length, predicted.length);
        for (int label : predicted)
            assertTrue(label >= 0 && label < k);
    }

    @Test
    void lloydFit_lossDecreases() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 100, 4, DIMENSION, 10f);

        KMeans.ClusteringResult r1 = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(4).withMaxIterations(1).withRandom(new Random(SEED)).build().fit(data);

        KMeans.ClusteringResult r100 = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(4).withMaxIterations(100).withRandom(new Random(SEED)).build().fit(data);

        assertTrue(r100.getLoss() <= r1.getLoss(),
            "Loss after 100 iterations (" + r100.getLoss() + ") should be <= loss after 1 (" + r1.getLoss() + ")");
    }

    @ParameterizedTest
    @EnumSource(Metric.Type.class)
    void lloydFit_allMetricTypes(Metric.Type metricType) {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 30, 3, DIMENSION, 50f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.LLOYD, metricType, Metric.Engine.SCALAR
        ).withClusterCount(3).withMaxIterations(50).withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, 3);
    }

    @Test
    void lloydFit_clusterCountGreaterThanData_throws() {
        float[][] data = new float[3][DIMENSION];

        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(10).withRandom(new Random(SEED)).build();

        assertThrows(IllegalArgumentException.class, () -> kmeans.fit(data));
    }

    @Test
    void lloydFit_nullData_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).build();

        assertThrows(IllegalArgumentException.class, () -> kmeans.fit(null));
    }

    @Test
    void lloydFit_emptyData_throws() {
        KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
            KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).build();

        assertThrows(IllegalArgumentException.class, () -> kmeans.fit(new float[0][]));
    }

    // ==================== MiniBatch KMeans ====================

    @Test
    void miniBatchFit_wellSeparatedClusters_assignsCorrectly() {
        Random rng = new Random(SEED);
        int k = 3;
        int pointsPerCluster = 50;
        float[][] data = generateClusters(rng, pointsPerCluster, k, DIMENSION, 100f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(k).withBatchSize(32).withMaxIterations(200)
            .withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, k);
        assertHomogeneousClusters(result.getClusterAssignments(), pointsPerCluster, k);
    }

    @Test
    void miniBatchFit_singleCluster() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 30, 1, DIMENSION, 0f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(1).withBatchSize(16).withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, 1);
    }

    @Test
    void miniBatchFit_batchSizeLargerThanData() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 10, 2, DIMENSION, 50f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withBatchSize(1000).withMaxIterations(50)
            .withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, 2);
    }

    @Test
    void miniBatchPredict() {
        Random rng = new Random(SEED);
        int k = 3;
        float[][] trainData = generateClusters(rng, 50, k, DIMENSION, 100f);

        MiniBatchKMeans kmeans = new MiniBatchKMeans(k, 32,
            Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 200, 1e-4f, 50, new Random(SEED));

        MiniBatchKMeans.Result model = kmeans.fit(trainData);
        float[][] testData = generateClusters(new Random(123), 10, k, DIMENSION, 100f);
        int[] predicted = kmeans.predict(testData, model);

        assertEquals(testData.length, predicted.length);
        for (int label : predicted)
            assertTrue(label >= 0 && label < k);
    }

    @Test
    void miniBatchFit_earlyStop() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 50, 2, DIMENSION, 100f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(2).withBatchSize(64).withMaxIterations(10000)
            .withMaxNoImprovementIterations(5).withTolerance(1e-6f)
            .withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, 2);
    }

    @Test
    void miniBatchFit_clusterSizesConsistentWithAssignments() {
        Random rng = new Random(SEED);
        int k = 3;
        float[][] data = generateClusters(rng, 50, k, DIMENSION, 100f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withClusterCount(k).withBatchSize(32).withMaxIterations(200)
            .withRandom(new Random(SEED)).build().fit(data);

        int[] assignments = result.getClusterAssignments();
        int[] sizes = result.getClusterSizes();

        int[] recomputedSizes = new int[k];
        for (int label : assignments)
            recomputedSizes[label]++;

        for (int c = 0; c < k; c++)
            assertEquals(recomputedSizes[c], sizes[c],
                "clusterSizes[" + c + "] inconsistent with assignments");
    }

    // ==================== Hierarchical KMeans ====================

    @Test
    void hierarchicalFit_wellSeparatedClusters() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 50, 4, DIMENSION, 100f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(2).withMaxDepth(4).withMaxIterationsPerLevel(30)
            .withRandom(new Random(SEED)).build().fit(data);

        assertNotNull(result.getCentroids());
        assertTrue(result.getCentroids().length > 0);
        assertValidResult(result, data.length, result.getCentroids().length);
    }

    @Test
    void hierarchicalFit_smallData_becomesLeaf() {
        float[][] data = new float[3][DIMENSION];
        Random rng = new Random(SEED);
        for (float[] row : data)
            for (int d = 0; d < DIMENSION; d++)
                row[d] = rng.nextFloat();

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR
        ).withBranchFactor(2).withMaxDepth(4).withMinClusterSize(10)
            .withRandom(new Random(SEED)).build().fit(data);

        assertEquals(1, result.getCentroids().length);
        assertValidResult(result, data.length, 1);
    }

    @Test
    void hierarchicalPredict() {
        Random rng = new Random(SEED);
        float[][] trainData = generateClusters(rng, 50, 4, DIMENSION, 100f);

        HierarchicalKMeans kmeans = new HierarchicalKMeans(2, 4, 4, 30, 1e-4f,
            new Random(SEED), Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);

        HierarchicalKMeans.Result model = kmeans.fit(trainData);
        float[][] testData = generateClusters(new Random(123), 10, 4, DIMENSION, 100f);
        int[] predicted = kmeans.predict(testData, model);

        int leafCount = model.getCentroids().length;
        assertEquals(testData.length, predicted.length);
        for (int label : predicted)
            assertTrue(label >= 0 && label < leafCount);
    }

    @Test
    void hierarchicalFit_depthOne_singleSplit() {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 30, 2, DIMENSION, 100f);

        HierarchicalKMeans kmeans = new HierarchicalKMeans(2, 2, 2, 30, 1e-4f,
            new Random(SEED), Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR);

        HierarchicalKMeans.Result result = kmeans.fit(data);

        assertTrue(result.getCentroids().length >= 2);
        assertValidResult(result, data.length, result.getCentroids().length);
    }

    @ParameterizedTest
    @EnumSource(Metric.Type.class)
    void hierarchicalFit_allMetricTypes(Metric.Type metricType) {
        Random rng = new Random(SEED);
        float[][] data = generateClusters(rng, 30, 2, DIMENSION, 50f);

        KMeans.ClusteringResult result = KMeans.newBuilder(
            KMeans.Type.HIERARCHICAL, metricType, Metric.Engine.SCALAR
        ).withBranchFactor(2).withMaxDepth(3).withMaxIterationsPerLevel(30)
            .withRandom(new Random(SEED)).build().fit(data);

        assertValidResult(result, data.length, result.getCentroids().length);
    }

    // ==================== Cross-algorithm ====================

    @Test
    void allAlgorithms_sameData_produceValidResults() {
        Random rng = new Random(SEED);
        int k = 4;
        float[][] data = generateClusters(rng, 40, k, DIMENSION, 50f);

        for (KMeans.Type type : KMeans.Type.values()) {
            KMeans.Builder builder = KMeans.newBuilder(type, Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR)
                .withRandom(new Random(SEED));

            switch (type) {
                case LLOYD:
                    builder.withClusterCount(k).withMaxIterations(100);
                    break;
                case MINI_BATCH:
                    builder.withClusterCount(k).withBatchSize(32).withMaxIterations(200);
                    break;
                case HIERARCHICAL:
                    builder.withBranchFactor(2).withMaxDepth(4).withMaxIterationsPerLevel(30);
                    break;
            }

            KMeans.ClusteringResult result = builder.build().fit(data);

            assertNotNull(result.getCentroids(), type + ": centroids null");
            assertTrue(result.getCentroids().length > 0, type + ": no centroids");
            assertValidResult(result, data.length, result.getCentroids().length);
            assertTrue(result.getLoss() >= 0, type + ": negative loss");
        }
    }

    // ==================== Helpers ====================

    private void assertValidResult(KMeans.ClusteringResult result, int expectedSamples, int expectedClusters) {
        assertNotNull(result);

        float[][] centroids = result.getCentroids();
        assertNotNull(centroids);
        assertEquals(expectedClusters, centroids.length);

        int[] assignments = result.getClusterAssignments();
        assertNotNull(assignments);
        assertEquals(expectedSamples, assignments.length);

        for (int label : assignments)
            assertTrue(label >= 0 && label < expectedClusters,
                "Label " + label + " out of range [0, " + expectedClusters + ")");

        int[] sizes = result.getClusterSizes();
        assertNotNull(sizes);
        assertEquals(expectedClusters, sizes.length);

        int totalSize = 0;
        for (int s : sizes) {
            assertTrue(s >= 0);
            totalSize += s;
        }
        assertEquals(expectedSamples, totalSize);

        assertTrue(Float.isFinite(result.getLoss()), "Loss should be finite, got " + result.getLoss());
    }

    private void assertHomogeneousClusters(int[] assignments, int pointsPerCluster, int k) {
        Set<Integer>[] labelsPerGroup = new HashSet[k];
        for (int g = 0; g < k; g++)
            labelsPerGroup[g] = new HashSet<>();

        for (int i = 0; i < assignments.length; i++) {
            int group = i / pointsPerCluster;
            labelsPerGroup[group].add(assignments[i]);
        }

        for (int g = 0; g < k; g++) {
            assertEquals(1, labelsPerGroup[g].size(),
                "Group " + g + " should map to exactly one cluster, got " + labelsPerGroup[g]);
        }

        Set<Integer> allLabels = new HashSet<>();
        for (int g = 0; g < k; g++)
            allLabels.addAll(labelsPerGroup[g]);
        assertEquals(k, allLabels.size(), "Each group should map to a distinct cluster");
    }
}
