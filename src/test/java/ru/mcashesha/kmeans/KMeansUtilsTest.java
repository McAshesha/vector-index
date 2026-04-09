package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

class KMeansUtilsTest {

    private static final long SEED = 42L;
    private static final int DIM = 8;
    private static boolean engineAvailable;

    @BeforeAll
    static void checkEngineAvailable() {
        try {
            Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
            engineAvailable = true;
        } catch (ExceptionInInitializerError | NoClassDefFoundError | UnsatisfiedLinkError e) {
            engineAvailable = false;
        }
    }

    private static void requireEngine() {
        assumeTrue(engineAvailable, "Metric.Engine not available (native lib missing)");
    }

    // ==================== validateAndGetDimension ====================

    @Test
    void validateAndGetDimension_validData_returnsDimension() {
        float[][] data = {{1, 2, 3}, {4, 5, 6}};
        assertEquals(3, KMeansUtils.validateAndGetDimension(data));
    }

    @Test
    void validateAndGetDimension_singlePoint_returnsDimension() {
        float[][] data = {{1, 2, 3, 4, 5}};
        assertEquals(5, KMeansUtils.validateAndGetDimension(data));
    }

    @Test
    void validateAndGetDimension_nullFirstPoint_throws() {
        float[][] data = {null, {1, 2}};
        assertThrows(IllegalArgumentException.class, () -> KMeansUtils.validateAndGetDimension(data));
    }

    @Test
    void validateAndGetDimension_zeroDimension_throws() {
        float[][] data = {new float[0]};
        assertThrows(IllegalArgumentException.class, () -> KMeansUtils.validateAndGetDimension(data));
    }

    @Test
    void validateAndGetDimension_inconsistentDimensions_throws() {
        float[][] data = {{1, 2, 3}, {4, 5}};
        assertThrows(IllegalArgumentException.class, () -> KMeansUtils.validateAndGetDimension(data));
    }

    @Test
    void validateAndGetDimension_nullSecondPoint_throws() {
        float[][] data = {{1, 2, 3}, null};
        assertThrows(IllegalArgumentException.class, () -> KMeansUtils.validateAndGetDimension(data));
    }

    // ==================== initializeCentroidsKMeansPlusPlus ====================

    @Test
    void initCentroids_returnsCorrectCount() {
        requireEngine();
        Random rng = new Random(SEED);
        float[][] data = generateData(rng, 100, DIM);
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, data.length, DIM, 5, distFn, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        assertEquals(5, centroids.length);
        for (float[] c : centroids)
            assertEquals(DIM, c.length);
    }

    @Test
    void initCentroids_singleCluster_returnsOnePoint() {
        requireEngine();
        Random rng = new Random(SEED);
        float[][] data = generateData(rng, 50, DIM);
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, data.length, DIM, 1, distFn, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        assertEquals(1, centroids.length);
    }

    @Test
    void initCentroids_kEqualsN_returnsAllPoints() {
        requireEngine();
        float[][] data = {{1, 0}, {0, 1}, {1, 1}};
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, data.length, 2, 3, distFn, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        assertEquals(3, centroids.length);
    }

    @Test
    void initCentroids_centroidsAreDataPoints() {
        requireEngine();
        float[][] data = {{10, 0}, {0, 10}, {-10, 0}, {0, -10}};
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, data.length, 2, 4, distFn, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        for (float[] centroid : centroids) {
            boolean found = false;
            for (float[] point : data) {
                if (Arrays.equals(centroid, point)) {
                    found = true;
                    break;
                }
            }
            assertTrue(found, "Centroid should be a copy of a data point");
        }
    }

    @Test
    void initCentroids_cosineMetric_normalizesCentroids() {
        requireEngine();
        float[][] data = {{3, 4}, {0, 5}, {5, 0}};
        Metric.DistanceFunction distFn = Metric.Type.COSINE_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, data.length, 2, 3, distFn, Metric.Type.COSINE_DISTANCE, new Random(SEED));

        for (float[] c : centroids) {
            float norm = 0;
            for (float v : c) norm += v * v;
            assertEquals(1.0f, (float) Math.sqrt(norm), 1e-4f, "Centroid should be unit-normalized");
        }
    }

    // ==================== assignPointsToClusters ====================

    @Test
    void assignPoints_assignsToNearestCentroid() {
        requireEngine();
        float[][] data = {{0, 0}, {10, 10}, {0.1f, 0.1f}, {9.9f, 9.9f}};
        float[][] centroids = {{0, 0}, {10, 10}};
        int[] labels = new int[4];
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float loss = KMeansUtils.assignPointsToClusters(data, centroids, 2, labels, null, null, distFn);

        assertEquals(0, labels[0]);
        assertEquals(1, labels[1]);
        assertEquals(0, labels[2]);
        assertEquals(1, labels[3]);
        assertTrue(loss >= 0);
    }

    @Test
    void assignPoints_computesClusterSizes() {
        requireEngine();
        float[][] data = {{0, 0}, {10, 10}, {0.1f, 0.1f}, {9.9f, 9.9f}};
        float[][] centroids = {{0, 0}, {10, 10}};
        int[] labels = new int[4];
        int[] sizes = new int[2];
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        KMeansUtils.assignPointsToClusters(data, centroids, 2, labels, null, sizes, distFn);

        assertEquals(2, sizes[0]);
        assertEquals(2, sizes[1]);
    }

    @Test
    void assignPoints_computesPointErrors() {
        requireEngine();
        float[][] data = {{0, 0}, {3, 4}};
        float[][] centroids = {{0, 0}};
        int[] labels = new int[2];
        float[] errors = new float[2];
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        KMeansUtils.assignPointsToClusters(data, centroids, 1, labels, errors, null, distFn);

        assertEquals(0f, errors[0], 1e-6f);
        assertEquals(25f, errors[1], 1e-6f);
    }

    @Test
    void assignPoints_lossIsNonNegative() {
        requireEngine();
        Random rng = new Random(SEED);
        float[][] data = generateData(rng, 100, DIM);
        float[][] centroids = {data[0].clone(), data[50].clone()};
        int[] labels = new int[100];
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float loss = KMeansUtils.assignPointsToClusters(data, centroids, 2, labels, null, null, distFn);

        assertTrue(loss >= 0);
    }

    // ==================== recomputeCentroids ====================

    @Test
    void recomputeCentroids_computesMean() {
        float[][] data = {{0, 0}, {2, 4}, {4, 2}};
        int[] labels = {0, 0, 0};
        float[][] newCentroids = new float[1][2];
        int[] sizes = new int[1];

        KMeansUtils.recomputeCentroids(data, labels, newCentroids, sizes, 1, 2, Metric.Type.L2SQ_DISTANCE);

        assertEquals(2.0f, newCentroids[0][0], 1e-6f);
        assertEquals(2.0f, newCentroids[0][1], 1e-6f);
        assertEquals(3, sizes[0]);
    }

    @Test
    void recomputeCentroids_multipleClusters() {
        float[][] data = {{0, 0}, {2, 0}, {10, 0}, {12, 0}};
        int[] labels = {0, 0, 1, 1};
        float[][] newCentroids = new float[2][2];
        int[] sizes = new int[2];

        KMeansUtils.recomputeCentroids(data, labels, newCentroids, sizes, 2, 2, Metric.Type.L2SQ_DISTANCE);

        assertEquals(1.0f, newCentroids[0][0], 1e-6f);
        assertEquals(11.0f, newCentroids[1][0], 1e-6f);
        assertEquals(2, sizes[0]);
        assertEquals(2, sizes[1]);
    }

    @Test
    void recomputeCentroids_cosineMetric_normalizesResult() {
        float[][] data = {{3, 4}, {6, 8}};
        int[] labels = {0, 0};
        float[][] newCentroids = new float[1][2];
        int[] sizes = new int[1];

        KMeansUtils.recomputeCentroids(data, labels, newCentroids, sizes, 1, 2, Metric.Type.COSINE_DISTANCE);

        float norm = (float) Math.sqrt(newCentroids[0][0] * newCentroids[0][0]
            + newCentroids[0][1] * newCentroids[0][1]);
        assertEquals(1.0f, norm, 1e-4f);
    }

    @Test
    void recomputeCentroids_emptyCluster_centroidRemainsZero() {
        float[][] data = {{1, 1}, {2, 2}};
        int[] labels = {0, 0};
        float[][] newCentroids = new float[2][2];
        int[] sizes = new int[2];

        KMeansUtils.recomputeCentroids(data, labels, newCentroids, sizes, 2, 2, Metric.Type.L2SQ_DISTANCE);

        assertEquals(0, sizes[1]);
        assertEquals(0f, newCentroids[1][0], 1e-6f);
        assertEquals(0f, newCentroids[1][1], 1e-6f);
    }

    // ==================== normalizeSingleCentroid ====================

    @Test
    void normalizeSingleCentroid_producesUnitVector() {
        float[] vec = {3f, 4f};
        KMeansUtils.normalizeSingleCentroid(vec);

        float norm = (float) Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
        assertEquals(1.0f, norm, 1e-6f);
        assertEquals(0.6f, vec[0], 1e-6f);
        assertEquals(0.8f, vec[1], 1e-6f);
    }

    @Test
    void normalizeSingleCentroid_zeroVector_unchanged() {
        float[] vec = {0f, 0f, 0f};
        KMeansUtils.normalizeSingleCentroid(vec);
        assertArrayEquals(new float[]{0f, 0f, 0f}, vec);
    }

    @Test
    void normalizeSingleCentroid_alreadyNormalized_unchanged() {
        float[] vec = {1f, 0f, 0f};
        KMeansUtils.normalizeSingleCentroid(vec);
        assertEquals(1f, vec[0], 1e-6f);
        assertEquals(0f, vec[1], 1e-6f);
    }

    // ==================== normalizeCentroids ====================

    @Test
    void normalizeCentroids_allBecomesUnit() {
        float[][] centroids = {{3, 4}, {5, 12}, {8, 15}};
        KMeansUtils.normalizeCentroids(centroids);

        for (float[] c : centroids) {
            float norm = 0;
            for (float v : c) norm += v * v;
            assertEquals(1.0f, (float) Math.sqrt(norm), 1e-5f);
        }
    }

    // ==================== computeDistanceMatrix ====================

    @Test
    void computeDistanceMatrix_isSymmetric() {
        requireEngine();
        float[][] points = {{0, 0}, {1, 0}, {0, 1}};
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] matrix = KMeansUtils.computeDistanceMatrix(points, distFn);

        assertEquals(3, matrix.length);
        for (int i = 0; i < 3; i++) {
            assertEquals(3, matrix[i].length);
            assertEquals(0f, matrix[i][i], 1e-6f, "Diagonal should be zero");
            for (int j = i + 1; j < 3; j++)
                assertEquals(matrix[i][j], matrix[j][i], 1e-6f, "Matrix should be symmetric");
        }
    }

    @Test
    void computeDistanceMatrix_knownValues() {
        requireEngine();
        float[][] points = {{0, 0}, {3, 4}};
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] matrix = KMeansUtils.computeDistanceMatrix(points, distFn);

        assertEquals(25f, matrix[0][1], 1e-6f);
        assertEquals(25f, matrix[1][0], 1e-6f);
    }

    @Test
    void computeDistanceMatrix_singlePoint() {
        requireEngine();
        float[][] points = {{1, 2, 3}};
        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        float[][] matrix = KMeansUtils.computeDistanceMatrix(points, distFn);

        assertEquals(1, matrix.length);
        assertEquals(0f, matrix[0][0], 1e-6f);
    }

    // ==================== assignPointsToClustersWithPruning ====================

    @Test
    void assignWithPruning_producesValidResults() {
        requireEngine();
        Random rng = new Random(SEED);
        float[][] data = generateData(rng, 200, DIM);
        float[][] centroids = new float[8][];
        for (int i = 0; i < 8; i++)
            centroids[i] = data[i * 25].clone();

        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        float[][] centroidDistances = KMeansUtils.precomputeCentroidDistances(centroids, distFn);

        int[] labelsPrune = new int[200];
        int[] sizesPrune = new int[8];
        KMeansUtils.assignPointsToClustersWithPruning(
            data, centroids, centroidDistances, 8, labelsPrune, null, sizesPrune, distFn);

        for (int label : labelsPrune)
            assertTrue(label >= 0 && label < 8);
        int totalPrune = 0;
        for (int s : sizesPrune) totalPrune += s;
        assertEquals(200, totalPrune);
    }

    // ==================== handleEmptyClusters ====================

    @Test
    void handleEmptyClusters_fillsEmptyCluster() {
        float[][] data = {{0, 0}, {1, 0}, {2, 0}, {100, 0}};
        float[][] centroids = {{1, 0}, {50, 0}};
        int[] clusterSizes = {3, 0};
        int[] labels = {0, 0, 0, 0};
        float[] pointErrors = {1f, 0f, 1f, 99f};
        boolean[] taken = new boolean[4];

        KMeansUtils.handleEmptyClusters(data, centroids, clusterSizes, labels, pointErrors, taken,
            2, 2, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        assertEquals(1, clusterSizes[1], "Empty cluster should get exactly 1 point");
        boolean foundCluster1 = false;
        for (int l : labels)
            if (l == 1) foundCluster1 = true;
        assertTrue(foundCluster1, "At least one point should be reassigned to empty cluster");
    }

    // ==================== Helper ====================

    private static float[][] generateData(Random rng, int count, int dimension) {
        float[][] data = new float[count][dimension];
        for (float[] row : data)
            for (int d = 0; d < dimension; d++)
                row[d] = rng.nextFloat() * 2f - 1f;
        return data;
    }
}
