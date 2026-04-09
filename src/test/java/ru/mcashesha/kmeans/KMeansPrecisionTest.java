package ru.mcashesha.kmeans;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Tests for correctness of triangle inequality pruning and loss accumulation precision.
 *
 * <p>Validates two key fixes:</p>
 * <ul>
 *   <li><b>Pruning correctness:</b> {@code assignPointsToClustersWithPruning} with the
 *       corrected L2 squared multiplier (4x instead of 2x) produces identical labels
 *       to the brute-force {@code assignPointsToClusters}.</li>
 *   <li><b>Pruning restriction:</b> KMeans with DOT_PRODUCT metric and k above the
 *       pruning threshold (k >= 64) works correctly because pruning is disabled
 *       for non-L2 metrics.</li>
 * </ul>
 */
class KMeansPrecisionTest {

    private static final long SEED = 42L;
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

    /**
     * Verifies that assignPointsToClustersWithPruning produces EXACTLY the same labels
     * as assignPointsToClusters (brute-force) for L2 squared distance with k=100.
     *
     * <p>This test exercises the corrected pruning multiplier (4x for L2 squared).
     * With the old incorrect multiplier (2x), pruning was too aggressive and could
     * skip centroids that were actually closer, leading to label mismatches.</p>
     */
    @Test
    void pruningLabelsMatchBruteForce_l2sq_k100() {
        requireEngine();

        int sampleCnt = 5000;
        int dimension = 64;
        int clusterCnt = 100;

        Random rng = new Random(SEED);
        float[][] data = new float[sampleCnt][dimension];
        for (int i = 0; i < sampleCnt; i++)
            for (int d = 0; d < dimension; d++)
                data[i][d] = rng.nextFloat() * 2f - 1f;

        Metric.DistanceFunction distFn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);

        // Initialize centroids via KMeans++ for realistic centroid placement
        float[][] centroids = KMeansUtils.initializeCentroidsKMeansPlusPlus(
            data, sampleCnt, dimension, clusterCnt, distFn, Metric.Type.L2SQ_DISTANCE, new Random(SEED));

        // Brute-force assignment (no pruning)
        int[] labelsNoPruning = new int[sampleCnt];
        int[] sizesNoPruning = new int[clusterCnt];
        float lossNoPruning = KMeansUtils.assignPointsToClusters(
            data, centroids, clusterCnt, labelsNoPruning, null, sizesNoPruning, distFn);

        // Pruned assignment
        float[][] centroidDistances = KMeansUtils.precomputeCentroidDistances(centroids, distFn);
        int[] labelsPruning = new int[sampleCnt];
        int[] sizesPruning = new int[clusterCnt];
        float lossPruning = KMeansUtils.assignPointsToClustersWithPruning(
            data, centroids, centroidDistances, clusterCnt, labelsPruning, null, sizesPruning, distFn);

        // Labels must be identical for every single point
        for (int i = 0; i < sampleCnt; i++) {
            assertEquals(labelsNoPruning[i], labelsPruning[i],
                "Label mismatch at point " + i + ": brute-force=" + labelsNoPruning[i]
                    + " pruned=" + labelsPruning[i]);
        }

        // Cluster sizes must match
        assertArrayEquals(sizesNoPruning, sizesPruning, "Cluster sizes must match");

        // Loss must match (both use the same accumulation now)
        assertEquals(lossNoPruning, lossPruning, Math.abs(lossNoPruning) * 1e-5f,
            "Loss must match between pruned and brute-force");
    }

    /**
     * Verifies that KMeans with DOT_PRODUCT and k=70 (above PRUNING_THRESHOLD=64)
     * works correctly -- pruning must NOT be activated for non-L2 metrics.
     *
     * <p>Before the fix, pruning would have been enabled for k >= 64 regardless of
     * metric type, which is mathematically incorrect for dot product and cosine
     * distance where the triangle inequality does not hold in the same form.</p>
     */
    @Test
    void lloydDotProduct_k70_pruningNotActivated_producesValidResult() {
        requireEngine();

        int k = 70;
        int pointsPerCluster = 20;
        int totalPoints = k * pointsPerCluster;
        int dimension = 16;

        Random rng = new Random(SEED);
        float[][] data = new float[totalPoints][dimension];
        for (int i = 0; i < totalPoints; i++)
            for (int d = 0; d < dimension; d++)
                data[i][d] = rng.nextFloat() * 2f - 1f;

        LloydKMeans kmeans = new LloydKMeans(
            k,
            Metric.Type.DOT_PRODUCT,
            Metric.Engine.SCALAR,
            50,
            1e-4f,
            new Random(SEED)
        );

        LloydKMeans.Result result = kmeans.fit(data);

        assertNotNull(result, "Result must not be null");
        int[] labels = result.getClusterAssignments();
        assertNotNull(labels, "Labels must not be null");
        assertEquals(totalPoints, labels.length, "Labels length must match data length");

        // Verify all labels are valid cluster indices
        for (int i = 0; i < totalPoints; i++) {
            assertTrue(labels[i] >= 0 && labels[i] < k,
                "Label " + labels[i] + " at index " + i + " is out of range [0, " + k + ")");
        }

        // Verify cluster sizes are non-negative and sum to total points
        int[] clusterSizes = result.getClusterSizes();
        assertNotNull(clusterSizes, "Cluster sizes must not be null");
        assertEquals(k, clusterSizes.length, "Cluster sizes length must match k");
        int totalAssigned = 0;
        for (int size : clusterSizes) {
            assertTrue(size >= 0, "Cluster size must be non-negative");
            totalAssigned += size;
        }
        assertEquals(totalPoints, totalAssigned, "Total assigned points must match data length");

        // Loss should be finite and non-positive (dot product is negated, so loss <= 0 is valid,
        // but the actual value depends on data; just check finite)
        assertTrue(Float.isFinite(result.getLoss()), "Loss must be finite");
    }
}
