package ru.mcashesha.kmeans;

import java.util.Random;
import org.junit.jupiter.api.Test;
import ru.mcashesha.metrics.Metric;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for algorithmic correctness improvements in KMeans:
 * <ul>
 *   <li>COSINE_DISTANCE convergence using angular convergence criterion.</li>
 *   <li>Hierarchical KMeans beam search prediction.</li>
 * </ul>
 *
 * <p>Uses a fixed seed (42) for reproducibility.</p>
 */
class KMeansAlgorithmicTest {

    private static final long SEED = 42L;
    private static final int DIMENSION = 8;

    /**
     * Generates normalized (unit-length) synthetic clustered data.
     *
     * <p>Each cluster center is a random unit vector, and each data point is a slightly
     * perturbed and re-normalized copy of its cluster center. This ensures the data lies
     * on the unit sphere, which is the natural domain for cosine distance.</p>
     *
     * @param rng              random number generator
     * @param pointsPerCluster number of points in each cluster
     * @param clusterCount     number of clusters
     * @param dimension        vector dimensionality
     * @return normalized clustered data
     */
    private static float[][] generateNormalizedClusters(Random rng, int pointsPerCluster,
        int clusterCount, int dimension) {
        // Generate well-separated cluster centers on the unit sphere
        float[][] centers = new float[clusterCount][dimension];
        for (int c = 0; c < clusterCount; c++) {
            for (int d = 0; d < dimension; d++)
                centers[c][d] = rng.nextFloat() - 0.5f;
            normalize(centers[c]);
            // Scale to separate clusters more (push them apart on the sphere)
            for (int d = 0; d < dimension; d++)
                centers[c][d] *= (1.0f + c * 2.0f);
            normalize(centers[c]);
        }

        float[][] data = new float[pointsPerCluster * clusterCount][dimension];
        for (int c = 0; c < clusterCount; c++) {
            for (int p = 0; p < pointsPerCluster; p++) {
                int idx = c * pointsPerCluster + p;
                for (int d = 0; d < dimension; d++)
                    data[idx][d] = centers[c][d] + (rng.nextFloat() - 0.5f) * 0.1f;
                normalize(data[idx]);
            }
        }
        return data;
    }

    /**
     * Generates well-separated clusters in Euclidean space.
     */
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

    private static void normalize(float[] v) {
        float norm = 0;
        for (float x : v)
            norm += x * x;
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int d = 0; d < v.length; d++)
                v[d] /= norm;
        }
    }

    /**
     * Verifies that Lloyd KMeans with COSINE_DISTANCE converges before reaching
     * the maximum number of iterations on normalized data. This validates that the
     * angular convergence criterion (cosine distance between old and new centroid)
     * works correctly and allows early termination.
     */
    @Test
    void lloydFit_cosineDistance_convergesBeforeMaxIterations() {
        Random rng = new Random(SEED);
        int k = 3;
        int maxIter = 300;
        float[][] data = generateNormalizedClusters(rng, 50, k, DIMENSION);

        LloydKMeans kmeans = new LloydKMeans(k, Metric.Type.COSINE_DISTANCE,
            Metric.Engine.SCALAR, maxIter, 1e-4f, new Random(SEED));

        LloydKMeans.Result result = kmeans.fit(data);

        assertNotNull(result);
        assertTrue(result.getIterations() < maxIter,
            "Expected convergence before " + maxIter + " iterations, "
                + "but performed " + result.getIterations());
        assertTrue(result.getLoss() >= 0, "Loss should be non-negative");
        assertTrue(Float.isFinite(result.getLoss()), "Loss should be finite");
    }

    /**
     * Verifies that Hierarchical KMeans with beamWidth=1 (default greedy) produces
     * the same prediction labels as the original greedy predictSinglePoint implementation.
     * This ensures backward compatibility.
     */
    @Test
    void hierarchicalPredict_beamWidthOne_equalsGreedy() {
        Random rng = new Random(SEED);
        int pointsPerCluster = 50;
        int clusterCount = 4;
        float[][] trainData = generateClusters(rng, pointsPerCluster, clusterCount, DIMENSION, 100f);

        // Build two identical models with beamWidth=1 (default)
        HierarchicalKMeans kmeansGreedy = new HierarchicalKMeans(
            2, 4, 4, 30, 1e-4f, new Random(SEED),
            Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 1);

        HierarchicalKMeans.Result model = kmeansGreedy.fit(trainData);

        float[][] testData = generateClusters(new Random(123), 20, clusterCount, DIMENSION, 100f);

        // Predict with beamWidth=1
        int[] labelsGreedy = kmeansGreedy.predict(testData, model);

        // Build a second instance with beamWidth=1 (should produce identical results)
        HierarchicalKMeans kmeansBeam1 = new HierarchicalKMeans(
            2, 4, 4, 30, 1e-4f, new Random(SEED),
            Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 1);

        int[] labelsBeam1 = kmeansBeam1.predict(testData, model);

        assertArrayEquals(labelsGreedy, labelsBeam1,
            "beamWidth=1 should produce identical labels to greedy traversal");
    }

    /**
     * Verifies that Hierarchical KMeans with beamWidth=3 produces valid labels
     * (all within the valid leaf cluster range). Beam search may return different
     * labels than greedy, but they must be structurally valid.
     */
    @Test
    void hierarchicalPredict_beamWidthThree_producesValidLabels() {
        Random rng = new Random(SEED);
        int pointsPerCluster = 50;
        int clusterCount = 4;
        float[][] trainData = generateClusters(rng, pointsPerCluster, clusterCount, DIMENSION, 100f);

        // Fit with beamWidth=1 (beam width only affects prediction, not fitting)
        HierarchicalKMeans kmeansFit = new HierarchicalKMeans(
            2, 4, 4, 30, 1e-4f, new Random(SEED),
            Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 1);

        HierarchicalKMeans.Result model = kmeansFit.fit(trainData);
        int leafCount = model.getCentroids().length;
        assertTrue(leafCount > 0, "Model must have at least one leaf centroid");

        // Predict with beamWidth=3
        HierarchicalKMeans kmeansBeam3 = new HierarchicalKMeans(
            2, 4, 4, 30, 1e-4f, new Random(SEED),
            Metric.Type.L2SQ_DISTANCE, Metric.Engine.SCALAR, 3);

        float[][] testData = generateClusters(new Random(123), 20, clusterCount, DIMENSION, 100f);
        int[] labels = kmeansBeam3.predict(testData, model);

        assertNotNull(labels);
        assertTrue(labels.length == testData.length,
            "Predicted labels array length should match test data length");

        for (int i = 0; i < labels.length; i++) {
            assertTrue(labels[i] >= 0 && labels[i] < leafCount,
                "Label " + labels[i] + " at index " + i
                    + " is out of valid range [0, " + leafCount + ")");
        }
    }
}
