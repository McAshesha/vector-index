package ru.mcashesha.metrics;

import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the dual-accumulator optimized VectorAPI implementation.
 *
 * <p>These tests verify that the pipeline-optimized VectorAPI methods (which use
 * two independent FMA accumulators per quantity to saturate the FMA pipeline)
 * produce results consistent with the Scalar baseline across a wide range of
 * vector dimensions. Special attention is paid to dimensions that exercise all
 * three loop phases: the dual-vector SIMD loop, the single-vector SIMD remainder
 * loop, and the scalar tail.</p>
 *
 * <p>All tests use a fixed seed (42) for reproducibility.</p>
 */
class VectorAPIPipelineTest {

    private static final float TOLERANCE = 1e-3f;
    private static final long SEED = 42L;

    private final Metric scalar = new Scalar();
    private final Metric vectorApi = new VectorAPI();

    /**
     * Generates a random float array with values uniformly distributed in [-1, 1].
     */
    private static float[] randomVector(Random rng, int dimension) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++)
            v[i] = rng.nextFloat() * 2f - 1f;
        return v;
    }

    // ==================== Cross-engine consistency across many dimensions ====================

    /**
     * Verifies that the optimized VectorAPI l2Distance matches Scalar across 50 random
     * vector pairs for each of many dimensions. The dimension list includes values that
     * are: below species length (1, 7), exact species length multiples (8, 16, 32, 64,
     * 128, 256, 512, 1024), one less than species length multiples (15, 31, 63, 127,
     * 255, 511), and other non-aligned sizes to exercise all three loop phases.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024})
    void l2Distance_matchesScalar_50pairs(int dimension) {
        Random rng = new Random(SEED);
        for (int pair = 0; pair < 50; pair++) {
            float[] a = randomVector(rng, dimension);
            float[] b = randomVector(rng, dimension);
            float expected = scalar.l2Distance(a, b);
            float actual = vectorApi.l2Distance(a, b);
            assertEquals(expected, actual, TOLERANCE,
                "l2Distance mismatch for dim=" + dimension + " pair=" + pair
                    + ": scalar=" + expected + " vectorApi=" + actual);
        }
    }

    /**
     * Verifies that the optimized VectorAPI dotProduct matches Scalar across 50 random
     * vector pairs for each dimension.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024})
    void dotProduct_matchesScalar_50pairs(int dimension) {
        Random rng = new Random(SEED);
        for (int pair = 0; pair < 50; pair++) {
            float[] a = randomVector(rng, dimension);
            float[] b = randomVector(rng, dimension);
            float expected = scalar.dotProduct(a, b);
            float actual = vectorApi.dotProduct(a, b);
            assertEquals(expected, actual, TOLERANCE,
                "dotProduct mismatch for dim=" + dimension + " pair=" + pair
                    + ": scalar=" + expected + " vectorApi=" + actual);
        }
    }

    /**
     * Verifies that the optimized VectorAPI cosineDistance matches Scalar across 50 random
     * vector pairs for each dimension.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024})
    void cosineDistance_matchesScalar_50pairs(int dimension) {
        Random rng = new Random(SEED);
        for (int pair = 0; pair < 50; pair++) {
            float[] a = randomVector(rng, dimension);
            float[] b = randomVector(rng, dimension);
            float expected = scalar.cosineDistance(a, b);
            float actual = vectorApi.cosineDistance(a, b);
            assertEquals(expected, actual, TOLERANCE,
                "cosineDistance mismatch for dim=" + dimension + " pair=" + pair
                    + ": scalar=" + expected + " vectorApi=" + actual);
        }
    }

    // ==================== Edge cases: very small dimensions ====================

    /**
     * Edge case: dimension=1, which is smaller than any SIMD vector width.
     * Only the scalar tail loop should execute.
     */
    @Test
    void edgeCase_dimension1() {
        float[] a = {0.5f};
        float[] b = {-0.3f};

        assertEquals(scalar.l2Distance(a, b), vectorApi.l2Distance(a, b), TOLERANCE, "l2 dim=1");
        assertEquals(scalar.dotProduct(a, b), vectorApi.dotProduct(a, b), TOLERANCE, "dot dim=1");
        assertEquals(scalar.cosineDistance(a, b), vectorApi.cosineDistance(a, b), TOLERANCE, "cosine dim=1");
    }

    /**
     * Edge case: dimension=2.
     */
    @Test
    void edgeCase_dimension2() {
        float[] a = {0.5f, -0.7f};
        float[] b = {-0.3f, 0.9f};

        assertEquals(scalar.l2Distance(a, b), vectorApi.l2Distance(a, b), TOLERANCE, "l2 dim=2");
        assertEquals(scalar.dotProduct(a, b), vectorApi.dotProduct(a, b), TOLERANCE, "dot dim=2");
        assertEquals(scalar.cosineDistance(a, b), vectorApi.cosineDistance(a, b), TOLERANCE, "cosine dim=2");
    }

    /**
     * Edge case: dimension=3.
     */
    @Test
    void edgeCase_dimension3() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        assertEquals(scalar.l2Distance(a, b), vectorApi.l2Distance(a, b), TOLERANCE, "l2 dim=3");
        assertEquals(scalar.dotProduct(a, b), vectorApi.dotProduct(a, b), TOLERANCE, "dot dim=3");
        assertEquals(scalar.cosineDistance(a, b), vectorApi.cosineDistance(a, b), TOLERANCE, "cosine dim=3");
    }

    // ==================== Cosine distance special cases ====================

    /**
     * Cosine distance of identical vectors should be approximately 0 (cosine similarity = 1).
     * Tests across several dimensions to exercise all loop phases.
     */
    @Test
    void cosineDistance_identicalVectors_isApproximatelyZero() {
        float[][] testVectors = {
            {1f},
            {1f, 2f, 3f},
            randomVector(new Random(SEED), 64),
            randomVector(new Random(SEED), 256),
            randomVector(new Random(SEED), 512)
        };

        for (float[] v : testVectors) {
            float result = vectorApi.cosineDistance(v, v);
            assertEquals(0f, result, 1e-6f,
                "cosineDistance of identical vectors should be ~0, got " + result
                    + " for dim=" + v.length);
        }
    }

    /**
     * Cosine distance of opposite (anti-parallel) vectors should be approximately 2.
     * cos(180 degrees) = -1, so 1 - (-1) = 2.
     */
    @Test
    void cosineDistance_oppositeVectors_isApproximatelyTwo() {
        // Test with various dimensions
        int[] dims = {1, 3, 8, 32, 128};
        Random rng = new Random(SEED);

        for (int dim : dims) {
            float[] a = randomVector(rng, dim);
            float[] b = new float[dim];
            for (int i = 0; i < dim; i++)
                b[i] = -a[i];

            float result = vectorApi.cosineDistance(a, b);
            assertEquals(2f, result, 1e-6f,
                "cosineDistance of opposite vectors should be ~2, got " + result
                    + " for dim=" + dim);
        }
    }

    // ==================== L2 distance special cases ====================

    /**
     * L2 distance between two zero vectors should be exactly 0.
     */
    @Test
    void l2Distance_zeroVectors_isZero() {
        int[] dims = {1, 3, 8, 16, 32, 64, 128, 256};

        for (int dim : dims) {
            float[] a = new float[dim]; // all zeros
            float[] b = new float[dim]; // all zeros

            float result = vectorApi.l2Distance(a, b);
            assertEquals(0f, result, 1e-6f,
                "l2Distance of zero vectors should be 0, got " + result
                    + " for dim=" + dim);
        }
    }

    // ==================== Dot product special cases ====================

    /**
     * Dot product of orthogonal vectors should be approximately 0.
     * Tests standard basis vectors which are trivially orthogonal.
     */
    @Test
    void dotProduct_orthogonalVectors_isZero() {
        // Test with standard basis vectors of various dimensions
        int[] dims = {2, 3, 8, 16, 32, 64};

        for (int dim : dims) {
            // e_0 and e_1 are always orthogonal
            float[] a = new float[dim];
            float[] b = new float[dim];
            a[0] = 1f;
            b[dim > 1 ? 1 : 0] = 1f;

            if (dim > 1) {
                float result = vectorApi.dotProduct(a, b);
                assertEquals(0f, result, 1e-6f,
                    "dotProduct of orthogonal vectors should be 0, got " + result
                        + " for dim=" + dim);
            }
        }
    }
}
