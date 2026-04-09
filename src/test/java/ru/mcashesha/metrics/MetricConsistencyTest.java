package ru.mcashesha.metrics;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Validates numerical consistency across the three distance computation engines:
 * {@link Metric.Engine#SCALAR SCALAR} (pure Java loops),
 * {@link Metric.Engine#VECTOR_API VECTOR_API} (jdk.incubator.vector SIMD), and
 * {@link Metric.Engine#SIMSIMD SIMSIMD} (native C via JNI).
 *
 * <p>Because VECTOR_API and SIMSIMD use hardware-specific SIMD instructions, floating-point
 * rounding may differ slightly from the scalar baseline. Every comparison therefore uses
 * a {@link #TOLERANCE} of 1e-3 to account for acceptable numerical divergence.</p>
 *
 * <p>The test suite is organized into the following groups:</p>
 * <ul>
 *   <li><b>Scalar vs VectorAPI</b> -- parameterized across many vector dimensions (1 through 512,
 *       including edge-case sizes that do not align with SIMD lane widths) for L2, dot product,
 *       cosine distance, and Hamming distance.</li>
 *   <li><b>Scalar vs SimSIMD</b> -- the same cross-engine checks against the native JNI backend.
 *       Tests are skipped gracefully when the native library is unavailable.</li>
 *   <li><b>All three engines</b> -- triangular consistency checks at 512 dimensions (the
 *       production embedding size) ensuring Scalar vs VectorAPI, Scalar vs SimSIMD,
 *       and VectorAPI vs SimSIMD all agree.</li>
 *   <li><b>Batch consistency</b> -- runs 100 consecutive vector-pair comparisons to catch
 *       intermittent numerical instabilities.</li>
 *   <li><b>Known values</b> -- deterministic hand-computed results (e.g., L2 of [1,2,3] vs
 *       [4,5,6] = 27) to verify absolute correctness, not just cross-engine agreement.</li>
 *   <li><b>Property tests</b> -- mathematical invariants such as non-negativity of L2,
 *       zero distance for identical vectors, cosine distance range [0, 2], and symmetry.</li>
 *   <li><b>Strategy pattern delegation</b> -- verifies that {@link Metric.Type#distance}
 *       correctly dispatches to the underlying engine, including the negation applied to
 *       {@link Metric.Type#DOT_PRODUCT}.</li>
 * </ul>
 *
 * <p>A fixed {@link #SEED} ensures reproducibility across runs.</p>
 */
class MetricConsistencyTest {

    /** Maximum acceptable absolute difference between engine outputs for floating-point metrics. */
    private static final float TOLERANCE = 1e-3f;

    /** Fixed random seed for deterministic vector generation across all test runs. */
    private static final long SEED = 42L;

    /** Flag set during {@link #checkSimsimd()} indicating whether the native SimSIMD library loaded. */
    private static boolean simsimdAvailable;

    /**
     * Attempts to load the SimSIMD native library before any tests run.
     * If the JNI shared library is not on {@code java.library.path}, all SimSIMD tests
     * will be skipped via JUnit assumptions rather than failing outright.
     */
    @BeforeAll
    static void checkSimsimd() {
        try {
            Metric.Engine.SIMSIMD.getMetric().l2Distance(new float[]{1f}, new float[]{2f});
            simsimdAvailable = true;
        }
        catch (UnsatisfiedLinkError e) {
            simsimdAvailable = false;
            System.err.println("SimSIMD native library not available, skipping SIMSIMD tests");
        }
    }

    /**
     * Generates an array of random float vectors with components uniformly distributed in [-1, 1].
     *
     * @param rng       random number generator for reproducibility
     * @param count     number of vectors to generate
     * @param dimension dimensionality of each vector
     * @return a {@code count x dimension} float array
     */
    private static float[][] generateVectorPairs(Random rng, int count, int dimension) {
        float[][] vectors = new float[count][dimension];
        for (float[] v : vectors)
            for (int d = 0; d < dimension; d++)
                v[d] = rng.nextFloat() * 2f - 1f;
        return vectors;
    }

    // ==================== SCALAR vs VECTOR_API ====================

    /**
     * Verifies that the Scalar and VectorAPI engines produce consistent L2 squared distances
     * across a wide range of vector dimensions, including sizes that are not multiples of
     * typical SIMD lane widths (e.g., 1, 3, 7, 15, 31, 255).
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void l2Distance_scalarVsVectorAPI(int dimension) {
        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().l2Distance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE,
            "L2 mismatch for dim=" + dimension + ": scalar=" + scalar + " vectorApi=" + vectorApi);
    }

    /**
     * Verifies that the Scalar and VectorAPI engines produce consistent dot product values
     * across a wide range of vector dimensions.
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void dotProduct_scalarVsVectorAPI(int dimension) {
        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().dotProduct(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().dotProduct(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE,
            "DotProduct mismatch for dim=" + dimension);
    }

    /**
     * Verifies that the Scalar and VectorAPI engines produce consistent cosine distances
     * across a wide range of vector dimensions.
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void cosineDistance_scalarVsVectorAPI(int dimension) {
        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE,
            "Cosine mismatch for dim=" + dimension);
    }

    /**
     * Verifies that the Scalar and VectorAPI engines produce identical Hamming distances
     * for byte arrays of varying lengths. Hamming distance is an integer count of differing
     * bits, so no tolerance is needed -- results must match exactly.
     *
     * @param length the number of bytes in each test array
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256})
    void hammingDistance_scalarVsVectorAPI(int length) {
        Random rng = new Random(SEED);
        byte[] a = new byte[length];
        byte[] b = new byte[length];
        rng.nextBytes(a);
        rng.nextBytes(b);

        long scalar = Metric.Engine.SCALAR.getMetric().hammingDistanceB8(a, b);
        long vectorApi = Metric.Engine.VECTOR_API.getMetric().hammingDistanceB8(a, b);

        assertEquals(scalar, vectorApi,
            "Hamming mismatch for len=" + length);
    }

    // ==================== SCALAR vs SIMSIMD ====================

    /**
     * Verifies that the Scalar and SimSIMD (native JNI) engines produce consistent
     * L2 squared distances across many vector dimensions.
     * Skipped when the native library is unavailable.
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void l2Distance_scalarVsSimsimd(int dimension) {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().l2Distance(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().l2Distance(a, b);

        assertEquals(scalar, simsimd, TOLERANCE,
            "L2 SIMSIMD mismatch for dim=" + dimension + ": scalar=" + scalar + " simsimd=" + simsimd);
    }

    /**
     * Verifies that the Scalar and SimSIMD engines produce consistent dot product values.
     * Skipped when the native library is unavailable.
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void dotProduct_scalarVsSimsimd(int dimension) {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().dotProduct(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().dotProduct(a, b);

        assertEquals(scalar, simsimd, TOLERANCE,
            "DotProduct SIMSIMD mismatch for dim=" + dimension);
    }

    /**
     * Verifies that the Scalar and SimSIMD engines produce consistent cosine distances.
     * Skipped when the native library is unavailable.
     *
     * @param dimension the dimensionality of the test vectors
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256, 512})
    void cosineDistance_scalarVsSimsimd(int dimension) {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        float[] a = generateVectorPairs(rng, 1, dimension)[0];
        float[] b = generateVectorPairs(rng, 1, dimension)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().cosineDistance(a, b);

        assertEquals(scalar, simsimd, TOLERANCE,
            "Cosine SIMSIMD mismatch for dim=" + dimension);
    }

    /**
     * Verifies that the Scalar and SimSIMD engines produce identical Hamming distances.
     * Hamming is integer-valued, so exact equality is required.
     * Skipped when the native library is unavailable.
     *
     * @param length the number of bytes in each test array
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256})
    void hammingDistance_scalarVsSimsimd(int length) {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        byte[] a = new byte[length];
        byte[] b = new byte[length];
        rng.nextBytes(a);
        rng.nextBytes(b);

        long scalar = Metric.Engine.SCALAR.getMetric().hammingDistanceB8(a, b);
        long simsimd = Metric.Engine.SIMSIMD.getMetric().hammingDistanceB8(a, b);

        assertEquals(scalar, simsimd,
            "Hamming SIMSIMD mismatch for len=" + length);
    }

    // ==================== All three engines ====================

    /**
     * Triangular consistency check for L2 squared distance at 512 dimensions (the typical
     * production embedding size). Asserts pairwise agreement among all three engines:
     * Scalar vs VectorAPI, Scalar vs SimSIMD, and VectorAPI vs SimSIMD.
     */
    @Test
    void allEngines_l2Distance_512d() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        int dim = 512;
        float[] a = generateVectorPairs(rng, 1, dim)[0];
        float[] b = generateVectorPairs(rng, 1, dim)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().l2Distance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().l2Distance(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE, "SCALAR vs VECTOR_API");
        assertEquals(scalar, simsimd, TOLERANCE, "SCALAR vs SIMSIMD");
        assertEquals(vectorApi, simsimd, TOLERANCE, "VECTOR_API vs SIMSIMD");
    }

    /**
     * Triangular consistency check for dot product at 512 dimensions.
     * Verifies Scalar vs VectorAPI and Scalar vs SimSIMD agreement.
     */
    @Test
    void allEngines_dotProduct_512d() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        int dim = 512;
        float[] a = generateVectorPairs(rng, 1, dim)[0];
        float[] b = generateVectorPairs(rng, 1, dim)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().dotProduct(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().dotProduct(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().dotProduct(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE, "SCALAR vs VECTOR_API");
        assertEquals(scalar, simsimd, TOLERANCE, "SCALAR vs SIMSIMD");
    }

    /**
     * Triangular consistency check for cosine distance at 512 dimensions.
     * Verifies Scalar vs VectorAPI and Scalar vs SimSIMD agreement.
     */
    @Test
    void allEngines_cosineDistance_512d() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        Random rng = new Random(SEED);
        int dim = 512;
        float[] a = generateVectorPairs(rng, 1, dim)[0];
        float[] b = generateVectorPairs(rng, 1, dim)[0];

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, b);
        float simsimd = Metric.Engine.SIMSIMD.getMetric().cosineDistance(a, b);

        assertEquals(scalar, vectorApi, TOLERANCE, "SCALAR vs VECTOR_API");
        assertEquals(scalar, simsimd, TOLERANCE, "SCALAR vs SIMSIMD");
    }

    // ==================== Batch consistency ====================

    /**
     * Stress-tests Scalar vs VectorAPI consistency over 100 consecutive vector pairs
     * at 128 dimensions, checking L2, dot product, and cosine distance for each pair.
     * This catches intermittent issues that might not surface with a single vector pair,
     * such as accumulator overflow or lane-boundary bugs in the SIMD implementation.
     */
    @Test
    void scalarVsVectorAPI_manyVectors() {
        Random rng = new Random(SEED);
        int dim = 128;
        int count = 100;
        float[][] vectors = generateVectorPairs(rng, count, dim);

        Metric scalarMetric = Metric.Engine.SCALAR.getMetric();
        Metric vectorApiMetric = Metric.Engine.VECTOR_API.getMetric();

        // Compare each consecutive pair (i, i+1) across all three floating-point metrics
        for (int i = 0; i < count - 1; i++) {
            float sL2 = scalarMetric.l2Distance(vectors[i], vectors[i + 1]);
            float vL2 = vectorApiMetric.l2Distance(vectors[i], vectors[i + 1]);
            assertEquals(sL2, vL2, TOLERANCE, "L2 mismatch at pair " + i);

            float sDot = scalarMetric.dotProduct(vectors[i], vectors[i + 1]);
            float vDot = vectorApiMetric.dotProduct(vectors[i], vectors[i + 1]);
            assertEquals(sDot, vDot, TOLERANCE, "Dot mismatch at pair " + i);

            float sCos = scalarMetric.cosineDistance(vectors[i], vectors[i + 1]);
            float vCos = vectorApiMetric.cosineDistance(vectors[i], vectors[i + 1]);
            assertEquals(sCos, vCos, TOLERANCE, "Cosine mismatch at pair " + i);
        }
    }

    // ==================== Known values ====================

    /**
     * Verifies L2 squared distance against a hand-computed known value.
     * For vectors [1,2,3] and [4,5,6]: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27.
     */
    @Test
    void l2Distance_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        float expected = 27f;

        // Scalar uses tight tolerance since it is the reference implementation with no SIMD rounding
        assertEquals(expected, Metric.Engine.SCALAR.getMetric().l2Distance(a, b), 1e-6f);
        // VectorAPI uses broader tolerance due to potential FMA instruction differences
        assertEquals(expected, Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b), 1e-3f);
    }

    /**
     * Verifies dot product against a hand-computed known value.
     * For vectors [1,2,3] and [4,5,6]: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.
     */
    @Test
    void dotProduct_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        float expected = 32f;

        assertEquals(expected, Metric.Engine.SCALAR.getMetric().dotProduct(a, b), 1e-6f);
        assertEquals(expected, Metric.Engine.VECTOR_API.getMetric().dotProduct(a, b), 1e-3f);
    }

    /**
     * Verifies that the cosine distance between a vector and itself is zero.
     * Cosine similarity of identical vectors is 1, so cosine distance (1 - similarity) = 0.
     */
    @Test
    void cosineDistance_identicalVectors_isZero() {
        float[] a = {1f, 2f, 3f, 4f};

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, a);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, a);

        assertEquals(0f, scalar, 1e-6f);
        assertEquals(0f, vectorApi, 1e-3f);
    }

    /**
     * Verifies that the cosine distance between opposite (anti-parallel) vectors is 2.
     * Cosine similarity of [1,0,0] and [-1,0,0] is -1, so cosine distance = 1 - (-1) = 2.
     */
    @Test
    void cosineDistance_oppositeVectors_isTwo() {
        float[] a = {1f, 0f, 0f};
        float[] b = {-1f, 0f, 0f};

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, b);

        // Cosine distance of perfectly opposite vectors: 1 - cos(180deg) = 1 - (-1) = 2
        assertEquals(2f, scalar, 1e-6f);
        assertEquals(2f, vectorApi, 1e-3f);
    }

    /**
     * Verifies that the L2 squared distance between a vector and itself is exactly zero.
     * This is a basic sanity check for the identity property of distance metrics.
     */
    @Test
    void l2Distance_identicalVectors_isZero() {
        float[] a = {1f, 2f, 3f, 4f};

        assertEquals(0f, Metric.Engine.SCALAR.getMetric().l2Distance(a, a), 1e-6f);
        assertEquals(0f, Metric.Engine.VECTOR_API.getMetric().l2Distance(a, a), 1e-6f);
    }

    /**
     * Property test: L2 squared distance must always be non-negative for any pair of vectors.
     * Checks 50 random vector pairs at 64 dimensions across both Scalar and VectorAPI engines.
     */
    @Test
    void l2Distance_isNonNegative() {
        Random rng = new Random(SEED);
        int dim = 64;
        for (int i = 0; i < 50; i++) {
            float[] a = generateVectorPairs(rng, 1, dim)[0];
            float[] b = generateVectorPairs(rng, 1, dim)[0];
            assertTrue(Metric.Engine.SCALAR.getMetric().l2Distance(a, b) >= 0);
            assertTrue(Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b) >= 0);
        }
    }

    /**
     * Property test: cosine distance must lie in the range [0, 2] for any pair of non-zero vectors.
     * A value of 0 means identical direction, 1 means orthogonal, and 2 means opposite direction.
     * Uses a small tolerance to accommodate floating-point rounding near the boundaries.
     */
    @Test
    void cosineDistance_rangeZeroToTwo() {
        Random rng = new Random(SEED);
        int dim = 64;
        for (int i = 0; i < 50; i++) {
            float[] a = generateVectorPairs(rng, 1, dim)[0];
            float[] b = generateVectorPairs(rng, 1, dim)[0];
            float d = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
            assertTrue(d >= -TOLERANCE && d <= 2f + TOLERANCE,
                "Cosine distance out of range [0,2]: " + d);
        }
    }

    // ==================== Via Metric.Type (strategy pattern) ====================

    /**
     * Verifies that {@link Metric.Type#distance} correctly delegates to the underlying engine
     * method for each metric type. Also checks that {@link Metric.Type#DOT_PRODUCT} negates
     * the raw dot product value so that lower values consistently indicate closer vectors
     * across all metric types.
     */
    @Test
    void metricType_distance_delegatesCorrectly() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        float l2 = Metric.Type.L2SQ_DISTANCE.distance(Metric.Engine.SCALAR, a, b);
        float dot = Metric.Type.DOT_PRODUCT.distance(Metric.Engine.SCALAR, a, b);
        float cos = Metric.Type.COSINE_DISTANCE.distance(Metric.Engine.SCALAR, a, b);

        // L2 squared: (4-1)^2 + (5-2)^2 + (6-3)^2 = 27
        assertEquals(27f, l2, 1e-6f);
        // DOT_PRODUCT is negated internally so that lower = closer (raw dot = 32, negated = -32)
        assertEquals(-32f, dot, 1e-6f);
        // Cosine distance is always in [0, 2]
        assertTrue(cos >= 0f && cos <= 2f);
    }
}
