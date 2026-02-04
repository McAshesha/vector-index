package ru.mcashesha.metrics;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

class MetricConsistencyTest {

    private static final float TOLERANCE = 1e-3f;
    private static final long SEED = 42L;

    private static boolean simsimdAvailable;

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

    private static float[][] generateVectorPairs(Random rng, int count, int dimension) {
        float[][] vectors = new float[count][dimension];
        for (float[] v : vectors)
            for (int d = 0; d < dimension; d++)
                v[d] = rng.nextFloat() * 2f - 1f;
        return vectors;
    }

    // ==================== SCALAR vs VECTOR_API ====================

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

    @Test
    void scalarVsVectorAPI_manyVectors() {
        Random rng = new Random(SEED);
        int dim = 128;
        int count = 100;
        float[][] vectors = generateVectorPairs(rng, count, dim);

        Metric scalarMetric = Metric.Engine.SCALAR.getMetric();
        Metric vectorApiMetric = Metric.Engine.VECTOR_API.getMetric();

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

    @Test
    void l2Distance_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        float expected = 27f;

        assertEquals(expected, Metric.Engine.SCALAR.getMetric().l2Distance(a, b), 1e-6f);
        assertEquals(expected, Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b), 1e-3f);
    }

    @Test
    void dotProduct_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        float expected = 32f;

        assertEquals(expected, Metric.Engine.SCALAR.getMetric().dotProduct(a, b), 1e-6f);
        assertEquals(expected, Metric.Engine.VECTOR_API.getMetric().dotProduct(a, b), 1e-3f);
    }

    @Test
    void cosineDistance_identicalVectors_isZero() {
        float[] a = {1f, 2f, 3f, 4f};

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, a);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, a);

        assertEquals(0f, scalar, 1e-6f);
        assertEquals(0f, vectorApi, 1e-3f);
    }

    @Test
    void cosineDistance_oppositeVectors_isTwo() {
        float[] a = {1f, 0f, 0f};
        float[] b = {-1f, 0f, 0f};

        float scalar = Metric.Engine.SCALAR.getMetric().cosineDistance(a, b);
        float vectorApi = Metric.Engine.VECTOR_API.getMetric().cosineDistance(a, b);

        assertEquals(2f, scalar, 1e-6f);
        assertEquals(2f, vectorApi, 1e-3f);
    }

    @Test
    void l2Distance_identicalVectors_isZero() {
        float[] a = {1f, 2f, 3f, 4f};

        assertEquals(0f, Metric.Engine.SCALAR.getMetric().l2Distance(a, a), 1e-6f);
        assertEquals(0f, Metric.Engine.VECTOR_API.getMetric().l2Distance(a, a), 1e-6f);
    }

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

    @Test
    void metricType_distance_delegatesCorrectly() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        float l2 = Metric.Type.L2SQ_DISTANCE.distance(Metric.Engine.SCALAR, a, b);
        float dot = Metric.Type.DOT_PRODUCT.distance(Metric.Engine.SCALAR, a, b);
        float cos = Metric.Type.COSINE_DISTANCE.distance(Metric.Engine.SCALAR, a, b);

        assertEquals(27f, l2, 1e-6f);
        assertEquals(-32f, dot, 1e-6f);
        assertTrue(cos >= 0f && cos <= 2f);
    }
}
