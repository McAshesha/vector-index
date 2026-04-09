package ru.mcashesha.metrics;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

class ScalarMetricTest {

    private static final long SEED = 42L;
    private final Metric scalar = new Scalar();
    private final Metric vectorApi = new VectorAPI();

    /**
     * With lazy SIMSIMD initialization, Engine class loading no longer fails when the native
     * library is unavailable. SCALAR is always available, so this flag is always true.
     * Kept for backward compatibility with the assumeTrue() guards below.
     */
    private static boolean engineAvailable;

    @BeforeAll
    static void checkEngineAvailable() {
        // Engine.SCALAR.isAvailable() always returns true since Scalar is pure Java.
        // Previously this check was needed because constructing the Engine enum triggered
        // SimSIMD class loading, which could throw ExceptionInInitializerError and kill
        // the entire enum. With lazy initialization, this is no longer a concern.
        engineAvailable = Metric.Engine.SCALAR.isAvailable();
    }

    // ==================== L2 Distance ====================

    @Test
    void l2Distance_identicalVectors_zero() {
        float[] a = {1f, 2f, 3f, 4f};
        assertEquals(0f, scalar.l2Distance(a, a), 1e-6f);
    }

    @Test
    void l2Distance_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        assertEquals(27f, scalar.l2Distance(a, b), 1e-6f);
    }

    @Test
    void l2Distance_isSymmetric() {
        float[] a = {1.5f, 2.3f, -0.7f};
        float[] b = {4.1f, -1.2f, 3.3f};
        assertEquals(scalar.l2Distance(a, b), scalar.l2Distance(b, a), 1e-6f);
    }

    @Test
    void l2Distance_singleDimension() {
        float[] a = {3f};
        float[] b = {7f};
        assertEquals(16f, scalar.l2Distance(a, b), 1e-6f);
    }

    @Test
    void l2Distance_zeroVectors() {
        float[] a = {0f, 0f, 0f};
        float[] b = {0f, 0f, 0f};
        assertEquals(0f, scalar.l2Distance(a, b), 1e-6f);
    }

    @Test
    void l2Distance_negativeComponents() {
        float[] a = {-1f, -2f};
        float[] b = {1f, 2f};
        assertEquals(20f, scalar.l2Distance(a, b), 1e-6f);
    }

    @Test
    void l2Distance_highDimensional_isNonNegative() {
        Random rng = new Random(SEED);
        int dim = 512;
        float[] a = new float[dim];
        float[] b = new float[dim];
        for (int i = 0; i < dim; i++) {
            a[i] = rng.nextFloat() * 2 - 1;
            b[i] = rng.nextFloat() * 2 - 1;
        }
        assertTrue(scalar.l2Distance(a, b) >= 0);
    }

    @Test
    void l2Distance_triangleInequality() {
        float[] a = {0f, 0f};
        float[] b = {3f, 0f};
        float[] c = {3f, 4f};
        // sqrt(L2) obeys triangle inequality: sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))
        double dAC = Math.sqrt(scalar.l2Distance(a, c));
        double dAB = Math.sqrt(scalar.l2Distance(a, b));
        double dBC = Math.sqrt(scalar.l2Distance(b, c));
        assertTrue(dAC <= dAB + dBC + 1e-6);
    }

    // ==================== Dot Product ====================

    @Test
    void dotProduct_knownValue() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        assertEquals(32f, scalar.dotProduct(a, b), 1e-6f);
    }

    @Test
    void dotProduct_isSymmetric() {
        float[] a = {1.5f, -2.3f, 0.7f};
        float[] b = {4.1f, 1.2f, -3.3f};
        assertEquals(scalar.dotProduct(a, b), scalar.dotProduct(b, a), 1e-6f);
    }

    @Test
    void dotProduct_orthogonalVectors_isZero() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(0f, scalar.dotProduct(a, b), 1e-6f);
    }

    @Test
    void dotProduct_sameDirection_positive() {
        float[] a = {1f, 2f, 3f};
        float[] b = {2f, 4f, 6f};
        assertTrue(scalar.dotProduct(a, b) > 0);
    }

    @Test
    void dotProduct_oppositeDirection_negative() {
        float[] a = {1f, 2f, 3f};
        float[] b = {-1f, -2f, -3f};
        assertTrue(scalar.dotProduct(a, b) < 0);
    }

    @Test
    void dotProduct_singleDimension() {
        float[] a = {3f};
        float[] b = {5f};
        assertEquals(15f, scalar.dotProduct(a, b), 1e-6f);
    }

    @Test
    void dotProduct_zeroVector_isZero() {
        float[] a = {1f, 2f, 3f};
        float[] b = {0f, 0f, 0f};
        assertEquals(0f, scalar.dotProduct(a, b), 1e-6f);
    }

    @Test
    void dotProduct_selfDot_equalsSquaredNorm() {
        float[] a = {3f, 4f};
        assertEquals(25f, scalar.dotProduct(a, a), 1e-6f); // 9 + 16
    }

    // ==================== Cosine Distance ====================

    @Test
    void cosineDistance_identicalVectors_zero() {
        float[] a = {1f, 2f, 3f};
        assertEquals(0f, scalar.cosineDistance(a, a), 1e-6f);
    }

    @Test
    void cosineDistance_oppositeVectors_two() {
        float[] a = {1f, 0f};
        float[] b = {-1f, 0f};
        assertEquals(2f, scalar.cosineDistance(a, b), 1e-6f);
    }

    @Test
    void cosineDistance_orthogonalVectors_one() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(1f, scalar.cosineDistance(a, b), 1e-5f);
    }

    @Test
    void cosineDistance_isSymmetric() {
        float[] a = {1.5f, 2.3f, -0.7f};
        float[] b = {4.1f, -1.2f, 3.3f};
        assertEquals(scalar.cosineDistance(a, b), scalar.cosineDistance(b, a), 1e-6f);
    }

    @Test
    void cosineDistance_parallelVectors_zero() {
        float[] a = {1f, 2f, 3f};
        float[] b = {2f, 4f, 6f};
        assertEquals(0f, scalar.cosineDistance(a, b), 1e-5f);
    }

    @Test
    void cosineDistance_rangeZeroToTwo() {
        Random rng = new Random(SEED);
        for (int i = 0; i < 100; i++) {
            float[] a = {rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1};
            float[] b = {rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1};
            float d = scalar.cosineDistance(a, b);
            assertTrue(d >= -1e-3f && d <= 2.001f,
                "Cosine distance out of range: " + d);
        }
    }

    @Test
    void cosineDistance_singleDimension_sameSign_zero() {
        float[] a = {5f};
        float[] b = {3f};
        assertEquals(0f, scalar.cosineDistance(a, b), 1e-6f);
    }

    @Test
    void cosineDistance_scaledVector_sameResult() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        float[] a2 = {10f, 20f, 30f}; // a * 10
        assertEquals(scalar.cosineDistance(a, b), scalar.cosineDistance(a2, b), 1e-5f);
    }

    // ==================== Hamming Distance ====================

    @Test
    void hammingDistance_identicalArrays_zero() {
        byte[] a = {1, 2, 3, 4};
        assertEquals(0, scalar.hammingDistanceB8(a, a));
    }

    @Test
    void hammingDistance_knownValue() {
        byte[] a = {0b00000000};
        byte[] b = {(byte) 0b11111111};
        assertEquals(8, scalar.hammingDistanceB8(a, b));
    }

    @Test
    void hammingDistance_singleBitDifference() {
        byte[] a = {0b00000000};
        byte[] b = {0b00000001};
        assertEquals(1, scalar.hammingDistanceB8(a, b));
    }

    @Test
    void hammingDistance_isSymmetric() {
        byte[] a = {(byte) 0xAB, (byte) 0xCD};
        byte[] b = {(byte) 0x12, (byte) 0x34};
        assertEquals(scalar.hammingDistanceB8(a, b), scalar.hammingDistanceB8(b, a));
    }

    @Test
    void hammingDistance_emptyArrays_zero() {
        byte[] a = {};
        byte[] b = {};
        assertEquals(0, scalar.hammingDistanceB8(a, b));
    }

    @Test
    void hammingDistance_multipleBytes() {
        byte[] a = {0, 0, 0};
        byte[] b = {(byte) 0xFF, (byte) 0xFF, (byte) 0xFF};
        assertEquals(24, scalar.hammingDistanceB8(a, b));
    }

    @Test
    void hammingDistance_negativeByteValues() {
        byte[] a = {-1};  // 0xFF
        byte[] b = {0};   // 0x00
        assertEquals(8, scalar.hammingDistanceB8(a, b));
    }

    @Test
    void hammingDistance_twoBitsDiffer() {
        byte[] a = {0b00000000};
        byte[] b = {0b00000011};
        assertEquals(2, scalar.hammingDistanceB8(a, b));
    }

    // ==================== Scalar vs VectorAPI (direct instantiation) ====================

    @Test
    void scalarVsVectorAPI_l2Distance_consistent() {
        float[] a = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] b = {8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f};
        assertEquals(scalar.l2Distance(a, b), vectorApi.l2Distance(a, b), 1e-3f);
    }

    @Test
    void scalarVsVectorAPI_dotProduct_consistent() {
        float[] a = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] b = {8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f};
        assertEquals(scalar.dotProduct(a, b), vectorApi.dotProduct(a, b), 1e-3f);
    }

    @Test
    void scalarVsVectorAPI_cosineDistance_consistent() {
        float[] a = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] b = {8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f};
        assertEquals(scalar.cosineDistance(a, b), vectorApi.cosineDistance(a, b), 1e-3f);
    }

    @Test
    void scalarVsVectorAPI_hammingDistance_consistent() {
        Random rng = new Random(SEED);
        byte[] a = new byte[64];
        byte[] b = new byte[64];
        rng.nextBytes(a);
        rng.nextBytes(b);
        assertEquals(scalar.hammingDistanceB8(a, b), vectorApi.hammingDistanceB8(a, b));
    }

    @Test
    void scalarVsVectorAPI_manyDimensions_l2() {
        Random rng = new Random(SEED);
        for (int dim : new int[]{1, 3, 7, 15, 16, 31, 32, 64, 128, 255, 512}) {
            float[] a = new float[dim];
            float[] b = new float[dim];
            for (int i = 0; i < dim; i++) {
                a[i] = rng.nextFloat() * 2 - 1;
                b[i] = rng.nextFloat() * 2 - 1;
            }
            assertEquals(scalar.l2Distance(a, b), vectorApi.l2Distance(a, b), 1e-2f,
                "L2 mismatch for dim=" + dim);
        }
    }

    // ==================== Metric.Type and Engine (require native lib) ====================

    @Test
    void metricType_l2sq_delegatesCorrectly() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        assertEquals(scalar.l2Distance(a, b), Metric.Type.L2SQ_DISTANCE.distance(Metric.Engine.SCALAR, a, b), 1e-6f);
    }

    @Test
    void metricType_dotProduct_negated() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        assertEquals(-scalar.dotProduct(a, b), Metric.Type.DOT_PRODUCT.distance(Metric.Engine.SCALAR, a, b), 1e-6f);
    }

    @Test
    void metricType_cosine_delegatesCorrectly() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        assertEquals(scalar.cosineDistance(a, b), Metric.Type.COSINE_DISTANCE.distance(Metric.Engine.SCALAR, a, b), 1e-6f);
    }

    @Test
    void resolveFunction_l2sq_matchesDirect() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        Metric.DistanceFunction fn = Metric.Type.L2SQ_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        assertEquals(27f, fn.compute(a, b), 1e-6f);
    }

    @Test
    void resolveFunction_dotProduct_isNegated() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        Metric.DistanceFunction fn = Metric.Type.DOT_PRODUCT.resolveFunction(Metric.Engine.SCALAR);
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        assertEquals(-32f, fn.compute(a, b), 1e-6f);
    }

    @Test
    void resolveFunction_cosine_matchesDirect() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        Metric.DistanceFunction fn = Metric.Type.COSINE_DISTANCE.resolveFunction(Metric.Engine.SCALAR);
        float[] a = {1, 0, 0};
        float[] b = {0, 1, 0};
        assertEquals(1f, fn.compute(a, b), 1e-5f);
    }

    @Test
    void resolveFunction_consistentWithDistance() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        float[] a = {1.5f, -2.3f, 0.7f, 4.1f};
        float[] b = {-0.5f, 3.1f, -1.2f, 2.8f};
        for (Metric.Type type : Metric.Type.values()) {
            float viaDistance = type.distance(Metric.Engine.SCALAR, a, b);
            float viaResolve = type.resolveFunction(Metric.Engine.SCALAR).compute(a, b);
            assertEquals(viaDistance, viaResolve, 1e-6f,
                type + ": resolveFunction and distance() should agree");
        }
    }

    @Test
    void engine_getMetric_returnsNonNull() {
        assumeTrue(engineAvailable, "Metric.Engine not available");
        for (Metric.Engine engine : Metric.Engine.values())
            assertNotNull(engine.getMetric(), engine + " metric should not be null");
    }
}
