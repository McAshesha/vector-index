package ru.mcashesha.metrics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the lazy initialization and availability behavior of {@link Metric.Engine}.
 *
 * <p>The key invariant being tested: SCALAR and VECTOR_API must always be usable,
 * regardless of whether the SimSIMD native library is available. Previously,
 * constructing any Engine constant would trigger eager loading of SimSIMD's native
 * library (because {@code new SimSIMD()} was called in the enum constant declaration),
 * and if the native library was missing, an {@link UnsatisfiedLinkError} would
 * propagate through the enum's class initializer, making ALL engines inaccessible.</p>
 *
 * <p>With lazy initialization, the SimSIMD instance is created only when
 * {@link Metric.Engine#getMetric()} is called on the SIMSIMD constant. This test
 * class verifies that this isolation works correctly.</p>
 */
class MetricEngineAvailabilityTest {

    // ==================== isAvailable() ====================

    /**
     * Verifies that SCALAR is always reported as available.
     * Scalar uses pure Java loops and has no native dependencies whatsoever.
     */
    @Test
    void scalar_isAlwaysAvailable() {
        assertTrue(Metric.Engine.SCALAR.isAvailable(),
                "SCALAR engine should always be available (pure Java, no native dependencies)");
    }

    /**
     * Verifies that VECTOR_API is always reported as available.
     * VectorAPI uses jdk.incubator.vector which is part of the JDK -- no native library needed.
     */
    @Test
    void vectorApi_isAlwaysAvailable() {
        assertTrue(Metric.Engine.VECTOR_API.isAvailable(),
                "VECTOR_API engine should always be available (uses JDK built-in vector API)");
    }

    /**
     * Verifies that SIMSIMD's isAvailable() does not throw an exception regardless of
     * whether the native library is present. It should return a boolean cleanly.
     */
    @Test
    void simsimd_isAvailable_neverThrows() {
        // This should never throw, even if the native library is missing.
        // It should return true if the lib is present, false if not.
        assertDoesNotThrow(() -> Metric.Engine.SIMSIMD.isAvailable(),
                "SIMSIMD.isAvailable() must not throw; it should return false if native lib is missing");
    }

    // ==================== Engine.values() ====================

    /**
     * Verifies that {@link Metric.Engine#values()} does not throw an exception.
     * This is the critical regression test: before lazy initialization, calling
     * Engine.values() could fail with ExceptionInInitializerError if SimSIMD's
     * native library was unavailable, because the enum's class initializer would
     * invoke {@code new SimSIMD()} which triggered native library loading.
     */
    @Test
    void engineValues_doesNotThrow() {
        Metric.Engine[] engines = assertDoesNotThrow(
                Metric.Engine::values,
                "Engine.values() must not throw even if SimSIMD native library is unavailable");
        assertEquals(3, engines.length,
                "There should be exactly 3 engine constants: SCALAR, VECTOR_API, SIMSIMD");
    }

    // ==================== getMetric() ====================

    /**
     * Verifies that SCALAR's getMetric() returns a non-null Scalar instance.
     * This confirms that the eager initialization path works correctly.
     */
    @Test
    void scalar_getMetric_returnsScalarInstance() {
        Metric metric = Metric.Engine.SCALAR.getMetric();
        assertNotNull(metric, "SCALAR.getMetric() should never return null");
        assertInstanceOf(Scalar.class, metric,
                "SCALAR.getMetric() should return an instance of Scalar");
    }

    /**
     * Verifies that VECTOR_API's getMetric() returns a non-null VectorAPI instance.
     * This confirms that the eager initialization path works correctly.
     */
    @Test
    void vectorApi_getMetric_returnsVectorAPIInstance() {
        Metric metric = Metric.Engine.VECTOR_API.getMetric();
        assertNotNull(metric, "VECTOR_API.getMetric() should never return null");
        assertInstanceOf(VectorAPI.class, metric,
                "VECTOR_API.getMetric() should return an instance of VectorAPI");
    }

    /**
     * Verifies that SIMSIMD's getMetric() returns a non-null SimSIMD instance.
     * The instance is always created (lazy holder pattern), but its methods will
     * throw UnsatisfiedLinkError if the native library is not available.
     */
    @Test
    void simsimd_getMetric_returnsNonNull() {
        // getMetric() should always return a non-null SimSIMD instance, regardless of
        // whether the native library is loaded. The error is deferred to method invocation.
        Metric metric = assertDoesNotThrow(
                () -> Metric.Engine.SIMSIMD.getMetric(),
                "SIMSIMD.getMetric() should not throw; errors are deferred to method calls");
        assertNotNull(metric, "SIMSIMD.getMetric() should never return null");
        assertInstanceOf(SimSIMD.class, metric,
                "SIMSIMD.getMetric() should return an instance of SimSIMD");
    }

    // ==================== resolveFunction for SCALAR and VECTOR_API ====================

    /**
     * Verifies that resolveFunction works for SCALAR across all metric types,
     * independently of SimSIMD availability. This confirms that the Type enum's
     * resolveFunction method does not inadvertently trigger SimSIMD initialization
     * when called with SCALAR.
     */
    @Test
    void resolveFunction_scalar_worksForAllMetricTypes() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        for (Metric.Type type : Metric.Type.values()) {
            Metric.DistanceFunction fn = assertDoesNotThrow(
                    () -> type.resolveFunction(Metric.Engine.SCALAR),
                    type + ": resolveFunction(SCALAR) should not throw");
            assertDoesNotThrow(
                    () -> fn.compute(a, b),
                    type + ": resolved SCALAR function should compute without error");
        }
    }

    /**
     * Verifies that resolveFunction works for VECTOR_API across all metric types,
     * independently of SimSIMD availability.
     */
    @Test
    void resolveFunction_vectorApi_worksForAllMetricTypes() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        for (Metric.Type type : Metric.Type.values()) {
            Metric.DistanceFunction fn = assertDoesNotThrow(
                    () -> type.resolveFunction(Metric.Engine.VECTOR_API),
                    type + ": resolveFunction(VECTOR_API) should not throw");
            assertDoesNotThrow(
                    () -> fn.compute(a, b),
                    type + ": resolved VECTOR_API function should compute without error");
        }
    }

    // ==================== SCALAR computation correctness ====================

    /**
     * Verifies that SCALAR engine computes correct known L2 distance values
     * through the Engine.getMetric() accessor, confirming the full path from
     * Engine enum through to the Scalar implementation works end-to-end.
     */
    @Test
    void scalar_computesCorrectL2Distance() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        float result = Metric.Engine.SCALAR.getMetric().l2Distance(a, b);
        assertEquals(27f, result, 1e-6f, "SCALAR L2 distance should be 27 for [1,2,3] vs [4,5,6]");
    }

    /**
     * Verifies that Metric.Type.distance() works correctly through the SCALAR engine.
     * This tests the full dispatch chain: Type enum -> Engine.getMetric() -> Scalar impl.
     */
    @Test
    void metricType_distance_worksWithScalar() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        // L2SQ should give 27
        assertEquals(27f,
                Metric.Type.L2SQ_DISTANCE.distance(Metric.Engine.SCALAR, a, b),
                1e-6f);

        // DOT_PRODUCT should give -32 (negated)
        assertEquals(-32f,
                Metric.Type.DOT_PRODUCT.distance(Metric.Engine.SCALAR, a, b),
                1e-6f);

        // COSINE_DISTANCE should be in [0, 2]
        float cos = Metric.Type.COSINE_DISTANCE.distance(Metric.Engine.SCALAR, a, b);
        assertTrue(cos >= 0f && cos <= 2f,
                "Cosine distance should be in [0, 2], got: " + cos);
    }

    // ==================== Isolation: SCALAR unaffected by SIMSIMD ====================

    /**
     * Verifies that accessing SCALAR and VECTOR_API works correctly even after
     * SIMSIMD has been referenced. This is the core isolation test: referencing
     * SIMSIMD (even if the native library is missing) must not corrupt or disable
     * the other engines.
     */
    @Test
    void scalarAndVectorApi_workAfterSimsimdReferenced() {
        // First, reference SIMSIMD to trigger any lazy initialization
        Metric.Engine.SIMSIMD.isAvailable(); // may return true or false

        // Now verify SCALAR still works perfectly
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        float scalarL2 = Metric.Engine.SCALAR.getMetric().l2Distance(a, b);
        assertEquals(27f, scalarL2, 1e-6f, "SCALAR must work after SIMSIMD has been referenced");

        float vectorApiL2 = Metric.Engine.VECTOR_API.getMetric().l2Distance(a, b);
        assertEquals(27f, vectorApiL2, 1e-3f, "VECTOR_API must work after SIMSIMD has been referenced");
    }

    /**
     * Verifies that getMetric() returns the same instance on repeated calls (singleton behavior).
     * For SCALAR and VECTOR_API this is trivial (field access). For SIMSIMD this confirms
     * the holder pattern returns a stable singleton.
     */
    @Test
    void getMetric_returnsSameInstance_onRepeatedCalls() {
        assertSame(Metric.Engine.SCALAR.getMetric(), Metric.Engine.SCALAR.getMetric(),
                "SCALAR.getMetric() should return the same instance on every call");
        assertSame(Metric.Engine.VECTOR_API.getMetric(), Metric.Engine.VECTOR_API.getMetric(),
                "VECTOR_API.getMetric() should return the same instance on every call");
        assertSame(Metric.Engine.SIMSIMD.getMetric(), Metric.Engine.SIMSIMD.getMetric(),
                "SIMSIMD.getMetric() should return the same instance on every call (holder pattern)");
    }
}
