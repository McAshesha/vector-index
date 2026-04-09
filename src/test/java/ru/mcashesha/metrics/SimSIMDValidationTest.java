package ru.mcashesha.metrics;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Validates the SimSIMD native JNI implementation's input validation behavior.
 *
 * <p>This test class focuses on the array length validation added as a security fix
 * to the native {@code simsimd_jni.c} code.  Previously, all four native functions
 * only read the length of array {@code a}, which meant that a shorter array {@code b}
 * would cause a buffer over-read in the SIMD kernel -- potentially crashing the JVM
 * or leaking heap data.</p>
 *
 * <p>The native code now checks both array lengths and throws
 * {@link IllegalArgumentException} when they differ.  These tests verify that
 * behavior, along with basic correctness for equal-length and empty arrays.</p>
 *
 * <p>All tests that invoke SimSIMD are guarded by a JUnit assumption: if the native
 * library is not available (e.g., the build was Java-only), the tests are skipped
 * rather than failing.</p>
 */
class SimSIMDValidationTest {

    /** Flag indicating whether the SimSIMD native library loaded successfully. */
    private static boolean simsimdAvailable;

    /** The SimSIMD metric instance, or {@code null} if the native library is unavailable. */
    private static Metric simsimd;

    /**
     * Attempts to load the SimSIMD native library before any tests run.
     * If loading fails, all tests in this class will be skipped via assumptions.
     */
    @BeforeAll
    static void checkSimsimd() {
        try {
            simsimd = Metric.Engine.SIMSIMD.getMetric();
            // Verify it actually works with a trivial call
            simsimd.l2Distance(new float[]{1f}, new float[]{2f});
            simsimdAvailable = true;
        } catch (UnsatisfiedLinkError | ExceptionInInitializerError | NoClassDefFoundError e) {
            simsimdAvailable = false;
            System.err.println("SimSIMD native library not available, skipping SimSIMDValidationTest");
        }
    }

    // ==================== Equal-length arrays: correctness ====================

    /**
     * Verifies that l2Distance produces a correct result when given equal-length arrays.
     * Uses a known-value pair: [1,2,3] vs [4,5,6] -> (3^2 + 3^2 + 3^2) = 27.
     */
    @Test
    void l2Distance_equalLengthArrays_returnsCorrectResult() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        float result = simsimd.l2Distance(a, b);
        assertEquals(27f, result, 1e-3f, "L2 squared distance of [1,2,3] vs [4,5,6] should be 27");
    }

    /**
     * Verifies that dotProduct produces a correct result when given equal-length arrays.
     * Uses a known-value pair: [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32.
     */
    @Test
    void dotProduct_equalLengthArrays_returnsCorrectResult() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};

        float result = simsimd.dotProduct(a, b);
        assertEquals(32f, result, 1e-3f, "Dot product of [1,2,3] . [4,5,6] should be 32");
    }

    /**
     * Verifies that cosineDistance produces a correct result when given equal-length arrays.
     * Identical vectors should have cosine distance of 0.
     */
    @Test
    void cosineDistance_equalLengthArrays_returnsCorrectResult() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};

        float result = simsimd.cosineDistance(a, a);
        assertEquals(0f, result, 1e-3f, "Cosine distance of a vector with itself should be 0");
    }

    /**
     * Verifies that hammingDistanceB8 produces a correct result for equal-length arrays.
     * [0x00] vs [0xFF] should differ in all 8 bits.
     */
    @Test
    void hammingDistanceB8_equalLengthArrays_returnsCorrectResult() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        byte[] a = {0x00};
        byte[] b = {(byte) 0xFF};

        long result = simsimd.hammingDistanceB8(a, b);
        assertEquals(8L, result, "Hamming distance of 0x00 vs 0xFF should be 8");
    }

    // ==================== Mismatched-length arrays: IllegalArgumentException ====================

    /**
     * Verifies that l2Distance throws IllegalArgumentException when array lengths differ.
     * This validates the native-side security fix that prevents buffer over-read.
     */
    @Test
    void l2Distance_mismatchedLengths_throwsIllegalArgumentException() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f};

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> simsimd.l2Distance(a, b),
            "l2Distance should throw IllegalArgumentException for mismatched array lengths"
        );
        assertTrue(ex.getMessage().contains("array lengths must be equal"),
            "Exception message should indicate array length mismatch");
    }

    /**
     * Verifies that dotProduct throws IllegalArgumentException when array lengths differ.
     */
    @Test
    void dotProduct_mismatchedLengths_throwsIllegalArgumentException() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f};

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> simsimd.dotProduct(a, b),
            "dotProduct should throw IllegalArgumentException for mismatched array lengths"
        );
        assertTrue(ex.getMessage().contains("array lengths must be equal"),
            "Exception message should indicate array length mismatch");
    }

    /**
     * Verifies that cosineDistance throws IllegalArgumentException when array lengths differ.
     */
    @Test
    void cosineDistance_mismatchedLengths_throwsIllegalArgumentException() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f};

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> simsimd.cosineDistance(a, b),
            "cosineDistance should throw IllegalArgumentException for mismatched array lengths"
        );
        assertTrue(ex.getMessage().contains("array lengths must be equal"),
            "Exception message should indicate array length mismatch");
    }

    /**
     * Verifies that hammingDistanceB8 throws IllegalArgumentException when array lengths differ.
     */
    @Test
    void hammingDistanceB8_mismatchedLengths_throwsIllegalArgumentException() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        byte[] a = {0x01, 0x02, 0x03};
        byte[] b = {0x04, 0x05};

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> simsimd.hammingDistanceB8(a, b),
            "hammingDistanceB8 should throw IllegalArgumentException for mismatched array lengths"
        );
        assertTrue(ex.getMessage().contains("array lengths must be equal"),
            "Exception message should indicate array length mismatch");
    }

    /**
     * Verifies that the mismatch check works when 'b' is longer than 'a'
     * (not just when 'a' is longer), ensuring the validation is symmetric.
     */
    @Test
    void l2Distance_bLongerThanA_throwsIllegalArgumentException() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {1f, 2f};
        float[] b = {3f, 4f, 5f};

        assertThrows(
            IllegalArgumentException.class,
            () -> simsimd.l2Distance(a, b),
            "l2Distance should throw IllegalArgumentException when b is longer than a"
        );
    }

    // ==================== Empty arrays: return 0 without error ====================

    /**
     * Verifies that l2Distance returns 0 for empty (length-0) arrays without throwing.
     * Empty arrays are a valid edge case: equal lengths (both 0), distance is trivially 0.
     */
    @Test
    void l2Distance_emptyArrays_returnsZero() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {};
        float[] b = {};

        float result = simsimd.l2Distance(a, b);
        assertEquals(0f, result, 1e-6f, "L2 distance of empty arrays should be 0");
    }

    /**
     * Verifies that dotProduct returns 0 for empty (length-0) arrays without throwing.
     */
    @Test
    void dotProduct_emptyArrays_returnsZero() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {};
        float[] b = {};

        float result = simsimd.dotProduct(a, b);
        assertEquals(0f, result, 1e-6f, "Dot product of empty arrays should be 0");
    }

    /**
     * Verifies that cosineDistance returns 0 for empty (length-0) arrays without throwing.
     */
    @Test
    void cosineDistance_emptyArrays_returnsZero() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        float[] a = {};
        float[] b = {};

        float result = simsimd.cosineDistance(a, b);
        assertEquals(0f, result, 1e-3f, "Cosine distance of empty arrays should be 0");
    }

    /**
     * Verifies that hammingDistanceB8 returns 0 for empty (length-0) arrays without throwing.
     */
    @Test
    void hammingDistanceB8_emptyArrays_returnsZero() {
        assumeTrue(simsimdAvailable, "SimSIMD not available");

        byte[] a = {};
        byte[] b = {};

        long result = simsimd.hammingDistanceB8(a, b);
        assertEquals(0L, result, "Hamming distance of empty arrays should be 0");
    }
}
