package ru.mcashesha.metrics;

/**
 * JNI bridge to the SimSIMD native C library for high-performance distance computation.
 *
 * <p>SimSIMD is a C library that provides highly optimized SIMD implementations of
 * distance functions, compiled with aggressive optimization flags ({@code -O3 -march=native
 * -ffast-math}). It automatically detects and uses the best available instruction set
 * (AVX-512, AVX2, SSE4, NEON, SVE, etc.) at runtime.</p>
 *
 * <p>All distance methods are declared {@code native} and implemented in the
 * {@code simsimd_jni} shared library, which is loaded at class initialization time.
 * The native side uses {@code GetPrimitiveArrayCritical()} for zero-copy access to
 * the Java array data, avoiding the overhead of copying array contents between the
 * JVM heap and native memory.</p>
 *
 * <h3>Loading</h3>
 * <p>The native library is loaded automatically by {@link NativeLibLoader}, which first
 * attempts to extract it from a bundled JAR resource ({@code /native/<os>-<arch>/<libname>})
 * into a temporary directory.  If the resource is not found, it falls back to
 * {@link System#loadLibrary(String)}, which searches {@code java.library.path}.
 * In most cases no JVM flags are required to use this engine.</p>
 *
 * @see Metric
 * @see Scalar   Pure Java baseline for comparison
 * @see VectorAPI Java Vector API alternative
 */
class SimSIMD implements Metric {

    static {
        NativeLibLoader.load();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of squared L2 distance,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     */
    @Override public native float l2Distance(float[] a, float[] b);

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of inner (dot) product,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     */
    @Override public native float dotProduct(float[] a, float[] b);

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of cosine distance,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     */
    @Override public native float cosineDistance(float[] a, float[] b);

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of Hamming distance
     * over byte arrays, which uses hardware-specific SIMD instructions selected
     * at runtime.</p>
     */
    @Override public native long hammingDistanceB8(byte[] a, byte[] b);

}
