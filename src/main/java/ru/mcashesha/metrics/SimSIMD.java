package ru.mcashesha.metrics;

/**
 * JNI bridge to the SimSIMD native C library for high-performance distance computation.
 *
 * <p>SimSIMD is a C library that provides highly optimized SIMD implementations of
 * distance functions, compiled with aggressive optimization flags ({@code -O3 -march=native
 * -ffast-math}). It automatically detects and uses the best available instruction set
 * (AVX-512, AVX2, SSE4, NEON, SVE, etc.) at runtime.</p>
 *
 * <p>The native JNI shared library is loaded at class initialization time. If loading fails,
 * the error is captured in {@link #LOAD_ERROR} rather than propagated, so that the class
 * remains usable for availability checks. Actual native method invocations will fail with
 * an {@link UnsatisfiedLinkError} if the library was not loaded successfully.</p>
 *
 * <h3>Loading</h3>
 * <p>The native library is loaded automatically by {@link NativeLibLoader}, which first
 * attempts to extract it from a bundled JAR resource ({@code /native/<os>-<arch>/<libname>})
 * into a temporary directory.  If the resource is not found, it falls back to
 * {@link System#loadLibrary(String)}, which searches {@code java.library.path}.
 * In most cases no JVM flags are required to use this engine.</p>
 *
 * <h3>Fail-safe static initializer</h3>
 * <p>Unlike the original design where the static initializer would throw
 * {@link UnsatisfiedLinkError} on load failure (killing the entire class and any enum
 * that references it), this version catches the error and stores it. This is critical
 * because {@link Metric.Engine} is an enum that holds all three engine constants --
 * if SimSIMD's class initializer fails, the JVM marks the class as unusable, which
 * causes {@code ExceptionInInitializerError} to propagate out of {@code Engine}'s
 * static initialization, making SCALAR and VECTOR_API inaccessible too.</p>
 *
 * @see Metric
 * @see Scalar   Pure Java baseline for comparison
 * @see VectorAPI Java Vector API alternative
 */
class SimSIMD implements Metric {

    /**
     * Stores the error encountered during native library loading, or {@code null} if loading
     * succeeded. This field allows the rest of the system to query whether SimSIMD is usable
     * without the static initializer propagating an {@link UnsatisfiedLinkError} that would
     * kill the entire class (and, by extension, any enum or class that references SimSIMD
     * at initialization time).
     *
     * <p>The field is set exactly once during class loading and never modified afterward,
     * so it is safe to read from any thread without synchronization.</p>
     */
    static final Throwable LOAD_ERROR;

    static {
        Throwable error = null;
        try {
            NativeLibLoader.load();
        } catch (UnsatisfiedLinkError | Exception e) {
            // Capture the error instead of letting it propagate. This prevents the class
            // initializer from failing, which would make SimSIMD permanently unusable
            // (ExceptionInInitializerError on first attempt, NoClassDefFoundError on subsequent).
            // Callers should check LOAD_ERROR or call isAvailable() before invoking native methods.
            error = e;
        }
        LOAD_ERROR = error;
    }

    /**
     * Verifies that the native library was loaded successfully before any native method call.
     * If loading failed, throws an {@link UnsatisfiedLinkError} with the original cause attached,
     * providing a clear diagnostic message that includes the root cause of the load failure.
     *
     * <p>This guard is called at the top of every public {@link Metric} method to ensure
     * a clear, informative error rather than an opaque JNI linkage failure.</p>
     */
    private void ensureLoaded() {
        if (LOAD_ERROR != null) {
            UnsatisfiedLinkError ule = new UnsatisfiedLinkError(
                    "SimSIMD native library failed to load: " + LOAD_ERROR.getMessage());
            ule.initCause(LOAD_ERROR);
            throw ule;
        }
    }

    // --- Static native method declarations ---
    // These are the actual JNI entry points whose C-side names are derived from the fully
    // qualified class name + method name (e.g., Java_ru_mcashesha_metrics_SimSIMD_l2Distance).
    // They are declared static because the C code uses jclass (not jobject) as the second
    // parameter, matching the original "static native" signature that javac -h generated
    // the JNI header from. The public instance methods below delegate to these after
    // verifying that the library loaded successfully via ensureLoaded().
    private static native float l2DistanceNative(float[] a, float[] b);
    private static native float dotProductNative(float[] a, float[] b);
    private static native float cosineDistanceNative(float[] a, float[] b);
    private static native long hammingDistanceB8Native(byte[] a, byte[] b);

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of squared L2 distance,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     *
     * @throws UnsatisfiedLinkError if the native library could not be loaded
     */
    @Override
    public float l2Distance(float[] a, float[] b) {
        ensureLoaded();
        return l2DistanceNative(a, b);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of inner (dot) product,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     *
     * @throws UnsatisfiedLinkError if the native library could not be loaded
     */
    @Override
    public float dotProduct(float[] a, float[] b) {
        ensureLoaded();
        return dotProductNative(a, b);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of cosine distance,
     * which uses hardware-specific SIMD instructions selected at runtime.</p>
     *
     * @throws UnsatisfiedLinkError if the native library could not be loaded
     */
    @Override
    public float cosineDistance(float[] a, float[] b) {
        ensureLoaded();
        return cosineDistanceNative(a, b);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates to the SimSIMD native implementation of Hamming distance
     * over byte arrays, which uses hardware-specific SIMD instructions selected
     * at runtime.</p>
     *
     * @throws UnsatisfiedLinkError if the native library could not be loaded
     */
    @Override
    public long hammingDistanceB8(byte[] a, byte[] b) {
        ensureLoaded();
        return hammingDistanceB8Native(a, b);
    }

    /**
     * Returns {@code true} if the native SimSIMD library was loaded successfully
     * and all native methods are available for use.
     *
     * <p>This method never throws. It is safe to call from any context, including
     * during class initialization, to decide whether to use the SIMSIMD engine
     * or fall back to SCALAR/VECTOR_API.</p>
     *
     * @return {@code true} if SimSIMD is operational, {@code false} otherwise
     */
    static boolean isAvailable() {
        return LOAD_ERROR == null;
    }

}
