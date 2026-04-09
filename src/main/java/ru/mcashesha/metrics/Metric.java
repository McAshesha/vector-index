package ru.mcashesha.metrics;

/**
 * Core interface defining distance computation contracts for vector similarity search.
 *
 * <p>Provides four distance functions that can be implemented by different computation
 * engines (pure Java, SIMD via Vector API, or native C via SimSIMD). All float-based
 * distance functions follow the convention that <em>lower values indicate greater
 * similarity</em>, enabling uniform comparisons regardless of the metric type.</p>
 *
 * <p>The interface is organized around two supporting enums:
 * <ul>
 *   <li>{@link Type} -- selects which distance metric to use and handles any
 *       necessary sign transformations (e.g., negating dot product).</li>
 *   <li>{@link Engine} -- selects the concrete implementation that performs the
 *       computation (scalar loops, hardware SIMD, or native JNI).</li>
 * </ul>
 *
 * @see Scalar   Pure Java loop-based implementation
 * @see VectorAPI SIMD-accelerated implementation via jdk.incubator.vector
 * @see SimSIMD  Native C implementation via JNI (SimSIMD library)
 */
public interface Metric {

    /**
     * Computes the squared Euclidean (L2) distance between two vectors.
     *
     * <p>Defined as {@code sum((a[i] - b[i])^2)} for all dimensions. The result is
     * <em>not</em> square-rooted, since the square root is a monotonic transformation
     * and omitting it avoids unnecessary computation in nearest-neighbor comparisons.</p>
     *
     * @param a first vector
     * @param b second vector; must have the same length as {@code a}
     * @return the sum of squared component-wise differences (always non-negative)
     */
    float l2Distance(float[] a, float[] b);

    /**
     * Computes the inner (dot) product of two vectors.
     *
     * <p>Defined as {@code sum(a[i] * b[i])} for all dimensions. A higher dot product
     * indicates greater similarity for normalized vectors. Note that when used via
     * {@link Type#DOT_PRODUCT}, the result is <em>negated</em> so that lower values
     * indicate greater similarity, consistent with the other metric types.</p>
     *
     * @param a first vector
     * @param b second vector; must have the same length as {@code a}
     * @return the inner product of the two vectors
     */
    float dotProduct(float[] a, float[] b);

    /**
     * Computes the cosine distance between two vectors.
     *
     * <p>Defined as {@code 1 - cos(theta)}, where {@code cos(theta)} is the cosine
     * similarity {@code dot(a,b) / (||a|| * ||b||)}. The result ranges from 0
     * (identical direction) to 2 (opposite direction).</p>
     *
     * @param a first vector
     * @param b second vector; must have the same length as {@code a}
     * @return the cosine distance (always in [0, 2])
     */
    float cosineDistance(float[] a, float[] b);

    /**
     * Computes the Hamming distance between two byte arrays, counting differing bits.
     *
     * <p>Each byte is treated as 8 independent bits. The distance is the total number
     * of bit positions where the two arrays differ, computed via XOR followed by
     * population count on each byte.</p>
     *
     * @param a first byte array
     * @param b second byte array; must have the same length as {@code a}
     * @return the total number of differing bits across all bytes
     */
    long hammingDistanceB8(byte[] a, byte[] b);

    /**
     * Functional interface representing a distance function between two float vectors.
     *
     * <p>Used on the hot path to avoid virtual dispatch overhead. Instead of calling
     * {@link Type#distance(Engine, float[], float[])} (which involves enum dispatch
     * and interface method invocation), callers can resolve a {@code DistanceFunction}
     * once via {@link Type#resolveFunction(Engine)} and invoke it directly as a
     * method reference or lambda, allowing the JIT compiler to inline aggressively.</p>
     */
    @FunctionalInterface
    interface DistanceFunction {
        /**
         * Computes the distance between two float vectors.
         *
         * @param a first vector
         * @param b second vector; must have the same length as {@code a}
         * @return the distance value (lower = more similar)
         */
        float compute(float[] a, float[] b);
    }

    /**
     * Enumerates the supported distance metric types.
     *
     * <p>Each constant delegates to the corresponding method on a {@link Metric}
     * implementation, applying any necessary transformations to ensure the universal
     * invariant: <em>lower distance values indicate greater similarity</em>.</p>
     */
    enum Type {
        /**
         * Squared Euclidean (L2) distance.
         *
         * <p>Delegates directly to {@link Metric#l2Distance(float[], float[])}.
         * No transformation needed since L2 distance is naturally non-negative
         * with zero indicating identical vectors.</p>
         */
        L2SQ_DISTANCE() {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.metric.l2Distance(a, b);
            }
        },
        /**
         * Dot product distance (negated).
         *
         * <p>Delegates to {@link Metric#dotProduct(float[], float[])} and
         * <em>negates</em> the result. This is because a higher raw dot product
         * means greater similarity, but the system-wide convention requires
         * lower values to mean greater similarity. By negating, the most similar
         * vectors receive the lowest (most negative) distance values.</p>
         */
        DOT_PRODUCT {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                // Negate so that higher dot product (= more similar) becomes lower distance
                return -engine.metric.dotProduct(a, b);
            }
        },
        /**
         * Cosine distance (1 - cosine similarity).
         *
         * <p>Delegates directly to {@link Metric#cosineDistance(float[], float[])}.
         * No transformation needed since cosine distance naturally satisfies the
         * lower-is-closer convention (0 = identical direction, 2 = opposite).</p>
         */
        COSINE_DISTANCE {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.metric.cosineDistance(a, b);
            }
        };

        /**
         * Computes the distance between two vectors using the specified engine.
         *
         * <p>This method involves virtual dispatch through both the enum and the
         * engine's metric implementation. For hot-path usage, prefer
         * {@link #resolveFunction(Engine)} to obtain a direct method reference.</p>
         *
         * @param engine the computation engine to use
         * @param a      first vector
         * @param b      second vector; must have the same length as {@code a}
         * @return the distance value (lower = more similar)
         */
        public abstract float distance(Engine engine, float[] a, float[] b);

        /**
         * Resolves a {@link DistanceFunction} method reference for the given engine.
         *
         * <p>Returns a lambda or method reference that captures the engine's concrete
         * {@link Metric} implementation, bypassing the enum dispatch on every call.
         * This is critical for hot-path performance: the JIT compiler can inline the
         * resolved function, eliminating both the enum switch and the interface
         * virtual dispatch.</p>
         *
         * <p>For {@link #DOT_PRODUCT}, the returned function wraps the dot product
         * in a negation lambda to maintain the lower-is-closer invariant.</p>
         *
         * @param engine the computation engine whose metric implementation to bind
         * @return a {@link DistanceFunction} that computes this metric type using the
         *         engine's implementation
         * @throws IllegalStateException if the metric type is unrecognized
         */
        public DistanceFunction resolveFunction(Engine engine) {
            Metric m = engine.metric;
            switch (this) {
                case L2SQ_DISTANCE: return m::l2Distance;
                case DOT_PRODUCT: return (a, b) -> -m.dotProduct(a, b);
                case COSINE_DISTANCE: return m::cosineDistance;
                default: throw new IllegalStateException("Unknown metric type: " + this);
            }
        }
    }

    /**
     * Enumerates the available distance computation engines.
     *
     * <p>Each engine wraps a concrete {@link Metric} implementation that uses a
     * different strategy for computing vector distances:
     * <ul>
     *   <li>{@link #SCALAR} -- pure Java loops; portable, no special requirements.</li>
     *   <li>{@link #VECTOR_API} -- uses {@code jdk.incubator.vector} for hardware
     *       SIMD acceleration; requires {@code --add-modules jdk.incubator.vector}.</li>
     *   <li>{@link #SIMSIMD} -- delegates to the SimSIMD native C library via JNI;
     *       requires the {@code simsimd_jni} shared library on the library path.</li>
     * </ul>
     *
     * <p>Engines are singletons -- each enum constant holds exactly one {@link Metric}
     * instance, created at class load time.</p>
     */
    enum Engine {
        /** Pure Java scalar (loop-based) distance computation. */
        SCALAR(new Scalar()),

        /** SIMD-accelerated distance computation via Java Vector API (jdk.incubator.vector). */
        VECTOR_API(new VectorAPI()),

        /** Native C distance computation via SimSIMD JNI bridge. */
        SIMSIMD(new SimSIMD());

        /** The concrete {@link Metric} implementation used by this engine. */
        final Metric metric;

        /**
         * Constructs an engine with the given metric implementation.
         *
         * @param metric the concrete distance computation implementation
         */
        Engine(Metric metric) {
            this.metric = metric;
        }

        /**
         * Returns the concrete {@link Metric} implementation held by this engine.
         *
         * @return the metric implementation
         */
        Metric getMetric() {
            return metric;
        }
    }

}
