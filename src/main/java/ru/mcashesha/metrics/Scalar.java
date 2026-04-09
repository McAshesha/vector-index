package ru.mcashesha.metrics;

/**
 * Pure Java scalar (loop-based) implementation of all distance functions.
 *
 * <p>This implementation uses simple {@code for} loops with no SIMD intrinsics
 * or native calls. It serves as the portable baseline that works on any JVM
 * without special flags or native libraries. While it is the slowest engine,
 * it is useful for correctness validation and environments where SIMD or JNI
 * are unavailable.</p>
 *
 * <p>All methods assume that the input arrays have the same length. Passing
 * arrays of different lengths will result in either an {@link ArrayIndexOutOfBoundsException}
 * or incorrect results (if {@code b} is longer than {@code a}, extra elements are ignored).</p>
 *
 * @see Metric
 * @see VectorAPI  SIMD-accelerated alternative
 * @see SimSIMD   Native C alternative via JNI
 */
class Scalar implements Metric, OffsetMetric {

    /**
     * {@inheritDoc}
     *
     * <p>Computes the squared L2 distance by iterating through all dimensions,
     * computing the difference at each position, squaring it, and accumulating
     * the sum. The square root is intentionally omitted since it is a monotonic
     * transformation that does not affect nearest-neighbor ordering.</p>
     */
    @Override public float l2Distance(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        float sumSq = 0;

        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];

            sumSq += diff * diff;
        }

        return sumSq;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes the dot product by accumulating the element-wise product of
     * corresponding components across all dimensions.</p>
     */
    @Override public float dotProduct(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        float sum = 0;

        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes cosine distance in a single pass by simultaneously accumulating
     * three values: the dot product of {@code a} and {@code b}, the squared norm
     * of {@code a}, and the squared norm of {@code b}. The final result is
     * {@code 1 - dot / sqrt(sumA * sumB)}, which is the cosine distance.</p>
     *
     * <p>The intermediate squared norms are multiplied as doubles before taking
     * the square root to reduce floating-point overflow risk for high-dimensional
     * vectors with large component values.</p>
     */
    @Override public float cosineDistance(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        float dot = 0, sumA = 0, sumB = 0;

        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];

            sumA += a[i] * a[i];

            sumB += b[i] * b[i];
        }

        // Cast to double before multiplying norms to avoid overflow, then sqrt
        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes the Hamming distance by XOR-ing each pair of bytes and counting
     * the set bits (population count) in the result. The {@code & 0xFF} mask
     * promotes the signed byte to an unsigned int before passing to
     * {@link Integer#bitCount(int)}, ensuring correct handling of negative byte values.</p>
     */
    @Override public long hammingDistanceB8(byte[] a, byte[] b) {
        MetricValidation.validateByteArrays(a, b);
        long distance = 0;

        for (int i = 0; i < a.length; i++) {
            // Mask to unsigned int (0-255) to ensure bitCount works correctly
            int xorVal = (a[i] ^ b[i]) & 0xFF;

            distance += Integer.bitCount(xorVal);
        }

        return distance;
    }

    // ======================== Offset-based methods for flat array layout ========================

    /** {@inheritDoc} */
    @Override public float l2Distance(float[] a, float[] b, int bOffset, int length) {
        float sumSq = 0;
        for (int i = 0; i < length; i++) {
            float diff = a[i] - b[bOffset + i];
            sumSq += diff * diff;
        }
        return sumSq;
    }

    /** {@inheritDoc} */
    @Override public float dotProduct(float[] a, float[] b, int bOffset, int length) {
        float sum = 0;
        for (int i = 0; i < length; i++)
            sum += a[i] * b[bOffset + i];
        return sum;
    }

    /** {@inheritDoc} */
    @Override public float cosineDistance(float[] a, float[] b, int bOffset, int length) {
        float dot = 0, sumA = 0, sumB = 0;
        for (int i = 0; i < length; i++) {
            float bi = b[bOffset + i];
            dot += a[i] * bi;
            sumA += a[i] * a[i];
            sumB += bi * bi;
        }
        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

}
