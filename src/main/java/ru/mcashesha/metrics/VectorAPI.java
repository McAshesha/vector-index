package ru.mcashesha.metrics;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated implementation of distance functions using Java's Vector API
 * ({@code jdk.incubator.vector}).
 *
 * <p>This implementation leverages hardware SIMD instructions (SSE, AVX, AVX-512,
 * NEON, etc.) to process multiple floating-point elements in parallel within a
 * single CPU instruction. It uses {@link FloatVector#SPECIES_PREFERRED} to
 * automatically select the widest SIMD register width available on the current
 * hardware, maximizing throughput without manual platform tuning.</p>
 *
 * <h3>Performance characteristics</h3>
 * <ul>
 *   <li>Fused multiply-add (FMA) operations are used wherever possible for both
 *       better numerical accuracy (single rounding instead of two) and higher
 *       throughput (FMA is a single instruction on modern CPUs).</li>
 *   <li>Each method uses a two-phase approach: a vectorized main loop that processes
 *       {@code SPECIES_PREFERRED.length()} elements per iteration, followed by a
 *       scalar tail loop for any remaining elements that do not fill a full vector.</li>
 * </ul>
 *
 * <p>Requires JVM flag {@code --add-modules jdk.incubator.vector} at both compile
 * and runtime.</p>
 *
 * @see Metric
 * @see Scalar   Pure Java baseline for comparison
 * @see SimSIMD  Native C alternative via JNI
 */
class VectorAPI implements Metric {

    /**
     * The preferred float vector species for the current hardware.
     *
     * <p>Automatically selects the widest available SIMD register width. For example,
     * on a machine with AVX-512 this will be 512-bit (16 floats per vector), on AVX2
     * it will be 256-bit (8 floats), and on ARM NEON it will be 128-bit (4 floats).</p>
     */
    static final VectorSpecies<Float> floatSpecies = FloatVector.SPECIES_PREFERRED;

    /**
     * The preferred byte vector species for the current hardware.
     *
     * <p>Used by the Hamming distance computation. The byte species width corresponds
     * to the same physical SIMD register width as the float species.</p>
     */
    static final VectorSpecies<Byte> byteSpecies = ByteVector.SPECIES_PREFERRED;

    /**
     * {@inheritDoc}
     *
     * <p>Uses vectorized subtract + FMA to accumulate squared differences. The FMA
     * instruction computes {@code diff * diff + sumVec} in a single operation,
     * providing both better throughput and improved numerical precision compared
     * to separate multiply and add operations.</p>
     */
    @Override public float l2Distance(float[] a, float[] b) {
        // Accumulator vector -- each lane independently sums partial squared differences
        FloatVector sumVec = FloatVector.zero(floatSpecies);

        int index = 0;

        // loopBound rounds down to the nearest multiple of the vector length
        int upperBound = floatSpecies.loopBound(a.length);

        // Main SIMD loop: process one full vector width per iteration
        for (; index < upperBound; index += floatSpecies.length()) {
            FloatVector vectorA = FloatVector.fromArray(floatSpecies, a, index);

            FloatVector vectorB = FloatVector.fromArray(floatSpecies, b, index);

            FloatVector vectorDiff = vectorA.sub(vectorB);

            // FMA: sumVec += diff * diff (fused multiply-add for accuracy and speed)
            sumVec = vectorDiff.fma(vectorDiff, sumVec);
        }

        // Horizontal reduction: sum all lanes of the accumulator vector into a scalar
        float sumSquares = sumVec.reduceLanes(VectorOperators.ADD);

        // Scalar tail: handle remaining elements that don't fill a full SIMD vector
        for (; index < a.length; index++) {
            float diff = a[index] - b[index];

            sumSquares += diff * diff;
        }

        return sumSquares;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Uses FMA to accumulate the element-wise products. The FMA instruction
     * computes {@code a[i] * b[i] + sum} in a single fused operation.</p>
     */
    @Override public float dotProduct(float[] a, float[] b) {
        // Accumulator vector for partial dot product sums across SIMD lanes
        FloatVector sumVec = FloatVector.zero(floatSpecies);

        int i = 0;

        int upperBound = floatSpecies.loopBound(a.length);

        // Main SIMD loop: FMA accumulates a[i]*b[i] into sumVec
        for (; i < upperBound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            // FMA: sumVec += va * vb (fused multiply-add)
            sumVec = va.fma(vb, sumVec);
        }

        // Horizontal reduction: collapse SIMD lanes into a single scalar sum
        float sum = sumVec.reduceLanes(VectorOperators.ADD);

        // Scalar tail for remaining elements
        for (; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes all three required accumulators (dot product, norm of a, norm of b)
     * simultaneously in a single pass using three independent FMA accumulator vectors.
     * This maximizes data locality by reading each element of {@code a} and {@code b}
     * only once from memory.</p>
     */
    @Override public float cosineDistance(float[] a, float[] b) {
        // Three independent accumulator vectors for the dot product and squared norms
        FloatVector dotVec = FloatVector.zero(floatSpecies);
        FloatVector sumAVec = FloatVector.zero(floatSpecies);
        FloatVector sumBVec = FloatVector.zero(floatSpecies);

        int i = 0, bound = floatSpecies.loopBound(a.length);

        // Main SIMD loop: accumulate dot(a,b), ||a||^2, and ||b||^2 in parallel
        for (; i < bound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            dotVec = va.fma(vb, dotVec);    // dotVec += va * vb

            sumAVec = va.fma(va, sumAVec);  // sumAVec += va * va (squared norm of a)

            sumBVec = vb.fma(vb, sumBVec);  // sumBVec += vb * vb (squared norm of b)
        }

        // Horizontal reduction of all three accumulator vectors
        float dot = dotVec.reduceLanes(VectorOperators.ADD);
        float sumA = sumAVec.reduceLanes(VectorOperators.ADD);
        float sumB = sumBVec.reduceLanes(VectorOperators.ADD);

        // Scalar tail for remaining elements
        for (; i < a.length; i++) {
            dot += a[i] * b[i];

            sumA += a[i] * a[i];

            sumB += b[i] * b[i];
        }

        // cosine_distance = 1 - cos(theta) = 1 - dot / (||a|| * ||b||)
        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

    /**
     * {@inheritDoc}
     *
     * <p>Uses a vectorized XOR followed by reinterpretation to {@link LongVector}
     * for efficient population counting. The approach works as follows:
     * <ol>
     *   <li>Load byte vectors from both arrays and XOR them to find differing bits.</li>
     *   <li>Reinterpret the XOR result as a {@link LongVector} -- this is a zero-cost
     *       bitcast that groups bytes into 64-bit longs without any data movement.</li>
     *   <li>Apply {@link Long#bitCount(long)} on each long lane to count set bits.</li>
     * </ol>
     *
     * <p>The long-based popcount is used because the Vector API does not provide a
     * direct per-byte popcount operation. Grouping into longs reduces the number of
     * popcount calls by a factor of 8 compared to per-byte processing.</p>
     */
    @Override public long hammingDistanceB8(byte[] a, byte[] b) {
        long distance = 0;

        int index = 0;

        int upperBound = byteSpecies.loopBound(a.length);
        // Number of 64-bit long lanes within one byte vector (e.g., 4 for a 256-bit vector)
        int longsPerVector = byteSpecies.length() / Long.BYTES;

        // Main SIMD loop: XOR byte vectors, then popcount via long reinterpretation
        for (; index < upperBound; index += byteSpecies.length()) {
            ByteVector vectorA = ByteVector.fromArray(byteSpecies, a, index);

            ByteVector vectorB = ByteVector.fromArray(byteSpecies, b, index);

            // XOR to find differing bits, then reinterpret as longs for efficient popcount
            LongVector longXor = vectorA.lanewise(VectorOperators.XOR, vectorB)
                .reinterpretAsLongs();

            // Count set bits in each 64-bit lane and accumulate
            for (int lane = 0; lane < longsPerVector; lane++)
                distance += Long.bitCount(longXor.lane(lane));
        }

        // Scalar tail: process remaining bytes individually
        for (; index < a.length; index++) {
            // Mask to unsigned int before bitCount to handle negative byte values correctly
            int xorValue = (a[index] ^ b[index]) & 0xFF;

            distance += Integer.bitCount(xorValue);
        }

        return distance;
    }

}
