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
 *   <li>Dual accumulators are used to saturate the FMA pipeline. Modern CPUs can
 *       execute 2 FMA operations per cycle but each has 4-5 cycles of latency.
 *       By maintaining two independent accumulator vectors, the CPU can issue an
 *       FMA to the second accumulator while the first is still in-flight, effectively
 *       doubling throughput.</li>
 *   <li>Each method uses a three-phase approach:
 *       (1) a dual-vector main loop processing {@code 2 * SPECIES_PREFERRED.length()}
 *           elements per iteration,
 *       (2) a single-vector loop for the remainder that is at least one full vector
 *           wide but not two, and
 *       (3) a scalar tail loop for any remaining elements that do not fill a full
 *           vector.</li>
 * </ul>
 *
 * <p>Requires JVM flag {@code --add-modules jdk.incubator.vector} at both compile
 * and runtime.</p>
 *
 * @see Metric
 * @see Scalar   Pure Java baseline for comparison
 * @see SimSIMD  Native C alternative via JNI
 */
class VectorAPI implements Metric, OffsetMetric {

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
     *
     * <p>Dual accumulators saturate the FMA pipeline (2 FMA/cycle throughput vs
     * 4-5 cycle latency). Processing 2 vector widths per iteration hides the
     * reduction latency.</p>
     */
    @Override public float l2Distance(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        // Dual accumulator vectors -- two independent accumulators allow the CPU to
        // issue FMA to sumVec1 while sumVec0's FMA is still in the pipeline, effectively
        // doubling throughput since modern CPUs can execute 2 FMA/cycle but each has
        // 4-5 cycles of latency.
        FloatVector sumVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumVec1 = FloatVector.zero(floatSpecies);

        int index = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- process 2 full vector widths per iteration.
        // The bound ensures we have at least 2 full vectors of data remaining.
        int upperBound2 = a.length - doubleStride + 1;

        for (; index < upperBound2; index += doubleStride) {
            FloatVector vectorA0 = FloatVector.fromArray(floatSpecies, a, index);
            FloatVector vectorB0 = FloatVector.fromArray(floatSpecies, b, index);
            FloatVector vectorDiff0 = vectorA0.sub(vectorB0);
            sumVec0 = vectorDiff0.fma(vectorDiff0, sumVec0);

            FloatVector vectorA1 = FloatVector.fromArray(floatSpecies, a, index + speciesLength);
            FloatVector vectorB1 = FloatVector.fromArray(floatSpecies, b, index + speciesLength);
            FloatVector vectorDiff1 = vectorA1.sub(vectorB1);
            sumVec1 = vectorDiff1.fma(vectorDiff1, sumVec1);
        }

        // Phase 2: Single-vector SIMD loop -- handle remainder that is at least one
        // full vector wide but didn't fit into the dual-vector loop.
        int upperBound1 = floatSpecies.loopBound(a.length);

        for (; index < upperBound1; index += speciesLength) {
            FloatVector vectorA = FloatVector.fromArray(floatSpecies, a, index);
            FloatVector vectorB = FloatVector.fromArray(floatSpecies, b, index);
            FloatVector vectorDiff = vectorA.sub(vectorB);
            sumVec0 = vectorDiff.fma(vectorDiff, sumVec0);
        }

        // Merge the two accumulators and horizontally reduce all lanes into a scalar
        float sumSquares = sumVec0.add(sumVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail -- handle remaining elements that don't fill a full SIMD vector
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
     *
     * <p>Dual accumulators saturate the FMA pipeline (2 FMA/cycle throughput vs
     * 4-5 cycle latency). Processing 2 vector widths per iteration hides the
     * reduction latency.</p>
     */
    @Override public float dotProduct(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        // Dual accumulator vectors for partial dot product sums across SIMD lanes.
        // Two independent accumulators allow the CPU to issue FMA to sumVec1 while
        // sumVec0's FMA is still in the pipeline.
        FloatVector sumVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumVec1 = FloatVector.zero(floatSpecies);

        int i = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- FMA accumulates a[i]*b[i] into two
        // independent accumulators, processing 2 vector widths per iteration.
        int upperBound2 = a.length - doubleStride + 1;

        for (; i < upperBound2; i += doubleStride) {
            FloatVector va0 = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb0 = FloatVector.fromArray(floatSpecies, b, i);
            sumVec0 = va0.fma(vb0, sumVec0);

            FloatVector va1 = FloatVector.fromArray(floatSpecies, a, i + speciesLength);
            FloatVector vb1 = FloatVector.fromArray(floatSpecies, b, i + speciesLength);
            sumVec1 = va1.fma(vb1, sumVec1);
        }

        // Phase 2: Single-vector SIMD loop for remainder kicking after dual loop.
        int upperBound1 = floatSpecies.loopBound(a.length);

        for (; i < upperBound1; i += speciesLength) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);
            sumVec0 = va.fma(vb, sumVec0);
        }

        // Merge dual accumulators and horizontally reduce into a scalar sum
        float sum = sumVec0.add(sumVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail for remaining elements
        for (; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes all three required accumulators (dot product, norm of a, norm of b)
     * simultaneously in a single pass. Each of the three quantities uses two independent
     * accumulator vectors (6 total), maximizing FMA pipeline utilization.</p>
     *
     * <p>Dual accumulators saturate the FMA pipeline (2 FMA/cycle throughput vs 4-5 cycle
     * latency). With 6 independent FMA streams (2 per quantity), the CPU can keep its
     * FMA units busy across iterations. Processing 2 vector widths per iteration hides
     * the reduction latency and ensures that the 6 FMA operations in each half-iteration
     * are spread across the pipeline stages without stalling.</p>
     */
    @Override public float cosineDistance(float[] a, float[] b) {
        MetricValidation.validateFloatArrays(a, b);
        // Six independent accumulator vectors: two for each of the three quantities
        // (dot product, squared norm of a, squared norm of b). This maximizes pipeline
        // utilization by providing enough independent FMA chains to keep the execution
        // units saturated despite the 4-5 cycle FMA latency.
        FloatVector dotVec0 = FloatVector.zero(floatSpecies);
        FloatVector dotVec1 = FloatVector.zero(floatSpecies);
        FloatVector sumAVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumAVec1 = FloatVector.zero(floatSpecies);
        FloatVector sumBVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumBVec1 = FloatVector.zero(floatSpecies);

        int i = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- accumulate dot(a,b), ||a||^2, and ||b||^2
        // using two independent accumulators per quantity. Each iteration processes
        // 2 * speciesLength elements and issues 6 FMA operations (3 per vector width).
        int upperBound2 = a.length - doubleStride + 1;

        for (; i < upperBound2; i += doubleStride) {
            // First vector width: feeds accumulators 0
            FloatVector va0 = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb0 = FloatVector.fromArray(floatSpecies, b, i);
            dotVec0 = va0.fma(vb0, dotVec0);
            sumAVec0 = va0.fma(va0, sumAVec0);
            sumBVec0 = vb0.fma(vb0, sumBVec0);

            // Second vector width: feeds accumulators 1, independent from accumulators 0
            FloatVector va1 = FloatVector.fromArray(floatSpecies, a, i + speciesLength);
            FloatVector vb1 = FloatVector.fromArray(floatSpecies, b, i + speciesLength);
            dotVec1 = va1.fma(vb1, dotVec1);
            sumAVec1 = va1.fma(va1, sumAVec1);
            sumBVec1 = vb1.fma(vb1, sumBVec1);
        }

        // Phase 2: Single-vector SIMD loop -- handle remainder that fits one full
        // vector width but not two. Only feeds accumulators 0 since there's no
        // second chunk to pipeline against.
        int upperBound1 = floatSpecies.loopBound(a.length);

        for (; i < upperBound1; i += speciesLength) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);
            dotVec0 = va.fma(vb, dotVec0);
            sumAVec0 = va.fma(va, sumAVec0);
            sumBVec0 = vb.fma(vb, sumBVec0);
        }

        // Merge the paired accumulators and horizontally reduce all lanes to scalars
        float dot = dotVec0.add(dotVec1).reduceLanes(VectorOperators.ADD);
        float sumA = sumAVec0.add(sumAVec1).reduceLanes(VectorOperators.ADD);
        float sumB = sumBVec0.add(sumBVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail for remaining elements
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
        MetricValidation.validateByteArrays(a, b);
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

    // ======================== Offset-based methods for flat array layout ========================
    // Dual accumulators match the optimization in the non-offset methods, ensuring the
    // hot-path scan loop in IVFIndexFlat benefits from full FMA pipeline saturation.

    /** {@inheritDoc} */
    @Override public float l2Distance(float[] a, float[] b, int bOffset, int length) {
        // Dual accumulator vectors for FMA pipeline saturation
        FloatVector sumVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumVec1 = FloatVector.zero(floatSpecies);

        int index = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- process 2 full vector widths per iteration
        int upperBound2 = length - doubleStride + 1;

        for (; index < upperBound2; index += doubleStride) {
            FloatVector vectorA0 = FloatVector.fromArray(floatSpecies, a, index);
            FloatVector vectorB0 = FloatVector.fromArray(floatSpecies, b, bOffset + index);
            FloatVector vectorDiff0 = vectorA0.sub(vectorB0);
            sumVec0 = vectorDiff0.fma(vectorDiff0, sumVec0);

            FloatVector vectorA1 = FloatVector.fromArray(floatSpecies, a, index + speciesLength);
            FloatVector vectorB1 = FloatVector.fromArray(floatSpecies, b, bOffset + index + speciesLength);
            FloatVector vectorDiff1 = vectorA1.sub(vectorB1);
            sumVec1 = vectorDiff1.fma(vectorDiff1, sumVec1);
        }

        // Phase 2: Single-vector SIMD loop for remainder that fits one full vector width
        int upperBound1 = floatSpecies.loopBound(length);

        for (; index < upperBound1; index += speciesLength) {
            FloatVector vectorA = FloatVector.fromArray(floatSpecies, a, index);
            FloatVector vectorB = FloatVector.fromArray(floatSpecies, b, bOffset + index);
            FloatVector vectorDiff = vectorA.sub(vectorB);
            sumVec0 = vectorDiff.fma(vectorDiff, sumVec0);
        }

        // Merge the two accumulators and horizontally reduce all lanes into a scalar
        float sumSquares = sumVec0.add(sumVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail for remaining elements
        for (; index < length; index++) {
            float diff = a[index] - b[bOffset + index];
            sumSquares += diff * diff;
        }
        return sumSquares;
    }

    /** {@inheritDoc} */
    @Override public float dotProduct(float[] a, float[] b, int bOffset, int length) {
        // Dual accumulator vectors for FMA pipeline saturation
        FloatVector sumVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumVec1 = FloatVector.zero(floatSpecies);

        int i = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- FMA accumulates a[i]*b[bOffset+i] into two
        // independent accumulators, processing 2 vector widths per iteration
        int upperBound2 = length - doubleStride + 1;

        for (; i < upperBound2; i += doubleStride) {
            FloatVector va0 = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb0 = FloatVector.fromArray(floatSpecies, b, bOffset + i);
            sumVec0 = va0.fma(vb0, sumVec0);

            FloatVector va1 = FloatVector.fromArray(floatSpecies, a, i + speciesLength);
            FloatVector vb1 = FloatVector.fromArray(floatSpecies, b, bOffset + i + speciesLength);
            sumVec1 = va1.fma(vb1, sumVec1);
        }

        // Phase 2: Single-vector SIMD loop for remainder
        int upperBound1 = floatSpecies.loopBound(length);

        for (; i < upperBound1; i += speciesLength) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb = FloatVector.fromArray(floatSpecies, b, bOffset + i);
            sumVec0 = va.fma(vb, sumVec0);
        }

        // Merge dual accumulators and horizontally reduce into a scalar sum
        float sum = sumVec0.add(sumVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail for remaining elements
        for (; i < length; i++)
            sum += a[i] * b[bOffset + i];
        return sum;
    }

    /** {@inheritDoc} */
    @Override public float cosineDistance(float[] a, float[] b, int bOffset, int length) {
        // Six independent accumulator vectors: two for each of the three quantities
        // (dot product, squared norm of a, squared norm of b)
        FloatVector dotVec0 = FloatVector.zero(floatSpecies);
        FloatVector dotVec1 = FloatVector.zero(floatSpecies);
        FloatVector sumAVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumAVec1 = FloatVector.zero(floatSpecies);
        FloatVector sumBVec0 = FloatVector.zero(floatSpecies);
        FloatVector sumBVec1 = FloatVector.zero(floatSpecies);

        int i = 0;
        int speciesLength = floatSpecies.length();
        int doubleStride = 2 * speciesLength;

        // Phase 1: Dual-vector SIMD loop -- accumulate dot(a,b), ||a||^2, and ||b||^2
        // using two independent accumulators per quantity
        int upperBound2 = length - doubleStride + 1;

        for (; i < upperBound2; i += doubleStride) {
            // First vector width: feeds accumulators 0
            FloatVector va0 = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb0 = FloatVector.fromArray(floatSpecies, b, bOffset + i);
            dotVec0 = va0.fma(vb0, dotVec0);
            sumAVec0 = va0.fma(va0, sumAVec0);
            sumBVec0 = vb0.fma(vb0, sumBVec0);

            // Second vector width: feeds accumulators 1, independent from accumulators 0
            FloatVector va1 = FloatVector.fromArray(floatSpecies, a, i + speciesLength);
            FloatVector vb1 = FloatVector.fromArray(floatSpecies, b, bOffset + i + speciesLength);
            dotVec1 = va1.fma(vb1, dotVec1);
            sumAVec1 = va1.fma(va1, sumAVec1);
            sumBVec1 = vb1.fma(vb1, sumBVec1);
        }

        // Phase 2: Single-vector SIMD loop for remainder that fits one full vector width
        int upperBound1 = floatSpecies.loopBound(length);

        for (; i < upperBound1; i += speciesLength) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);
            FloatVector vb = FloatVector.fromArray(floatSpecies, b, bOffset + i);
            dotVec0 = va.fma(vb, dotVec0);
            sumAVec0 = va.fma(va, sumAVec0);
            sumBVec0 = vb.fma(vb, sumBVec0);
        }

        // Merge the paired accumulators and horizontally reduce all lanes to scalars
        float dot = dotVec0.add(dotVec1).reduceLanes(VectorOperators.ADD);
        float sumA = sumAVec0.add(sumAVec1).reduceLanes(VectorOperators.ADD);
        float sumB = sumBVec0.add(sumBVec1).reduceLanes(VectorOperators.ADD);

        // Phase 3: Scalar tail for remaining elements
        for (; i < length; i++) {
            float bi = b[bOffset + i];
            dot += a[i] * bi;
            sumA += a[i] * a[i];
            sumB += bi * bi;
        }
        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

}
