package ru.mcashesha.metrics;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

class VectorAPI implements Metric {

    static final VectorSpecies<Float> floatSpecies = FloatVector.SPECIES_PREFERRED;

    static final VectorSpecies<Byte> byteSpecies = ByteVector.SPECIES_PREFERRED;

    @Override public float l2Distance(float[] a, float[] b) {
        FloatVector sumVec = FloatVector.zero(floatSpecies);

        int index = 0;

        int upperBound = floatSpecies.loopBound(a.length);

        for (; index < upperBound; index += floatSpecies.length()) {
            FloatVector vectorA = FloatVector.fromArray(floatSpecies, a, index);

            FloatVector vectorB = FloatVector.fromArray(floatSpecies, b, index);

            FloatVector vectorDiff = vectorA.sub(vectorB);

            sumVec = vectorDiff.fma(vectorDiff, sumVec);
        }

        float sumSquares = sumVec.reduceLanes(VectorOperators.ADD);

        for (; index < a.length; index++) {
            float diff = a[index] - b[index];

            sumSquares += diff * diff;
        }

        return sumSquares;
    }

    @Override public float dotProduct(float[] a, float[] b) {
        FloatVector sumVec = FloatVector.zero(floatSpecies);

        int i = 0;

        int upperBound = floatSpecies.loopBound(a.length);

        for (; i < upperBound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            sumVec = va.fma(vb, sumVec);
        }

        float sum = sumVec.reduceLanes(VectorOperators.ADD);

        for (; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    @Override public float cosineDistance(float[] a, float[] b) {
        FloatVector dotVec = FloatVector.zero(floatSpecies);
        FloatVector sumAVec = FloatVector.zero(floatSpecies);
        FloatVector sumBVec = FloatVector.zero(floatSpecies);

        int i = 0, bound = floatSpecies.loopBound(a.length);

        for (; i < bound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            dotVec = va.fma(vb, dotVec);

            sumAVec = va.fma(va, sumAVec);

            sumBVec = vb.fma(vb, sumBVec);
        }

        float dot = dotVec.reduceLanes(VectorOperators.ADD);
        float sumA = sumAVec.reduceLanes(VectorOperators.ADD);
        float sumB = sumBVec.reduceLanes(VectorOperators.ADD);

        for (; i < a.length; i++) {
            dot += a[i] * b[i];

            sumA += a[i] * a[i];

            sumB += b[i] * b[i];
        }

        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

    @Override public long hammingDistanceB8(byte[] a, byte[] b) {
        long distance = 0;

        int index = 0;

        int upperBound = byteSpecies.loopBound(a.length);
        int longsPerVector = byteSpecies.length() / Long.BYTES;

        for (; index < upperBound; index += byteSpecies.length()) {
            ByteVector vectorA = ByteVector.fromArray(byteSpecies, a, index);

            ByteVector vectorB = ByteVector.fromArray(byteSpecies, b, index);

            LongVector longXor = vectorA.lanewise(VectorOperators.XOR, vectorB)
                .reinterpretAsLongs();

            for (int lane = 0; lane < longsPerVector; lane++)
                distance += Long.bitCount(longXor.lane(lane));
        }

        for (; index < a.length; index++) {
            int xorValue = (a[index] ^ b[index]) & 0xFF;

            distance += Integer.bitCount(xorValue);
        }

        return distance;
    }

}
