package ru.mcashesha.metrics;

class Scalar implements Metric {

    @Override public float l2Distance(float[] a, float[] b) {
        float sumSq = 0;

        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];

            sumSq += diff * diff;
        }

        return sumSq;
    }

    @Override public float dotProduct(float[] a, float[] b) {
        float sum = 0;

        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    @Override public float cosineDistance(float[] a, float[] b) {
        float dot = 0, sumA = 0, sumB = 0;

        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];

            sumA += a[i] * a[i];

            sumB += b[i] * b[i];
        }

        return 1 - (float)(dot / Math.sqrt((double) sumA * sumB));
    }

    @Override public long hammingDistanceB8(byte[] a, byte[] b) {
        long distance = 0;

        for (int i = 0; i < a.length; i++) {
            int xorVal = (a[i] ^ b[i]) & 0xFF;

            distance += Integer.bitCount(xorVal);
        }

        return distance;
    }

}
