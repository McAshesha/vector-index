package ru.mcashesha.metrics;

class SimSIMD implements Metric {

    static {
        System.loadLibrary("simsimd_jni");
    }

    @Override public native float l2Distance(float[] a, float[] b);

    @Override public native float dotProduct(float[] a, float[] b);

    @Override public native float cosineDistance(float[] a, float[] b);

    @Override public native long hammingDistanceB8(byte[] a, byte[] b);

}
