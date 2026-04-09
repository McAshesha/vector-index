package ru.mcashesha.metrics;

/**
 * Optional fail-fast validation for metric input arrays.
 *
 * <p>Enabled via {@code -Dvectorindex.validate.inputs=true}. When disabled (the default),
 * validation methods return immediately with zero overhead. When enabled, they check for
 * null arrays and length mismatches, throwing {@link IllegalArgumentException} on violations.</p>
 *
 * <p>This is useful during development and testing to get clear error messages instead of
 * {@link NullPointerException} or {@link ArrayIndexOutOfBoundsException}.</p>
 */
// Optional fail-fast validation. Enabled via -Dvectorindex.validate.inputs=true
final class MetricValidation {

    // Not final to allow test overrides; the JIT will still optimize the branch.
    static boolean VALIDATE = Boolean.getBoolean("vectorindex.validate.inputs");

    static void validateFloatArrays(float[] a, float[] b) {
        if (!VALIDATE) return;
        if (a == null || b == null) throw new IllegalArgumentException("vectors must not be null");
        if (a.length != b.length) throw new IllegalArgumentException(
            "vectors must have same length, got " + a.length + " and " + b.length);
    }

    static void validateByteArrays(byte[] a, byte[] b) {
        if (!VALIDATE) return;
        if (a == null || b == null) throw new IllegalArgumentException("byte arrays must not be null");
        if (a.length != b.length) throw new IllegalArgumentException(
            "byte arrays must have same length, got " + a.length + " and " + b.length);
    }

    private MetricValidation() {}
}
