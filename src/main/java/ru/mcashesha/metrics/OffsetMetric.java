package ru.mcashesha.metrics;

/**
 * Extension of {@link Metric} that supports offset-based distance computation against flat arrays.
 *
 * <p>Standard {@link Metric} methods accept two separate {@code float[]} arrays. This interface
 * adds overloads where the second operand is a sub-region of a larger flat array, specified by
 * a starting offset and a length. This enables zero-copy distance computation when vectors are
 * stored contiguously in a single {@code float[]} (flat layout), avoiding the need to extract
 * temporary sub-arrays on the hot path.</p>
 *
 * <p>In all offset methods, vector {@code a} is read from index 0 to {@code length - 1}, and
 * vector {@code b} is read from index {@code bOffset} to {@code bOffset + length - 1}.</p>
 *
 * @see Metric
 * @see Scalar
 * @see VectorAPI
 */
interface OffsetMetric {

    /**
     * Computes the squared Euclidean (L2) distance between vector {@code a} and a sub-region
     * of flat array {@code b}.
     *
     * @param a       first vector (read from index 0 to {@code length - 1})
     * @param b       flat array containing packed vectors
     * @param bOffset starting index in {@code b} for the second vector
     * @param length  number of dimensions to compare
     * @return the sum of squared differences (always non-negative)
     */
    float l2Distance(float[] a, float[] b, int bOffset, int length);

    /**
     * Computes the dot product between vector {@code a} and a sub-region of flat array {@code b}.
     *
     * @param a       first vector (read from index 0 to {@code length - 1})
     * @param b       flat array containing packed vectors
     * @param bOffset starting index in {@code b} for the second vector
     * @param length  number of dimensions to compare
     * @return the inner product
     */
    float dotProduct(float[] a, float[] b, int bOffset, int length);

    /**
     * Computes the cosine distance between vector {@code a} and a sub-region of flat array {@code b}.
     *
     * @param a       first vector (read from index 0 to {@code length - 1})
     * @param b       flat array containing packed vectors
     * @param bOffset starting index in {@code b} for the second vector
     * @param length  number of dimensions to compare
     * @return the cosine distance (in [0, 2])
     */
    float cosineDistance(float[] a, float[] b, int bOffset, int length);
}
