#include "ru_mcashesha_metrics_SimSIMD.h"

#include <simsimd/simsimd.h>
#include <math.h>

/* ======================== L2 distance ======================== */
/*
 * float l2Distance(float[] a, float[] b)
 */
JNIEXPORT jfloat JNICALL
Java_ru_mcashesha_metrics_SimSIMD_l2Distance(
    JNIEnv *env,
    jclass clazz,
    jfloatArray a,
    jfloatArray b)
{
    (void) clazz;

    jsize len = (*env)->GetArrayLength(env, a);

    jfloat *ptrA = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *ptrB = (*env)->GetPrimitiveArrayCritical(env, b, NULL);

    if (ptrA == NULL || ptrB == NULL)
    {
        if (ptrA != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
        }
        if (ptrB != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);
        }
        return 0.0f;
    }

    simsimd_distance_t distSq = 0.0;
    simsimd_l2sq_f32(
        (simsimd_f32_t const *) ptrA,
        (simsimd_f32_t const *) ptrB,
        (simsimd_size_t) len,
        &distSq);

    (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);

    return (jfloat) distSq;
}

/* ======================== dot product ======================== */
/*
 * float dotProduct(float[] a, float[] b)
 */
JNIEXPORT jfloat JNICALL
Java_ru_mcashesha_metrics_SimSIMD_dotProduct(
    JNIEnv *env,
    jclass clazz,
    jfloatArray a,
    jfloatArray b)
{
    (void) clazz;

    jsize len = (*env)->GetArrayLength(env, a);

    jfloat *ptrA = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *ptrB = (*env)->GetPrimitiveArrayCritical(env, b, NULL);

    if (ptrA == NULL || ptrB == NULL)
    {
        if (ptrA != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
        }
        if (ptrB != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);
        }
        return 0.0f;
    }

    simsimd_distance_t product = 0.0;
    simsimd_dot_f32(
        (simsimd_f32_t const *) ptrA,
        (simsimd_f32_t const *) ptrB,
        (simsimd_size_t) len,
        &product);

    (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);

    return (jfloat) product;
}

/* ======================== cosine distance ======================== */
/*
 * float cosineDistance(float[] a, float[] b)
 */
JNIEXPORT jfloat JNICALL
Java_ru_mcashesha_metrics_SimSIMD_cosineDistance(
    JNIEnv *env,
    jclass clazz,
    jfloatArray a,
    jfloatArray b)
{
    (void) clazz;

    jsize len = (*env)->GetArrayLength(env, a);

    jfloat *ptrA = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *ptrB = (*env)->GetPrimitiveArrayCritical(env, b, NULL);

    if (ptrA == NULL || ptrB == NULL)
    {
        if (ptrA != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
        }
        if (ptrB != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);
        }
        return 0.0f;
    }

    simsimd_distance_t distance = 0.0;
    simsimd_cos_f32(
        (simsimd_f32_t const *) ptrA,
        (simsimd_f32_t const *) ptrB,
        (simsimd_size_t) len,
        &distance);

    (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);

    return (jfloat) distance;
}

/* ======================== hamming B8 ======================== */
/*
 * long hammingDistanceB8(byte[] a, byte[] b)
 */
JNIEXPORT jlong JNICALL
Java_ru_mcashesha_metrics_SimSIMD_hammingDistanceB8(
    JNIEnv *env,
    jclass clazz,
    jbyteArray a,
    jbyteArray b)
{
    (void) clazz;

    jsize len = (*env)->GetArrayLength(env, a);

    jbyte *ptrA = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jbyte *ptrB = (*env)->GetPrimitiveArrayCritical(env, b, NULL);

    if (ptrA == NULL || ptrB == NULL)
    {
        if (ptrA != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
        }
        if (ptrB != NULL)
        {
            (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);
        }
        return 0L;
    }

    simsimd_distance_t distance = 0.0;
    simsimd_hamming_b8(
        (simsimd_b8_t const *) ptrA,
        (simsimd_b8_t const *) ptrB,
        (simsimd_size_t) len,
        &distance);

    (*env)->ReleasePrimitiveArrayCritical(env, a, ptrA, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, b, ptrB, JNI_ABORT);

    return (jlong) distance;
}
