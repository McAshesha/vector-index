package ru.mcashesha.metrics;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MetricValidation} input validation and its integration
 * with {@link Scalar} and {@link VectorAPI}.
 */
class MetricValidationTest {

    // ==================== Tests with validation disabled (default) ====================

    @Nested
    class WhenValidationDisabled {

        @Test
        void validateFlag_isFalseByDefault() {
            assertFalse(MetricValidation.VALIDATE,
                "VALIDATE should be false by default (no system property set)");
        }

        @Test
        void validateFloatArrays_withNull_doesNotThrow() {
            assertDoesNotThrow(() -> MetricValidation.validateFloatArrays(null, null));
        }

        @Test
        void validateByteArrays_withNull_doesNotThrow() {
            assertDoesNotThrow(() -> MetricValidation.validateByteArrays(null, null));
        }

        @Test
        void validateFloatArrays_withMismatchedLengths_doesNotThrow() {
            assertDoesNotThrow(() ->
                MetricValidation.validateFloatArrays(new float[3], new float[5]));
        }

        @Test
        void validateByteArrays_withMismatchedLengths_doesNotThrow() {
            assertDoesNotThrow(() ->
                MetricValidation.validateByteArrays(new byte[3], new byte[5]));
        }

        @Test
        void scalar_l2Distance_validInputs_worksNormally() {
            Scalar scalar = new Scalar();
            float[] a = {1f, 2f, 3f};
            float[] b = {4f, 5f, 6f};
            assertEquals(27f, scalar.l2Distance(a, b), 1e-6f);
        }

        @Test
        void vectorAPI_l2Distance_validInputs_worksNormally() {
            VectorAPI vectorApi = new VectorAPI();
            float[] a = {1f, 2f, 3f};
            float[] b = {4f, 5f, 6f};
            assertEquals(27f, vectorApi.l2Distance(a, b), 1e-3f);
        }
    }

    // ==================== Tests with validation enabled ====================

    @Nested
    class WhenValidationEnabled {

        @BeforeAll
        static void enableValidation() {
            MetricValidation.VALIDATE = true;
        }

        @AfterAll
        static void restoreValidation() {
            MetricValidation.VALIDATE = false;
        }

        // -- Float array validation --

        @Test
        void validateFloatArrays_nullFirst_throwsIAE() {
            assertTrue(MetricValidation.VALIDATE, "VALIDATE should be true");
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateFloatArrays(null, new float[3]));
            assertTrue(ex.getMessage().contains("null"));
        }

        @Test
        void validateFloatArrays_nullSecond_throwsIAE() {
            assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateFloatArrays(new float[3], null));
        }

        @Test
        void validateFloatArrays_bothNull_throwsIAE() {
            assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateFloatArrays(null, null));
        }

        @Test
        void validateFloatArrays_lengthMismatch_throwsIAE() {
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateFloatArrays(new float[3], new float[5]));
            assertTrue(ex.getMessage().contains("3") && ex.getMessage().contains("5"));
        }

        @Test
        void validateFloatArrays_validInputs_doesNotThrow() {
            assertDoesNotThrow(() ->
                MetricValidation.validateFloatArrays(new float[3], new float[3]));
        }

        // -- Byte array validation --

        @Test
        void validateByteArrays_nullFirst_throwsIAE() {
            assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateByteArrays(null, new byte[3]));
        }

        @Test
        void validateByteArrays_nullSecond_throwsIAE() {
            assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateByteArrays(new byte[3], null));
        }

        @Test
        void validateByteArrays_lengthMismatch_throwsIAE() {
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> MetricValidation.validateByteArrays(new byte[3], new byte[5]));
            assertTrue(ex.getMessage().contains("3") && ex.getMessage().contains("5"));
        }

        @Test
        void validateByteArrays_validInputs_doesNotThrow() {
            assertDoesNotThrow(() ->
                MetricValidation.validateByteArrays(new byte[3], new byte[3]));
        }

        // -- Integration with Scalar --

        @Test
        void scalar_l2Distance_nullInput_throwsIAE() {
            Scalar scalar = new Scalar();
            assertThrows(IllegalArgumentException.class,
                () -> scalar.l2Distance(null, new float[3]));
        }

        @Test
        void scalar_dotProduct_nullInput_throwsIAE() {
            Scalar scalar = new Scalar();
            assertThrows(IllegalArgumentException.class,
                () -> scalar.dotProduct(null, new float[3]));
        }

        @Test
        void scalar_cosineDistance_nullInput_throwsIAE() {
            Scalar scalar = new Scalar();
            assertThrows(IllegalArgumentException.class,
                () -> scalar.cosineDistance(null, new float[3]));
        }

        @Test
        void scalar_hammingDistanceB8_nullInput_throwsIAE() {
            Scalar scalar = new Scalar();
            assertThrows(IllegalArgumentException.class,
                () -> scalar.hammingDistanceB8(null, new byte[3]));
        }

        @Test
        void scalar_l2Distance_lengthMismatch_throwsIAE() {
            Scalar scalar = new Scalar();
            assertThrows(IllegalArgumentException.class,
                () -> scalar.l2Distance(new float[3], new float[5]));
        }

        // -- Integration with VectorAPI --

        @Test
        void vectorAPI_l2Distance_nullInput_throwsIAE() {
            VectorAPI vectorApi = new VectorAPI();
            assertThrows(IllegalArgumentException.class,
                () -> vectorApi.l2Distance(null, new float[3]));
        }

        @Test
        void vectorAPI_dotProduct_nullInput_throwsIAE() {
            VectorAPI vectorApi = new VectorAPI();
            assertThrows(IllegalArgumentException.class,
                () -> vectorApi.dotProduct(null, new float[3]));
        }

        @Test
        void vectorAPI_cosineDistance_lengthMismatch_throwsIAE() {
            VectorAPI vectorApi = new VectorAPI();
            assertThrows(IllegalArgumentException.class,
                () -> vectorApi.cosineDistance(new float[3], new float[5]));
        }

        @Test
        void vectorAPI_hammingDistanceB8_nullInput_throwsIAE() {
            VectorAPI vectorApi = new VectorAPI();
            assertThrows(IllegalArgumentException.class,
                () -> vectorApi.hammingDistanceB8(null, new byte[3]));
        }
    }
}
