package ru.mcashesha.metrics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link NativeLibLoader} idempotency and deterministic caching.
 */
class NativeLibLoaderTest {

    /**
     * Calling {@link NativeLibLoader#load()} multiple times must not throw.
     * The first call may succeed or fail depending on whether the native library
     * is available, but subsequent calls must behave identically (idempotent).
     */
    @Test
    void load_calledMultipleTimes_isIdempotent() {
        // The native library may or may not be available in the test environment.
        // We capture the outcome of the first call and verify all subsequent calls
        // produce the same outcome.
        Throwable firstError = null;
        try {
            NativeLibLoader.load();
        } catch (UnsatisfiedLinkError e) {
            firstError = e;
        }

        // Second and third calls must behave identically
        for (int i = 0; i < 2; i++) {
            if (firstError == null) {
                // First call succeeded -- subsequent calls must also succeed (no-op)
                assertDoesNotThrow(NativeLibLoader::load,
                    "Repeated load() call #" + (i + 2) + " should not throw after successful first load");
            } else {
                // First call failed -- subsequent calls should also fail with UnsatisfiedLinkError
                // (since the loaded flag was never set to true)
                assertThrows(UnsatisfiedLinkError.class, NativeLibLoader::load,
                    "Repeated load() call #" + (i + 2) + " should throw after failed first load");
            }
        }
    }
}
