package ru.mcashesha.metrics;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Loads the {@code simsimd_jni} native library at runtime, transparently handling
 * extraction from the JAR when the library is bundled as a classpath resource.
 *
 * <h3>Loading strategy</h3>
 * <ol>
 *   <li><b>JAR resource</b> -- looks for
 *       {@code /native/<os>-<arch>/<libname>} on the classpath
 *       (e.g. {@code /native/linux-amd64/libsimsimd_jni.so}),
 *       extracts it to a temporary directory, and loads via {@link System#load(String)}.</li>
 *   <li><b>Fallback</b> -- if the resource is not found (e.g. the JAR was built
 *       on a different platform), falls back to {@link System#loadLibrary(String)},
 *       which searches {@code java.library.path} and the OS library path.</li>
 * </ol>
 *
 * <p>The loader is thread-safe and idempotent: multiple calls to {@link #load()}
 * are harmless and only the first one performs actual work.</p>
 */
final class NativeLibLoader {

    private static final String LIB_NAME = "simsimd_jni";
    private static volatile boolean loaded = false;

    private NativeLibLoader() { }

    /**
     * Ensures the native library is loaded. Safe to call from multiple threads.
     *
     * @throws UnsatisfiedLinkError if the library cannot be loaded by any strategy
     */
    static void load() {
        if (loaded) return;
        synchronized (NativeLibLoader.class) {
            if (loaded) return;
            doLoad();
            loaded = true;
        }
    }

    private static void doLoad() {
        // Strategy 1: extract from JAR resource
        try {
            loadFromResource();
            return;
        } catch (Exception e) {
            // Strategy 2: standard library path
            try {
                System.loadLibrary(LIB_NAME);
                return;
            } catch (UnsatisfiedLinkError ule) {
                ule.addSuppressed(e);
                throw ule;
            }
        }
    }

    private static void loadFromResource() throws IOException {
        String os = detectOs();
        String arch = detectArch();
        String libFileName = System.mapLibraryName(LIB_NAME);
        String resourcePath = "/native/" + os + "-" + arch + "/" + libFileName;

        try (InputStream in = NativeLibLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new FileNotFoundException(
                        "Native library not found in JAR: " + resourcePath);
            }

            Path tempDir = Files.createTempDirectory("simsimd_jni");
            Path tempLib = tempDir.resolve(libFileName);
            Files.copy(in, tempLib, StandardCopyOption.REPLACE_EXISTING);

            // Best-effort cleanup on JVM shutdown
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();

            System.load(tempLib.toAbsolutePath().toString());
        }
    }

    private static String detectOs() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("linux")) return "linux";
        if (os.contains("mac") || os.contains("darwin")) return "macos";
        if (os.contains("win")) return "windows";
        throw new UnsupportedOperationException("Unsupported OS: " + os);
    }

    private static String detectArch() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.equals("amd64") || arch.equals("x86_64")) return "amd64";
        if (arch.equals("aarch64") || arch.equals("arm64")) return "aarch64";
        throw new UnsupportedOperationException("Unsupported architecture: " + arch);
    }
}
