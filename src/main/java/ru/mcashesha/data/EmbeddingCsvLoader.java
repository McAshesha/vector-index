package ru.mcashesha.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for loading embedding vectors from CSV files.
 *
 * <p>Expected CSV format (with header row):</p>
 * <pre>
 *   id, metadata, "[v0, v1, ..., v511]"
 * </pre>
 *
 * <p>Each row after the header must have at least three comma-separated fields:</p>
 * <ol>
 *   <li><b>id</b> -- an identifier (ignored during parsing; vectors are returned in file order).</li>
 *   <li><b>metadata</b> -- arbitrary metadata string (ignored during parsing).</li>
 *   <li><b>embedding</b> -- a bracket-delimited, comma-separated list of 512 float values,
 *       optionally enclosed in double quotes (e.g., {@code "[0.1, 0.2, ..., 0.9]"}).</li>
 * </ol>
 *
 * <p>The loader validates that each embedding has exactly {@value #EMBEDDING_DIMENSION} dimensions
 * and throws {@link IllegalArgumentException} if a mismatch is found.</p>
 *
 * <p><b>Note:</b> Error messages from validation are in Russian (as originally authored) and have
 * been intentionally preserved.</p>
 */
public final class EmbeddingCsvLoader {

    /** The expected number of dimensions for each embedding vector. */
    private static final int EMBEDDING_DIMENSION = 512;

    /**
     * Loads embedding vectors from the specified CSV file.
     *
     * <p>The first line of the file is treated as a header and is skipped. Blank lines are
     * silently skipped. Lines with fewer than three comma-separated fields are also skipped.
     * Each valid embedding is parsed and validated to have exactly {@value #EMBEDDING_DIMENSION}
     * dimensions.</p>
     *
     * @param csvPath the path to the CSV file to load
     * @return a 2D float array where each row is an embedding vector of length {@value #EMBEDDING_DIMENSION};
     *         the array length equals the number of valid data rows in the CSV
     * @throws IOException              if an I/O error occurs reading the file
     * @throws IllegalArgumentException if the file is empty, or any embedding has a dimension
     *                                  other than {@value #EMBEDDING_DIMENSION}
     */
    public static float[][] loadEmbeddings(Path csvPath) throws IOException {
        List<float[]> vectors = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;

            // Read and discard the header line
            line = reader.readLine();
            if (line == null)
                throw new IllegalArgumentException("CSV файл пустой: " + csvPath);

            // Process each subsequent data line
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;

                // Split into at most 3 fields: id, metadata, embedding.
                // Using limit=3 ensures the embedding field (which contains commas) is not split further.
                String[] parts = line.split(",", 3);
                if (parts.length < 3)
                    continue;

                // The third field contains the embedding vector as a bracket-delimited float array
                String embeddingField = parts[2].trim();
                float[] vec = parseEmbedding(embeddingField);

                if (vec.length != EMBEDDING_DIMENSION) {
                    throw new IllegalArgumentException(
                        "Ожидалась размерность " + EMBEDDING_DIMENSION +
                            ", но получено " + vec.length + " для строки: " + line
                    );
                }

                vectors.add(vec);
            }
        }

        // Convert the list to a 2D array for efficient indexed access
        float[][] result = new float[vectors.size()][EMBEDDING_DIMENSION];
        for (int i = 0; i < vectors.size(); i++)
            result[i] = vectors.get(i);

        return result;
    }

    /**
     * Parses a single embedding vector from its string representation.
     *
     * <p>The method handles the following formats by stripping outer characters:</p>
     * <ul>
     *   <li>Optional surrounding double quotes: {@code "..."}</li>
     *   <li>Optional surrounding square brackets: {@code [...]}</li>
     * </ul>
     *
     * <p>After stripping, the remaining string is split on commas and each token is parsed
     * as a float value.</p>
     *
     * @param embeddingField the raw string field from the CSV, e.g., {@code "[0.1, 0.2, ..., 0.9]"}
     * @return a float array containing the parsed embedding values
     * @throws IllegalArgumentException if the field is empty after stripping delimiters,
     *                                  or if the number of values does not equal {@value #EMBEDDING_DIMENSION}
     * @throws NumberFormatException    if any token cannot be parsed as a float
     */
    private static float[] parseEmbedding(String embeddingField) {
        String s = embeddingField.trim();

        // Strip optional surrounding double quotes (CSV quoting)
        if (s.length() >= 2 && s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"')
            s = s.substring(1, s.length() - 1).trim();

        // Strip optional surrounding square brackets
        if (!s.isEmpty() && s.charAt(0) == '[')
            s = s.substring(1);
        if (!s.isEmpty() && s.charAt(s.length() - 1) == ']')
            s = s.substring(0, s.length() - 1);

        s = s.trim();
        if (s.isEmpty())
            throw new IllegalArgumentException("Пустой embedding: \"" + embeddingField + "\"");

        // Split the cleaned string by commas to get individual float tokens
        String[] tokens = s.split(",");
        if (tokens.length != EMBEDDING_DIMENSION) {
            throw new IllegalArgumentException(
                "Ожидалось " + EMBEDDING_DIMENSION + " значений, но получено " +
                    tokens.length + " в поле: \"" + embeddingField + "\""
            );
        }

        // Parse each token as a float
        float[] result = new float[EMBEDDING_DIMENSION];
        for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
            String t = tokens[i].trim();
            result[i] = Float.parseFloat(t);
        }

        return result;
    }

}
