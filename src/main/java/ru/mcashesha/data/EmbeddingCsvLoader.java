package ru.mcashesha.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class EmbeddingCsvLoader {

    private static final int EMBEDDING_DIMENSION = 512;

    public static float[][] loadEmbeddings(Path csvPath) throws IOException {
        List<float[]> vectors = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;

            line = reader.readLine();
            if (line == null)
                throw new IllegalArgumentException("CSV файл пустой: " + csvPath);

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;

                String[] parts = line.split(",", 3);
                if (parts.length < 3)
                    continue;

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

        float[][] result = new float[vectors.size()][EMBEDDING_DIMENSION];
        for (int i = 0; i < vectors.size(); i++)
            result[i] = vectors.get(i);

        return result;
    }

    private static float[] parseEmbedding(String embeddingField) {
        String s = embeddingField.trim();

        if (s.length() >= 2 && s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"')
            s = s.substring(1, s.length() - 1).trim();

        if (!s.isEmpty() && s.charAt(0) == '[')
            s = s.substring(1);
        if (!s.isEmpty() && s.charAt(s.length() - 1) == ']')
            s = s.substring(0, s.length() - 1);

        s = s.trim();
        if (s.isEmpty())
            throw new IllegalArgumentException("Пустой embedding: \"" + embeddingField + "\"");

        String[] tokens = s.split(",");
        if (tokens.length != EMBEDDING_DIMENSION) {
            throw new IllegalArgumentException(
                "Ожидалось " + EMBEDDING_DIMENSION + " значений, но получено " +
                    tokens.length + " в поле: \"" + embeddingField + "\""
            );
        }

        float[] result = new float[EMBEDDING_DIMENSION];
        for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
            String t = tokens[i].trim();
            result[i] = Float.parseFloat(t);
        }

        return result;
    }

}
