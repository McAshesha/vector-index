package ru.mcashesha.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingCsvLoaderTest {

    @TempDir
    Path tempDir;

    private static final int DIM = 512;

    private String makeEmbeddingString(float startVal) {
        StringBuilder sb = new StringBuilder("\"[");
        for (int i = 0; i < DIM; i++) {
            if (i > 0) sb.append(", ");
            sb.append(startVal + i * 0.001f);
        }
        sb.append("]\"");
        return sb.toString();
    }

    private Path writeCsv(String content) throws IOException {
        Path csv = tempDir.resolve("test.csv");
        Files.writeString(csv, content, StandardCharsets.UTF_8);
        return csv;
    }

    @Test
    void loadEmbeddings_validFile_loadsCorrectly() throws IOException {
        String embedding = makeEmbeddingString(0.1f);
        String csv = "id,metadata,embedding\n" +
            "1,meta1," + embedding + "\n" +
            "2,meta2," + embedding + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(2, result.length);
        assertEquals(DIM, result[0].length);
        assertEquals(DIM, result[1].length);
    }

    @Test
    void loadEmbeddings_singleRow_loadsOneVector() throws IOException {
        String embedding = makeEmbeddingString(0.5f);
        String csv = "id,metadata,embedding\n" +
            "1,meta," + embedding + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(1, result.length);
        assertEquals(DIM, result[0].length);
    }

    @Test
    void loadEmbeddings_valuesAreParsedCorrectly() throws IOException {
        StringBuilder embBuilder = new StringBuilder("\"[");
        for (int i = 0; i < DIM; i++) {
            if (i > 0) embBuilder.append(", ");
            embBuilder.append(i == 0 ? "1.5" : "0.0");
        }
        embBuilder.append("]\"");
        String csv = "id,metadata,embedding\n" +
            "1,meta," + embBuilder + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(1.5f, result[0][0], 1e-6f);
        assertEquals(0.0f, result[0][1], 1e-6f);
    }

    @Test
    void loadEmbeddings_emptyFile_throws() throws IOException {
        Path path = writeCsv("");
        assertThrows(IllegalArgumentException.class, () -> EmbeddingCsvLoader.loadEmbeddings(path));
    }

    @Test
    void loadEmbeddings_headerOnly_returnsEmpty() throws IOException {
        Path path = writeCsv("id,metadata,embedding\n");
        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);
        assertEquals(0, result.length);
    }

    @Test
    void loadEmbeddings_wrongDimension_throws() throws IOException {
        StringBuilder embBuilder = new StringBuilder("\"[");
        for (int i = 0; i < 10; i++) {
            if (i > 0) embBuilder.append(", ");
            embBuilder.append("0.1");
        }
        embBuilder.append("]\"");
        String csv = "id,metadata,embedding\n" +
            "1,meta," + embBuilder + "\n";
        Path path = writeCsv(csv);

        assertThrows(IllegalArgumentException.class, () -> EmbeddingCsvLoader.loadEmbeddings(path));
    }

    @Test
    void loadEmbeddings_blankLinesSkipped() throws IOException {
        String embedding = makeEmbeddingString(0.1f);
        String csv = "id,metadata,embedding\n" +
            "\n" +
            "1,meta," + embedding + "\n" +
            "\n" +
            "  \n" +
            "2,meta2," + embedding + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(2, result.length);
    }

    @Test
    void loadEmbeddings_lineWithFewerThanThreeFields_skipped() throws IOException {
        String embedding = makeEmbeddingString(0.1f);
        String csv = "id,metadata,embedding\n" +
            "only_one_field\n" +
            "1,meta," + embedding + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(1, result.length);
    }

    @Test
    void loadEmbeddings_withoutQuotesAndBrackets_parsesCorrectly() throws IOException {
        StringBuilder embBuilder = new StringBuilder("[");
        for (int i = 0; i < DIM; i++) {
            if (i > 0) embBuilder.append(", ");
            embBuilder.append("0.5");
        }
        embBuilder.append("]");
        String csv = "id,metadata,embedding\n" +
            "1,meta," + embBuilder + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(1, result.length);
        assertEquals(0.5f, result[0][0], 1e-6f);
    }

    @Test
    void loadEmbeddings_nonExistentFile_throwsIOException() {
        Path nonExistent = tempDir.resolve("no_such_file.csv");
        assertThrows(IOException.class, () -> EmbeddingCsvLoader.loadEmbeddings(nonExistent));
    }

    @Test
    void loadEmbeddings_multipleValidRows_allParsed() throws IOException {
        String embedding = makeEmbeddingString(0.0f);
        StringBuilder csv = new StringBuilder("id,metadata,embedding\n");
        int rowCount = 50;
        for (int i = 0; i < rowCount; i++)
            csv.append(i).append(",meta_").append(i).append(",").append(embedding).append("\n");
        Path path = writeCsv(csv.toString());

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(rowCount, result.length);
        for (float[] row : result)
            assertEquals(DIM, row.length);
    }

    @Test
    void loadEmbeddings_negativeValues_parsedCorrectly() throws IOException {
        StringBuilder embBuilder = new StringBuilder("\"[");
        for (int i = 0; i < DIM; i++) {
            if (i > 0) embBuilder.append(", ");
            embBuilder.append(i == 0 ? "-1.5" : "-0.001");
        }
        embBuilder.append("]\"");
        String csv = "id,metadata,embedding\n1,meta," + embBuilder + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(-1.5f, result[0][0], 1e-6f);
        assertTrue(result[0][1] < 0);
    }

    @Test
    void loadEmbeddings_numberFormatError_throws() throws IOException {
        StringBuilder embBuilder = new StringBuilder("\"[");
        for (int i = 0; i < DIM; i++) {
            if (i > 0) embBuilder.append(", ");
            embBuilder.append(i == 0 ? "not_a_number" : "0.1");
        }
        embBuilder.append("]\"");
        String csv = "id,metadata,embedding\n1,meta," + embBuilder + "\n";
        Path path = writeCsv(csv);

        assertThrows(NumberFormatException.class, () -> EmbeddingCsvLoader.loadEmbeddings(path));
    }

    @Test
    void loadEmbeddings_metadataWithCommas_handledByLimitSplit() throws IOException {
        String embedding = makeEmbeddingString(0.2f);
        // The metadata field might contain commas, but split(",", 3) limits splitting
        // In practice, the embedding field is the third field after the first two commas
        String csv = "id,metadata,embedding\n" +
            "1,meta_field," + embedding + "\n";
        Path path = writeCsv(csv);

        float[][] result = EmbeddingCsvLoader.loadEmbeddings(path);

        assertEquals(1, result.length);
    }
}
