package ru.mcashesha;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import ru.mcashesha.data.EmbeddingCsvLoader;
import ru.mcashesha.ivf.IVFIndex;
import ru.mcashesha.ivf.IVFIndexFlat;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

public class VectorIndexBenchmark {
    private static final int SEED = 42;
    private static final int TOP_K = 100;
    private static final int NUM_QUERIES = 50;
    private static final int WARMUP_QUERIES = 5;

    private static final Metric.Type METRIC_TYPE = Metric.Type.L2SQ_DISTANCE;
    private static final Metric.Engine METRIC_ENGINE = Metric.Engine.VECTOR_API;

    public static void main(String[] args) throws IOException {
        System.out.println("Loading embeddings...");
        float[][] data = EmbeddingCsvLoader.loadEmbeddings(Paths.get("embeddings.csv"));
        System.out.printf("Loaded %d vectors of dimension %d%n%n", data.length, data[0].length);

        Random random = new Random(SEED);
        float[][] queries = generateQueries(data, NUM_QUERIES + WARMUP_QUERIES, random);

        System.out.println("Computing brute-force ground truth...");
        int[][] groundTruth = computeGroundTruth(data, queries, TOP_K);
        System.out.println("Ground truth computed.\n");

        List<BenchmarkResult> results = new ArrayList<>();

        // Test Lloyd with best k values and multiple nProbes per build
        System.out.println("=".repeat(80));
        System.out.println("LLOYD KMEANS");
        System.out.println("=".repeat(80));
        int[] lloydClusters = {64, 128};
        testLloyd(data, queries, groundTruth, lloydClusters, results, new Random(SEED));

        // Test MiniBatch - faster build times expected
        System.out.println("\n" + "=".repeat(80));
        System.out.println("MINIBATCH KMEANS");
        System.out.println("=".repeat(80));
        int[] mbClusters = {64, 128, 256};
        int[] batchSizes = {512, 1024};
        testMiniBatch(data, queries, groundTruth, mbClusters, batchSizes, results, new Random(SEED));

        // Test Hierarchical
        System.out.println("\n" + "=".repeat(80));
        System.out.println("HIERARCHICAL KMEANS");
        System.out.println("=".repeat(80));
        testHierarchical(data, queries, groundTruth, results, new Random(SEED));

        // Print summary
        printSummary(results);
    }

    static void testLloyd(float[][] data, float[][] queries, int[][] groundTruth,
                          int[] clusters, List<BenchmarkResult> results, Random random) {
        for (int k : clusters) {
            System.out.printf("\nBuilding Lloyd k=%d...%n", k);
            KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
                    KMeans.Type.LLOYD, METRIC_TYPE, METRIC_ENGINE)
                .withClusterCount(k)
                .withMaxIterations(100)
                .withTolerance(1e-4f)
                .withRandom(random)
                .build();

            long buildStart = System.currentTimeMillis();
            IVFIndex idx = new IVFIndexFlat(kmeans);
            idx.build(data);
            long buildTime = System.currentTimeMillis() - buildStart;
            System.out.printf("Build time: %d ms, clusters: %d%n", buildTime, idx.getCountClusters());

            // Test multiple nProbe values with same index
            int[] nProbes = {4, 8, 16, 32, Math.min(64, k)};
            for (int nProbe : nProbes) {
                if (nProbe > k) continue;
                BenchmarkResult r = benchmarkSearch(idx, "Lloyd", String.format("k=%d", k),
                    queries, groundTruth, nProbe, buildTime);
                results.add(r);
                printResult(r);
            }
        }
    }

    static void testMiniBatch(float[][] data, float[][] queries, int[][] groundTruth,
                              int[] clusters, int[] batchSizes, List<BenchmarkResult> results, Random random) {
        for (int k : clusters) {
            for (int batch : batchSizes) {
                System.out.printf("\nBuilding MiniBatch k=%d, batch=%d...%n", k, batch);
                KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
                        KMeans.Type.MINI_BATCH, METRIC_TYPE, METRIC_ENGINE)
                    .withClusterCount(k)
                    .withBatchSize(batch)
                    .withMaxIterations(300)
                    .withMaxNoImprovementIterations(30)
                    .withTolerance(1e-4f)
                    .withRandom(random)
                    .build();

                long buildStart = System.currentTimeMillis();
                IVFIndex idx = new IVFIndexFlat(kmeans);
                idx.build(data);
                long buildTime = System.currentTimeMillis() - buildStart;
                System.out.printf("Build time: %d ms, clusters: %d%n", buildTime, idx.getCountClusters());

                int[] nProbes = {4, 8, 16, 32, Math.min(64, k)};
                for (int nProbe : nProbes) {
                    if (nProbe > k) continue;
                    BenchmarkResult r = benchmarkSearch(idx, "MiniBatch", String.format("k=%d,b=%d", k, batch),
                        queries, groundTruth, nProbe, buildTime);
                    results.add(r);
                    printResult(r);
                }
            }
        }
    }

    static void testHierarchical(float[][] data, float[][] queries, int[][] groundTruth,
                                  List<BenchmarkResult> results, Random random) {
        // Test configurations that give different cluster counts
        int[][] configs = {
            {4, 4, 100},   // ~64 clusters
            {8, 3, 100},   // ~64 clusters
            {4, 5, 50},    // ~256 clusters
            {8, 4, 50},    // ~512 clusters
            {16, 3, 50},   // ~256 clusters
        };

        for (int[] cfg : configs) {
            int branch = cfg[0], depth = cfg[1], minSize = cfg[2];
            System.out.printf("\nBuilding Hierarchical branch=%d, depth=%d, minSize=%d...%n", branch, depth, minSize);

            KMeans<? extends KMeans.ClusteringResult> kmeans = KMeans.newBuilder(
                    KMeans.Type.HIERARCHICAL, METRIC_TYPE, METRIC_ENGINE)
                .withBranchFactor(branch)
                .withMaxDepth(depth)
                .withMinClusterSize(minSize)
                .withMaxIterationsPerLevel(30)
                .withTolerance(1e-4f)
                .withRandom(random)
                .build();

            long buildStart = System.currentTimeMillis();
            IVFIndex idx = new IVFIndexFlat(kmeans);
            idx.build(data);
            long buildTime = System.currentTimeMillis() - buildStart;
            int actualClusters = idx.getCountClusters();
            System.out.printf("Build time: %d ms, clusters: %d%n", buildTime, actualClusters);

            int[] nProbes = {4, 8, 16, 32, Math.min(64, actualClusters)};
            for (int nProbe : nProbes) {
                if (nProbe > actualClusters) continue;
                BenchmarkResult r = benchmarkSearch(idx, "Hierarchical",
                    String.format("bf=%d,d=%d,min=%d", branch, depth, minSize),
                    queries, groundTruth, nProbe, buildTime);
                results.add(r);
                printResult(r);
            }
        }
    }

    static BenchmarkResult benchmarkSearch(IVFIndex idx, String algorithm, String params,
                                            float[][] queries, int[][] groundTruth, int nProbe, long buildTime) {
        // Warmup
        for (int i = 0; i < WARMUP_QUERIES; i++) {
            idx.search(queries[i], TOP_K, nProbe);
        }

        // Benchmark
        long searchStart = System.nanoTime();
        double totalRecall = 0;
        for (int i = WARMUP_QUERIES; i < queries.length; i++) {
            List<IVFIndex.SearchResult> results = idx.search(queries[i], TOP_K, nProbe);
            totalRecall += computeRecall(results, groundTruth[i]);
        }
        long searchTimeNs = System.nanoTime() - searchStart;

        double avgRecall = totalRecall / NUM_QUERIES;
        double avgSearchTimeUs = (searchTimeNs / 1000.0) / NUM_QUERIES;

        return new BenchmarkResult(algorithm, params, nProbe, idx.getCountClusters(),
            buildTime, avgSearchTimeUs, avgRecall);
    }

    static double computeRecall(List<IVFIndex.SearchResult> results, int[] truth) {
        Set<Integer> truthSet = new HashSet<>();
        for (int id : truth) truthSet.add(id);

        int hits = 0;
        for (IVFIndex.SearchResult r : results) {
            if (truthSet.contains(r.id)) hits++;
        }
        return (double) hits / truth.length;
    }

    static int[][] computeGroundTruth(float[][] data, float[][] queries, int topK) {
        Metric.DistanceFunction distFn = METRIC_TYPE.resolveFunction(METRIC_ENGINE);
        int[][] truth = new int[queries.length][topK];

        for (int q = 0; q < queries.length; q++) {
            float[] query = queries[q];
            float[] distances = new float[data.length];
            int[] indices = new int[data.length];

            for (int i = 0; i < data.length; i++) {
                distances[i] = distFn.compute(query, data[i]);
                indices[i] = i;
            }

            // Partial sort
            for (int k = 0; k < topK; k++) {
                int minIdx = k;
                for (int i = k + 1; i < data.length; i++) {
                    if (distances[i] < distances[minIdx]) minIdx = i;
                }
                float tmpD = distances[k]; distances[k] = distances[minIdx]; distances[minIdx] = tmpD;
                int tmpI = indices[k]; indices[k] = indices[minIdx]; indices[minIdx] = tmpI;
                truth[q][k] = indices[k];
            }
        }
        return truth;
    }

    static float[][] generateQueries(float[][] data, int numQueries, Random random) {
        float[][] queries = new float[numQueries][];
        for (int i = 0; i < numQueries; i++) {
            queries[i] = data[random.nextInt(data.length)].clone();
        }
        return queries;
    }

    static void printResult(BenchmarkResult r) {
        System.out.printf("  nProbe=%-3d | Search: %8.1f μs | Recall@%d: %.4f%n",
            r.nProbe, r.avgSearchTimeUs, TOP_K, r.recall);
    }

    static void printSummary(List<BenchmarkResult> results) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("ИТОГОВАЯ ТАБЛИЦА - ЛУЧШИЕ КОНФИГУРАЦИИ");
        System.out.println("=".repeat(100));

        // Best per algorithm with recall >= 0.95
        System.out.println("\n### Лучшие по скорости (Recall >= 0.95):");
        System.out.println("-".repeat(100));
        System.out.printf("%-15s %-25s %-8s %-10s %-12s %-12s %-10s%n",
            "Алгоритм", "Параметры", "nProbe", "Кластеры", "Build (ms)", "Search (μs)", "Recall");
        System.out.println("-".repeat(100));

        Map<String, BenchmarkResult> bestFast = new LinkedHashMap<>();
        for (BenchmarkResult r : results) {
            if (r.recall >= 0.95) {
                BenchmarkResult cur = bestFast.get(r.algorithm);
                if (cur == null || r.avgSearchTimeUs < cur.avgSearchTimeUs) {
                    bestFast.put(r.algorithm, r);
                }
            }
        }
        for (BenchmarkResult r : bestFast.values()) {
            System.out.printf("%-15s %-25s %-8d %-10d %-12d %-12.1f %-10.4f%n",
                r.algorithm, r.params, r.nProbe, r.clusters, r.buildTimeMs, r.avgSearchTimeUs, r.recall);
        }

        // Best per algorithm with recall >= 0.99
        System.out.println("\n### Лучшие по скорости (Recall >= 0.99):");
        System.out.println("-".repeat(100));
        System.out.printf("%-15s %-25s %-8s %-10s %-12s %-12s %-10s%n",
            "Алгоритм", "Параметры", "nProbe", "Кластеры", "Build (ms)", "Search (μs)", "Recall");
        System.out.println("-".repeat(100));

        Map<String, BenchmarkResult> bestAccurate = new LinkedHashMap<>();
        for (BenchmarkResult r : results) {
            if (r.recall >= 0.99) {
                BenchmarkResult cur = bestAccurate.get(r.algorithm);
                if (cur == null || r.avgSearchTimeUs < cur.avgSearchTimeUs) {
                    bestAccurate.put(r.algorithm, r);
                }
            }
        }
        for (BenchmarkResult r : bestAccurate.values()) {
            System.out.printf("%-15s %-25s %-8d %-10d %-12d %-12.1f %-10.4f%n",
                r.algorithm, r.params, r.nProbe, r.clusters, r.buildTimeMs, r.avgSearchTimeUs, r.recall);
        }

        // Best build time with recall >= 0.95
        System.out.println("\n### Лучшие по времени построения (Recall >= 0.95):");
        System.out.println("-".repeat(100));
        System.out.printf("%-15s %-25s %-8s %-10s %-12s %-12s %-10s%n",
            "Алгоритм", "Параметры", "nProbe", "Кластеры", "Build (ms)", "Search (μs)", "Recall");
        System.out.println("-".repeat(100));

        Map<String, BenchmarkResult> bestBuild = new LinkedHashMap<>();
        for (BenchmarkResult r : results) {
            if (r.recall >= 0.95) {
                BenchmarkResult cur = bestBuild.get(r.algorithm);
                if (cur == null || r.buildTimeMs < cur.buildTimeMs) {
                    bestBuild.put(r.algorithm, r);
                }
            }
        }
        for (BenchmarkResult r : bestBuild.values()) {
            System.out.printf("%-15s %-25s %-8d %-10d %-12d %-12.1f %-10.4f%n",
                r.algorithm, r.params, r.nProbe, r.clusters, r.buildTimeMs, r.avgSearchTimeUs, r.recall);
        }

        // Recommendations
        System.out.println("\n" + "=".repeat(100));
        System.out.println("РЕКОМЕНДАЦИИ");
        System.out.println("=".repeat(100));

        BenchmarkResult overallBest = null;
        for (BenchmarkResult r : results) {
            if (r.recall >= 0.95) {
                if (overallBest == null || r.avgSearchTimeUs < overallBest.avgSearchTimeUs) {
                    overallBest = r;
                }
            }
        }

        if (overallBest != null) {
            System.out.println("\n🏆 ЛУЧШИЙ БАЛАНС (Recall >= 0.95, минимальное время поиска):");
            System.out.printf("   Алгоритм: %s%n", overallBest.algorithm);
            System.out.printf("   Параметры: %s%n", overallBest.params);
            System.out.printf("   nProbe: %d%n", overallBest.nProbe);
            System.out.printf("   Кластеров: %d%n", overallBest.clusters);
            System.out.printf("   Время построения: %d ms%n", overallBest.buildTimeMs);
            System.out.printf("   Время поиска: %.1f μs%n", overallBest.avgSearchTimeUs);
            System.out.printf("   Recall@%d: %.4f%n", TOP_K, overallBest.recall);
        }

        // Full results table
        System.out.println("\n" + "=".repeat(100));
        System.out.println("ВСЕ РЕЗУЛЬТАТЫ (отсортировано по Recall)");
        System.out.println("=".repeat(100));
        System.out.printf("%-15s %-25s %-8s %-10s %-12s %-12s %-10s%n",
            "Алгоритм", "Параметры", "nProbe", "Кластеры", "Build (ms)", "Search (μs)", "Recall");
        System.out.println("-".repeat(100));

        results.stream()
            .sorted((a, b) -> Double.compare(b.recall, a.recall))
            .forEach(r -> System.out.printf("%-15s %-25s %-8d %-10d %-12d %-12.1f %-10.4f%n",
                r.algorithm, r.params, r.nProbe, r.clusters, r.buildTimeMs, r.avgSearchTimeUs, r.recall));
    }

    static class BenchmarkResult {
        final String algorithm;
        final String params;
        final int nProbe;
        final int clusters;
        final long buildTimeMs;
        final double avgSearchTimeUs;
        final double recall;

        BenchmarkResult(String algorithm, String params, int nProbe, int clusters,
                        long buildTimeMs, double avgSearchTimeUs, double recall) {
            this.algorithm = algorithm;
            this.params = params;
            this.nProbe = nProbe;
            this.clusters = clusters;
            this.buildTimeMs = buildTimeMs;
            this.avgSearchTimeUs = avgSearchTimeUs;
            this.recall = recall;
        }
    }
}
