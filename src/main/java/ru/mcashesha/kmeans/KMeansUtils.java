package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

final class KMeansUtils {
    private KMeansUtils() {}

    private static final int PARALLEL_THRESHOLD = 1000;

    static int validateAndGetDimension(float[][] data) {
        if (data[0] == null)
            throw new IllegalArgumentException("points must be non-null");

        int dimension = data[0].length;
        if (dimension == 0)
            throw new IllegalArgumentException("points must have positive dimension");

        for (int i = 1; i < data.length; i++) {
            float[] point = data[i];
            if (point == null || point.length != dimension)
                throw new IllegalArgumentException("all points must be non-null and have the same dimension");
        }

        return dimension;
    }

    static float[][] initializeCentroidsKMeansPlusPlus(
        float[][] data,
        int sampleCnt,
        int dimension,
        int clusterCnt,
        Metric.DistanceFunction distFn,
        Metric.Type metricType,
        Random random) {

        float[][] centroids = new float[clusterCnt][dimension];

        int firstIdx = random.nextInt(sampleCnt);
        System.arraycopy(data[firstIdx], 0, centroids[0], 0, dimension);

        float[] minDistances = new float[sampleCnt];

        // Parallelize initial distance computation
        float[] firstCentroid = centroids[0];
        if (sampleCnt >= PARALLEL_THRESHOLD) {
            IntStream.range(0, sampleCnt).parallel().forEach(i ->
                minDistances[i] = distFn.compute(data[i], firstCentroid));
        } else {
            for (int i = 0; i < sampleCnt; i++)
                minDistances[i] = distFn.compute(data[i], firstCentroid);
        }

        for (int c = 1; c < clusterCnt; c++) {
            // Sum weights - parallel reduction for large datasets
            float totalWeight;
            if (sampleCnt >= PARALLEL_THRESHOLD) {
                totalWeight = (float) IntStream.range(0, sampleCnt).parallel()
                    .mapToDouble(i -> Math.max(0f, minDistances[i]))
                    .sum();
            } else {
                totalWeight = 0f;
                for (int i = 0; i < sampleCnt; i++)
                    totalWeight += Math.max(0f, minDistances[i]);
            }

            int chosenIdx;

            if (totalWeight == 0f)
                chosenIdx = random.nextInt(sampleCnt);
            else {
                // Sequential - inherently not parallelizable due to cumulative probability
                float threshold = random.nextFloat() * totalWeight;
                float cumulative = 0f;
                chosenIdx = sampleCnt - 1;

                for (int i = 0; i < sampleCnt; i++) {
                    cumulative += Math.max(0f, minDistances[i]);
                    if (cumulative >= threshold) {
                        chosenIdx = i;
                        break;
                    }
                }
            }

            System.arraycopy(data[chosenIdx], 0, centroids[c], 0, dimension);

            // Parallelize distance update
            float[] newCentroid = centroids[c];
            if (sampleCnt >= PARALLEL_THRESHOLD) {
                IntStream.range(0, sampleCnt).parallel().forEach(i -> {
                    float distance = distFn.compute(data[i], newCentroid);
                    if (distance < minDistances[i])
                        minDistances[i] = distance;
                });
            } else {
                for (int i = 0; i < sampleCnt; i++) {
                    float distance = distFn.compute(data[i], newCentroid);
                    if (distance < minDistances[i])
                        minDistances[i] = distance;
                }
            }
        }

        if (metricType == Metric.Type.COSINE_DISTANCE)
            normalizeCentroids(centroids);

        return centroids;
    }

    static float assignPointsToClusters(
        float[][] data,
        float[][] centroids,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        if (clusterSizes != null)
            Arrays.fill(clusterSizes, 0);

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            return assignPointsToClustersParallel(data, centroids, clusterCnt, labels, pointErrors, clusterSizes, distFn);
        }

        // Sequential path for small datasets
        float loss = 0f;

        for (int i = 0; i < sampleCnt; i++) {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;

            if (clusterSizes != null)
                clusterSizes[nearestClusterIdx]++;

            loss += nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        }

        return loss;
    }

    private static float assignPointsToClustersParallel(
        float[][] data,
        float[][] centroids,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        // Parallel assignment - each point is independent
        float[] distances = new float[sampleCnt];

        IntStream.range(0, sampleCnt).parallel().forEach(i -> {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;
            distances[i] = nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        });

        // Compute cluster sizes sequentially (avoid atomic contention)
        if (clusterSizes != null) {
            for (int i = 0; i < sampleCnt; i++)
                clusterSizes[labels[i]]++;
        }

        // Parallel reduction for loss using IntStream to sum distances
        return (float) IntStream.range(0, sampleCnt).parallel()
            .mapToDouble(i -> distances[i])
            .sum();
    }

    static void recomputeCentroids(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension,
        Metric.Type metricType) {

        int sampleCnt = data.length;

        for (int c = 0; c < clusterCnt; c++) {
            Arrays.fill(newCentroids[c], 0);
            clusterSizes[c] = 0;
        }

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            recomputeCentroidsParallel(data, labels, newCentroids, clusterSizes, clusterCnt, dimension);
        } else {
            // Sequential path
            for (int i = 0; i < sampleCnt; i++) {
                int clusterIdx = labels[i];
                clusterSizes[clusterIdx]++;

                float[] centroidSum = newCentroids[clusterIdx];
                float[] point = data[i];

                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += point[d];
            }
        }

        // Final averaging
        for (int c = 0; c < clusterCnt; c++) {
            int size = clusterSizes[c];
            if (size > 0) {
                float invSize = 1.0f / size;
                float[] centroid = newCentroids[c];
                for (int d = 0; d < dimension; d++)
                    centroid[d] *= invSize;
            }
        }

        if (metricType == Metric.Type.COSINE_DISTANCE)
            normalizeCentroids(newCentroids);
    }

    private static void recomputeCentroidsParallel(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension) {

        int sampleCnt = data.length;
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunkSize = (sampleCnt + numThreads - 1) / numThreads;

        // Thread-local accumulators: [threadIdx][clusterIdx][dimension]
        float[][][] localSums = new float[numThreads][clusterCnt][dimension];
        int[][] localCounts = new int[numThreads][clusterCnt];

        // Parallel accumulation into thread-local buffers
        IntStream.range(0, numThreads).parallel().forEach(threadIdx -> {
            int start = threadIdx * chunkSize;
            int end = Math.min(start + chunkSize, sampleCnt);

            float[][] mySums = localSums[threadIdx];
            int[] myCounts = localCounts[threadIdx];

            for (int i = start; i < end; i++) {
                int clusterIdx = labels[i];
                myCounts[clusterIdx]++;

                float[] centroidSum = mySums[clusterIdx];
                float[] point = data[i];

                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += point[d];
            }
        });

        // Merge thread-local results (can be parallelized per cluster)
        IntStream.range(0, clusterCnt).parallel().forEach(c -> {
            float[] centroidSum = newCentroids[c];
            int totalCount = 0;

            for (int t = 0; t < numThreads; t++) {
                totalCount += localCounts[t][c];
                float[] threadSum = localSums[t][c];
                for (int d = 0; d < dimension; d++)
                    centroidSum[d] += threadSum[d];
            }

            clusterSizes[c] = totalCount;
        });
    }

    static void handleEmptyClusters(
        float[][] data,
        float[][] newCentroids,
        int[] clusterSizes,
        int[] labels,
        float[] pointErrors,
        boolean[] taken,
        int clusterCnt,
        int dimension,
        Metric.Type metricType,
        Random random) {

        int sampleCnt = data.length;
        Arrays.fill(taken, false);

        for (int emptyCluster = 0; emptyCluster < clusterCnt; emptyCluster++) {
            if (clusterSizes[emptyCluster] != 0)
                continue;

            int chosenIdx = choosePointFromLargestCluster(labels, clusterSizes, pointErrors, taken, clusterCnt);

            if (chosenIdx == -1)
                chosenIdx = chooseGlobalWorstPoint(labels, pointErrors, taken);

            if (chosenIdx == -1)
                chosenIdx = random.nextInt(sampleCnt);

            taken[chosenIdx] = true;

            int oldCluster = labels[chosenIdx];
            int oldSize = clusterSizes[oldCluster];

            labels[chosenIdx] = emptyCluster;
            clusterSizes[emptyCluster] = 1;

            if (oldCluster >= 0 && oldSize > 0) {
                clusterSizes[oldCluster] = oldSize - 1;

                if (clusterSizes[oldCluster] > 0) {
                    float[] donorCentroid = newCentroids[oldCluster];
                    float[] point = data[chosenIdx];
                    float invNewSize = 1.0f / clusterSizes[oldCluster];
                    for (int d = 0; d < dimension; d++)
                        donorCentroid[d] = (donorCentroid[d] * oldSize - point[d]) * invNewSize;
                    if (metricType == Metric.Type.COSINE_DISTANCE)
                        normalizeSingleCentroid(donorCentroid);
                }
            }

            System.arraycopy(data[chosenIdx], 0, newCentroids[emptyCluster], 0, dimension);
        }
    }

    static void normalizeCentroids(float[][] centroids) {
        if (centroids.length >= 32) {
            IntStream.range(0, centroids.length).parallel()
                .forEach(c -> normalizeSingleCentroid(centroids[c]));
        } else {
            for (float[] centroid : centroids)
                normalizeSingleCentroid(centroid);
        }
    }

    static void normalizeSingleCentroid(float[] centroid) {
        float norm = 0f;
        for (float v : centroid)
            norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            float invNorm = 1.0f / norm;
            for (int d = 0; d < centroid.length; d++)
                centroid[d] *= invNorm;
        }
    }

    private static int choosePointFromLargestCluster(
        int[] labels,
        int[] clusterSizes,
        float[] pointErrors,
        boolean[] taken,
        int clusterCnt) {

        int largestCluster = -1;
        int maxSize = 0;
        for (int c = 0; c < clusterCnt; c++) {
            if (clusterSizes[c] > maxSize) {
                maxSize = clusterSizes[c];
                largestCluster = c;
            }
        }
        if (largestCluster < 0 || maxSize <= 1)
            return -1;

        int n = labels.length;
        final int targetCluster = largestCluster;

        if (n >= PARALLEL_THRESHOLD) {
            // Parallel argmax
            return IntStream.range(0, n).parallel()
                .filter(i -> !taken[i] && labels[i] == targetCluster)
                .boxed()
                .max((a, b) -> {
                    float errA = pointErrors != null ? pointErrors[a] : 1.0f;
                    float errB = pointErrors != null ? pointErrors[b] : 1.0f;
                    return Float.compare(errA, errB);
                })
                .orElse(-1);
        }

        // Sequential path
        int bestIdx = -1;
        float bestError = -1f;
        for (int i = 0; i < n; i++) {
            if (taken[i])
                continue;
            if (labels[i] == largestCluster) {
                float err = pointErrors != null ? pointErrors[i] : 1.0f;
                if (err > bestError) {
                    bestError = err;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    private static int chooseGlobalWorstPoint(
        int[] labels,
        float[] pointErrors,
        boolean[] taken) {

        int n = labels.length;

        if (n >= PARALLEL_THRESHOLD) {
            // Parallel argmax
            return IntStream.range(0, n).parallel()
                .filter(i -> !taken[i] && labels[i] >= 0)
                .boxed()
                .max((a, b) -> {
                    float errA = pointErrors != null ? pointErrors[a] : 1.0f;
                    float errB = pointErrors != null ? pointErrors[b] : 1.0f;
                    return Float.compare(errA, errB);
                })
                .orElse(-1);
        }

        // Sequential path
        int bestIdx = -1;
        float bestError = -1f;

        for (int i = 0; i < n; i++) {
            if (taken[i])
                continue;
            if (labels[i] < 0)
                continue;
            float err = pointErrors != null ? pointErrors[i] : 1.0f;
            if (err > bestError) {
                bestError = err;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * Computes pairwise distance matrix between all points.
     * Useful for analysis and debugging.
     * @return Symmetric distance matrix where result[i][j] = distance(points[i], points[j])
     */
    static float[][] computeDistanceMatrix(float[][] points, Metric.DistanceFunction distFn) {
        int n = points.length;
        float[][] distances = new float[n][n];

        if (n >= 100) {
            // Parallel computation - only upper triangle, then mirror
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int j = i + 1; j < n; j++) {
                    float d = distFn.compute(points[i], points[j]);
                    distances[i][j] = d;
                    distances[j][i] = d;
                }
            });
        } else {
            // Sequential for small matrices
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    float d = distFn.compute(points[i], points[j]);
                    distances[i][j] = d;
                    distances[j][i] = d;
                }
            }
        }

        return distances;
    }

    /**
     * Precomputes centroid-to-centroid distances for triangle inequality pruning.
     * @return Distance matrix between all centroids
     */
    static float[][] precomputeCentroidDistances(float[][] centroids, Metric.DistanceFunction distFn) {
        return computeDistanceMatrix(centroids, distFn);
    }

    /**
     * Assignment with triangle inequality pruning.
     * Uses precomputed centroid distances to skip impossible clusters.
     */
    static float assignPointsToClustersWithPruning(
        float[][] data,
        float[][] centroids,
        float[][] centroidDistances,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;

        if (clusterSizes != null)
            Arrays.fill(clusterSizes, 0);

        if (sampleCnt >= PARALLEL_THRESHOLD) {
            return assignPointsToClustersWithPruningParallel(
                data, centroids, centroidDistances, clusterCnt, labels, pointErrors, clusterSizes, distFn);
        }

        // Sequential path with pruning
        float loss = 0f;

        for (int i = 0; i < sampleCnt; i++) {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                // Triangle inequality pruning:
                // If d(centroid[nearest], centroid[c]) > 2 * d(point, centroid[nearest])
                // then centroid[c] cannot be closer to point than centroid[nearest]
                if (centroidDistances[nearestClusterIdx][c] > 2 * nearestDistance)
                    continue;

                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;

            if (clusterSizes != null)
                clusterSizes[nearestClusterIdx]++;

            loss += nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        }

        return loss;
    }

    private static float assignPointsToClustersWithPruningParallel(
        float[][] data,
        float[][] centroids,
        float[][] centroidDistances,
        int clusterCnt,
        int[] labels,
        float[] pointErrors,
        int[] clusterSizes,
        Metric.DistanceFunction distFn) {

        int sampleCnt = data.length;
        float[] distances = new float[sampleCnt];

        IntStream.range(0, sampleCnt).parallel().forEach(i -> {
            float[] point = data[i];

            int nearestClusterIdx = 0;
            float nearestDistance = distFn.compute(point, centroids[0]);

            for (int c = 1; c < clusterCnt; c++) {
                // Triangle inequality pruning
                if (centroidDistances[nearestClusterIdx][c] > 2 * nearestDistance)
                    continue;

                float distance = distFn.compute(point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;
            distances[i] = nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        });

        // Compute cluster sizes sequentially
        if (clusterSizes != null) {
            for (int i = 0; i < sampleCnt; i++)
                clusterSizes[labels[i]]++;
        }

        // Parallel reduction for loss
        return (float) IntStream.range(0, sampleCnt).parallel()
            .mapToDouble(i -> distances[i])
            .sum();
    }
}
