package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import ru.mcashesha.metrics.Metric;

final class KMeansUtils {
    private KMeansUtils() {}

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

        float[] firstCentroid = centroids[0];
        for (int i = 0; i < sampleCnt; i++)
            minDistances[i] = distFn.compute(data[i], firstCentroid);

        for (int c = 1; c < clusterCnt; c++) {
            float totalWeight = 0f;

            for (int i = 0; i < sampleCnt; i++)
                totalWeight += Math.max(0f, minDistances[i]);

            int chosenIdx;

            if (totalWeight == 0f)
                chosenIdx = random.nextInt(sampleCnt);
            else {
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

            float[] newCentroid = centroids[c];
            for (int i = 0; i < sampleCnt; i++) {
                float distance = distFn.compute(data[i], newCentroid);
                if (distance < minDistances[i])
                    minDistances[i] = distance;
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
        float loss = 0f;

        if (clusterSizes != null)
            Arrays.fill(clusterSizes, 0);

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

    static void recomputeCentroids(
        float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes,
        int clusterCnt,
        int dimension,
        Metric.Type metricType) {

        for (int c = 0; c < clusterCnt; c++) {
            Arrays.fill(newCentroids[c], 0);
            clusterSizes[c] = 0;
        }

        for (int i = 0; i < data.length; i++) {
            int clusterIdx = labels[i];
            clusterSizes[clusterIdx]++;

            float[] centroidSum = newCentroids[clusterIdx];
            float[] point = data[i];

            for (int d = 0; d < dimension; d++)
                centroidSum[d] += point[d];
        }

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
        for (float[] centroid : centroids)
            normalizeSingleCentroid(centroid);
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

        int bestIdx = -1;
        float bestError = -1f;
        for (int i = 0; i < labels.length; i++) {
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

        int bestIdx = -1;
        float bestError = -1f;

        for (int i = 0; i < labels.length; i++) {
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
}
