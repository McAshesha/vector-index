package ru.mcashesha.kmeans;

import java.util.Random;
import ru.mcashesha.metrics.Metric;

public interface KMeans<R extends KMeans.ClusteringResult> {

    static Builder newBuilder(Type type,
        Metric.Type metricType,
        Metric.Engine metricEngine) {
        return new Builder(type, metricType, metricEngine);
    }

    R fit(float[][] data);

    int[] predict(float[][] data, R model);

    Metric.Type getMetricType();

    Metric.Engine getMetricEngine();

    enum Type {
        HIERARCHICAL,
        LLOYD,
        MINI_BATCH
    }

    interface ClusteringResult {
        float[][] getCentroids();

        int[] getClusterAssignments();

        float getLoss();

        int[] getClusterSizes();
    }

    final class Builder {
        private final Type type;
        private final Metric.Type metricType;
        private final Metric.Engine metricEngine;

        private int clusterCount = 16;
        private int batchSize = 1024;
        private int maxIterations = 300;
        private int maxNoImprovementIterations = 50;
        private float tolerance = 1e-4f;

        private int branchFactor = 2;
        private int maxDepth = 6;
        private int minClusterSize = 4;
        private int maxIterationsPerLevel = 50;
        private boolean minClusterSizeOverridden;

        private Random random = new Random();

        private Builder(Type type,
            Metric.Type metricType,
            Metric.Engine metricEngine) {
            if (type == null)
                throw new IllegalArgumentException("type must be non-null");
            if (metricType == null || metricEngine == null)
                throw new IllegalArgumentException("metricType and metricEngine must be non-null");

            this.type = type;
            this.metricType = metricType;
            this.metricEngine = metricEngine;
        }

        public Builder withClusterCount(int clusterCount) {
            this.clusterCount = clusterCount;
            return this;
        }

        public Builder withBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder withMaxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder withTolerance(float tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        public Builder withMaxNoImprovementIterations(int maxNoImprovementIterations) {
            this.maxNoImprovementIterations = maxNoImprovementIterations;
            return this;
        }

        public Builder withBranchFactor(int branchFactor) {
            this.branchFactor = branchFactor;
            if (!minClusterSizeOverridden)
                this.minClusterSize = Math.max(2 * branchFactor, 2);
            return this;
        }

        public Builder withMaxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder withMinClusterSize(int minClusterSize) {
            this.minClusterSize = minClusterSize;
            this.minClusterSizeOverridden = true;
            return this;
        }

        public Builder withMaxIterationsPerLevel(int maxIterationsPerLevel) {
            this.maxIterationsPerLevel = maxIterationsPerLevel;
            return this;
        }

        public Builder withRandom(Random random) {
            if (random == null)
                throw new IllegalArgumentException("random must be non-null");
            this.random = random;
            return this;
        }

        public KMeans<? extends ClusteringResult> build() {
            switch (type) {
                case LLOYD: {
                    return new LloydKMeans(
                        clusterCount,
                        metricType,
                        metricEngine,
                        maxIterations,
                        tolerance,
                        random
                    );
                }
                case MINI_BATCH: {
                    return new MiniBatchKMeans(
                        clusterCount,
                        batchSize,
                        metricType,
                        metricEngine,
                        maxIterations,
                        tolerance,
                        maxNoImprovementIterations,
                        random
                    );
                }
                case HIERARCHICAL: {
                    return new HierarchicalKMeans(
                        branchFactor,
                        maxDepth,
                        minClusterSize,
                        maxIterationsPerLevel,
                        tolerance,
                        random,
                        metricType,
                        metricEngine
                    );
                }
                default: {
                    throw new IllegalStateException("Unsupported KMeans type: " + type);
                }
            }
        }
    }
}
