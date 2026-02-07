package ru.mcashesha.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import ru.mcashesha.metrics.Metric;

class HierarchicalKMeans implements KMeans<HierarchicalKMeans.Result> {

    private final int branchFactor;
    private final int maxDepth;
    private final int minClusterSize;
    private final int maxIterationsPerLevel;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public HierarchicalKMeans(int branchFactor,
        int maxDepth,
        int minClusterSize,
        int maxIterationsPerLevel,
        float tolerance,
        Random random,
        Metric.Type metricType,
        Metric.Engine metricEngine) {

        if (branchFactor <= 1)
            throw new IllegalArgumentException("branchFactor must be >= 2");
        if (maxDepth <= 0)
            throw new IllegalArgumentException("maxDepth must be > 0");
        if (minClusterSize <= 0)
            throw new IllegalArgumentException("minClusterSize must be > 0");
        if (maxIterationsPerLevel <= 0)
            throw new IllegalArgumentException("maxIterationsPerLevel must be > 0");
        if (tolerance < 0.0f)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.branchFactor = branchFactor;
        this.maxDepth = maxDepth;
        this.minClusterSize = minClusterSize;
        this.maxIterationsPerLevel = maxIterationsPerLevel;
        this.tolerance = tolerance;
        this.random = random;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
    }

    @Override public Metric.Type getMetricType() {
        return metricType;
    }

    @Override public Metric.Engine getMetricEngine() {
        return metricEngine;
    }

    @Override public Result fit(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");

        int sampleCnt = data.length;
        int dimension = KMeansUtils.validateAndGetDimension(data);

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);

        // Parallel indices initialization for large datasets
        int[] allIndices = new int[sampleCnt];
        if (sampleCnt >= PARALLEL_CENTROID_THRESHOLD) {
            IntStream.range(0, sampleCnt).parallel().forEach(i -> allIndices[i] = i);
        } else {
            for (int i = 0; i < sampleCnt; i++)
                allIndices[i] = i;
        }

        Node root = buildNode(data, allIndices, 0, dimension);

        List<float[]> leafCentroidsList = new ArrayList<>();
        int[] leafAssignments = new int[sampleCnt];
        FloatWrapper lossAccumulator = new FloatWrapper();

        assignLeafIdsAndCollect(root, data, leafCentroidsList, leafAssignments, lossAccumulator, distFn);

        float[][] leafCentroids = leafCentroidsList.toArray(new float[0][]);
        int[] clusterSizes = computeClusterSizes(leafAssignments, leafCentroids.length);

        return new Result(root, leafAssignments, leafCentroids, lossAccumulator.val, clusterSizes);
    }

    private static final int PARALLEL_PREDICT_THRESHOLD = 1000;

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");
        if (model.getRoot() == null)
            throw new IllegalArgumentException("model root must be non-null");

        int dimension = KMeansUtils.validateAndGetDimension(data);
        if (model.getRoot().getCentroid().length != dimension)
            throw new IllegalArgumentException("data dimension must match tree centroid dimension");

        Metric.DistanceFunction distFn = metricType.resolveFunction(metricEngine);
        int[] labels = new int[data.length];

        if (data.length >= PARALLEL_PREDICT_THRESHOLD) {
            // Parallel prediction - each point traversal is independent
            Node root = model.getRoot();
            IntStream.range(0, data.length).parallel().forEach(i ->
                labels[i] = predictSinglePoint(data[i], root, distFn));
        } else {
            // Sequential path for small datasets
            for (int i = 0; i < data.length; i++)
                labels[i] = predictSinglePoint(data[i], model.getRoot(), distFn);
        }

        return labels;
    }

    private int predictSinglePoint(float[] point, Node node, Metric.DistanceFunction distFn) {
        while (!node.isLeaf()) {
            Node[] children = node.getChildren();
            if (children == null || children.length == 0)
                break;

            Node bestChild = children[0];
            float bestDistance = distFn.compute(point, children[0].centroid);

            for (int c = 1; c < children.length; c++) {
                float distance = distFn.compute(point, children[c].centroid);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestChild = children[c];
                }
            }

            node = bestChild;
        }

        if (node.getLeafId() < 0)
            throw new IllegalStateException("Leaf node has no leafId assigned");

        return node.getLeafId();
    }

    private Node buildNode(float[][] data,
        int[] indices,
        int level,
        int dimension) {
        int sampleCnt = indices.length;

        float[] centroid = computeCentroid(data, indices, dimension);

        if (level >= maxDepth - 1 || sampleCnt < minClusterSize)
            return new Node(level, centroid, null, indices);

        int locClusterCnt = Math.min(branchFactor, sampleCnt);
        if (locClusterCnt < 2)
            return new Node(level, centroid, null, indices);

        // Parallel subset creation for large nodes
        float[][] subset = new float[sampleCnt][];
        if (sampleCnt >= PARALLEL_CENTROID_THRESHOLD) {
            IntStream.range(0, sampleCnt).parallel().forEach(i ->
                subset[i] = data[indices[i]]);
        } else {
            for (int i = 0; i < sampleCnt; i++)
                subset[i] = data[indices[i]];
        }

        LloydKMeans kmeans = new LloydKMeans(
            locClusterCnt,
            metricType,
            metricEngine,
            maxIterationsPerLevel,
            tolerance,
            random
        );

        LloydKMeans.Result kmResult = kmeans.fit(subset);
        int[] labels = kmResult.getClusterAssignments();

        int[] clusterSizes = new int[locClusterCnt];
        for (int label : labels) {
            if (label < 0 || label >= locClusterCnt)
                throw new IllegalStateException("KMeans produced invalid label: " + label);
            clusterSizes[label]++;
        }

        int nonEmptyClusterCnt = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0)
                nonEmptyClusterCnt++;
        }

        if (nonEmptyClusterCnt <= 1)
            return new Node(level, centroid, null, indices);

        int[] clusterIdToChildIdx = new int[locClusterCnt];
        Arrays.fill(clusterIdToChildIdx, -1);

        int childIdx = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0) {
                clusterIdToChildIdx[c] = childIdx;
                childIdx++;
            }
        }

        int[][] childIndices = new int[nonEmptyClusterCnt][];

        int[] childSizes = new int[nonEmptyClusterCnt];
        for (int c = 0; c < locClusterCnt; c++) {
            int mappedChildIdx = clusterIdToChildIdx[c];
            if (mappedChildIdx >= 0)
                childSizes[mappedChildIdx] = clusterSizes[c];
        }

        for (int i = 0; i < nonEmptyClusterCnt; i++)
            childIndices[i] = new int[childSizes[i]];

        // Parallel child indices distribution for large nodes
        if (sampleCnt >= PARALLEL_CENTROID_THRESHOLD) {
            AtomicIntegerArray atomicOffsets = new AtomicIntegerArray(nonEmptyClusterCnt);
            IntStream.range(0, sampleCnt).parallel().forEach(i -> {
                int originalCluster = labels[i];
                int mappedChild = clusterIdToChildIdx[originalCluster];
                int pos = atomicOffsets.getAndIncrement(mappedChild);
                childIndices[mappedChild][pos] = indices[i];
            });
        } else {
            int[] offsets = new int[nonEmptyClusterCnt];
            for (int i = 0; i < sampleCnt; i++) {
                int originalCluster = labels[i];
                int mappedChild = clusterIdToChildIdx[originalCluster];
                int pos = offsets[mappedChild]++;
                childIndices[mappedChild][pos] = indices[i];
            }
        }

        // Parallel tree building: children at same level are independent
        Node[] children = new Node[nonEmptyClusterCnt];
        if (nonEmptyClusterCnt >= 2 && level < maxDepth - 2) {
            // Use parallel streams for building children
            int[][] finalChildIndices = childIndices;
            IntStream.range(0, nonEmptyClusterCnt).parallel().forEach(i ->
                children[i] = buildNode(data, finalChildIndices[i], level + 1, dimension));
        } else {
            // Sequential for small number of children or near leaf level
            for (int i = 0; i < nonEmptyClusterCnt; i++)
                children[i] = buildNode(data, childIndices[i], level + 1, dimension);
        }

        return new Node(level, centroid, children, null);
    }

    private static final int PARALLEL_CENTROID_THRESHOLD = 5000;

    private float[] computeCentroid(float[][] data,
        int[] indices,
        int dimension) {
        float[] centroid = new float[dimension];
        int cnt = indices.length;
        if (cnt == 0)
            return centroid;

        if (cnt >= PARALLEL_CENTROID_THRESHOLD) {
            // Parallel centroid computation with thread-local sums
            int numThreads = Runtime.getRuntime().availableProcessors();
            int chunkSize = (cnt + numThreads - 1) / numThreads;
            float[][] localSums = new float[numThreads][dimension];

            IntStream.range(0, numThreads).parallel().forEach(threadIdx -> {
                int start = threadIdx * chunkSize;
                int end = Math.min(start + chunkSize, cnt);
                float[] mySum = localSums[threadIdx];

                for (int i = start; i < end; i++) {
                    float[] point = data[indices[i]];
                    for (int d = 0; d < dimension; d++)
                        mySum[d] += point[d];
                }
            });

            // Merge
            for (int t = 0; t < numThreads; t++) {
                for (int d = 0; d < dimension; d++)
                    centroid[d] += localSums[t][d];
            }
        } else {
            // Sequential path
            for (int idx : indices) {
                float[] point = data[idx];
                for (int d = 0; d < dimension; d++)
                    centroid[d] += point[d];
            }
        }

        float invCnt = 1.0f / (float)cnt;
        for (int d = 0; d < dimension; d++)
            centroid[d] *= invCnt;

        if (metricType == Metric.Type.COSINE_DISTANCE)
            KMeansUtils.normalizeSingleCentroid(centroid);

        return centroid;
    }

    private static final int PARALLEL_LEAF_LOSS_THRESHOLD = 1000;

    private void assignLeafIdsAndCollect(Node node,
        float[][] data,
        List<float[]> leafCentroidsList,
        int[] leafAssignments,
        FloatWrapper lossAccumulator,
        Metric.DistanceFunction distFn) {
        if (node == null)
            return;

        if (node.isLeaf()) {
            int leafId = leafCentroidsList.size();
            node.leafId = leafId;
            leafCentroidsList.add(node.centroid);

            if (node.pointIndices != null) {
                int[] indices = node.pointIndices;
                float[] centroid = node.centroid;

                if (indices.length >= PARALLEL_LEAF_LOSS_THRESHOLD) {
                    // Parallel loss computation for large leaves
                    double loss = IntStream.of(indices).parallel()
                        .mapToDouble(idx -> distFn.compute(data[idx], centroid))
                        .sum();
                    lossAccumulator.val += (float) loss;

                    // Sequential assignment (fast, no synchronization needed)
                    for (int idx : indices)
                        leafAssignments[idx] = leafId;
                } else {
                    // Sequential path for small leaves
                    for (int idx : indices) {
                        leafAssignments[idx] = leafId;
                        lossAccumulator.val += distFn.compute(data[idx], centroid);
                    }
                }
            }
        }
        else {
            Node[] children = node.getChildren();
            if (children != null) {
                for (Node child : children)
                    assignLeafIdsAndCollect(child, data, leafCentroidsList, leafAssignments,
                        lossAccumulator, distFn);
            }
        }
    }

    private int[] computeClusterSizes(int[] assignments, int clusterCnt) {
        int[] sizes = new int[clusterCnt];

        for (int label : assignments) {
            if (label < 0 || label >= clusterCnt) {
                throw new IllegalStateException(
                    "invalid cluster label " + label + " (expected 0.." + (clusterCnt - 1) + ")"
                );
            }
            sizes[label]++;
        }

        return sizes;
    }

    private static final class FloatWrapper {
        float val;
    }

    public static final class Node {
        private final int level;
        private final float[] centroid;
        private final Node[] children;
        private final int[] pointIndices;
        private int leafId = -1;

        Node(int level,
            float[] centroid,
            Node[] children,
            int[] pointIndices) {
            this.level = level;
            this.centroid = centroid;
            this.children = children;
            this.pointIndices = pointIndices;
        }

        public int getLevel() {
            return level;
        }

        public float[] getCentroid() {
            return centroid;
        }

        public Node[] getChildren() {
            return children;
        }

        public int[] getPointIndices() {
            return pointIndices;
        }

        public int getLeafId() {
            return leafId;
        }

        public boolean isLeaf() {
            return children == null || children.length == 0;
        }
    }

    static class Result implements ClusteringResult {
        private final Node root;
        private final int[] leafAssignments;
        private final float[][] leafCentroids;
        private final float loss;
        private final int[] clusterSizes;

        Result(Node root,
            int[] leafAssignments,
            float[][] leafCentroids,
            float loss,
            int[] clusterSizes) {
            this.root = root;
            this.leafAssignments = leafAssignments;
            this.leafCentroids = leafCentroids;
            this.loss = loss;
            this.clusterSizes = clusterSizes;
        }

        public Node getRoot() {
            return root;
        }

        @Override public int[] getClusterAssignments() {
            return leafAssignments;
        }

        @Override public float[][] getCentroids() {
            return leafCentroids;
        }

        @Override public float getLoss() {
            return loss;
        }

        @Override public int[] getClusterSizes() {
            return clusterSizes;
        }
    }
}
