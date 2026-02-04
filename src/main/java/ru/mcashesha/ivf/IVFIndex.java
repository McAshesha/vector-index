package ru.mcashesha.ivf;

import java.util.List;
import ru.mcashesha.metrics.Metric;

public interface IVFIndex {
    void build(float[][] vectors, int[] ids);

    void build(float[][] vectors);

    List<SearchResult> search(float[] query, int topK, int nProbe);

    int getDimension();

    int getCountClusters();

    Metric.Type getMetricType();

    Metric.Engine getMetricEngine();

    final class SearchResult {
        public final int id;
        public final float distance;
        public final int clusterId;

        public SearchResult(int id, float distance, int clusterId) {
            this.id = id;
            this.distance = distance;
            this.clusterId = clusterId;
        }
    }
}
