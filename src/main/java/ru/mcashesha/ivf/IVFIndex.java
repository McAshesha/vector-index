package ru.mcashesha.ivf;

import java.util.List;
import ru.mcashesha.metrics.Metric;

/**
 * Interface for an Inverted File (IVF) index that supports approximate nearest neighbor (ANN) search.
 *
 * <p>An IVF index partitions the vector space into clusters (via KMeans or similar algorithms)
 * and, at search time, only examines a subset of clusters ({@code nProbe}) to find approximate
 * nearest neighbors. This trades a small amount of recall accuracy for significant speedup
 * compared to exhaustive brute-force search.</p>
 *
 * <h3>Typical usage:</h3>
 * <pre>{@code
 *   IVFIndex index = new IVFIndexFlat(kMeans);
 *   index.build(vectors, ids);
 *   List<SearchResult> results = index.search(query, topK, nProbe);
 * }</pre>
 *
 * @see IVFIndexFlat
 */
public interface IVFIndex {

    /**
     * Builds the index from the given vectors and associated IDs.
     *
     * <p>This method clusters the vectors using the configured KMeans algorithm, then
     * reorders and stores the data internally for efficient search. After this call,
     * the index is ready for queries via {@link #search} or {@link #searchBatch}.</p>
     *
     * @param vectors the dataset of vectors to index; each vector must have the same dimension
     * @param ids     custom integer identifiers for each vector; must have the same length as {@code vectors}.
     *                These IDs are returned in {@link SearchResult} to identify matched vectors.
     * @throws IllegalArgumentException if vectors are null, empty, or have inconsistent dimensions,
     *                                  or if ids length does not match vectors length
     */
    void build(float[][] vectors, int[] ids);

    /**
     * Builds the index from the given vectors, using sequential indices (0, 1, 2, ...) as IDs.
     *
     * <p>Convenience overload equivalent to {@code build(vectors, null)}.</p>
     *
     * @param vectors the dataset of vectors to index; each vector must have the same dimension
     * @throws IllegalArgumentException if vectors are null, empty, or have inconsistent dimensions
     */
    void build(float[][] vectors);

    /**
     * Performs a single approximate nearest neighbor search.
     *
     * <p>The search probes {@code nProbe} clusters (out of the total number of clusters)
     * closest to the query vector, then returns the top-{@code topK} nearest vectors
     * found within those clusters, sorted by ascending distance.</p>
     *
     * @param query  the query vector; must match the dimension of the indexed vectors
     * @param topK   the number of nearest neighbors to return (must be &gt; 0)
     * @param nProbe the number of clusters to probe; higher values improve recall at the cost of speed.
     *               Clamped to [1, totalClusters].
     * @return a list of up to {@code topK} search results sorted by ascending distance
     * @throws IllegalStateException    if the index has not been built yet
     * @throws IllegalArgumentException if the query is null, has wrong dimension, or topK &le; 0
     */
    List<SearchResult> search(float[] query, int topK, int nProbe);

    /**
     * Performs parallel approximate nearest neighbor search for multiple queries.
     *
     * <p>Each query is processed independently and in parallel. The returned list
     * has the same size as {@code queries}, where the i-th element contains the
     * results for the i-th query.</p>
     *
     * @param queries array of query vectors; each must match the index dimension
     * @param topK    the number of nearest neighbors to return per query (must be &gt; 0)
     * @param nProbe  the number of clusters to probe per query
     * @return a list of result lists, one per query, each sorted by ascending distance
     * @throws IllegalStateException    if the index has not been built yet
     * @throws IllegalArgumentException if queries are null/empty, any query has wrong dimension, or topK &le; 0
     */
    List<List<SearchResult>> searchBatch(float[][] queries, int topK, int nProbe);

    /**
     * Returns the dimensionality of the indexed vectors.
     *
     * @return the number of components in each vector
     */
    int getDimension();

    /**
     * Returns the number of clusters (Voronoi cells) in the index.
     *
     * @return the total number of clusters produced by the KMeans algorithm
     */
    int getCountClusters();

    /**
     * Returns the distance metric type used by this index (e.g., L2, dot product, cosine).
     *
     * @return the metric type
     */
    Metric.Type getMetricType();

    /**
     * Returns the distance computation engine used by this index (e.g., Scalar, VectorAPI, SimSIMD).
     *
     * @return the metric engine
     */
    Metric.Engine getMetricEngine();

    /**
     * An immutable result record from an approximate nearest neighbor search.
     *
     * <p>Each result contains the ID of the matched vector, the computed distance
     * from the query to that vector, and the cluster in which the vector resides.
     * Lower distance values indicate closer (more similar) vectors.</p>
     */
    final class SearchResult {

        /** The identifier of the matched vector, as provided during {@link IVFIndex#build}. */
        public final int id;

        /**
         * The distance from the query vector to this result vector, computed using the
         * index's configured distance metric. Lower values indicate closer matches.
         * For dot product, this is the negated dot product so that lower = more similar.
         */
        public final float distance;

        /** The index of the cluster (Voronoi cell) in which this vector resides. */
        public final int clusterId;

        /**
         * Constructs a new search result.
         *
         * @param id        the identifier of the matched vector
         * @param distance  the distance from the query to the matched vector
         * @param clusterId the cluster index containing the matched vector
         */
        public SearchResult(int id, float distance, int clusterId) {
            this.id = id;
            this.distance = distance;
            this.clusterId = clusterId;
        }
    }
}
