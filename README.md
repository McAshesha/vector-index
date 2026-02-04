# Vector Index Bench

A JMH benchmarking suite for vector similarity search using Inverted File (IVF) indexes in Java. Compares three distance computation engines and three clustering algorithms across multiple distance metrics.

## Overview

This project implements an IVF (Inverted File) flat index for approximate nearest neighbor search and benchmarks its build and search performance. The core components are:

- **Distance engines** — three interchangeable backends for computing vector distances
- **Clustering algorithms** — three KMeans variants for partitioning vector space
- **IVF index** — clusters vectors at build time, probes nearest clusters at search time

All combinations of engine, algorithm, and metric type can be benchmarked via JMH.

## Distance Engines

| Engine | Implementation | Description |
|--------|---------------|-------------|
| **Scalar** | Pure Java loops | Baseline. Clean loops suitable for JIT auto-vectorization |
| **VectorAPI** | `jdk.incubator.vector` | Explicit SIMD via Java Vector API (FMA accumulation, deferred horizontal reduction) |
| **SimSIMD** | C via JNI | Native SIMD using [SimSIMD](https://github.com/ashvardanian/SimSIMD). Uses `GetPrimitiveArrayCritical` for zero-copy array access |

Each engine implements four distance functions:

- `l2Distance` — L2 squared distance
- `dotProduct` — dot product (negated internally for distance semantics: lower = closer)
- `cosineDistance` — cosine distance (1 - cosine similarity)
- `hammingDistanceB8` — Hamming distance over packed byte arrays

## Clustering Algorithms

All algorithms use KMeans++ initialization and support pluggable distance metrics.

| Algorithm | Class | Description |
|-----------|-------|-------------|
| **Lloyd** | `LloydKMeans` | Classic iterative KMeans. Full dataset pass per iteration. Convergence via max centroid shift (L2). Handles empty clusters by redistributing worst-fitting points from the largest cluster |
| **MiniBatch** | `MiniBatchKMeans` | Stochastic variant. Samples a mini-batch per iteration and applies weighted centroid updates. Early stopping on no improvement. Final pass reassigns all points and recomputes centroids |
| **Hierarchical** | `HierarchicalKMeans` | Recursively partitions data into a tree using Lloyd at each level. Configurable branch factor, max depth, and minimum leaf size. Leaf centroids form the final cluster set |

Configuration via builder:

```java
KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
    .withClusterCount(64)
    .withMaxIterations(100)
    .withTolerance(1e-3f)
    .build();
```

## IVF Index

`IVFIndexFlat` implements the IVF search strategy:

1. **Build** — fits a KMeans model on the dataset, creates per-cluster inverted lists of vector indices
2. **Search** — computes query-to-centroid distances, selects `nProbe` nearest clusters, scans those clusters with a max-heap for top-K extraction, returns results sorted by distance

## Prerequisites

- Java 17+
- Maven 3.8+
- CMake 3.16+
- C compiler (GCC or Clang)

## Build

```bash
mvn clean package
```

This will:
1. Run CMake to fetch SimSIMD and build the native JNI library (`libsimsimd_jni.so`)
2. Compile Java sources with `--add-modules jdk.incubator.vector`
3. Create an executable uber JAR at `target/benchmarks.jar`

## Running Tests

```bash
mvn test
```

The test suite includes:
- **MetricConsistencyTest** — validates that all three engines produce consistent results for all distance functions
- **KMeansTest** — tests all clustering algorithms across all metric types
- **IVFIndexFlatTest** — tests index build and search correctness

## Running Benchmarks

Benchmarks require an `embeddings.csv` file with 512-dimensional vectors (CSV format: `id, metadata, "[v0, v1, ..., v511]"`).

```bash
# Run all benchmarks
java -jar target/benchmarks.jar

# Run build benchmarks only
java -jar target/benchmarks.jar IVFIndexBuildBenchmarks

# Run search benchmarks only
java -jar target/benchmarks.jar IVFIndexSearchBenchmarks

# Filter by parameters
java -jar target/benchmarks.jar -p metricEngineName=SIMSIMD -p metricTypeName=L2SQ_DISTANCE
```

### Benchmark Configuration

**Build benchmarks** (`IVFIndexBuildBenchmarks`):
- Mode: Average time (ms)
- 1 warmup iteration (30s), 5 measurement iterations (120s each), 1 fork

**Search benchmarks** (`IVFIndexSearchBenchmarks`):
- Mode: Sample time (us)
- 5 warmup iterations (10s), 10 measurement iterations (30s each), 3 forks
- TOP_K = 100, nProbe = 16 (Lloyd/MiniBatch), 8 (Hierarchical)

Parameters for both:
- `metricTypeName`: `L2SQ_DISTANCE`, `DOT_PRODUCT`, `COSINE_DISTANCE`
- `metricEngineName`: `SCALAR`, `VECTOR_API`, `SIMSIMD`

### Quick Demo (non-JMH)

```bash
java -cp target/benchmarks.jar ru.mcashesha.VectorIndexBenchmark
```

Builds a hierarchical IVF index and runs a single search query, printing build and search times.

## Project Structure

```
src/
  main/
    java/ru/mcashesha/
      metrics/
        Metric.java          # Interface + Type/Engine enums + DistanceFunction
        Scalar.java           # Pure Java implementation
        VectorAPI.java        # Java Vector API (SIMD) implementation
        SimSIMD.java          # JNI wrapper for native SimSIMD
      kmeans/
        KMeans.java           # Interface + Builder + ClusteringResult
        LloydKMeans.java      # Classic Lloyd's algorithm
        MiniBatchKMeans.java  # Mini-batch variant
        HierarchicalKMeans.java # Hierarchical tree-based clustering
      ivf/
        IVFIndex.java         # Index interface + SearchResult
        IVFIndexFlat.java     # Flat IVF implementation with inverted lists
      data/
        EmbeddingCsvLoader.java # CSV embedding reader (512-dim)
      IVFIndexBuildBenchmarks.java   # JMH build benchmarks
      IVFIndexSearchBenchmarks.java  # JMH search benchmarks
      VectorIndexBenchmark.java      # Non-JMH demo
    native/
      CMakeLists.txt          # Builds libsimsimd_jni.so
      simsimd_jni.c           # JNI bridge to SimSIMD
      ru_mcashesha_metrics_SimSIMD.h  # JNI header
  test/
    java/ru/mcashesha/
      metrics/MetricConsistencyTest.java  # 114 tests
      kmeans/KMeansTest.java              # 25 tests
      ivf/IVFIndexFlatTest.java           # 20 tests
```

## Architecture

```
CSV ──> EmbeddingCsvLoader ──> float[][]
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               LloydKMeans  MiniBatchKMeans  HierarchicalKMeans
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                          ClusteringResult
                        (centroids, assignments)
                                  │
                                  ▼
                           IVFIndexFlat.build()
                        (inverted lists per cluster)
                                  │
                                  ▼
                           IVFIndexFlat.search()
                    (nProbe clusters ──> top-K heap)
                                  │
                                  ▼
                         List<SearchResult>
```

Distance computation is pluggable at every level via `Metric.Type` and `Metric.Engine`.

## Parameter Tuning Results

Benchmarks run on a dataset of **183,615 vectors** with **512 dimensions** (image embeddings). Search accuracy measured as Recall@100 against brute-force ground truth.

### Algorithm Comparison Summary

| Algorithm | Build Time | Search Time | Recall | Use Case |
|-----------|------------|-------------|--------|----------|
| **MiniBatch** | 4-18 sec | 4-15 ms | 95-99% | Best overall balance |
| **Hierarchical** | 9-17 sec | 4-16 ms | 95-99% | Large cluster counts |
| **Lloyd** | 79-149 sec | 4-14 ms | 96-99% | Maximum centroid quality |

### Optimal Parameters by Recall Target

#### Recall >= 95% (Fast Search)

| Algorithm | Parameters | nProbe | Clusters | Build | Search | Recall |
|-----------|------------|--------|----------|-------|--------|--------|
| **MiniBatch** | k=64, batch=512 | 8 | 64 | 4.1 sec | 7.0 ms | 97.8% |
| **MiniBatch** | k=128, batch=512 | 8 | 128 | 7.9 sec | 3.9 ms | 95.1% |
| **Hierarchical** | bf=8, d=3, min=100 | 8 | 64 | 8.6 sec | 7.7 ms | 97.2% |
| **Lloyd** | k=128 | 8 | 128 | 149 sec | 3.7 ms | 96.2% |

#### Recall >= 99% (High Accuracy)

| Algorithm | Parameters | nProbe | Clusters | Build | Search | Recall |
|-----------|------------|--------|----------|-------|--------|--------|
| **MiniBatch** | k=256, batch=1024 | 32 | 256 | 18.4 sec | 7.7 ms | 99.3% |
| **Hierarchical** | bf=8, d=4, min=50 | 64 | 512 | 15.3 sec | 8.7 ms | 99.5% |
| **Lloyd** | k=64 | 16 | 64 | 79 sec | 13.5 ms | 99.6% |

#### Recall = 100% (Perfect Accuracy)

| Algorithm | Parameters | nProbe | Clusters | Build | Search | Recall |
|-----------|------------|--------|----------|-------|--------|--------|
| **MiniBatch** | k=64, batch=512 | 64 | 64 | 4.1 sec | 52.9 ms | 100% |
| **Lloyd** | k=64 | 64 | 64 | 79 sec | 53.5 ms | 100% |

### Recommended Configurations

#### MiniBatch KMeans (Best Overall)

Build time **20x faster** than Lloyd with comparable search quality.

```java
// Optimal balance: fast build + high recall
KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.MINI_BATCH, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
    .withClusterCount(64)           // 64-128 clusters
    .withBatchSize(512)             // optimal batch size
    .withMaxIterations(300)
    .withMaxNoImprovementIterations(30)
    .withTolerance(1e-4f)
    .withRandom(new Random(42))
    .build();

IVFIndex index = new IVFIndexFlat(kmeans);
index.build(data);

// Search with nProbe=8 for ~98% recall, nProbe=16 for ~99% recall
List<SearchResult> results = index.search(query, 100, 8);
```

#### Hierarchical KMeans (Large Cluster Counts)

Good for 256+ clusters with reasonable build time.

```java
KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.HIERARCHICAL, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
    .withBranchFactor(8)            // 8-16 optimal
    .withMaxDepth(4)                // depth 3-4
    .withMinClusterSize(50)
    .withMaxIterationsPerLevel(30)
    .withTolerance(1e-4f)
    .withRandom(new Random(42))
    .build();

// Search with nProbe=32-64 for 97-99% recall
List<SearchResult> results = index.search(query, 100, 32);
```

#### Lloyd KMeans (Maximum Quality)

Use when index is built once and used extensively.

```java
KMeans<?> kmeans = KMeans.newBuilder(KMeans.Type.LLOYD, Metric.Type.L2SQ_DISTANCE, Metric.Engine.VECTOR_API)
    .withClusterCount(64)           // 64-128 clusters
    .withMaxIterations(100)
    .withTolerance(1e-4f)
    .withRandom(new Random(42))
    .build();

// Search with nProbe=8-16 for 98-99% recall
List<SearchResult> results = index.search(query, 100, 8);
```

### nProbe Selection Guide

The `nProbe` parameter controls the recall/speed trade-off:

```
nProbe ~ clusters * (target_recall / 100)^2

Examples (64 clusters):
  - 95% recall: nProbe ~ 64 * 0.90 = 8
  - 99% recall: nProbe ~ 64 * 0.98 = 16
  - 100% recall: nProbe = 64 (full scan)
```

| Clusters | nProbe | Expected Recall |
|----------|--------|-----------------|
| 64 | 4 | ~92% |
| 64 | 8 | ~97% |
| 64 | 16 | ~99% |
| 64 | 32 | ~99.9% |
| 128 | 8 | ~95% |
| 128 | 16 | ~98% |
| 128 | 32 | ~99.8% |
| 256 | 16 | ~97% |
| 256 | 32 | ~99% |
| 256 | 64 | ~99.9% |

### Key Findings

1. **MiniBatch KMeans is the best default choice** — 20x faster build than Lloyd with minimal accuracy loss

2. **More clusters = faster search, lower recall per nProbe** — double clusters, halve search time, but need higher nProbe

3. **Batch size 512-1024 is optimal for MiniBatch** — larger batches slightly improve quality but increase build time

4. **Hierarchical shines with 256+ clusters** — tree structure provides good partitioning at scale

5. **Lloyd produces highest quality centroids** — use when build time is not a concern
