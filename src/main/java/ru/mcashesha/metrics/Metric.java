package ru.mcashesha.metrics;

public interface Metric {

    float l2Distance(float[] a, float[] b);

    float dotProduct(float[] a, float[] b);

    float cosineDistance(float[] a, float[] b);

    long hammingDistanceB8(byte[] a, byte[] b);

    @FunctionalInterface
    interface DistanceFunction {
        float compute(float[] a, float[] b);
    }

    enum Type {
        L2SQ_DISTANCE() {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.metric.l2Distance(a, b);
            }
        },
        DOT_PRODUCT {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return -engine.metric.dotProduct(a, b);
            }
        },
        COSINE_DISTANCE {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.metric.cosineDistance(a, b);
            }
        };

        public abstract float distance(Engine engine, float[] a, float[] b);

        public DistanceFunction resolveFunction(Engine engine) {
            Metric m = engine.metric;
            switch (this) {
                case L2SQ_DISTANCE: return m::l2Distance;
                case DOT_PRODUCT: return (a, b) -> -m.dotProduct(a, b);
                case COSINE_DISTANCE: return m::cosineDistance;
                default: throw new IllegalStateException("Unknown metric type: " + this);
            }
        }
    }

    enum Engine {
        SCALAR(new Scalar()),
        VECTOR_API(new VectorAPI()),
        SIMSIMD(new SimSIMD());

        final Metric metric;

        Engine(Metric metric) {
            this.metric = metric;
        }

        Metric getMetric() {
            return metric;
        }
    }

}
