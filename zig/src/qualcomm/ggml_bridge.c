#include "ggml_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"
#include "ggml.h"

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1.0;
    }
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

double tf_qualcomm_monotonic_ms(void) {
    return monotonic_ms();
}

size_t tf_qualcomm_peak_rss_kib(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return (size_t) usage.ru_maxrss;
}

static ggml_backend_t init_hexagon_backend(void) {
    ggml_backend_reg_t registry = ggml_backend_hexagon_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
}

struct tf_qualcomm_backend {
    ggml_backend_t backend;
};

void *tf_qualcomm_backend_create(const char *backend_name, int cpu_threads) {
    const int use_htp = strcmp(backend_name, "htp") == 0;
    if (!use_htp && strcmp(backend_name, "cpu") != 0) {
        return NULL;
    }
    ggml_backend_t backend = use_htp ? init_hexagon_backend() : ggml_backend_cpu_init();
    if (backend == NULL) {
        return NULL;
    }
    if (!use_htp) {
        ggml_backend_cpu_set_n_threads(backend, cpu_threads);
    }
    struct tf_qualcomm_backend *context = malloc(sizeof(*context));
    if (context == NULL) {
        ggml_backend_free(backend);
        return NULL;
    }
    context->backend = backend;
    return context;
}

void tf_qualcomm_backend_destroy(void *opaque) {
    if (opaque == NULL) {
        return;
    }
    struct tf_qualcomm_backend *context = opaque;
    ggml_backend_free(context->backend);
    free(context);
}

int tf_qualcomm_backend_matmul(void *opaque,
                               size_t k,
                               size_t m,
                               size_t n,
                               size_t iterations,
                               const float *weight,
                               const float *input,
                               float *output,
                               double *ms_per_iteration) {
    if (opaque == NULL || iterations == 0) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    ggml_backend_t backend = context->backend;

    const struct ggml_init_params params = {
        .mem_size = 4u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (ctx == NULL) {
        return 2;
    }
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) m);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
    struct ggml_tensor *y = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == NULL) {
        ggml_free(ctx);
        return 3;
    }
    ggml_backend_tensor_set(a, weight, 0, k * m * sizeof(*weight));
    ggml_backend_tensor_set(b, input, 0, k * n * sizeof(*input));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return 4;
    }
    const double started = monotonic_ms();
    for (size_t i = 0; i < iterations; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return 5;
        }
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(y, output, 0, m * n * sizeof(*output));
    *ms_per_iteration = (stopped - started) / (double) iterations;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return 0;
}

int tf_qualcomm_matmul(const char *backend_name,
                       size_t k,
                       size_t m,
                       size_t n,
                       size_t iterations,
                       int cpu_threads,
                       const float *weight,
                       const float *input,
                       float *output,
                       double *ms_per_iteration) {
    void *context = tf_qualcomm_backend_create(backend_name, cpu_threads);
    if (context == NULL) {
        return 2;
    }
    const int status = tf_qualcomm_backend_matmul(context, k, m, n, iterations, weight, input, output, ms_per_iteration);
    tf_qualcomm_backend_destroy(context);
    return status;
}
