#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static int parse_positive(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > INT32_MAX) {
        return -1;
    }
    *out = (int) value;
    return 0;
}

static void fill_inputs(float *a, float *b, int k, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            const int value = ((row * 17 + col * 13) % 31) - 15;
            a[(size_t) row * (size_t) k + (size_t) col] = (float) value / 16.0f;
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            const int value = ((row * 11 + col * 7) % 29) - 14;
            b[(size_t) row * (size_t) k + (size_t) col] = (float) value / 16.0f;
        }
    }
}

static float reference_max_abs(const float *a, const float *b, const float *got, int k, int m, int n) {
    float max_abs = 0.0f;
    for (int row_b = 0; row_b < n; ++row_b) {
        for (int row_a = 0; row_a < m; ++row_a) {
            float sum = 0.0f;
            for (int col = 0; col < k; ++col) {
                sum += a[(size_t) row_a * (size_t) k + (size_t) col] *
                       b[(size_t) row_b * (size_t) k + (size_t) col];
            }
            const float delta = fabsf(got[(size_t) row_b * (size_t) m + (size_t) row_a] - sum);
            if (delta > max_abs) {
                max_abs = delta;
            }
        }
    }
    return max_abs;
}

static ggml_backend_t init_hexagon_backend(void) {
    ggml_backend_reg_t registry = ggml_backend_hexagon_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s cpu|htp K M N ITERATIONS CPU_THREADS\n", argv[0]);
        return 64;
    }

    int k = 0;
    int m = 0;
    int n = 0;
    int iterations = 0;
    int cpu_threads = 0;
    if (parse_positive(argv[2], &k) || parse_positive(argv[3], &m) || parse_positive(argv[4], &n) ||
        parse_positive(argv[5], &iterations) || parse_positive(argv[6], &cpu_threads)) {
        fprintf(stderr, "all dimensions, iteration count and CPU thread count must be positive integers\n");
        return 64;
    }

    const int use_htp = strcmp(argv[1], "htp") == 0;
    if (!use_htp && strcmp(argv[1], "cpu") != 0) {
        fprintf(stderr, "backend must be cpu or htp\n");
        return 64;
    }

    ggml_backend_t backend = use_htp ? init_hexagon_backend() : ggml_backend_cpu_init();
    if (backend == NULL) {
        fprintf(stderr, "failed to initialize %s backend\n", argv[1]);
        return 2;
    }
    if (!use_htp) {
        ggml_backend_cpu_set_n_threads(backend, cpu_threads);
    }

    const size_t a_count = (size_t) k * (size_t) m;
    const size_t b_count = (size_t) k * (size_t) n;
    const size_t y_count = (size_t) m * (size_t) n;
    float *a_data = malloc(a_count * sizeof(*a_data));
    float *b_data = malloc(b_count * sizeof(*b_data));
    float *y_data = malloc(y_count * sizeof(*y_data));
    if (a_data == NULL || b_data == NULL || y_data == NULL) {
        fprintf(stderr, "host allocation failed\n");
        free(a_data);
        free(b_data);
        free(y_data);
        ggml_backend_free(backend);
        return 3;
    }
    fill_inputs(a_data, b_data, k, m, n);

    const struct ggml_init_params params = {
        .mem_size = 4u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "ggml context initialization failed\n");
        free(a_data);
        free(b_data);
        free(y_data);
        ggml_backend_free(backend);
        return 4;
    }

    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor *y = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == NULL) {
        fprintf(stderr, "backend tensor allocation failed\n");
        ggml_free(ctx);
        free(a_data);
        free(b_data);
        free(y_data);
        ggml_backend_free(backend);
        return 5;
    }

    ggml_backend_tensor_set(a, a_data, 0, a_count * sizeof(*a_data));
    ggml_backend_tensor_set(b, b_data, 0, b_count * sizeof(*b_data));

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "warmup graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        free(a_data);
        free(b_data);
        free(y_data);
        ggml_backend_free(backend);
        return 6;
    }

    const double started_ms = monotonic_ms();
    for (int i = 0; i < iterations; ++i) {
        status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "measured graph compute failed: %s\n", ggml_status_to_string(status));
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            free(a_data);
            free(b_data);
            free(y_data);
            ggml_backend_free(backend);
            return 7;
        }
    }
    const double stopped_ms = monotonic_ms();

    ggml_backend_tensor_get(y, y_data, 0, y_count * sizeof(*y_data));
    const float max_abs = reference_max_abs(a_data, b_data, y_data, k, m, n);
    const double total_ms = stopped_ms - started_ms;
    const double ms_per_iteration = total_ms / (double) iterations;
    const double operations = 2.0 * (double) k * (double) m * (double) n;
    const double gflops = operations / (ms_per_iteration * 1000000.0);

    printf("{\"tensorfold\":\"qualcomm-matmul\",\"backend\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,"
           "\"iterations\":%d,\"cpu_threads\":%d,\"ms_per_iteration\":%.6f,\"gflops\":%.6f,"
           "\"max_abs_error\":%.9g,\"passed\":%s}\n",
           argv[1], k, m, n, iterations, cpu_threads, ms_per_iteration, gflops, max_abs,
           max_abs <= 1e-3f ? "true" : "false");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    free(a_data);
    free(b_data);
    free(y_data);
    ggml_backend_free(backend);
    return max_abs <= 1e-3f ? 0 : 8;
}
