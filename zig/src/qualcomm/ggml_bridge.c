#include "ggml_bridge.h"

#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
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

struct tf_head_candidate {
    size_t id;
    float logit;
};

struct tf_head_worker {
    const uint16_t *embedding;
    const float *state;
    size_t hidden;
    size_t begin;
    size_t end;
    struct tf_head_candidate top[5];
};

static void head_insert(struct tf_head_candidate top[5], size_t id, float logit) {
    if (logit <= top[4].logit) {
        return;
    }
    top[4] = (struct tf_head_candidate) { .id = id, .logit = logit };
    for (size_t i = 4; i > 0 && top[i].logit > top[i - 1].logit; --i) {
        const struct tf_head_candidate swap = top[i];
        top[i] = top[i - 1];
        top[i - 1] = swap;
    }
}

static void *head_worker_main(void *opaque) {
    struct tf_head_worker *worker = opaque;
    for (size_t i = 0; i < 5; ++i) {
        worker->top[i] = (struct tf_head_candidate) { .id = 0, .logit = -__builtin_inff() };
    }
    for (size_t token = worker->begin; token < worker->end; ++token) {
        const float16_t *weight = (const float16_t *) (worker->embedding + token * worker->hidden);
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        size_t d = 0;
        for (; d + 8 <= worker->hidden; d += 8) {
            const float16x8_t packed = vld1q_f16(weight + d);
            sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(packed)), vld1q_f32(worker->state + d));
            sum1 = vfmaq_f32(sum1, vcvt_f32_f16(vget_high_f16(packed)), vld1q_f32(worker->state + d + 4));
        }
        float logit = vaddvq_f32(sum0) + vaddvq_f32(sum1);
        for (; d < worker->hidden; ++d) {
            logit += (float) weight[d] * worker->state[d];
        }
        head_insert(worker->top, token, logit);
    }
    return NULL;
}

int tf_qualcomm_tied_head_f16(const unsigned short *embedding,
                              const float *state,
                              size_t vocab,
                              size_t hidden,
                              int cpu_threads,
                              size_t *top_ids,
                              float *top_logits) {
    if (embedding == NULL || state == NULL || top_ids == NULL || top_logits == NULL || vocab == 0 || hidden == 0) {
        return 1;
    }
    size_t thread_count = cpu_threads > 0 ? (size_t) cpu_threads : 1;
    if (thread_count > 8) {
        thread_count = 8;
    }
    if (thread_count > vocab) {
        thread_count = vocab;
    }
    pthread_t threads[8];
    struct tf_head_worker workers[8];
    size_t created = 0;
    for (size_t i = 0; i < thread_count; ++i) {
        workers[i] = (struct tf_head_worker) {
            .embedding = (const uint16_t *) embedding,
            .state = state,
            .hidden = hidden,
            .begin = vocab * i / thread_count,
            .end = vocab * (i + 1) / thread_count,
        };
        if (pthread_create(&threads[i], NULL, head_worker_main, &workers[i]) != 0) {
            break;
        }
        created += 1;
    }
    if (created != thread_count) {
        for (size_t i = 0; i < created; ++i) {
            pthread_join(threads[i], NULL);
        }
        return 2;
    }
    for (size_t i = 0; i < created; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            return 3;
        }
    }
    struct tf_head_candidate top[5];
    for (size_t i = 0; i < 5; ++i) {
        top[i] = (struct tf_head_candidate) { .id = 0, .logit = -__builtin_inff() };
    }
    for (size_t thread = 0; thread < thread_count; ++thread) {
        for (size_t i = 0; i < 5; ++i) {
            head_insert(top, workers[thread].top[i].id, workers[thread].top[i].logit);
        }
    }
    for (size_t i = 0; i < 5; ++i) {
        top_ids[i] = top[i].id;
        top_logits[i] = top[i].logit;
    }
    return 0;
}

static ggml_backend_t init_hexagon_backend(void) {
    ggml_backend_reg_t registry = ggml_backend_hexagon_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
}

#define TF_QUALCOMM_MATMUL_SLOTS 16

struct tf_qualcomm_matmul_slot {
    size_t k;
    size_t m;
    size_t n;
    enum ggml_type weight_type;
    struct ggml_context *ctx;
    struct ggml_tensor *weight;
    struct ggml_tensor *input;
    struct ggml_tensor *output;
    struct ggml_cgraph *graph;
    ggml_backend_buffer_t buffer;
};

struct tf_qualcomm_backend {
    ggml_backend_t backend;
    struct tf_qualcomm_matmul_slot slots[TF_QUALCOMM_MATMUL_SLOTS];
    size_t slot_count;
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
    struct tf_qualcomm_backend *context = calloc(1, sizeof(*context));
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
    for (size_t i = 0; i < context->slot_count; ++i) {
        ggml_backend_buffer_free(context->slots[i].buffer);
        ggml_free(context->slots[i].ctx);
    }
    ggml_backend_free(context->backend);
    free(context);
}

static struct tf_qualcomm_matmul_slot *get_matmul_slot(struct tf_qualcomm_backend *context,
                                                        size_t k,
                                                        size_t m,
                                                        size_t n,
                                                        enum ggml_type weight_type) {
    for (size_t i = 0; i < context->slot_count; ++i) {
        struct tf_qualcomm_matmul_slot *slot = &context->slots[i];
        if (slot->k == k && slot->m == m && slot->n == n && slot->weight_type == weight_type) {
            return slot;
        }
    }
    if (context->slot_count == TF_QUALCOMM_MATMUL_SLOTS) {
        return NULL;
    }

    struct tf_qualcomm_matmul_slot *slot = &context->slots[context->slot_count];
    const struct ggml_init_params params = {
        .mem_size = 4u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) {
        return NULL;
    }
    slot->weight = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) k, (int64_t) m);
    slot->input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
    slot->output = ggml_mul_mat(slot->ctx, slot->weight, slot->input);
    slot->graph = ggml_new_graph_custom(slot->ctx, 8, false);
    ggml_build_forward_expand(slot->graph, slot->output);
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->k = k;
    slot->m = m;
    slot->n = n;
    slot->weight_type = weight_type;
    context->slot_count += 1;
    return slot;
}

static int backend_matmul_typed(void *opaque,
                                size_t k,
                                size_t m,
                                size_t n,
                                size_t iterations,
                                int warmup,
                                enum ggml_type weight_type,
                                const void *weight,
                                const float *input,
                                float *output,
                                double *ms_per_iteration) {
    if (opaque == NULL || iterations == 0) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_matmul_slot *slot = get_matmul_slot(context, k, m, n, weight_type);
    if (slot == NULL) {
        return 2;
    }
    ggml_backend_tensor_set(slot->weight, weight, 0, ggml_row_size(weight_type, k) * m);
    ggml_backend_tensor_set(slot->input, input, 0, k * n * sizeof(*input));
    if (warmup && ggml_backend_graph_compute(context->backend, slot->graph) != GGML_STATUS_SUCCESS) {
        return 4;
    }
    const double started = monotonic_ms();
    for (size_t i = 0; i < iterations; ++i) {
        if (ggml_backend_graph_compute(context->backend, slot->graph) != GGML_STATUS_SUCCESS) {
            return 5;
        }
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(slot->output, output, 0, m * n * sizeof(*output));
    *ms_per_iteration = (stopped - started) / (double) iterations;
    return 0;
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
    return backend_matmul_typed(opaque, k, m, n, iterations, 1, GGML_TYPE_F32, weight, input, output, ms_per_iteration);
}

int tf_qualcomm_backend_matmul_once(void *opaque,
                                    size_t k,
                                    size_t m,
                                    size_t n,
                                    const float *weight,
                                    const float *input,
                                    float *output,
                                    double *compute_ms) {
    return backend_matmul_typed(opaque, k, m, n, 1, 0, GGML_TYPE_F32, weight, input, output, compute_ms);
}

void tf_qualcomm_f32_to_f16(const float *input, unsigned short *output, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const float16x8_t packed = vcombine_f16(vcvt_f16_f32(vld1q_f32(input + i)), vcvt_f16_f32(vld1q_f32(input + i + 4)));
        vst1q_u16(output + i, vreinterpretq_u16_f16(packed));
    }
    for (; i < count; ++i) {
        output[i] = (uint16_t) vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(input[i]))), 0);
    }
}

int tf_qualcomm_backend_matmul_f16_once(void *opaque,
                                        size_t k,
                                        size_t m,
                                        size_t n,
                                        const unsigned short *weight,
                                        const float *input,
                                        float *output,
                                        double *compute_ms) {
    return backend_matmul_typed(opaque, k, m, n, 1, 0, GGML_TYPE_F16, weight, input, output, compute_ms);
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
