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

struct tf_gdn_worker {
    float *recurrent;
    const float *query;
    const float *key;
    const float *value;
    const float *beta;
    const float *decay;
    float *output;
    size_t key_heads;
    size_t value_heads;
    size_t key_width;
    size_t value_width;
    size_t begin_head;
    size_t end_head;
};

static void *gdn_worker_main(void *opaque) {
    struct tf_gdn_worker *worker = opaque;
    const size_t values_per_key = worker->value_heads / worker->key_heads;
    for (size_t head = worker->begin_head; head < worker->end_head; ++head) {
        const size_t key_head = head / values_per_key;
        const float *key_row = worker->key + key_head * worker->key_width;
        const float *query_row = worker->query + key_head * worker->key_width;
        const float32x4_t decay_vector = vdupq_n_f32(worker->decay[head]);
        for (size_t dv = 0; dv < worker->value_width; ++dv) {
            float *row = worker->recurrent + (head * worker->value_width + dv) * worker->key_width;
            float32x4_t memory_sum = vdupq_n_f32(0.0f);
            size_t dk = 0;
            for (; dk + 4 <= worker->key_width; dk += 4) {
                float32x4_t state = vmulq_f32(vld1q_f32(row + dk), decay_vector);
                vst1q_f32(row + dk, state);
                memory_sum = vfmaq_f32(memory_sum, state, vld1q_f32(key_row + dk));
            }
            float memory = vaddvq_f32(memory_sum);
            for (; dk < worker->key_width; ++dk) {
                row[dk] *= worker->decay[head];
                memory += row[dk] * key_row[dk];
            }

            const float delta = (worker->value[head * worker->value_width + dv] - memory) * worker->beta[head];
            const float32x4_t delta_vector = vdupq_n_f32(delta);
            float32x4_t output_sum = vdupq_n_f32(0.0f);
            dk = 0;
            for (; dk + 4 <= worker->key_width; dk += 4) {
                float32x4_t state = vfmaq_f32(vld1q_f32(row + dk), vld1q_f32(key_row + dk), delta_vector);
                vst1q_f32(row + dk, state);
                output_sum = vfmaq_f32(output_sum, state, vld1q_f32(query_row + dk));
            }
            float result = vaddvq_f32(output_sum);
            for (; dk < worker->key_width; ++dk) {
                row[dk] += key_row[dk] * delta;
                result += row[dk] * query_row[dk];
            }
            worker->output[head * worker->value_width + dv] = result;
        }
    }
    return NULL;
}

void tf_qualcomm_gdn_step_f32(float *recurrent,
                              const float *query,
                              const float *key,
                              const float *value,
                              const float *beta,
                              const float *decay,
                              float *output,
                              size_t key_heads,
                              size_t value_heads,
                              size_t key_width,
                              size_t value_width,
                              size_t thread_count) {
    if (thread_count == 0) {
        thread_count = 1;
    }
    if (thread_count > 8) {
        thread_count = 8;
    }
    if (thread_count > value_heads) {
        thread_count = value_heads;
    }
    pthread_t threads[8];
    struct tf_gdn_worker workers[8];
    for (size_t i = 0; i < thread_count; ++i) {
        workers[i] = (struct tf_gdn_worker) {
            .recurrent = recurrent,
            .query = query,
            .key = key,
            .value = value,
            .beta = beta,
            .decay = decay,
            .output = output,
            .key_heads = key_heads,
            .value_heads = value_heads,
            .key_width = key_width,
            .value_width = value_width,
            .begin_head = value_heads * i / thread_count,
            .end_head = value_heads * (i + 1) / thread_count,
        };
    }
    size_t created = 0;
    for (; created < thread_count; ++created) {
        if (pthread_create(&threads[created], NULL, gdn_worker_main, &workers[created]) != 0) {
            break;
        }
    }
    for (size_t i = created; i < thread_count; ++i) {
        gdn_worker_main(&workers[i]);
    }
    for (size_t i = 0; i < created; ++i) {
        pthread_join(threads[i], NULL);
    }
}

void tf_qualcomm_dense_f16(const unsigned short *weight,
                           const float *input,
                           float *output,
                           size_t rows,
                           size_t columns) {
    for (size_t row = 0; row < rows; ++row) {
        const float16_t *row_weight = (const float16_t *) (weight + row * columns);
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        size_t column = 0;
        for (; column + 8 <= columns; column += 8) {
            const float16x8_t packed = vld1q_f16(row_weight + column);
            sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(packed)), vld1q_f32(input + column));
            sum1 = vfmaq_f32(sum1, vcvt_f32_f16(vget_high_f16(packed)), vld1q_f32(input + column + 4));
        }
        float result = vaddvq_f32(sum0) + vaddvq_f32(sum1);
        for (; column < columns; ++column) {
            result += (float) row_weight[column] * input[column];
        }
        output[row] = result;
    }
}

static ggml_backend_t init_hexagon_backend(void) {
    ggml_backend_reg_t registry = ggml_backend_hexagon_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
}

#define TF_QUALCOMM_MATMUL_SLOTS 16
#define TF_QUALCOMM_RESIDENT_SLOTS 256

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
    int initialized;
};

struct tf_qualcomm_resident_slot {
    size_t k;
    size_t m;
    size_t batch_n;
    struct ggml_context *ctx;
    struct ggml_tensor *weight;
    struct ggml_tensor *batch_input;
    struct ggml_tensor *batch_output;
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_input;
    struct ggml_tensor *decode_output;
    struct ggml_cgraph *decode_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weight_loaded;
};

struct tf_qualcomm_resident_group_slot {
    size_t k;
    size_t m[3];
    size_t matrix_count;
    size_t batch_n;
    struct ggml_context *ctx;
    struct ggml_tensor *weight[3];
    struct ggml_tensor *batch_input;
    struct ggml_tensor *batch_output[3];
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_input;
    struct ggml_tensor *decode_output[3];
    struct ggml_cgraph *decode_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
};

struct tf_qualcomm_resident_mlp_slot {
    size_t input_width;
    size_t intermediate_width;
    size_t output_width;
    size_t batch_n;
    struct ggml_context *ctx;
    struct ggml_tensor *gate_weight;
    struct ggml_tensor *up_weight;
    struct ggml_tensor *down_weight;
    struct ggml_tensor *batch_input;
    struct ggml_tensor *batch_output;
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_input;
    struct ggml_tensor *decode_output;
    struct ggml_cgraph *decode_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
};

struct tf_qualcomm_resident_gdn_slot {
    size_t width;
    size_t heads;
    size_t batch_n;
    struct ggml_context *ctx;
    struct ggml_tensor *state;
    struct ggml_tensor *batch_query;
    struct ggml_tensor *batch_key;
    struct ggml_tensor *batch_value;
    struct ggml_tensor *batch_log_decay;
    struct ggml_tensor *batch_beta;
    struct ggml_tensor *batch_result;
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_query;
    struct ggml_tensor *decode_key;
    struct ggml_tensor *decode_value;
    struct ggml_tensor *decode_log_decay;
    struct ggml_tensor *decode_beta;
    struct ggml_tensor *decode_result;
    struct ggml_cgraph *decode_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
};

struct tf_qualcomm_backend {
    ggml_backend_t backend;
    struct tf_qualcomm_matmul_slot slots[TF_QUALCOMM_MATMUL_SLOTS];
    size_t slot_count;
    struct tf_qualcomm_resident_slot resident_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_group_slot resident_group_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_mlp_slot resident_mlp_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_gdn_slot resident_gdn_slots[TF_QUALCOMM_RESIDENT_SLOTS];
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
    for (size_t i = 0; i < TF_QUALCOMM_RESIDENT_SLOTS; ++i) {
        if (context->resident_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_slots[i].buffer);
            ggml_free(context->resident_slots[i].ctx);
        }
        if (context->resident_group_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_group_slots[i].buffer);
            ggml_free(context->resident_group_slots[i].ctx);
        }
        if (context->resident_mlp_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_mlp_slots[i].buffer);
            ggml_free(context->resident_mlp_slots[i].ctx);
        }
        if (context->resident_gdn_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_gdn_slots[i].buffer);
            ggml_free(context->resident_gdn_slots[i].ctx);
        }
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
    slot->initialized = 1;
    context->slot_count += 1;
    return slot;
}

static struct tf_qualcomm_resident_slot *get_resident_matmul_slot(struct tf_qualcomm_backend *context,
                                                                  size_t slot_id,
                                                                  size_t k,
                                                                  size_t m,
                                                                  size_t n) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) {
        return NULL;
    }
    struct tf_qualcomm_resident_slot *slot = &context->resident_slots[slot_id];
    if (slot->initialized) {
        if (slot->k != k || slot->m != m || (n != slot->batch_n && n != 1)) {
            return NULL;
        }
        return slot;
    }

    const struct ggml_init_params params = {
        .mem_size = 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) {
        return NULL;
    }
    slot->weight = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F16, (int64_t) k, (int64_t) m);
    slot->batch_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
    slot->batch_output = ggml_mul_mat(slot->ctx, slot->weight, slot->batch_input);
    slot->batch_graph = ggml_new_graph_custom(slot->ctx, 8, false);
    ggml_build_forward_expand(slot->batch_graph, slot->batch_output);
    if (n != 1) {
        slot->decode_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, 1);
        slot->decode_output = ggml_mul_mat(slot->ctx, slot->weight, slot->decode_input);
        slot->decode_graph = ggml_new_graph_custom(slot->ctx, 8, false);
        ggml_build_forward_expand(slot->decode_graph, slot->decode_output);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->k = k;
    slot->m = m;
    slot->batch_n = n;
    slot->initialized = 1;
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

int tf_qualcomm_backend_matmul_f16_resident_once(void *opaque,
                                                 size_t slot_id,
                                                 size_t k,
                                                 size_t m,
                                                 size_t n,
                                                 const unsigned short *weight,
                                                 const float *input,
                                                 float *output,
                                                 double *compute_ms,
                                                 int *weight_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weight_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_slot *slot = get_resident_matmul_slot(context, slot_id, k, m, n);
    if (slot == NULL) {
        return 2;
    }
    *weight_uploaded = 0;
    if (!slot->weight_loaded) {
        if (weight == NULL) {
            return 3;
        }
        ggml_backend_tensor_set(slot->weight, weight, 0, ggml_row_size(GGML_TYPE_F16, k) * m);
        slot->weight_loaded = 1;
        *weight_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = n == slot->batch_n ? slot->batch_input : slot->decode_input;
    struct ggml_tensor *output_tensor = n == slot->batch_n ? slot->batch_output : slot->decode_output;
    struct ggml_cgraph *graph = n == slot->batch_n ? slot->batch_graph : slot->decode_graph;
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) {
        return 4;
    }
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 5;
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, m * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static struct tf_qualcomm_resident_group_slot *get_resident_group_slot(struct tf_qualcomm_backend *context,
                                                                       size_t slot_id,
                                                                       size_t matrix_count,
                                                                       size_t k,
                                                                       const size_t m[3],
                                                                       size_t n) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS || matrix_count < 2 || matrix_count > 3) {
        return NULL;
    }
    struct tf_qualcomm_resident_group_slot *slot = &context->resident_group_slots[slot_id];
    if (slot->initialized) {
        if (slot->k != k || slot->matrix_count != matrix_count || (n != slot->batch_n && n != 1)) {
            return NULL;
        }
        for (size_t i = 0; i < matrix_count; ++i) {
            if (slot->m[i] != m[i]) {
                return NULL;
            }
        }
        return slot;
    }

    const struct ggml_init_params params = {
        .mem_size = 2u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) {
        return NULL;
    }
    slot->batch_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
    slot->batch_graph = ggml_new_graph_custom(slot->ctx, 16, false);
    for (size_t i = 0; i < matrix_count; ++i) {
        slot->weight[i] = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F16, (int64_t) k, (int64_t) m[i]);
        slot->batch_output[i] = ggml_mul_mat(slot->ctx, slot->weight[i], slot->batch_input);
        ggml_build_forward_expand(slot->batch_graph, slot->batch_output[i]);
    }
    if (n != 1) {
        slot->decode_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, 1);
        slot->decode_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        for (size_t i = 0; i < matrix_count; ++i) {
            slot->decode_output[i] = ggml_mul_mat(slot->ctx, slot->weight[i], slot->decode_input);
            ggml_build_forward_expand(slot->decode_graph, slot->decode_output[i]);
        }
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->k = k;
    slot->matrix_count = matrix_count;
    slot->batch_n = n;
    for (size_t i = 0; i < matrix_count; ++i) {
        slot->m[i] = m[i];
    }
    slot->initialized = 1;
    return slot;
}

int tf_qualcomm_backend_matmul_f16_resident_group_once(void *opaque,
                                                       size_t slot_id,
                                                       size_t matrix_count,
                                                       size_t k,
                                                       size_t m0,
                                                       size_t m1,
                                                       size_t m2,
                                                       size_t n,
                                                       const unsigned short *weight0,
                                                       const unsigned short *weight1,
                                                       const unsigned short *weight2,
                                                       const float *input,
                                                       float *output0,
                                                       float *output1,
                                                       float *output2,
                                                       double *compute_ms,
                                                       int *weights_uploaded) {
    if (opaque == NULL || input == NULL || output0 == NULL || output1 == NULL || compute_ms == NULL || weights_uploaded == NULL) {
        return 1;
    }
    const size_t m[3] = { m0, m1, m2 };
    const uint16_t *weights[3] = { weight0, weight1, weight2 };
    float *outputs[3] = { output0, output1, output2 };
    if (matrix_count == 3 && output2 == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_group_slot *slot = get_resident_group_slot(context, slot_id, matrix_count, k, m, n);
    if (slot == NULL) {
        return 2;
    }
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        for (size_t i = 0; i < matrix_count; ++i) {
            if (weights[i] == NULL) {
                return 3;
            }
            ggml_backend_tensor_set(slot->weight[i], weights[i], 0, ggml_row_size(GGML_TYPE_F16, k) * m[i]);
        }
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }

    struct ggml_tensor *input_tensor = n == slot->batch_n ? slot->batch_input : slot->decode_input;
    struct ggml_tensor **output_tensors = n == slot->batch_n ? slot->batch_output : slot->decode_output;
    struct ggml_cgraph *graph = n == slot->batch_n ? slot->batch_graph : slot->decode_graph;
    if (input_tensor == NULL || graph == NULL) {
        return 4;
    }
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 5;
    }
    const double stopped = monotonic_ms();
    for (size_t i = 0; i < matrix_count; ++i) {
        ggml_backend_tensor_get(output_tensors[i], outputs[i], 0, m[i] * n * sizeof(*outputs[i]));
    }
    *compute_ms = stopped - started;
    return 0;
}

static struct tf_qualcomm_resident_mlp_slot *get_resident_mlp_slot(struct tf_qualcomm_backend *context,
                                                                   size_t slot_id,
                                                                   size_t input_width,
                                                                   size_t intermediate_width,
                                                                   size_t output_width,
                                                                   size_t n) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) {
        return NULL;
    }
    struct tf_qualcomm_resident_mlp_slot *slot = &context->resident_mlp_slots[slot_id];
    if (slot->initialized) {
        if (slot->input_width != input_width || slot->intermediate_width != intermediate_width ||
            slot->output_width != output_width || (n != slot->batch_n && n != 1)) {
            return NULL;
        }
        return slot;
    }

    const struct ggml_init_params params = {
        .mem_size = 2u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) {
        return NULL;
    }
    slot->gate_weight = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F16, (int64_t) input_width, (int64_t) intermediate_width);
    slot->up_weight = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F16, (int64_t) input_width, (int64_t) intermediate_width);
    slot->down_weight = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F16, (int64_t) intermediate_width, (int64_t) output_width);

    slot->batch_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) input_width, (int64_t) n);
    struct ggml_tensor *batch_gate = ggml_mul_mat(slot->ctx, slot->gate_weight, slot->batch_input);
    struct ggml_tensor *batch_up = ggml_mul_mat(slot->ctx, slot->up_weight, slot->batch_input);
    struct ggml_tensor *batch_activated = ggml_mul(slot->ctx, ggml_silu(slot->ctx, batch_gate), batch_up);
    slot->batch_output = ggml_mul_mat(slot->ctx, slot->down_weight, batch_activated);
    slot->batch_graph = ggml_new_graph_custom(slot->ctx, 16, false);
    ggml_build_forward_expand(slot->batch_graph, slot->batch_output);

    if (n != 1) {
        slot->decode_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) input_width, 1);
        struct ggml_tensor *decode_gate = ggml_mul_mat(slot->ctx, slot->gate_weight, slot->decode_input);
        struct ggml_tensor *decode_up = ggml_mul_mat(slot->ctx, slot->up_weight, slot->decode_input);
        struct ggml_tensor *decode_activated = ggml_mul(slot->ctx, ggml_silu(slot->ctx, decode_gate), decode_up);
        slot->decode_output = ggml_mul_mat(slot->ctx, slot->down_weight, decode_activated);
        slot->decode_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        ggml_build_forward_expand(slot->decode_graph, slot->decode_output);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->input_width = input_width;
    slot->intermediate_width = intermediate_width;
    slot->output_width = output_width;
    slot->batch_n = n;
    slot->initialized = 1;
    return slot;
}

int tf_qualcomm_backend_mlp_f16_resident_once(void *opaque,
                                              size_t slot_id,
                                              size_t input_width,
                                              size_t intermediate_width,
                                              size_t output_width,
                                              size_t n,
                                              const unsigned short *gate_weight,
                                              const unsigned short *up_weight,
                                              const unsigned short *down_weight,
                                              const float *input,
                                              float *output,
                                              double *compute_ms,
                                              int *weights_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weights_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_mlp_slot *slot = get_resident_mlp_slot(context, slot_id, input_width, intermediate_width, output_width, n);
    if (slot == NULL) {
        return 2;
    }
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        if (gate_weight == NULL || up_weight == NULL || down_weight == NULL) {
            return 3;
        }
        ggml_backend_tensor_set(slot->gate_weight, gate_weight, 0, ggml_row_size(GGML_TYPE_F16, input_width) * intermediate_width);
        ggml_backend_tensor_set(slot->up_weight, up_weight, 0, ggml_row_size(GGML_TYPE_F16, input_width) * intermediate_width);
        ggml_backend_tensor_set(slot->down_weight, down_weight, 0, ggml_row_size(GGML_TYPE_F16, intermediate_width) * output_width);
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }

    struct ggml_tensor *input_tensor = n == slot->batch_n ? slot->batch_input : slot->decode_input;
    struct ggml_tensor *output_tensor = n == slot->batch_n ? slot->batch_output : slot->decode_output;
    struct ggml_cgraph *graph = n == slot->batch_n ? slot->batch_graph : slot->decode_graph;
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) {
        return 4;
    }
    ggml_backend_tensor_set(input_tensor, input, 0, input_width * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 5;
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, output_width * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static struct ggml_cgraph *build_gdn_graph(struct ggml_context *ctx,
                                           struct ggml_tensor *state,
                                           size_t width,
                                           size_t heads,
                                           size_t n,
                                           struct ggml_tensor **query,
                                           struct ggml_tensor **key,
                                           struct ggml_tensor **value,
                                           struct ggml_tensor **log_decay,
                                           struct ggml_tensor **beta,
                                           struct ggml_tensor **result) {
    *query = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) width, (int64_t) heads, (int64_t) n, 1);
    *key = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) width, (int64_t) heads, (int64_t) n, 1);
    *value = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) width, (int64_t) heads, (int64_t) n, 1);
    *log_decay = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, (int64_t) heads, (int64_t) n, 1);
    *beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, (int64_t) heads, (int64_t) n, 1);
    *result = ggml_gated_delta_net(ctx, *query, *key, *value, *log_decay, *beta, state, 1);
    const size_t output_bytes = width * heads * n * sizeof(float);
    struct ggml_tensor *new_state = ggml_view_4d(ctx,
                                                 *result,
                                                 (int64_t) width,
                                                 (int64_t) width,
                                                 (int64_t) heads,
                                                 1,
                                                 width * sizeof(float),
                                                 width * width * sizeof(float),
                                                 width * width * heads * sizeof(float),
                                                 output_bytes);
    struct ggml_tensor *state_copy = ggml_cpy(ctx, new_state, state);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, state_copy);
    return graph;
}

static struct tf_qualcomm_resident_gdn_slot *get_resident_gdn_slot(struct tf_qualcomm_backend *context,
                                                                   size_t slot_id,
                                                                   size_t width,
                                                                   size_t heads,
                                                                   size_t n) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) {
        return NULL;
    }
    struct tf_qualcomm_resident_gdn_slot *slot = &context->resident_gdn_slots[slot_id];
    if (slot->initialized) {
        if (slot->width != width || slot->heads != heads || (n != slot->batch_n && n != 1)) {
            return NULL;
        }
        return slot;
    }

    const struct ggml_init_params params = {
        .mem_size = 2u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) {
        return NULL;
    }
    slot->state = ggml_new_tensor_4d(slot->ctx, GGML_TYPE_F32, (int64_t) width, (int64_t) width, (int64_t) heads, 1);
    slot->batch_graph = build_gdn_graph(slot->ctx,
                                        slot->state,
                                        width,
                                        heads,
                                        n,
                                        &slot->batch_query,
                                        &slot->batch_key,
                                        &slot->batch_value,
                                        &slot->batch_log_decay,
                                        &slot->batch_beta,
                                        &slot->batch_result);
    if (n != 1) {
        slot->decode_graph = build_gdn_graph(slot->ctx,
                                             slot->state,
                                             width,
                                             heads,
                                             1,
                                             &slot->decode_query,
                                             &slot->decode_key,
                                             &slot->decode_value,
                                             &slot->decode_log_decay,
                                             &slot->decode_beta,
                                             &slot->decode_result);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    slot->width = width;
    slot->heads = heads;
    slot->batch_n = n;
    slot->initialized = 1;
    return slot;
}

int tf_qualcomm_backend_gdn_f32_resident_once(void *opaque,
                                              size_t slot_id,
                                              size_t width,
                                              size_t heads,
                                              size_t n,
                                              const float *query,
                                              const float *key,
                                              const float *value,
                                              const float *log_decay,
                                              const float *beta,
                                              const float *initial_state,
                                              float *output,
                                              double *compute_ms) {
    if (opaque == NULL || query == NULL || key == NULL || value == NULL || log_decay == NULL || beta == NULL ||
        output == NULL || compute_ms == NULL || width == 0 || heads == 0 || n == 0) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_gdn_slot *slot = get_resident_gdn_slot(context, slot_id, width, heads, n);
    if (slot == NULL) {
        return 2;
    }
    if (initial_state != NULL) {
        ggml_backend_tensor_set(slot->state, initial_state, 0, width * width * heads * sizeof(*initial_state));
    }

    struct ggml_tensor *query_tensor = n == slot->batch_n ? slot->batch_query : slot->decode_query;
    struct ggml_tensor *key_tensor = n == slot->batch_n ? slot->batch_key : slot->decode_key;
    struct ggml_tensor *value_tensor = n == slot->batch_n ? slot->batch_value : slot->decode_value;
    struct ggml_tensor *decay_tensor = n == slot->batch_n ? slot->batch_log_decay : slot->decode_log_decay;
    struct ggml_tensor *beta_tensor = n == slot->batch_n ? slot->batch_beta : slot->decode_beta;
    struct ggml_tensor *result_tensor = n == slot->batch_n ? slot->batch_result : slot->decode_result;
    struct ggml_cgraph *graph = n == slot->batch_n ? slot->batch_graph : slot->decode_graph;
    if (query_tensor == NULL || key_tensor == NULL || value_tensor == NULL || decay_tensor == NULL || beta_tensor == NULL ||
        result_tensor == NULL || graph == NULL) {
        return 3;
    }

    const size_t vector_bytes = width * heads * n * sizeof(*query);
    const size_t scalar_bytes = heads * n * sizeof(*log_decay);
    ggml_backend_tensor_set(query_tensor, query, 0, vector_bytes);
    ggml_backend_tensor_set(key_tensor, key, 0, vector_bytes);
    ggml_backend_tensor_set(value_tensor, value, 0, vector_bytes);
    ggml_backend_tensor_set(decay_tensor, log_decay, 0, scalar_bytes);
    ggml_backend_tensor_set(beta_tensor, beta, 0, scalar_bytes);
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 4;
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(result_tensor, output, 0, vector_bytes);
    *compute_ms = stopped - started;
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
