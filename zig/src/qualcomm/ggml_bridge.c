#include "ggml_bridge.h"

#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifndef TF_GGML_VULKAN_ONLY
#include "ggml-hexagon.h"
#endif
#include "ggml.h"
#ifdef TF_GGML_VULKAN
#include "ggml-vulkan.h"
#endif

// Experimental bridge-level storage selector.  The public APIs retain their
// q4 names so the Zig runner and protected Q4 control remain unchanged; an
// opt-in process environment variable lets us exercise Qualcomm's native
// repacked low-bit HMX kernels with the same decoded EXL3 weights.
static enum ggml_type tf_qualcomm_q4_storage_type(void) {
    const char *value = getenv("TENSORFOLD_Q4_STORAGE");
    if (value != NULL) {
        if (strcmp(value, "q3_k") == 0) return GGML_TYPE_Q3_K;
        if (strcmp(value, "q2_k") == 0) return GGML_TYPE_Q2_K;
    }
    return GGML_TYPE_Q4_0;
}

static int tf_qualcomm_is_repacked_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q2_K;
}

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

static inline float32x4_t tf_fast_exp_f32(float32x4_t x) {
    x = vmaxq_f32(vdupq_n_f32(-88.3762626647949f), vminq_f32(x, vdupq_n_f32(88.3762626647949f)));
    float32x4_t fx = vmlaq_n_f32(vdupq_n_f32(0.5f), x, 1.44269504088896341f);
    int32x4_t emm0 = vcvtq_s32_f32(fx);
    float32x4_t tmp = vcvtq_f32_s32(emm0);
    const uint32x4_t correction = vcgtq_f32(tmp, fx);
    tmp = vsubq_f32(tmp, vreinterpretq_f32_u32(vandq_u32(correction, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
    fx = tmp;
    x = vsubq_f32(x, vmulq_n_f32(fx, 0.693359375f));
    x = vsubq_f32(x, vmulq_n_f32(fx, -2.12194440e-4f));
    const float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));
    emm0 = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f));
    emm0 = vshlq_n_s32(emm0, 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(emm0));
}

void tf_qualcomm_silu_norm_f16_fast(const float *gate,
                                    const float *value,
                                    const unsigned char *norm,
                                    float inverse_rms,
                                    float *output,
                                    size_t count) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t inv = vdupq_n_f32(inverse_rms);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t g = vld1q_f32(gate + i);
        const float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, tf_fast_exp_f32(vnegq_f32(g))));
        float norm_values[4];
        memcpy(norm_values, norm + i * sizeof(float), sizeof(norm_values));
        const float32x4_t normalized = vmulq_f32(vmulq_f32(vld1q_f32(norm_values), vld1q_f32(value + i)), inv);
        const float32x4_t activated = vmulq_f32(vmulq_f32(g, sigmoid), normalized);
        vst1q_f32(output + i, vcvt_f32_f16(vcvt_f16_f32(activated)));
    }
    if (i < count) {
        float gate_tail[4] = { 0 };
        float value_tail[4] = { 0 };
        float norm_tail[4] = { 0 };
        float output_tail[4];
        const size_t tail = count - i;
        memcpy(gate_tail, gate + i, tail * sizeof(float));
        memcpy(value_tail, value + i, tail * sizeof(float));
        memcpy(norm_tail, norm + i * sizeof(float), tail * sizeof(float));
        const float32x4_t g = vld1q_f32(gate_tail);
        const float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, tf_fast_exp_f32(vnegq_f32(g))));
        const float32x4_t normalized = vmulq_f32(vmulq_f32(vld1q_f32(norm_tail), vld1q_f32(value_tail)), inv);
        const float32x4_t activated = vmulq_f32(vmulq_f32(g, sigmoid), normalized);
        vst1q_f32(output_tail, vcvt_f32_f16(vcvt_f16_f32(activated)));
        memcpy(output + i, output_tail, tail * sizeof(float));
    }
}

static ggml_backend_t init_hexagon_backend(void) {
#ifdef TF_GGML_VULKAN_ONLY
    return NULL;
#else
    ggml_backend_reg_t registry = ggml_backend_hexagon_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
#endif
}

#ifdef TF_GGML_VULKAN
static ggml_backend_t init_vulkan_backend(void) {
    ggml_backend_reg_t registry = ggml_backend_vk_reg();
    if (registry == NULL || ggml_backend_reg_dev_count(registry) == 0) {
        return NULL;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), NULL);
}
#endif

#define TF_QUALCOMM_MATMUL_SLOTS 16
#define TF_QUALCOMM_RESIDENT_SLOTS 256
#define TF_QUALCOMM_VERIFY_N 3
#define TF_QUALCOMM_VERIFY2_N 2
#define TF_QUALCOMM_VERIFY6_N 6
#define TF_QUALCOMM_VERIFY7_N 7

#define TF_RESIDENT_DYNAMIC_GRAPHS 4

struct tf_qualcomm_resident_variant {
    size_t n;
    struct ggml_context *ctx;
    struct ggml_tensor *input;
    struct ggml_tensor *output;
    struct ggml_cgraph *graph;
    ggml_backend_buffer_t buffer;
};

struct tf_qualcomm_resident_group_variant {
    size_t n;
    struct ggml_context *ctx;
    struct ggml_tensor *input;
    struct ggml_tensor *output[3];
    struct ggml_cgraph *graph;
    ggml_backend_buffer_t buffer;
};

struct tf_qualcomm_resident_tail_variant {
    size_t n;
    struct ggml_context *ctx;
    struct ggml_tensor *state;
    struct ggml_tensor *attention;
    struct ggml_tensor *output;
    struct ggml_cgraph *graph;
    ggml_backend_buffer_t buffer;
};

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
    enum ggml_type weight_type;
    struct ggml_context *ctx;
    struct ggml_tensor *weight;
    struct ggml_tensor *batch_input;
    struct ggml_tensor *batch_output;
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_input;
    struct ggml_tensor *decode_output;
    struct ggml_cgraph *decode_graph;
    struct ggml_tensor *verify_input;
    struct ggml_tensor *verify_output;
    struct ggml_cgraph *verify_graph;
    struct ggml_tensor *verify6_input;
    struct ggml_tensor *verify6_output;
    struct ggml_cgraph *verify6_graph;
    struct ggml_tensor *verify7_input;
    struct ggml_tensor *verify7_output;
    struct ggml_cgraph *verify7_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weight_loaded;
    struct tf_qualcomm_resident_variant dynamic[TF_RESIDENT_DYNAMIC_GRAPHS];
    size_t dynamic_count;
    size_t dynamic_next;
};

struct tf_qualcomm_exl3_slot {
    size_t k;
    size_t m;
    size_t words;
    size_t batch_n;
    struct ggml_context *ctx;
    struct ggml_tensor *trellis;
    struct ggml_tensor *suh;
    struct ggml_tensor *svh;
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
    enum ggml_type weight_type;
    struct ggml_context *ctx;
    struct ggml_tensor *weight[3];
    struct ggml_tensor *batch_input;
    struct ggml_tensor *batch_output[3];
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_input;
    struct ggml_tensor *decode_output[3];
    struct ggml_cgraph *decode_graph;
    struct ggml_tensor *verify_input;
    struct ggml_tensor *verify_output[3];
    struct ggml_cgraph *verify_graph;
    struct ggml_tensor *verify6_input;
    struct ggml_tensor *verify6_output[3];
    struct ggml_cgraph *verify6_graph;
    struct ggml_tensor *verify7_input;
    struct ggml_tensor *verify7_output[3];
    struct ggml_cgraph *verify7_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
    struct tf_qualcomm_resident_group_variant dynamic[TF_RESIDENT_DYNAMIC_GRAPHS];
    size_t dynamic_count;
    size_t dynamic_next;
};

struct tf_qualcomm_resident_mlp_slot {
    size_t input_width;
    size_t intermediate_width;
    size_t output_width;
    size_t batch_n;
    enum ggml_type weight_type;
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
    struct ggml_tensor *verify_input;
    struct ggml_tensor *verify_output;
    struct ggml_cgraph *verify_graph;
    struct ggml_tensor *verify6_input;
    struct ggml_tensor *verify6_output;
    struct ggml_cgraph *verify6_graph;
    struct ggml_tensor *verify7_input;
    struct ggml_tensor *verify7_output;
    struct ggml_cgraph *verify7_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
};

struct tf_qualcomm_resident_tail_slot {
    size_t attention_width;
    size_t hidden_width;
    size_t intermediate_width;
    size_t batch_n;
    enum ggml_type output_weight_type;
    struct ggml_context *ctx;
    struct ggml_tensor *output_weight;
    struct ggml_tensor *gate_weight;
    struct ggml_tensor *up_weight;
    struct ggml_tensor *down_weight;
    struct ggml_tensor *post_norm_weight;
    struct ggml_tensor *batch_state;
    struct ggml_tensor *batch_attention;
    struct ggml_tensor *batch_output;
    struct ggml_cgraph *batch_graph;
    struct ggml_tensor *decode_state;
    struct ggml_tensor *decode_attention;
    struct ggml_tensor *decode_output;
    struct ggml_cgraph *decode_graph;
    struct ggml_tensor *verify_state;
    struct ggml_tensor *verify_attention;
    struct ggml_tensor *verify_output;
    struct ggml_cgraph *verify_graph;
    struct ggml_tensor *verify6_state;
    struct ggml_tensor *verify6_attention;
    struct ggml_tensor *verify6_output;
    struct ggml_cgraph *verify6_graph;
    struct ggml_tensor *verify7_state;
    struct ggml_tensor *verify7_attention;
    struct ggml_tensor *verify7_output;
    struct ggml_cgraph *verify7_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
    struct tf_qualcomm_resident_tail_variant dynamic[TF_RESIDENT_DYNAMIC_GRAPHS];
    size_t dynamic_count;
    size_t dynamic_next;
};

// One compiled graph per row count; verify rounds revisit a slot with
// n in {1,2,3,6,7} after the prompt pass seeded batch_n, so the slot keeps a
// memoized variant per distinct n instead of rejecting the shape.
#define TF_LINEAR_TAIL_MAX_GRAPHS 8

struct tf_qualcomm_linear_tail_variant {
    size_t n;
    struct ggml_tensor *state;
    struct ggml_tensor *z;
    struct ggml_tensor *query;
    struct ggml_tensor *key;
    struct ggml_tensor *value;
    struct ggml_tensor *log_decay;
    struct ggml_tensor *beta;
    struct ggml_tensor *output;
    struct ggml_tensor *gdn_result;
    struct ggml_cgraph *graph;
};

struct tf_qualcomm_resident_linear_tail_slot {
    size_t width;
    size_t heads;
    size_t hidden_width;
    size_t intermediate_width;
    struct ggml_context *ctx;
    struct ggml_tensor *gdn_state;
    struct ggml_tensor *output_weight;
    struct ggml_tensor *gate_weight;
    struct ggml_tensor *up_weight;
    struct ggml_tensor *down_weight;
    struct ggml_tensor *linear_norm_weight;
    struct ggml_tensor *post_norm_weight;
    struct tf_qualcomm_linear_tail_variant variants[TF_LINEAR_TAIL_MAX_GRAPHS];
    size_t variant_count;
    ggml_backend_buffer_t buffer;
    int initialized;
    int weights_loaded;
};

struct tf_qualcomm_gdn_variant {
    size_t n;
    struct ggml_context *ctx;
    struct ggml_tensor *query;
    struct ggml_tensor *key;
    struct ggml_tensor *value;
    struct ggml_tensor *log_decay;
    struct ggml_tensor *beta;
    struct ggml_tensor *result;
    struct ggml_cgraph *graph;
    ggml_backend_buffer_t buffer;
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
    struct ggml_tensor *verify_query;
    struct ggml_tensor *verify_key;
    struct ggml_tensor *verify_value;
    struct ggml_tensor *verify_log_decay;
    struct ggml_tensor *verify_beta;
    struct ggml_tensor *verify_result;
    struct ggml_cgraph *verify_graph;
    struct ggml_tensor *verify2_query;
    struct ggml_tensor *verify2_key;
    struct ggml_tensor *verify2_value;
    struct ggml_tensor *verify2_log_decay;
    struct ggml_tensor *verify2_beta;
    struct ggml_tensor *verify2_result;
    struct ggml_cgraph *verify2_graph;
    struct ggml_tensor *verify6_query;
    struct ggml_tensor *verify6_key;
    struct ggml_tensor *verify6_value;
    struct ggml_tensor *verify6_log_decay;
    struct ggml_tensor *verify6_beta;
    struct ggml_tensor *verify6_result;
    struct ggml_cgraph *verify6_graph;
    struct ggml_tensor *verify7_query;
    struct ggml_tensor *verify7_key;
    struct ggml_tensor *verify7_value;
    struct ggml_tensor *verify7_log_decay;
    struct ggml_tensor *verify7_beta;
    struct ggml_tensor *verify7_result;
    struct ggml_cgraph *verify7_graph;
    ggml_backend_buffer_t buffer;
    int initialized;
    struct tf_qualcomm_gdn_variant dynamic[TF_RESIDENT_DYNAMIC_GRAPHS];
    size_t dynamic_count;
    size_t dynamic_next;
};

struct tf_qualcomm_backend {
    ggml_backend_t backend;
    struct tf_qualcomm_matmul_slot slots[TF_QUALCOMM_MATMUL_SLOTS];
    size_t slot_count;
    struct tf_qualcomm_resident_slot resident_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_exl3_slot exl3_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_group_slot resident_group_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_mlp_slot resident_mlp_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_tail_slot resident_tail_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_linear_tail_slot resident_linear_tail_slots[TF_QUALCOMM_RESIDENT_SLOTS];
    struct tf_qualcomm_resident_gdn_slot resident_gdn_slots[TF_QUALCOMM_RESIDENT_SLOTS];
};

void *tf_qualcomm_backend_create(const char *backend_name, int cpu_threads) {
    const int use_htp = strcmp(backend_name, "htp") == 0;
    const int use_cpu = strcmp(backend_name, "cpu") == 0;
    const int use_vulkan = strcmp(backend_name, "vulkan") == 0;
    if (!use_htp && !use_cpu && !use_vulkan) {
        return NULL;
    }
    ggml_backend_t backend = NULL;
    if (use_htp) {
        backend = init_hexagon_backend();
    } else if (use_cpu) {
        backend = ggml_backend_cpu_init();
    }
#ifdef TF_GGML_VULKAN
    else {
        backend = init_vulkan_backend();
    }
#else
    else {
        return NULL;
    }
#endif
    if (backend == NULL) {
        return NULL;
    }
    if (use_cpu) {
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

void tf_qualcomm_backend_reset_graphs(void *opaque) {
    if (opaque == NULL) {
        return;
    }
    struct tf_qualcomm_backend *context = opaque;
    for (size_t i = 0; i < context->slot_count; ++i) {
        ggml_backend_buffer_free(context->slots[i].buffer);
        ggml_free(context->slots[i].ctx);
    }
    memset(context->slots, 0, sizeof(context->slots));
    context->slot_count = 0;
    for (size_t i = 0; i < TF_QUALCOMM_RESIDENT_SLOTS; ++i) {
        for (size_t j = 0; j < context->resident_slots[i].dynamic_count; ++j) {
            if (context->resident_slots[i].dynamic[j].buffer != NULL) ggml_backend_buffer_free(context->resident_slots[i].dynamic[j].buffer);
            if (context->resident_slots[i].dynamic[j].ctx != NULL) ggml_free(context->resident_slots[i].dynamic[j].ctx);
        }
        for (size_t j = 0; j < context->resident_group_slots[i].dynamic_count; ++j) {
            if (context->resident_group_slots[i].dynamic[j].buffer != NULL) ggml_backend_buffer_free(context->resident_group_slots[i].dynamic[j].buffer);
            if (context->resident_group_slots[i].dynamic[j].ctx != NULL) ggml_free(context->resident_group_slots[i].dynamic[j].ctx);
        }
        for (size_t j = 0; j < context->resident_tail_slots[i].dynamic_count; ++j) {
            if (context->resident_tail_slots[i].dynamic[j].buffer != NULL) ggml_backend_buffer_free(context->resident_tail_slots[i].dynamic[j].buffer);
            if (context->resident_tail_slots[i].dynamic[j].ctx != NULL) ggml_free(context->resident_tail_slots[i].dynamic[j].ctx);
        }
        for (size_t j = 0; j < context->resident_gdn_slots[i].dynamic_count; ++j) {
            if (context->resident_gdn_slots[i].dynamic[j].buffer != NULL) ggml_backend_buffer_free(context->resident_gdn_slots[i].dynamic[j].buffer);
            if (context->resident_gdn_slots[i].dynamic[j].ctx != NULL) ggml_free(context->resident_gdn_slots[i].dynamic[j].ctx);
        }
        if (context->resident_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_slots[i].buffer);
            ggml_free(context->resident_slots[i].ctx);
        }
        if (context->exl3_slots[i].initialized) {
            ggml_backend_buffer_free(context->exl3_slots[i].buffer);
            ggml_free(context->exl3_slots[i].ctx);
        }
        if (context->resident_group_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_group_slots[i].buffer);
            ggml_free(context->resident_group_slots[i].ctx);
        }
        if (context->resident_mlp_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_mlp_slots[i].buffer);
            ggml_free(context->resident_mlp_slots[i].ctx);
        }
        if (context->resident_tail_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_tail_slots[i].buffer);
            ggml_free(context->resident_tail_slots[i].ctx);
        }
        if (context->resident_linear_tail_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_linear_tail_slots[i].buffer);
            ggml_free(context->resident_linear_tail_slots[i].ctx);
        }
        if (context->resident_gdn_slots[i].initialized) {
            ggml_backend_buffer_free(context->resident_gdn_slots[i].buffer);
            ggml_free(context->resident_gdn_slots[i].ctx);
        }
    }
    memset(context->resident_slots, 0, sizeof(context->resident_slots));
    memset(context->exl3_slots, 0, sizeof(context->exl3_slots));
    memset(context->resident_group_slots, 0, sizeof(context->resident_group_slots));
    memset(context->resident_mlp_slots, 0, sizeof(context->resident_mlp_slots));
    memset(context->resident_tail_slots, 0, sizeof(context->resident_tail_slots));
    memset(context->resident_linear_tail_slots, 0, sizeof(context->resident_linear_tail_slots));
    memset(context->resident_gdn_slots, 0, sizeof(context->resident_gdn_slots));
}

void tf_qualcomm_backend_destroy(void *opaque) {
    if (opaque == NULL) {
        return;
    }
    struct tf_qualcomm_backend *context = opaque;
    tf_qualcomm_backend_reset_graphs(context);
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
    if (tf_qualcomm_is_repacked_weight_type(weight_type)) {
        ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    slot->k = k;
    slot->m = m;
    slot->n = n;
    slot->weight_type = weight_type;
    slot->initialized = 1;
    context->slot_count += 1;
    return slot;
}

static struct tf_qualcomm_resident_variant *resident_matmul_dynamic(struct tf_qualcomm_resident_slot *slot, size_t n) {
    for (size_t i = 0; i < slot->dynamic_count; ++i) {
        if (slot->dynamic[i].n == n) return &slot->dynamic[i];
    }
    return NULL;
}

static int build_resident_matmul_dynamic(struct tf_qualcomm_backend *context,
                                         struct tf_qualcomm_resident_slot *slot,
                                         size_t n) {
    if (n == 0) return 1;
    const int replacing = slot->dynamic_count >= TF_RESIDENT_DYNAMIC_GRAPHS;
    const size_t index = replacing ? slot->dynamic_next : slot->dynamic_count;
    struct tf_qualcomm_resident_variant *variant = &slot->dynamic[index];
    if (replacing) {
        if (variant->buffer != NULL) ggml_backend_buffer_free(variant->buffer);
        if (variant->ctx != NULL) ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
    }
    const struct ggml_init_params params = {
        .mem_size = 256u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == NULL) return 1;
    variant->input = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, (int64_t) slot->k, (int64_t) n);
    variant->output = ggml_mul_mat(variant->ctx, slot->weight, variant->input);
    variant->graph = ggml_new_graph_custom(variant->ctx, 8, false);
    ggml_build_forward_expand(variant->graph, variant->output);
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, context->backend);
    if (variant->buffer == NULL) {
        ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
        return 1;
    }
    variant->n = n;
    if (replacing) {
        slot->dynamic_next = (slot->dynamic_next + 1) % TF_RESIDENT_DYNAMIC_GRAPHS;
    } else {
        slot->dynamic_count += 1;
    }
    return 0;
}

static struct tf_qualcomm_resident_slot *get_resident_matmul_slot(struct tf_qualcomm_backend *context,
                                                                  size_t slot_id,
                                                                  size_t k,
                                                                  size_t m,
                                                                  size_t n,
                                                                  enum ggml_type weight_type) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) {
        return NULL;
    }
    struct tf_qualcomm_resident_slot *slot = &context->resident_slots[slot_id];
    if (slot->initialized) {
        if (slot->k != k || slot->m != m || slot->weight_type != weight_type) return NULL;
        if (n == slot->batch_n || n == 1 || n == TF_QUALCOMM_VERIFY_N ||
            n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N ||
            resident_matmul_dynamic(slot, n) != NULL) return slot;
        return build_resident_matmul_dynamic(context, slot, n) == 0 ? slot : NULL;
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
    slot->weight = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) k, (int64_t) m);
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
    if (n != TF_QUALCOMM_VERIFY_N) {
        slot->verify_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY_N);
        slot->verify_output = ggml_mul_mat(slot->ctx, slot->weight, slot->verify_input);
        slot->verify_graph = ggml_new_graph_custom(slot->ctx, 8, false);
        ggml_build_forward_expand(slot->verify_graph, slot->verify_output);
    }
    if (n != TF_QUALCOMM_VERIFY6_N) {
        slot->verify6_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY6_N);
        slot->verify6_output = ggml_mul_mat(slot->ctx, slot->weight, slot->verify6_input);
        slot->verify6_graph = ggml_new_graph_custom(slot->ctx, 8, false);
        ggml_build_forward_expand(slot->verify6_graph, slot->verify6_output);
    }
    if (n != TF_QUALCOMM_VERIFY7_N) {
        slot->verify7_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY7_N);
        slot->verify7_output = ggml_mul_mat(slot->ctx, slot->weight, slot->verify7_input);
        slot->verify7_graph = ggml_new_graph_custom(slot->ctx, 8, false);
        ggml_build_forward_expand(slot->verify7_graph, slot->verify7_output);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    if (tf_qualcomm_is_repacked_weight_type(weight_type)) {
        ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    slot->k = k;
    slot->m = m;
    slot->batch_n = n;
    slot->weight_type = weight_type;
    slot->initialized = 1;
    return slot;
}

static struct ggml_tensor *resident_matmul_input(struct tf_qualcomm_resident_slot *slot, size_t n) {
    struct tf_qualcomm_resident_variant *dynamic = resident_matmul_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->input;
    if (n == slot->batch_n) return slot->batch_input;
    if (n == 1) return slot->decode_input;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_input;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_input;
    return slot->verify_input;
}

static struct ggml_tensor *resident_matmul_output(struct tf_qualcomm_resident_slot *slot, size_t n) {
    struct tf_qualcomm_resident_variant *dynamic = resident_matmul_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->output;
    if (n == slot->batch_n) return slot->batch_output;
    if (n == 1) return slot->decode_output;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_output;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_output;
    return slot->verify_output;
}

static struct ggml_cgraph *resident_matmul_graph(struct tf_qualcomm_resident_slot *slot, size_t n) {
    struct tf_qualcomm_resident_variant *dynamic = resident_matmul_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->graph;
    if (n == slot->batch_n) return slot->batch_graph;
    if (n == 1) return slot->decode_graph;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_graph;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_graph;
    return slot->verify_graph;
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

int tf_qualcomm_backend_matmul_q4(void *opaque,
                                 size_t k,
                                 size_t m,
                                 size_t n,
                                 size_t iterations,
                                 const float *weight,
                                 const float *input,
                                 float *output,
                                 double *ms_per_iteration) {
    if (opaque == NULL || weight == NULL || k % 32 != 0) {
        return 1;
    }
    const size_t quantized_bytes = ggml_row_size(GGML_TYPE_Q4_0, k) * m;
    void *quantized = malloc(quantized_bytes);
    if (quantized == NULL) {
        return 2;
    }
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, weight, quantized, 0, m, k, NULL);
    if (written != quantized_bytes) {
        free(quantized);
        return 3;
    }
    const int status = backend_matmul_typed(
        opaque, k, m, n, iterations, 1, GGML_TYPE_Q4_0, quantized, input, output, ms_per_iteration);
    free(quantized);
    return status;
}

int tf_qualcomm_backend_matmul_q8(void *opaque,
                                 size_t k,
                                 size_t m,
                                 size_t n,
                                 size_t iterations,
                                 const float *weight,
                                 const float *input,
                                 float *output,
                                 double *ms_per_iteration) {
    if (opaque == NULL || weight == NULL || k % 32 != 0) {
        return 1;
    }
    const size_t quantized_bytes = ggml_row_size(GGML_TYPE_Q8_0, k) * m;
    void *quantized = malloc(quantized_bytes);
    if (quantized == NULL) {
        return 2;
    }
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, weight, quantized, 0, m, k, NULL);
    if (written != quantized_bytes) {
        free(quantized);
        return 3;
    }
    const int status = backend_matmul_typed(
        opaque, k, m, n, iterations, 1, GGML_TYPE_Q8_0, quantized, input, output, ms_per_iteration);
    free(quantized);
    return status;
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
    struct tf_qualcomm_resident_slot *slot = get_resident_matmul_slot(context, slot_id, k, m, n, GGML_TYPE_F16);
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
    struct ggml_tensor *input_tensor = resident_matmul_input(slot, n);
    struct ggml_tensor *output_tensor = resident_matmul_output(slot, n);
    struct ggml_cgraph *graph = resident_matmul_graph(slot, n);
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

int tf_qualcomm_backend_matmul_q8_resident_once(void *opaque,
                                                size_t slot_id,
                                                size_t k,
                                                size_t m,
                                                size_t n,
                                                const float *weight,
                                                const float *input,
                                                float *output,
                                                double *compute_ms,
                                                int *weight_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weight_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_slot *slot = get_resident_matmul_slot(context, slot_id, k, m, n, GGML_TYPE_Q8_0);
    if (slot == NULL) {
        return 2;
    }
    *weight_uploaded = 0;
    if (!slot->weight_loaded) {
        if (weight == NULL) {
            return 3;
        }
        const size_t quantized_bytes = ggml_row_size(GGML_TYPE_Q8_0, k) * m;
        void *quantized = malloc(quantized_bytes);
        if (quantized == NULL) {
            return 4;
        }
        const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, weight, quantized, 0, m, k, NULL);
        if (written != quantized_bytes) {
            free(quantized);
            return 5;
        }
        ggml_backend_tensor_set(slot->weight, quantized, 0, quantized_bytes);
        free(quantized);
        slot->weight_loaded = 1;
        *weight_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_matmul_input(slot, n);
    struct ggml_tensor *output_tensor = resident_matmul_output(slot, n);
    struct ggml_cgraph *graph = resident_matmul_graph(slot, n);
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) {
        return 6;
    }
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 7;
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, m * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static int set_quant_weight_from_f16(struct ggml_tensor *tensor,
                                     const uint16_t *weight,
                                     size_t k,
                                     size_t m,
                                     enum ggml_type type) {
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t quantized_bytes = row_bytes * m;
    void *quantized = malloc(quantized_bytes);
    if (quantized == NULL) {
        return 1;
    }

    const size_t chunk_rows = 256;
    float *converted = malloc(chunk_rows * k * sizeof(*converted));
    if (converted == NULL) {
        free(quantized);
        return 2;
    }
    for (size_t row = 0; row < m; row += chunk_rows) {
        const size_t rows = m - row < chunk_rows ? m - row : chunk_rows;
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) weight + row * k,
                              converted,
                              (int64_t) (rows * k));
        const size_t written = ggml_quantize_chunk(type,
                                                   converted,
                                                   (char *) quantized + row * row_bytes,
                                                   0,
                                                   rows,
                                                   k,
                                                   NULL);
        if (written != rows * row_bytes) {
            free(converted);
            free(quantized);
            return 3;
        }
    }
    free(converted);
    ggml_backend_tensor_set(tensor, quantized, 0, quantized_bytes);
    free(quantized);
    return 0;
}

static int matmul_quant_f16_source_resident_once(void *opaque,
                                                 size_t slot_id,
                                                 size_t k,
                                                 size_t m,
                                                 size_t n,
                                                 const unsigned short *weight,
                                                 const float *input,
                                                 float *output,
                                                 double *compute_ms,
                                                 int *weight_uploaded,
                                                 enum ggml_type type) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weight_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_slot *slot = get_resident_matmul_slot(context, slot_id, k, m, n, type);
    if (slot == NULL) {
        return 2;
    }
    *weight_uploaded = 0;
    if (!slot->weight_loaded) {
        if (weight == NULL || set_quant_weight_from_f16(slot->weight, weight, k, m, type) != 0) {
            return 3;
        }
        slot->weight_loaded = 1;
        *weight_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_matmul_input(slot, n);
    struct ggml_tensor *output_tensor = resident_matmul_output(slot, n);
    struct ggml_cgraph *graph = resident_matmul_graph(slot, n);
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

int tf_qualcomm_backend_matmul_q8_f16_source_resident_once(void *opaque,
                                                           size_t slot_id,
                                                           size_t k,
                                                           size_t m,
                                                           size_t n,
                                                           const unsigned short *weight,
                                                           const float *input,
                                                           float *output,
                                                           double *compute_ms,
                                                           int *weight_uploaded) {
    return matmul_quant_f16_source_resident_once(opaque, slot_id, k, m, n, weight, input, output,
                                                  compute_ms, weight_uploaded, GGML_TYPE_Q8_0);
}

int tf_qualcomm_backend_matmul_q4_f16_source_resident_once(void *opaque,
                                                           size_t slot_id,
                                                           size_t k,
                                                           size_t m,
                                                           size_t n,
                                                           const unsigned short *weight,
                                                           const float *input,
                                                           float *output,
                                                           double *compute_ms,
                                                           int *weight_uploaded) {
    return matmul_quant_f16_source_resident_once(opaque, slot_id, k, m, n, weight, input, output,
                                                  compute_ms, weight_uploaded, GGML_TYPE_Q4_0);
}

int tf_qualcomm_backend_matmul_q4_resident_once(void *opaque,
                                                size_t slot_id,
                                                size_t k,
                                                size_t m,
                                                size_t n,
                                                const float *weight,
                                                const float *input,
                                                float *output,
                                                double *compute_ms,
                                                int *weight_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weight_uploaded == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    struct tf_qualcomm_resident_slot *slot = get_resident_matmul_slot(context, slot_id, k, m, n, qtype);
    if (slot == NULL) return 2;
    *weight_uploaded = 0;
    if (!slot->weight_loaded) {
        if (weight == NULL) return 3;
        const size_t bytes = ggml_row_size(qtype, k) * m;
        void *quantized = malloc(bytes);
        if (quantized == NULL) return 4;
        const size_t written = ggml_quantize_chunk(qtype, weight, quantized, 0, m, k, NULL);
        if (written != bytes) { free(quantized); return 5; }
        ggml_backend_tensor_set(slot->weight, quantized, 0, bytes);
        free(quantized);
        slot->weight_loaded = 1;
        *weight_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_matmul_input(slot, n);
    struct ggml_tensor *output_tensor = resident_matmul_output(slot, n);
    struct ggml_cgraph *graph = resident_matmul_graph(slot, n);
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) return 6;
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 7;
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, m * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static struct tf_qualcomm_exl3_slot *get_exl3_slot(struct tf_qualcomm_backend *context,
                                                    size_t slot_id,
                                                    size_t k,
                                                    size_t m,
                                                    size_t n,
                                                    size_t words) {
#ifdef TF_GGML_VULKAN_ONLY
    (void) context;
    (void) slot_id;
    (void) k;
    (void) m;
    (void) n;
    (void) words;
    return NULL;
#else
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS || words < 8 || words > 128 || words % 8 != 0 ||
        k % 128 != 0 || m % 128 != 0) {
        return NULL;
    }
    struct tf_qualcomm_exl3_slot *slot = &context->exl3_slots[slot_id];
    if (slot->initialized) {
        if (slot->k != k || slot->m != m || slot->words != words || (n != slot->batch_n && n != 1)) {
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
    slot->trellis = ggml_new_tensor_3d(slot->ctx, GGML_TYPE_I16, (int64_t) words, (int64_t) (m / 16), (int64_t) (k / 16));
    slot->suh = ggml_new_tensor_1d(slot->ctx, GGML_TYPE_F16, (int64_t) k);
    slot->svh = ggml_new_tensor_1d(slot->ctx, GGML_TYPE_F16, (int64_t) m);
    slot->batch_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, (int64_t) n);
    slot->batch_output = ggml_hexagon_exl3_mul_mat(slot->ctx, slot->trellis, slot->suh, slot->svh, slot->batch_input);
    slot->batch_graph = ggml_new_graph_custom(slot->ctx, 8, false);
    ggml_build_forward_expand(slot->batch_graph, slot->batch_output);
    if (n != 1) {
        slot->decode_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, 1);
        slot->decode_output = ggml_hexagon_exl3_mul_mat(slot->ctx, slot->trellis, slot->suh, slot->svh, slot->decode_input);
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
    slot->words = words;
    slot->batch_n = n;
    slot->initialized = 1;
    return slot;
#endif
}

int tf_qualcomm_backend_exl3_resident_once(void *opaque,
                                           size_t slot_id,
                                           size_t k,
                                           size_t m,
                                           size_t n,
                                           size_t words,
                                           const unsigned short *trellis,
                                           const unsigned short *suh,
                                           const unsigned short *svh,
                                           const float *input,
                                           float *output,
                                           double *compute_ms,
                                           int *weight_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weight_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_exl3_slot *slot = get_exl3_slot(context, slot_id, k, m, n, words);
    if (slot == NULL) {
        return 2;
    }
    *weight_uploaded = 0;
    if (!slot->weight_loaded) {
        if (trellis == NULL || suh == NULL || svh == NULL) {
            return 3;
        }
        const size_t trellis_bytes = (k / 16) * (m / 16) * words * sizeof(*trellis);
        ggml_backend_tensor_set(slot->trellis, trellis, 0, trellis_bytes);
        ggml_backend_tensor_set(slot->suh, suh, 0, k * sizeof(*suh));
        ggml_backend_tensor_set(slot->svh, svh, 0, m * sizeof(*svh));
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

int tf_qualcomm_backend_exl3_paged_once(void *opaque,
                                        size_t slot_id,
                                        size_t k,
                                        size_t m,
                                        size_t n,
                                        size_t words,
                                        const unsigned short *trellis,
                                        const unsigned short *suh,
                                        const unsigned short *svh,
                                        const float *input,
                                        float *output,
                                        double *compute_ms) {
    if (opaque == NULL || trellis == NULL || suh == NULL || svh == NULL ||
        input == NULL || output == NULL || compute_ms == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_exl3_slot *slot = get_exl3_slot(context, slot_id, k, m, n, words);
    if (slot == NULL) return 2;
    const size_t trellis_bytes = (k / 16) * (m / 16) * words * sizeof(*trellis);
    ggml_backend_tensor_set(slot->trellis, trellis, 0, trellis_bytes);
    ggml_backend_tensor_set(slot->suh, suh, 0, k * sizeof(*suh));
    ggml_backend_tensor_set(slot->svh, svh, 0, m * sizeof(*svh));
    struct ggml_tensor *input_tensor = n == slot->batch_n ? slot->batch_input : slot->decode_input;
    struct ggml_tensor *output_tensor = n == slot->batch_n ? slot->batch_output : slot->decode_output;
    struct ggml_cgraph *graph = n == slot->batch_n ? slot->batch_graph : slot->decode_graph;
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) return 3;
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 4;
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, m * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static struct tf_qualcomm_resident_group_variant *resident_group_dynamic(struct tf_qualcomm_resident_group_slot *slot,
                                                                         size_t n) {
    for (size_t i = 0; i < slot->dynamic_count; ++i) {
        if (slot->dynamic[i].n == n) return &slot->dynamic[i];
    }
    return NULL;
}

static int build_resident_group_dynamic(struct tf_qualcomm_backend *context,
                                        struct tf_qualcomm_resident_group_slot *slot,
                                        size_t n) {
    if (n == 0) return 1;
    const int replacing = slot->dynamic_count >= TF_RESIDENT_DYNAMIC_GRAPHS;
    const size_t index = replacing ? slot->dynamic_next : slot->dynamic_count;
    struct tf_qualcomm_resident_group_variant *variant = &slot->dynamic[index];
    if (replacing) {
        if (variant->buffer != NULL) ggml_backend_buffer_free(variant->buffer);
        if (variant->ctx != NULL) ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
    }
    const struct ggml_init_params params = {
        .mem_size = 512u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == NULL) return 1;
    variant->input = ggml_new_tensor_2d(variant->ctx, GGML_TYPE_F32, (int64_t) slot->k, (int64_t) n);
    variant->graph = ggml_new_graph_custom(variant->ctx, 16, false);
    for (size_t i = 0; i < slot->matrix_count; ++i) {
        variant->output[i] = ggml_mul_mat(variant->ctx, slot->weight[i], variant->input);
        ggml_build_forward_expand(variant->graph, variant->output[i]);
    }
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, context->backend);
    if (variant->buffer == NULL) {
        ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
        return 1;
    }
    variant->n = n;
    if (replacing) {
        slot->dynamic_next = (slot->dynamic_next + 1) % TF_RESIDENT_DYNAMIC_GRAPHS;
    } else {
        slot->dynamic_count += 1;
    }
    return 0;
}

static struct tf_qualcomm_resident_group_slot *get_resident_group_slot(struct tf_qualcomm_backend *context,
                                                                       size_t slot_id,
                                                                       size_t matrix_count,
                                                                       size_t k,
                                                                       const size_t m[3],
                                                                       size_t n,
                                                                       enum ggml_type weight_type) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS || matrix_count < 2 || matrix_count > 3) {
        return NULL;
    }
    struct tf_qualcomm_resident_group_slot *slot = &context->resident_group_slots[slot_id];
    if (slot->initialized) {
        if (slot->k != k || slot->matrix_count != matrix_count || slot->weight_type != weight_type) return NULL;
        for (size_t i = 0; i < matrix_count; ++i) {
            if (slot->m[i] != m[i]) return NULL;
        }
        if (n == slot->batch_n || n == 1 || n == TF_QUALCOMM_VERIFY_N ||
            n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N ||
            resident_group_dynamic(slot, n) != NULL) return slot;
        return build_resident_group_dynamic(context, slot, n) == 0 ? slot : NULL;
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
        slot->weight[i] = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) k, (int64_t) m[i]);
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
    if (n != TF_QUALCOMM_VERIFY_N) {
        slot->verify_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY_N);
        slot->verify_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        for (size_t i = 0; i < matrix_count; ++i) {
            slot->verify_output[i] = ggml_mul_mat(slot->ctx, slot->weight[i], slot->verify_input);
            ggml_build_forward_expand(slot->verify_graph, slot->verify_output[i]);
        }
    }
    if (n != TF_QUALCOMM_VERIFY6_N) {
        slot->verify6_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY6_N);
        slot->verify6_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        for (size_t i = 0; i < matrix_count; ++i) {
            slot->verify6_output[i] = ggml_mul_mat(slot->ctx, slot->weight[i], slot->verify6_input);
            ggml_build_forward_expand(slot->verify6_graph, slot->verify6_output[i]);
        }
    }
    if (n != TF_QUALCOMM_VERIFY7_N) {
        slot->verify7_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) k, TF_QUALCOMM_VERIFY7_N);
        slot->verify7_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        for (size_t i = 0; i < matrix_count; ++i) {
            slot->verify7_output[i] = ggml_mul_mat(slot->ctx, slot->weight[i], slot->verify7_input);
            ggml_build_forward_expand(slot->verify7_graph, slot->verify7_output[i]);
        }
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    if (tf_qualcomm_is_repacked_weight_type(weight_type)) {
        ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    slot->k = k;
    slot->matrix_count = matrix_count;
    slot->batch_n = n;
    slot->weight_type = weight_type;
    for (size_t i = 0; i < matrix_count; ++i) {
        slot->m[i] = m[i];
    }
    slot->initialized = 1;
    return slot;
}

static struct ggml_tensor *resident_group_input(struct tf_qualcomm_resident_group_slot *slot, size_t n) {
    struct tf_qualcomm_resident_group_variant *dynamic = resident_group_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->input;
    if (n == slot->batch_n) return slot->batch_input;
    if (n == 1) return slot->decode_input;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_input;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_input;
    return slot->verify_input;
}

static struct ggml_tensor **resident_group_outputs(struct tf_qualcomm_resident_group_slot *slot, size_t n) {
    struct tf_qualcomm_resident_group_variant *dynamic = resident_group_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->output;
    if (n == slot->batch_n) return slot->batch_output;
    if (n == 1) return slot->decode_output;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_output;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_output;
    return slot->verify_output;
}

static struct ggml_cgraph *resident_group_graph(struct tf_qualcomm_resident_group_slot *slot, size_t n) {
    struct tf_qualcomm_resident_group_variant *dynamic = resident_group_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->graph;
    if (n == slot->batch_n) return slot->batch_graph;
    if (n == 1) return slot->decode_graph;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_graph;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_graph;
    return slot->verify_graph;
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
    struct tf_qualcomm_resident_group_slot *slot = get_resident_group_slot(context, slot_id, matrix_count, k, m, n, GGML_TYPE_F16);
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

    struct ggml_tensor *input_tensor = resident_group_input(slot, n);
    struct ggml_tensor **output_tensors = resident_group_outputs(slot, n);
    struct ggml_cgraph *graph = resident_group_graph(slot, n);
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

static int set_quant_weight(struct ggml_tensor *tensor, const float *weight, size_t k, size_t m, enum ggml_type type) {
    const size_t bytes = ggml_row_size(type, k) * m;
    void *quantized = malloc(bytes);
    if (quantized == NULL) {
        return 1;
    }
    const size_t written = ggml_quantize_chunk(type, weight, quantized, 0, m, k, NULL);
    if (written != bytes) {
        free(quantized);
        return 2;
    }
    ggml_backend_tensor_set(tensor, quantized, 0, bytes);
    free(quantized);
    return 0;
}

static int set_q8_weight(struct ggml_tensor *tensor, const float *weight, size_t k, size_t m) {
    return set_quant_weight(tensor, weight, k, m, GGML_TYPE_Q8_0);
}

int tf_qualcomm_backend_matmul_q8_resident_group_once(void *opaque,
                                                      size_t slot_id,
                                                      size_t matrix_count,
                                                      size_t k,
                                                      size_t m0,
                                                      size_t m1,
                                                      size_t m2,
                                                      size_t n,
                                                      const float *weight0,
                                                      const float *weight1,
                                                      const float *weight2,
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
    const float *weights[3] = { weight0, weight1, weight2 };
    float *outputs[3] = { output0, output1, output2 };
    if (matrix_count == 3 && output2 == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_group_slot *slot = get_resident_group_slot(
        context, slot_id, matrix_count, k, m, n, GGML_TYPE_Q8_0);
    if (slot == NULL) {
        return 2;
    }
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        for (size_t i = 0; i < matrix_count; ++i) {
            if (weights[i] == NULL || set_q8_weight(slot->weight[i], weights[i], k, m[i]) != 0) {
                return 3;
            }
        }
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_group_input(slot, n);
    struct ggml_tensor **output_tensors = resident_group_outputs(slot, n);
    struct ggml_cgraph *graph = resident_group_graph(slot, n);
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

int tf_qualcomm_backend_matmul_q4_resident_group_once(void *opaque,
                                                      size_t slot_id,
                                                      size_t matrix_count,
                                                      size_t k,
                                                      size_t m0,
                                                      size_t m1,
                                                      size_t m2,
                                                      size_t n,
                                                      const float *weight0,
                                                      const float *weight1,
                                                      const float *weight2,
                                                      const float *input,
                                                      float *output0,
                                                      float *output1,
                                                      float *output2,
                                                      double *compute_ms,
                                                      int *weights_uploaded) {
    if (opaque == NULL || input == NULL || output0 == NULL || output1 == NULL || compute_ms == NULL || weights_uploaded == NULL) return 1;
    const size_t m[3] = { m0, m1, m2 };
    const float *weights[3] = { weight0, weight1, weight2 };
    float *outputs[3] = { output0, output1, output2 };
    if (matrix_count == 3 && output2 == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    struct tf_qualcomm_resident_group_slot *slot = get_resident_group_slot(context, slot_id, matrix_count, k, m, n, qtype);
    if (slot == NULL) return 2;
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        for (size_t i = 0; i < matrix_count; ++i) {
            if (weights[i] == NULL || set_quant_weight(slot->weight[i], weights[i], k, m[i], qtype) != 0) return 3;
        }
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_group_input(slot, n);
    struct ggml_tensor **output_tensors = resident_group_outputs(slot, n);
    struct ggml_cgraph *graph = resident_group_graph(slot, n);
    if (input_tensor == NULL || graph == NULL) return 4;
    ggml_backend_tensor_set(input_tensor, input, 0, k * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 5;
    const double stopped = monotonic_ms();
    for (size_t i = 0; i < matrix_count; ++i) ggml_backend_tensor_get(output_tensors[i], outputs[i], 0, m[i] * n * sizeof(*outputs[i]));
    *compute_ms = stopped - started;
    return 0;
}

static struct tf_qualcomm_resident_mlp_slot *get_resident_mlp_slot(struct tf_qualcomm_backend *context,
                                                                   size_t slot_id,
                                                                   size_t input_width,
                                                                   size_t intermediate_width,
                                                                   size_t output_width,
                                                                   size_t n,
                                                                   enum ggml_type weight_type) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) {
        return NULL;
    }
    struct tf_qualcomm_resident_mlp_slot *slot = &context->resident_mlp_slots[slot_id];
    if (slot->initialized) {
        if (slot->input_width != input_width || slot->intermediate_width != intermediate_width ||
            slot->output_width != output_width || slot->weight_type != weight_type ||
            (n != slot->batch_n && n != 1 && n != TF_QUALCOMM_VERIFY2_N && n != TF_QUALCOMM_VERIFY_N && n != TF_QUALCOMM_VERIFY6_N && n != TF_QUALCOMM_VERIFY7_N)) {
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
    slot->gate_weight = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) input_width, (int64_t) intermediate_width);
    slot->up_weight = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) input_width, (int64_t) intermediate_width);
    slot->down_weight = ggml_new_tensor_2d(slot->ctx, weight_type, (int64_t) intermediate_width, (int64_t) output_width);

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
    if (n != TF_QUALCOMM_VERIFY_N) {
        slot->verify_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) input_width, TF_QUALCOMM_VERIFY_N);
        struct ggml_tensor *verify_gate = ggml_mul_mat(slot->ctx, slot->gate_weight, slot->verify_input);
        struct ggml_tensor *verify_up = ggml_mul_mat(slot->ctx, slot->up_weight, slot->verify_input);
        struct ggml_tensor *verify_activated = ggml_mul(slot->ctx, ggml_silu(slot->ctx, verify_gate), verify_up);
        slot->verify_output = ggml_mul_mat(slot->ctx, slot->down_weight, verify_activated);
        slot->verify_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        ggml_build_forward_expand(slot->verify_graph, slot->verify_output);
    }
    if (n != TF_QUALCOMM_VERIFY6_N) {
        slot->verify6_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) input_width, TF_QUALCOMM_VERIFY6_N);
        struct ggml_tensor *verify6_gate = ggml_mul_mat(slot->ctx, slot->gate_weight, slot->verify6_input);
        struct ggml_tensor *verify6_up = ggml_mul_mat(slot->ctx, slot->up_weight, slot->verify6_input);
        struct ggml_tensor *verify6_activated = ggml_mul(slot->ctx, ggml_silu(slot->ctx, verify6_gate), verify6_up);
        slot->verify6_output = ggml_mul_mat(slot->ctx, slot->down_weight, verify6_activated);
        slot->verify6_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        ggml_build_forward_expand(slot->verify6_graph, slot->verify6_output);
    }
    if (n != TF_QUALCOMM_VERIFY7_N) {
        slot->verify7_input = ggml_new_tensor_2d(slot->ctx, GGML_TYPE_F32, (int64_t) input_width, TF_QUALCOMM_VERIFY7_N);
        struct ggml_tensor *verify7_gate = ggml_mul_mat(slot->ctx, slot->gate_weight, slot->verify7_input);
        struct ggml_tensor *verify7_up = ggml_mul_mat(slot->ctx, slot->up_weight, slot->verify7_input);
        struct ggml_tensor *verify7_activated = ggml_mul(slot->ctx, ggml_silu(slot->ctx, verify7_gate), verify7_up);
        slot->verify7_output = ggml_mul_mat(slot->ctx, slot->down_weight, verify7_activated);
        slot->verify7_graph = ggml_new_graph_custom(slot->ctx, 16, false);
        ggml_build_forward_expand(slot->verify7_graph, slot->verify7_output);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    if (tf_qualcomm_is_repacked_weight_type(weight_type)) {
        ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    slot->input_width = input_width;
    slot->intermediate_width = intermediate_width;
    slot->output_width = output_width;
    slot->batch_n = n;
    slot->weight_type = weight_type;
    slot->initialized = 1;
    return slot;
}

static struct ggml_tensor *resident_mlp_input(struct tf_qualcomm_resident_mlp_slot *slot, size_t n) {
    if (n == slot->batch_n) return slot->batch_input;
    if (n == 1) return slot->decode_input;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_input;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_input;
    return slot->verify_input;
}

static struct ggml_tensor *resident_mlp_output(struct tf_qualcomm_resident_mlp_slot *slot, size_t n) {
    if (n == slot->batch_n) return slot->batch_output;
    if (n == 1) return slot->decode_output;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_output;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_output;
    return slot->verify_output;
}

static struct ggml_cgraph *resident_mlp_graph(struct tf_qualcomm_resident_mlp_slot *slot, size_t n) {
    if (n == slot->batch_n) return slot->batch_graph;
    if (n == 1) return slot->decode_graph;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_graph;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_graph;
    return slot->verify_graph;
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
    struct tf_qualcomm_resident_mlp_slot *slot = get_resident_mlp_slot(context, slot_id, input_width, intermediate_width, output_width, n, GGML_TYPE_F16);
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

    struct ggml_tensor *input_tensor = resident_mlp_input(slot, n);
    struct ggml_tensor *output_tensor = resident_mlp_output(slot, n);
    struct ggml_cgraph *graph = resident_mlp_graph(slot, n);
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

int tf_qualcomm_backend_mlp_q8_resident_once(void *opaque,
                                             size_t slot_id,
                                             size_t input_width,
                                             size_t intermediate_width,
                                             size_t output_width,
                                             size_t n,
                                             const float *gate_weight,
                                             const float *up_weight,
                                             const float *down_weight,
                                             const float *input,
                                             float *output,
                                             double *compute_ms,
                                             int *weights_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weights_uploaded == NULL) {
        return 1;
    }
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_mlp_slot *slot = get_resident_mlp_slot(
        context, slot_id, input_width, intermediate_width, output_width, n, GGML_TYPE_Q8_0);
    if (slot == NULL) {
        return 2;
    }
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        if (gate_weight == NULL || up_weight == NULL || down_weight == NULL) {
            return 3;
        }
        if (set_q8_weight(slot->gate_weight, gate_weight, input_width, intermediate_width) != 0 ||
            set_q8_weight(slot->up_weight, up_weight, input_width, intermediate_width) != 0 ||
            set_q8_weight(slot->down_weight, down_weight, intermediate_width, output_width) != 0) {
            return 4;
        }
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_mlp_input(slot, n);
    struct ggml_tensor *output_tensor = resident_mlp_output(slot, n);
    struct ggml_cgraph *graph = resident_mlp_graph(slot, n);
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) {
        return 5;
    }
    ggml_backend_tensor_set(input_tensor, input, 0, input_width * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) {
        return 6;
    }
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, output_width * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

int tf_qualcomm_backend_mlp_q4_resident_once(void *opaque,
                                             size_t slot_id,
                                             size_t input_width,
                                             size_t intermediate_width,
                                             size_t output_width,
                                             size_t n,
                                             const float *gate_weight,
                                             const float *up_weight,
                                             const float *down_weight,
                                             const float *input,
                                             float *output,
                                             double *compute_ms,
                                             int *weights_uploaded) {
    if (opaque == NULL || input == NULL || output == NULL || compute_ms == NULL || weights_uploaded == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    struct tf_qualcomm_resident_mlp_slot *slot = get_resident_mlp_slot(context, slot_id, input_width, intermediate_width, output_width, n, qtype);
    if (slot == NULL) return 2;
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        if (gate_weight == NULL || up_weight == NULL || down_weight == NULL) return 3;
        if (set_quant_weight(slot->gate_weight, gate_weight, input_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->up_weight, up_weight, input_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->down_weight, down_weight, intermediate_width, output_width, qtype) != 0) return 4;
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }
    struct ggml_tensor *input_tensor = resident_mlp_input(slot, n);
    struct ggml_tensor *output_tensor = resident_mlp_output(slot, n);
    struct ggml_cgraph *graph = resident_mlp_graph(slot, n);
    if (input_tensor == NULL || output_tensor == NULL || graph == NULL) return 5;
    ggml_backend_tensor_set(input_tensor, input, 0, input_width * n * sizeof(*input));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 6;
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, output, 0, output_width * n * sizeof(*output));
    *compute_ms = stopped - started;
    return 0;
}

static struct ggml_tensor *round_f16_to_f32(struct ggml_context *ctx, struct ggml_tensor *value) {
    return ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_F16), GGML_TYPE_F32);
}

static void build_layer_tail_graph(struct ggml_context *ctx,
                                   struct tf_qualcomm_resident_tail_slot *slot,
                                   size_t n,
                                   struct ggml_tensor **state,
                                   struct ggml_tensor **attention,
                                   struct ggml_tensor **output) {
    *state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) slot->hidden_width, (int64_t) n);
    *attention = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) slot->attention_width, (int64_t) n);

    struct ggml_tensor *attention_update = ggml_mul_mat(ctx, slot->output_weight, *attention);
    struct ggml_tensor *post_attention = round_f16_to_f32(ctx, ggml_add(ctx, *state, attention_update));
    struct ggml_tensor *normalized = ggml_rms_norm(ctx, post_attention, 1.0e-6f);
    struct ggml_tensor *scaled = ggml_mul(ctx, normalized, slot->post_norm_weight);
    struct ggml_tensor *mlp_input = round_f16_to_f32(ctx, scaled);
    struct ggml_tensor *gate = ggml_mul_mat(ctx, slot->gate_weight, mlp_input);
    struct ggml_tensor *up = ggml_mul_mat(ctx, slot->up_weight, mlp_input);
    struct ggml_tensor *activated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    struct ggml_tensor *mlp_update = ggml_mul_mat(ctx, slot->down_weight, activated);
    *output = round_f16_to_f32(ctx, ggml_add(ctx, post_attention, mlp_update));

}

static struct tf_qualcomm_resident_tail_variant *resident_tail_dynamic(struct tf_qualcomm_resident_tail_slot *slot,
                                                                       size_t n) {
    for (size_t i = 0; i < slot->dynamic_count; ++i) {
        if (slot->dynamic[i].n == n) return &slot->dynamic[i];
    }
    return NULL;
}

static int build_resident_tail_dynamic(struct tf_qualcomm_backend *context,
                                       struct tf_qualcomm_resident_tail_slot *slot,
                                       size_t n) {
    if (n == 0) return 1;
    const int replacing = slot->dynamic_count >= TF_RESIDENT_DYNAMIC_GRAPHS;
    const size_t index = replacing ? slot->dynamic_next : slot->dynamic_count;
    struct tf_qualcomm_resident_tail_variant *variant = &slot->dynamic[index];
    if (replacing) {
        if (variant->buffer != NULL) ggml_backend_buffer_free(variant->buffer);
        if (variant->ctx != NULL) ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
    }
    const struct ggml_init_params params = {
        .mem_size = 4u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == NULL) return 1;
    build_layer_tail_graph(variant->ctx, slot, n, &variant->state, &variant->attention, &variant->output);
    variant->graph = ggml_new_graph_custom(variant->ctx, 32, false);
    ggml_build_forward_expand(variant->graph, variant->output);
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, context->backend);
    if (variant->buffer == NULL) {
        ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
        return 1;
    }
    variant->n = n;
    if (replacing) {
        slot->dynamic_next = (slot->dynamic_next + 1) % TF_RESIDENT_DYNAMIC_GRAPHS;
    } else {
        slot->dynamic_count += 1;
    }
    return 0;
}

static struct tf_qualcomm_resident_tail_slot *get_resident_tail_slot(struct tf_qualcomm_backend *context,
                                                                      size_t slot_id,
                                                                      size_t attention_width,
                                                                      size_t hidden_width,
                                                                      size_t intermediate_width,
                                                                      size_t n,
                                                                      enum ggml_type output_weight_type) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) return NULL;
    struct tf_qualcomm_resident_tail_slot *slot = &context->resident_tail_slots[slot_id];
    if (slot->initialized) {
        if (slot->attention_width != attention_width || slot->hidden_width != hidden_width ||
            slot->intermediate_width != intermediate_width || slot->output_weight_type != output_weight_type) return NULL;
        if (n == slot->batch_n || n == 1 || n == TF_QUALCOMM_VERIFY_N ||
            n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N ||
            resident_tail_dynamic(slot, n) != NULL) return slot;
        return build_resident_tail_dynamic(context, slot, n) == 0 ? slot : NULL;
    }

    const struct ggml_init_params params = {
        .mem_size = 4u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) return NULL;
    slot->attention_width = attention_width;
    slot->hidden_width = hidden_width;
    slot->intermediate_width = intermediate_width;
    slot->batch_n = n;
    slot->output_weight_type = output_weight_type;
    slot->output_weight = ggml_new_tensor_2d(slot->ctx, output_weight_type, (int64_t) attention_width, (int64_t) hidden_width);
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    slot->gate_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) hidden_width, (int64_t) intermediate_width);
    slot->up_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) hidden_width, (int64_t) intermediate_width);
    slot->down_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) intermediate_width, (int64_t) hidden_width);
    slot->post_norm_weight = ggml_new_tensor_1d(slot->ctx, GGML_TYPE_F32, (int64_t) hidden_width);

    build_layer_tail_graph(slot->ctx, slot, n, &slot->batch_state, &slot->batch_attention, &slot->batch_output);
    slot->batch_graph = ggml_new_graph_custom(slot->ctx, 32, false);
    ggml_build_forward_expand(slot->batch_graph, slot->batch_output);
    if (n != 1) {
        build_layer_tail_graph(slot->ctx, slot, 1, &slot->decode_state, &slot->decode_attention, &slot->decode_output);
        slot->decode_graph = ggml_new_graph_custom(slot->ctx, 32, false);
        ggml_build_forward_expand(slot->decode_graph, slot->decode_output);
    }
    if (n != TF_QUALCOMM_VERIFY_N) {
        build_layer_tail_graph(slot->ctx, slot, TF_QUALCOMM_VERIFY_N, &slot->verify_state, &slot->verify_attention, &slot->verify_output);
        slot->verify_graph = ggml_new_graph_custom(slot->ctx, 32, false);
        ggml_build_forward_expand(slot->verify_graph, slot->verify_output);
    }
    if (n != TF_QUALCOMM_VERIFY6_N) {
        build_layer_tail_graph(slot->ctx, slot, TF_QUALCOMM_VERIFY6_N, &slot->verify6_state, &slot->verify6_attention, &slot->verify6_output);
        slot->verify6_graph = ggml_new_graph_custom(slot->ctx, 32, false);
        ggml_build_forward_expand(slot->verify6_graph, slot->verify6_output);
    }
    if (n != TF_QUALCOMM_VERIFY7_N) {
        build_layer_tail_graph(slot->ctx, slot, TF_QUALCOMM_VERIFY7_N, &slot->verify7_state, &slot->verify7_attention, &slot->verify7_output);
        slot->verify7_graph = ggml_new_graph_custom(slot->ctx, 32, false);
        ggml_build_forward_expand(slot->verify7_graph, slot->verify7_output);
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    slot->initialized = 1;
    return slot;
}

static struct ggml_tensor *resident_tail_state(struct tf_qualcomm_resident_tail_slot *slot, size_t n) {
    struct tf_qualcomm_resident_tail_variant *dynamic = resident_tail_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->state;
    if (n == slot->batch_n) return slot->batch_state;
    if (n == 1) return slot->decode_state;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_state;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_state;
    return slot->verify_state;
}

static struct ggml_tensor *resident_tail_attention(struct tf_qualcomm_resident_tail_slot *slot, size_t n) {
    struct tf_qualcomm_resident_tail_variant *dynamic = resident_tail_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->attention;
    if (n == slot->batch_n) return slot->batch_attention;
    if (n == 1) return slot->decode_attention;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_attention;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_attention;
    return slot->verify_attention;
}

static struct ggml_tensor *resident_tail_output(struct tf_qualcomm_resident_tail_slot *slot, size_t n) {
    struct tf_qualcomm_resident_tail_variant *dynamic = resident_tail_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->output;
    if (n == slot->batch_n) return slot->batch_output;
    if (n == 1) return slot->decode_output;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_output;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_output;
    return slot->verify_output;
}

static struct ggml_cgraph *resident_tail_graph(struct tf_qualcomm_resident_tail_slot *slot, size_t n) {
    struct tf_qualcomm_resident_tail_variant *dynamic = resident_tail_dynamic(slot, n);
    if (dynamic != NULL) return dynamic->graph;
    if (n == slot->batch_n) return slot->batch_graph;
    if (n == 1) return slot->decode_graph;
    if (n == TF_QUALCOMM_VERIFY6_N) return slot->verify6_graph;
    if (n == TF_QUALCOMM_VERIFY7_N) return slot->verify7_graph;
    return slot->verify_graph;
}

int tf_qualcomm_backend_layer_tail_q4_resident_once(void *opaque,
                                                    size_t slot_id,
                                                    size_t attention_width,
                                                    size_t hidden_width,
                                                    size_t intermediate_width,
                                                    size_t n,
                                                    int output_q8,
                                                    const float *output_weight,
                                                    const float *gate_weight,
                                                    const float *up_weight,
                                                    const float *down_weight,
                                                    const float *post_norm_weight,
                                                    float *state,
                                                    const float *attention,
                                                    double *compute_ms,
                                                    int *weights_uploaded) {
    if (opaque == NULL || state == NULL || attention == NULL || compute_ms == NULL || weights_uploaded == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    const enum ggml_type output_type = output_q8 ? GGML_TYPE_Q8_0 : qtype;
    struct tf_qualcomm_resident_tail_slot *slot = get_resident_tail_slot(
        context, slot_id, attention_width, hidden_width, intermediate_width, n, output_type);
    if (slot == NULL) return 2;
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        if (output_weight == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL || post_norm_weight == NULL) return 3;
        const int output_status = output_q8
            ? set_q8_weight(slot->output_weight, output_weight, attention_width, hidden_width)
            : set_quant_weight(slot->output_weight, output_weight, attention_width, hidden_width, qtype);
        if (output_status != 0 ||
            set_quant_weight(slot->gate_weight, gate_weight, hidden_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->up_weight, up_weight, hidden_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->down_weight, down_weight, intermediate_width, hidden_width, qtype) != 0) return 4;
        ggml_backend_tensor_set(slot->post_norm_weight, post_norm_weight, 0, hidden_width * sizeof(*post_norm_weight));
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }

    struct ggml_tensor *state_tensor = resident_tail_state(slot, n);
    struct ggml_tensor *attention_tensor = resident_tail_attention(slot, n);
    struct ggml_tensor *output_tensor = resident_tail_output(slot, n);
    struct ggml_cgraph *graph = resident_tail_graph(slot, n);
    if (state_tensor == NULL || attention_tensor == NULL || output_tensor == NULL || graph == NULL) return 5;
    ggml_backend_tensor_set(state_tensor, state, 0, hidden_width * n * sizeof(*state));
    ggml_backend_tensor_set(attention_tensor, attention, 0, attention_width * n * sizeof(*attention));
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 6;
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, state, 0, hidden_width * n * sizeof(*state));
    *compute_ms = stopped - started;
    return 0;
}

static int linear_tail_variant_snapshot_n(size_t n) {
    // Verify restores read per-token states, so only the row counts the
    // verifier can reject with keep snapshot slots; everything else (prompt
    // batch, single-token decode) only needs the final state.
    return (n == TF_QUALCOMM_VERIFY2_N || n == TF_QUALCOMM_VERIFY_N ||
            n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N) ? (int) n : 1;
}

static struct tf_qualcomm_linear_tail_variant *linear_tail_variant_for(struct tf_qualcomm_resident_linear_tail_slot *slot,
                                                                       size_t n) {
    for (size_t i = 0; i < slot->variant_count; ++i) {
        if (slot->variants[i].n == n) return &slot->variants[i];
    }
    return NULL;
}

static struct ggml_cgraph *build_linear_tail_graph(struct ggml_context *ctx,
                                                    struct tf_qualcomm_resident_linear_tail_slot *slot,
                                                    size_t n,
                                                    int snapshots,
                                                    struct ggml_tensor **state,
                                                    struct ggml_tensor **z,
                                                    struct ggml_tensor **query,
                                                    struct ggml_tensor **key,
                                                    struct ggml_tensor **value,
                                                    struct ggml_tensor **log_decay,
                                                    struct ggml_tensor **beta,
                                                    struct ggml_tensor **output,
                                                    struct ggml_tensor **gdn_result) {
    const size_t linear_width = slot->width * slot->heads;
    const size_t vector_bytes = linear_width * n * sizeof(float);
    *state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) slot->hidden_width, (int64_t) n);
    *z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) linear_width, (int64_t) n);
    *query = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) slot->width, (int64_t) slot->heads, (int64_t) n, 1);
    *key = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) slot->width, (int64_t) slot->heads, (int64_t) n, 1);
    *value = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t) slot->width, (int64_t) slot->heads, (int64_t) n, 1);
    *log_decay = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, (int64_t) slot->heads, (int64_t) n, 1);
    *beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, (int64_t) slot->heads, (int64_t) n, 1);

    struct ggml_tensor *gdn = ggml_gated_delta_net(ctx, *query, *key, *value, *log_decay, *beta, slot->gdn_state, snapshots);
    *gdn_result = gdn;
    struct ggml_tensor *gdn_output = ggml_view_2d(ctx, gdn, (int64_t) linear_width, (int64_t) n, linear_width * sizeof(float), 0);
    // Quantize the recurrent rows exactly like the non-fused path's
    // roundBF16 before the norm; without it the HTP kernel's row results
    // carry batch-n low-bit noise and depth-2/6 greedy paths diverge.
    gdn_output = round_f16_to_f32(ctx, gdn_output);
    struct ggml_tensor *new_gdn_state = ggml_view_4d(ctx,
                                                     gdn,
                                                     (int64_t) slot->width,
                                                     (int64_t) slot->width,
                                                     (int64_t) slot->heads,
                                                     1,
                                                     slot->width * sizeof(float),
                                                     slot->width * slot->width * sizeof(float),
                                                     slot->width * slot->width * slot->heads * sizeof(float),
                                                     vector_bytes);
    struct ggml_tensor *state_copy = ggml_cpy(ctx, new_gdn_state, slot->gdn_state);
    struct ggml_tensor *gdn_rows = ggml_reshape_2d(ctx, gdn_output, (int64_t) slot->width, (int64_t) (slot->heads * n));
    struct ggml_tensor *normalized_rows = ggml_rms_norm(ctx, gdn_rows, 1.0e-6f);
    struct ggml_tensor *weighted_rows = ggml_mul(ctx, normalized_rows, slot->linear_norm_weight);
    struct ggml_tensor *z_rows = ggml_reshape_2d(ctx, *z, (int64_t) slot->width, (int64_t) (slot->heads * n));
    struct ggml_tensor *gated_rows = ggml_mul(ctx, ggml_silu(ctx, z_rows), weighted_rows);
    struct ggml_tensor *gated = ggml_reshape_2d(ctx, round_f16_to_f32(ctx, gated_rows), (int64_t) linear_width, (int64_t) n);

    struct ggml_tensor *attention_update = ggml_mul_mat(ctx, slot->output_weight, gated);
    struct ggml_tensor *post_attention = round_f16_to_f32(ctx, ggml_add(ctx, *state, attention_update));
    struct ggml_tensor *normalized = ggml_rms_norm(ctx, post_attention, 1.0e-6f);
    struct ggml_tensor *mlp_input = round_f16_to_f32(ctx, ggml_mul(ctx, normalized, slot->post_norm_weight));
    struct ggml_tensor *gate = ggml_mul_mat(ctx, slot->gate_weight, mlp_input);
    struct ggml_tensor *up = ggml_mul_mat(ctx, slot->up_weight, mlp_input);
    struct ggml_tensor *activated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    struct ggml_tensor *mlp_update = ggml_mul_mat(ctx, slot->down_weight, activated);
    *output = round_f16_to_f32(ctx, ggml_add(ctx, post_attention, mlp_update));

    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 48, false);
    ggml_build_forward_expand(graph, *output);
    ggml_build_forward_expand(graph, state_copy);
    return graph;
}

static int linear_tail_build_variant(struct tf_qualcomm_resident_linear_tail_slot *slot, size_t n) {
    if (linear_tail_variant_for(slot, n) != NULL) return 0;
    if (slot->variant_count >= TF_LINEAR_TAIL_MAX_GRAPHS) return 1;
    struct tf_qualcomm_linear_tail_variant *variant = &slot->variants[slot->variant_count];
    variant->graph = build_linear_tail_graph(slot->ctx, slot, n, linear_tail_variant_snapshot_n(n),
        &variant->state, &variant->z, &variant->query, &variant->key, &variant->value,
        &variant->log_decay, &variant->beta, &variant->output, &variant->gdn_result);
    if (variant->graph == NULL) return 1;
    variant->n = n;
    slot->variant_count++;
    return 0;
}

static struct tf_qualcomm_resident_linear_tail_slot *get_resident_linear_tail_slot(struct tf_qualcomm_backend *context,
                                                                                    size_t slot_id,
                                                                                    size_t width,
                                                                                    size_t heads,
                                                                                    size_t hidden_width,
                                                                                    size_t intermediate_width,
                                                                                    size_t n) {
    if (slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) return NULL;
    struct tf_qualcomm_resident_linear_tail_slot *slot = &context->resident_linear_tail_slots[slot_id];
    if (slot->initialized) {
        if (slot->width != width || slot->heads != heads || slot->hidden_width != hidden_width ||
            slot->intermediate_width != intermediate_width) return NULL;
        return linear_tail_variant_for(slot, n) == NULL ? NULL : slot;
    }
    const struct ggml_init_params params = {
        .mem_size = 6u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    slot->ctx = ggml_init(params);
    if (slot->ctx == NULL) return NULL;
    slot->width = width;
    slot->heads = heads;
    slot->hidden_width = hidden_width;
    slot->intermediate_width = intermediate_width;
    slot->variant_count = 0;
    const size_t linear_width = width * heads;
    slot->gdn_state = ggml_new_tensor_4d(slot->ctx, GGML_TYPE_F32, (int64_t) width, (int64_t) width, (int64_t) heads, 1);
    const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
    slot->output_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) linear_width, (int64_t) hidden_width);
    slot->gate_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) hidden_width, (int64_t) intermediate_width);
    slot->up_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) hidden_width, (int64_t) intermediate_width);
    slot->down_weight = ggml_new_tensor_2d(slot->ctx, qtype, (int64_t) intermediate_width, (int64_t) hidden_width);
    slot->linear_norm_weight = ggml_new_tensor_1d(slot->ctx, GGML_TYPE_F32, (int64_t) width);
    slot->post_norm_weight = ggml_new_tensor_1d(slot->ctx, GGML_TYPE_F32, (int64_t) hidden_width);
    // Compile every row count the caller can present before the buffer is
    // carved: the seeding n plus the fixed verify/decode sizes.
    static const size_t variant_n[] = { 1, TF_QUALCOMM_VERIFY2_N, TF_QUALCOMM_VERIFY_N, TF_QUALCOMM_VERIFY6_N, TF_QUALCOMM_VERIFY7_N };
    if (linear_tail_build_variant(slot, n) != 0) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    for (size_t i = 0; i < sizeof(variant_n) / sizeof(variant_n[0]); ++i) {
        if (linear_tail_build_variant(slot, variant_n[i]) != 0) {
            ggml_free(slot->ctx);
            memset(slot, 0, sizeof(*slot));
            return NULL;
        }
    }
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, context->backend);
    if (slot->buffer == NULL) {
        ggml_free(slot->ctx);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    ggml_backend_buffer_set_usage(slot->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    slot->initialized = 1;
    return slot;
}

int tf_qualcomm_backend_linear_tail_q4_resident_once(void *opaque,
                                                     size_t slot_id,
                                                     size_t width,
                                                     size_t heads,
                                                     size_t hidden_width,
                                                     size_t intermediate_width,
                                                     size_t n,
                                                     const float *output_weight,
                                                     const float *gate_weight,
                                                     const float *up_weight,
                                                     const float *down_weight,
                                                     const float *linear_norm_weight,
                                                     const float *post_norm_weight,
                                                     float *state,
                                                     const float *z,
                                                     const float *query,
                                                     const float *key,
                                                     const float *value,
                                                     const float *log_decay,
                                                     const float *beta,
                                                     const float *initial_gdn_state,
                                                     double *compute_ms,
                                                     int *weights_uploaded) {
    if (opaque == NULL || state == NULL || z == NULL || query == NULL || key == NULL || value == NULL ||
        log_decay == NULL || beta == NULL || compute_ms == NULL || weights_uploaded == NULL) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_linear_tail_slot *slot = get_resident_linear_tail_slot(
        context, slot_id, width, heads, hidden_width, intermediate_width, n);
    if (slot == NULL) return 2;
    *weights_uploaded = 0;
    if (!slot->weights_loaded) {
        if (output_weight == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL ||
            linear_norm_weight == NULL || post_norm_weight == NULL) return 3;
        const size_t linear_width = width * heads;
        const enum ggml_type qtype = tf_qualcomm_q4_storage_type();
        if (set_quant_weight(slot->output_weight, output_weight, linear_width, hidden_width, qtype) != 0 ||
            set_quant_weight(slot->gate_weight, gate_weight, hidden_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->up_weight, up_weight, hidden_width, intermediate_width, qtype) != 0 ||
            set_quant_weight(slot->down_weight, down_weight, intermediate_width, hidden_width, qtype) != 0) return 4;
        ggml_backend_tensor_set(slot->linear_norm_weight, linear_norm_weight, 0, width * sizeof(*linear_norm_weight));
        ggml_backend_tensor_set(slot->post_norm_weight, post_norm_weight, 0, hidden_width * sizeof(*post_norm_weight));
        slot->weights_loaded = 1;
        *weights_uploaded = 1;
    }
    if (initial_gdn_state != NULL) {
        ggml_backend_tensor_set(slot->gdn_state, initial_gdn_state, 0, width * width * heads * sizeof(*initial_gdn_state));
    }
    struct tf_qualcomm_linear_tail_variant *variant = linear_tail_variant_for(slot, n);
    if (variant == NULL) return 2;
    struct ggml_tensor *state_tensor = variant->state;
    struct ggml_tensor *z_tensor = variant->z;
    struct ggml_tensor *query_tensor = variant->query;
    struct ggml_tensor *key_tensor = variant->key;
    struct ggml_tensor *value_tensor = variant->value;
    struct ggml_tensor *decay_tensor = variant->log_decay;
    struct ggml_tensor *beta_tensor = variant->beta;
    struct ggml_tensor *output_tensor = variant->output;
    struct ggml_cgraph *graph = variant->graph;
    if (state_tensor == NULL || z_tensor == NULL || query_tensor == NULL || key_tensor == NULL || value_tensor == NULL ||
        decay_tensor == NULL || beta_tensor == NULL || output_tensor == NULL || graph == NULL) return 5;
    const size_t linear_width = width * heads;
    const size_t vector_bytes = linear_width * n * sizeof(float);
    const size_t scalar_bytes = heads * n * sizeof(float);
    ggml_backend_tensor_set(state_tensor, state, 0, hidden_width * n * sizeof(*state));
    ggml_backend_tensor_set(z_tensor, z, 0, vector_bytes);
    ggml_backend_tensor_set(query_tensor, query, 0, vector_bytes);
    ggml_backend_tensor_set(key_tensor, key, 0, vector_bytes);
    ggml_backend_tensor_set(value_tensor, value, 0, vector_bytes);
    ggml_backend_tensor_set(decay_tensor, log_decay, 0, scalar_bytes);
    ggml_backend_tensor_set(beta_tensor, beta, 0, scalar_bytes);
    const double started = monotonic_ms();
    if (ggml_backend_graph_compute(context->backend, graph) != GGML_STATUS_SUCCESS) return 6;
    const double stopped = monotonic_ms();
    ggml_backend_tensor_get(output_tensor, state, 0, hidden_width * n * sizeof(*state));
    *compute_ms = stopped - started;
    return 0;
}

int tf_qualcomm_backend_linear_tail_gdn_state_restore_snapshot(void *opaque,
                                                               size_t slot_id,
                                                               size_t n,
                                                               size_t tokens_back,
                                                               float *scratch,
                                                               size_t elements) {
    if (opaque == NULL || scratch == NULL || slot_id >= TF_QUALCOMM_RESIDENT_SLOTS ||
        (n != TF_QUALCOMM_VERIFY2_N && n != TF_QUALCOMM_VERIFY_N && n != TF_QUALCOMM_VERIFY6_N && n != TF_QUALCOMM_VERIFY7_N) || tokens_back >= n) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_linear_tail_slot *slot = &context->resident_linear_tail_slots[slot_id];
    if (!slot->initialized || elements != slot->width * slot->width * slot->heads) return 2;
    struct tf_qualcomm_linear_tail_variant *variant = linear_tail_variant_for(slot, n);
    if (variant == NULL || variant->gdn_result == NULL) return 3;
    if (variant->graph == NULL || (size_t) linear_tail_variant_snapshot_n(n) <= tokens_back) return 3;
    // The fused graph mirrors the plain GDN layout: token outputs first, then
    // one state snapshot per token, slot s holding the state s tokens back.
    const size_t output_bytes = slot->width * slot->heads * n * sizeof(*scratch);
    const size_t state_bytes = elements * sizeof(*scratch);
    ggml_backend_tensor_get(variant->gdn_result, scratch, output_bytes + tokens_back * state_bytes, state_bytes);
    ggml_backend_tensor_set(slot->gdn_state, scratch, 0, state_bytes);
    return 0;
}

static struct ggml_cgraph *build_gdn_graph(struct ggml_context *ctx,
                                           struct ggml_tensor *state,
                                           size_t width,
                                           size_t heads,
                                           size_t n,
                                           size_t snapshots,
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
    *result = ggml_gated_delta_net(ctx, *query, *key, *value, *log_decay, *beta, state, (int64_t) snapshots);
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

static struct tf_qualcomm_gdn_variant *resident_gdn_dynamic(struct tf_qualcomm_resident_gdn_slot *slot, size_t n) {
    for (size_t i = 0; i < slot->dynamic_count; ++i) {
        if (slot->dynamic[i].n == n) return &slot->dynamic[i];
    }
    return NULL;
}

static int build_resident_gdn_dynamic(struct tf_qualcomm_backend *context,
                                      struct tf_qualcomm_resident_gdn_slot *slot,
                                      size_t n) {
    if (n == 0) return 1;
    const int replacing = slot->dynamic_count >= TF_RESIDENT_DYNAMIC_GRAPHS;
    const size_t index = replacing ? slot->dynamic_next : slot->dynamic_count;
    struct tf_qualcomm_gdn_variant *variant = &slot->dynamic[index];
    if (replacing) {
        if (variant->buffer != NULL) ggml_backend_buffer_free(variant->buffer);
        if (variant->ctx != NULL) ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
    }
    const struct ggml_init_params params = {
        .mem_size = 2u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    variant->ctx = ggml_init(params);
    if (variant->ctx == NULL) return 1;
    variant->graph = build_gdn_graph(
        variant->ctx, slot->state, slot->width, slot->heads, n, 1,
        &variant->query, &variant->key, &variant->value,
        &variant->log_decay, &variant->beta, &variant->result);
    variant->buffer = ggml_backend_alloc_ctx_tensors(variant->ctx, context->backend);
    if (variant->buffer == NULL) {
        ggml_free(variant->ctx);
        memset(variant, 0, sizeof(*variant));
        return 1;
    }
    variant->n = n;
    if (replacing) {
        slot->dynamic_next = (slot->dynamic_next + 1) % TF_RESIDENT_DYNAMIC_GRAPHS;
    } else {
        slot->dynamic_count += 1;
    }
    return 0;
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
        if (slot->width != width || slot->heads != heads) return NULL;
        if (n == slot->batch_n || n == 1 || n == TF_QUALCOMM_VERIFY2_N || n == TF_QUALCOMM_VERIFY_N ||
            n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N ||
            resident_gdn_dynamic(slot, n) != NULL) return slot;
        return build_resident_gdn_dynamic(context, slot, n) == 0 ? slot : NULL;
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
                                        (n == TF_QUALCOMM_VERIFY2_N || n == TF_QUALCOMM_VERIFY_N || n == TF_QUALCOMM_VERIFY6_N || n == TF_QUALCOMM_VERIFY7_N) ? n : 1,
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
                                             1,
                                             &slot->decode_query,
                                             &slot->decode_key,
                                             &slot->decode_value,
                                             &slot->decode_log_decay,
                                             &slot->decode_beta,
                                             &slot->decode_result);
    }
    if (n != TF_QUALCOMM_VERIFY_N) {
        slot->verify_graph = build_gdn_graph(slot->ctx,
                                             slot->state,
                                             width,
                                             heads,
                                             TF_QUALCOMM_VERIFY_N,
                                             TF_QUALCOMM_VERIFY_N,
                                             &slot->verify_query,
                                             &slot->verify_key,
                                             &slot->verify_value,
                                             &slot->verify_log_decay,
                                             &slot->verify_beta,
                                             &slot->verify_result);
    }
    if (n != TF_QUALCOMM_VERIFY2_N) {
        slot->verify2_graph = build_gdn_graph(slot->ctx,
                                              slot->state,
                                              width,
                                              heads,
                                              TF_QUALCOMM_VERIFY2_N,
                                              TF_QUALCOMM_VERIFY2_N,
                                              &slot->verify2_query,
                                              &slot->verify2_key,
                                              &slot->verify2_value,
                                              &slot->verify2_log_decay,
                                              &slot->verify2_beta,
                                              &slot->verify2_result);
    }
    if (n != TF_QUALCOMM_VERIFY6_N) {
        slot->verify6_graph = build_gdn_graph(slot->ctx,
                                              slot->state,
                                              width,
                                              heads,
                                              TF_QUALCOMM_VERIFY6_N,
                                              TF_QUALCOMM_VERIFY6_N,
                                              &slot->verify6_query,
                                              &slot->verify6_key,
                                              &slot->verify6_value,
                                              &slot->verify6_log_decay,
                                              &slot->verify6_beta,
                                              &slot->verify6_result);
    }
    if (n != TF_QUALCOMM_VERIFY7_N) {
        slot->verify7_graph = build_gdn_graph(slot->ctx,
                                              slot->state,
                                              width,
                                              heads,
                                              TF_QUALCOMM_VERIFY7_N,
                                              TF_QUALCOMM_VERIFY7_N,
                                              &slot->verify7_query,
                                              &slot->verify7_key,
                                              &slot->verify7_value,
                                              &slot->verify7_log_decay,
                                              &slot->verify7_beta,
                                              &slot->verify7_result);
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

    struct tf_qualcomm_gdn_variant *dynamic = resident_gdn_dynamic(slot, n);
    struct ggml_tensor *query_tensor = dynamic != NULL ? dynamic->query : n == slot->batch_n ? slot->batch_query : n == 1 ? slot->decode_query : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_query : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_query : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_query : slot->verify_query;
    struct ggml_tensor *key_tensor = dynamic != NULL ? dynamic->key : n == slot->batch_n ? slot->batch_key : n == 1 ? slot->decode_key : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_key : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_key : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_key : slot->verify_key;
    struct ggml_tensor *value_tensor = dynamic != NULL ? dynamic->value : n == slot->batch_n ? slot->batch_value : n == 1 ? slot->decode_value : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_value : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_value : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_value : slot->verify_value;
    struct ggml_tensor *decay_tensor = dynamic != NULL ? dynamic->log_decay : n == slot->batch_n ? slot->batch_log_decay : n == 1 ? slot->decode_log_decay : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_log_decay : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_log_decay : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_log_decay : slot->verify_log_decay;
    struct ggml_tensor *beta_tensor = dynamic != NULL ? dynamic->beta : n == slot->batch_n ? slot->batch_beta : n == 1 ? slot->decode_beta : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_beta : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_beta : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_beta : slot->verify_beta;
    struct ggml_tensor *result_tensor = dynamic != NULL ? dynamic->result : n == slot->batch_n ? slot->batch_result : n == 1 ? slot->decode_result : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_result : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_result : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_result : slot->verify_result;
    struct ggml_cgraph *graph = dynamic != NULL ? dynamic->graph : n == slot->batch_n ? slot->batch_graph : n == 1 ? slot->decode_graph : n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_graph : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_graph : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_graph : slot->verify_graph;
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

int tf_qualcomm_backend_gdn_state_get(void *opaque,
                                     size_t slot_id,
                                     float *state,
                                     size_t elements) {
    if (opaque == NULL || state == NULL || slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_gdn_slot *slot = &context->resident_gdn_slots[slot_id];
    if (!slot->initialized || elements != slot->width * slot->width * slot->heads) return 2;
    ggml_backend_tensor_get(slot->state, state, 0, elements * sizeof(*state));
    return 0;
}

int tf_qualcomm_backend_gdn_state_set(void *opaque,
                                     size_t slot_id,
                                     const float *state,
                                     size_t elements) {
    if (opaque == NULL || state == NULL || slot_id >= TF_QUALCOMM_RESIDENT_SLOTS) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_gdn_slot *slot = &context->resident_gdn_slots[slot_id];
    if (!slot->initialized || elements != slot->width * slot->width * slot->heads) return 2;
    ggml_backend_tensor_set(slot->state, state, 0, elements * sizeof(*state));
    return 0;
}

int tf_qualcomm_backend_gdn_state_restore_snapshot(void *opaque,
                                                  size_t slot_id,
                                                  size_t n,
                                                  size_t tokens_back,
                                                  float *scratch,
                                                  size_t elements) {
    if (opaque == NULL || scratch == NULL || slot_id >= TF_QUALCOMM_RESIDENT_SLOTS ||
        (n != TF_QUALCOMM_VERIFY2_N && n != TF_QUALCOMM_VERIFY_N && n != TF_QUALCOMM_VERIFY6_N && n != TF_QUALCOMM_VERIFY7_N) || tokens_back >= n) return 1;
    struct tf_qualcomm_backend *context = opaque;
    struct tf_qualcomm_resident_gdn_slot *slot = &context->resident_gdn_slots[slot_id];
    if (!slot->initialized || elements != slot->width * slot->width * slot->heads) return 2;
    struct ggml_tensor *result = n == TF_QUALCOMM_VERIFY2_N ? slot->verify2_result : n == TF_QUALCOMM_VERIFY6_N ? slot->verify6_result : n == TF_QUALCOMM_VERIFY7_N ? slot->verify7_result : slot->verify_result;
    if (n == slot->batch_n) result = slot->batch_result;
    if (result == NULL) return 3;
    const size_t output_bytes = slot->width * slot->heads * n * sizeof(*scratch);
    const size_t state_bytes = elements * sizeof(*scratch);
    ggml_backend_tensor_get(result, scratch, output_bytes + tokens_back * state_bytes, state_bytes);
    ggml_backend_tensor_set(slot->state, scratch, 0, state_bytes);
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
