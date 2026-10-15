#pragma once

#include <stddef.h>

double tf_qualcomm_monotonic_ms(void);
size_t tf_qualcomm_peak_rss_kib(void);
int tf_qualcomm_tied_head_f16(const unsigned short *embedding,
                              const float *state,
                              size_t vocab,
                              size_t hidden,
                              int cpu_threads,
                              size_t *top_ids,
                              float *top_logits);
void *tf_qualcomm_backend_create(const char *backend_name, int cpu_threads);
void tf_qualcomm_backend_destroy(void *context);
int tf_qualcomm_backend_matmul(void *context,
                               size_t k,
                               size_t m,
                               size_t n,
                               size_t iterations,
                               const float *weight,
                               const float *input,
                               float *output,
                               double *ms_per_iteration);
int tf_qualcomm_backend_matmul_once(void *context,
                                    size_t k,
                                    size_t m,
                                    size_t n,
                                    const float *weight,
                                    const float *input,
                                    float *output,
                                    double *compute_ms);
void tf_qualcomm_f32_to_f16(const float *input, unsigned short *output, size_t count);
int tf_qualcomm_backend_matmul_f16_once(void *context,
                                        size_t k,
                                        size_t m,
                                        size_t n,
                                        const unsigned short *weight,
                                        const float *input,
                                        float *output,
                                        double *compute_ms);

int tf_qualcomm_matmul(const char *backend_name,
                       size_t k,
                       size_t m,
                       size_t n,
                       size_t iterations,
                       int cpu_threads,
                       const float *weight,
                       const float *input,
                       float *output,
                       double *ms_per_iteration);
