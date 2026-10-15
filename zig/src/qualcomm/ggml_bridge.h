#pragma once

#include <stddef.h>

double tf_qualcomm_monotonic_ms(void);
size_t tf_qualcomm_peak_rss_kib(void);
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
