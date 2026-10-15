#pragma once

#include <stddef.h>

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
