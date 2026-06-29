#ifndef __ALGEBRA_H__
#define __ALGEBRA_H__

#include <stdbool.h>
#include <hpc/types.h>

f64  vec_sum(f64* v, usize dim);
void vec_sq(f64* v, usize dim);

bool vec_eq(f64* u, f64* v, usize dim);
void vec_assign(f64* u, f64* v, usize dim);

void vec_add(f64* dst, f64* v, usize dim);
void vec_mul_scalar(f64* v, f64 scalar, usize dim);

void mean_vec(f64* dst, f64** set, usize size, usize dim);

#pragma omp declare target
f64 euclidean_dist(f64* u, f64* v, usize dim);
#pragma omp end declare target

#endif // !__ALGEBRA_H__
