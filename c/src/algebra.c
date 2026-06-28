#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <hpc/algebra.h>

f64 vec_sum(f64* v, usize dim)
{
	f64 sum = 0;
	for (usize i = 0; i < dim; i++)
		sum += v[i];
	return sum;
}

void vec_sq(f64* v, usize dim)
{
	for (usize i = 0; i < dim; i++)
		v[i] *= v[i];
}

bool vec_eq(f64* x, f64* y, usize dim)
{
	for (usize i = 0; i < dim; i++)
		if (x[i] != y[i])
			return false;
	return true;
}

void vec_assign(f64* dst, f64* src, usize dim)
{
	for (usize i = 0; i < dim; i++)
		dst[i] = src[i];
}

void vec_add(f64* dst, f64* v, usize dim)
{
	for (usize i = 0; i < dim; i++)
		dst[i] += v[i];
}

void vec_mul_scalar(f64* v, f64 scalar, usize dim)
{
	for (usize i = 0; i < dim; i++)
		v[i] *= scalar;
}

f64 euclidean_dist(f64* x, f64* y, usize dim)
{
	f64 sum = 0;
	for (usize i = 0; i < dim; i++) {
		f64 term = x[i] - y[i];
		sum += term * term;
	}
	return sqrt(sum);
}

void mean_vec(f64* dst, f64** set, usize size, usize dim)
{
	assert(size > 0);

	for (usize i = 0; i < size; i++)
		vec_add(dst, set[i], dim);

	vec_mul_scalar(dst, 1. / (f64)size, dim);
}
