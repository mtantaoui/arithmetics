#include <immintrin.h>

// Function declaration
int addition(int a, int b);

__m512 f32x16_set(float value);
__m512 f32x16_load(float *mem_addr);
__m512 f32x16_add(__m512 a, __m512 b);

void f32x16_store(float *mem_addr, __m512 a);
