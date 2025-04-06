#include <stdio.h>
#include <immintrin.h>

__m512 f32x16_set(float value)
{
    return _mm512_set1_ps(value);
}

__m512 f32x16_load(float *mem_addr)
{
    return _mm512_loadu_ps(mem_addr);
}

__m512 f32x16_add(__m512 a, __m512 b)
{
    return _mm512_add_ps(a, b);
}

void f32x16_store(float *mem_addr, __m512 a)
{
    return _mm512_store_ps(mem_addr, a);
}

// A simple C function that adds two integers
int addition(int a, int b)
{
    return a + b;
}

// int main()
// {
//     // Arrays to hold the input values and result
//     float arr1[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
//                       8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
//     float arr2[16] = {15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
//                       7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
//     float result[16];

//     // Load the arrays into AVX-512 registers
//     __m512 vec1 = f32x16_load(arr1);
//     __m512 vec2 = f32x16_load(arr2);

//     // Add the two vectors
//     __m512 sum = f32x16_add(vec1, vec2);

//     // Store the result back into the result array
//     f32x16_store(result, sum);

//     // Print the result
//     printf("Result of addition:\n");
//     for (int i = 0; i < 16; i++)
//     {
//         printf("%f ", result[i]);
//     }
//     printf("\n");

//     return 0;
// }