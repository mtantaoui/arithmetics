#include <stdio.h>
#include <immintrin.h>



__m512 f32x16_set(float value){
    return _mm512_set1_ps(value);
}

// A simple C function that adds two integers
int addition(int a, int b) {
    printf("C function called with %d and %d\n", a, b);
    return a + b;
}