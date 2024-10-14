#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
zero3(
            float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
            int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        mx[i] = 0.f;
        my[i] = 0.f;
        mz[i] = 0.f;

    }
}
