
#include "float3.h"

// dst += prefactor * dot(a,b)
extern "C" __global__ void
tangentspaceprojection(
            float* __restrict__ kx, float* __restrict__ ky, float* __restrict__ kz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 k = {kx[i], ky[i], kz[i]};
        float3 m = {mx[i], my[i], mz[i]};

        const float km = dot(k,m);
        
        kx[i] = kx[i] - km*mx[i];
        ky[i] = ky[i] - km*my[i];
        kz[i] = kz[i] - km*mz[i];
    }
}

