
#include <stdint.h>
#include "float3.h"

// rotates a vector v0 from the tangent space of m0 to the tangent space of m
// saves result in v

extern "C" __global__ void
tangentspacerotation(
        float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz,
        float* __restrict__ v0x,  float* __restrict__  v0y,  float* __restrict__ v0z,
        float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
        float* __restrict__ m0x,  float* __restrict__  m0y,  float* __restrict__ m0z,
        int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m  = { mx[i],  my[i],  mz[i]};
        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 v0 = {v0x[i], v0y[i], v0z[i]};

        const float SIN = dot(m,v0);
        const float COS = dot(m,m0);

        vx[i] = v0x[i]*COS-m0x[i]*SIN;
        vy[i] = v0y[i]*COS-m0y[i]*SIN;
        vz[i] = v0z[i]*COS-m0z[i]*SIN;

    }
}
