#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
rotationrodrigues(
            float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
            float* __restrict__ Bx,  float* __restrict__  By,  float* __restrict__ Bz,
            float dt, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 B = {Bx[i], By[i], Bz[i]};

        const float theta = dt*len(B);
        
        // Compute the prefactor efficiently to save time because sin and 1/x are costly.
        // I wonder how much difference this makes.
        float pref;
        if(theta<=1e-2){
            //pref = 1.0 - theta*theta*(1.0-theta*theta/20.0)/6.0;
            // don't divide when using float...
            const float theta2 = theta*theta;
            pref = 1.0 - 0.166667*theta2*(1.0-0.05*theta2); 
        }else
            pref = sin(theta)/theta;
        
        // update m and normalize but check if it is within the sample
        if(!is0(m))
            m = normalized(m*cos(theta) + dt*pref*B); 

        mx[i] = m.x;
        my[i] = m.y;
        mz[i] = m.z;

    }
}
