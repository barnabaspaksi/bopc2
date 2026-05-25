#include <iostream>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

__global__ void julia_kernel_gpu(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {
    // Map directly to 2D image coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard for 2D grid
    if (x >= res_x || y >= res_y) return;

    // Logic taken from the CPU kernel
    float scaledX = scale * x_scale * (float)(x - res_x / 2) / (res_x / 2);
    float scaledY = scale * y_scale * (float)(y - res_y / 2) / (res_y / 2);

    Complex z(scaledX, scaledY);

    int i = 0;
    for(i = 0; i < max_iter; i++) {
        z = z * z + c;
        if(z.magnitude2() > max_mag)
            break;
    }

    // Coalesced memory write using row-major order where x is row, y is column
    int index = x * res_y + y; 
    julia_set[index] = (float)i / max_iter; 
}

void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    // 1. Establish your good default block size (2D)
    dim3 block(32, 8); 
    if (global_block_x > 0 && global_block_y > 0) {
        block = dim3(global_block_x, global_block_y);
    }

    // 2. Compute 2D grid size (Ceil division)
    dim3 grid((res_x + block.x - 1) / block.x, 
              (res_y + block.y - 1) / block.y);

    size_t mem_size = res_x * res_y * sizeof(float);
    float *d_julia_set;
    
    cudaMalloc(&d_julia_set, mem_size);

    julia_kernel_gpu<<<grid, block>>>(d_julia_set, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
    cudaMemcpy(julia_set, d_julia_set, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_julia_set);
}
