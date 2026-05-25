#include <iostream>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

__global__ void julia_kernel_gpu(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale){
    //we no longer pass in the pixel coords but calculate them from the thread parameters
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                                    
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + local_tid;
    int total_pixels = res_x * res_y;
    //guard against too many thread launches 
    if (tid >= total_pixels) return;

    int x = tid / res_y; 
    int y = tid % res_y;
    //julia_set[x * res_y + y] = (float)(tid % 256);

    //logic yoinked from the cpu kernel
    float scaledX = scale * x_scale * (float) (x - res_x / 2) / (res_x / 2);
    float scaledY = scale * y_scale * (float) (y - res_y / 2) / (res_y / 2);

    Complex z(scaledX, scaledY);

    int i = 0;
    for(i = 0; i < max_iter; i++) {
        z = z * z + c;
        if(z.magnitude2() > max_mag)
            break;
    }
    julia_set[x * res_y + y] = (float)i / max_iter; //normalize to max_iter which happens in the calling code in the cpu version
}   



void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    int total_pixels = res_x * res_y;
    // compute a good default block size
    dim3 block(32,8); //8 warps for a 2d block
    if (global_block_x > 0) {
        block = dim3(global_block_x, global_block_y);
    }

    //array size 
    size_t mem_size = total_pixels * sizeof(float);
    float *d_julia_set;
    cudaMalloc(&d_julia_set, mem_size);

    int grid_size = (total_pixels + block.x * block.y - 1) / (block.x * block.y); //round up division

    julia_kernel_gpu<<<grid_size,block>>>(d_julia_set, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = cudaGetErrorString(err);
        std::cout << "CUDA Error for block size " << block.x << "x" << block.y << ": " << error_msg << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(julia_set, d_julia_set, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_julia_set);
    

}
