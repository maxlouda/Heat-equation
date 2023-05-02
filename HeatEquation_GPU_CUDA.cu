#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#define _USE_MATH_DEFINES

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "lodepng.h"

typedef uint8_t u8;
typedef uint32_t u32;
typedef float f32;
typedef double f64;
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

f64 measureTime() {
    struct timespec now;
    u32 t = timespec_get(&now, TIME_UTC);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

u32 __host__ __device__ getIndex(u32 i, u32 j, u32 width) {
    return i * width + j;
}

void __global__ heatFlowGlobal(f32* T_old, f32* T_new, u32 size_xy, f32 dx2, f32 eta, f32 dt) {
    u32 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0 && i < size_xy - 1) {
        u32 j = threadIdx.y + blockIdx.y * blockDim.y;
        if (j > 0 && j < size_xy - 1) {
            u32 k = i * size_xy + j;
            u32 kim1 = (i - 1) * size_xy + j;
            u32 kip1 = (i + 1) * size_xy + j;
            u32 kjm1 = i * size_xy + j - 1;
            u32 kjp1 = i * size_xy + j + 1;
            T_new[k] = T_old[k] + dt * eta * (T_old[kim1] + T_old[kip1] + T_old[kjm1] + T_old[kjp1] - 4.0 * T_old[k])/dx2;
        }
    }
}

__global__ void heatFlowShared(f32* T_old, f32* T_new, u32 size_xy, f32 dx2, f32 eta, f32 dt) {
    u32 i = threadIdx.x + blockIdx.x * blockDim.x;
    u32 j = threadIdx.y + blockIdx.y * blockDim.y;
    u32 k = i * size_xy + j;

    __shared__ f32 s_T_old[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];

    if (i < size_xy && j < size_xy) {
        s_T_old[threadIdx.x + 1][threadIdx.y + 1] = T_old[k];
        if (threadIdx.x == 0 && i > 0) {
            s_T_old[0][threadIdx.y + 1] = T_old[k - size_xy];
        }
        if (threadIdx.x == BLOCK_SIZE_X - 1 && i < size_xy - 1) {
            s_T_old[BLOCK_SIZE_X + 1][threadIdx.y + 1] = T_old[k + size_xy];
        }
        if (threadIdx.y == 0 && j > 0) {
            s_T_old[threadIdx.x + 1][0] = T_old[k - 1];
        }
        if (threadIdx.y == BLOCK_SIZE_Y - 1 && j < size_xy - 1) {
            s_T_old[threadIdx.x + 1][BLOCK_SIZE_Y + 1] = T_old[k + 1];
        }
    }

    __syncthreads();

    if (i > 0 && i < size_xy - 1 && j > 0 && j < size_xy - 1) {
        T_new[k] = s_T_old[threadIdx.x + 1][threadIdx.y + 1] +
            dt * eta *
            (s_T_old[threadIdx.x][threadIdx.y + 1] +
                s_T_old[threadIdx.x + 2][threadIdx.y + 1] +
                s_T_old[threadIdx.x + 1][threadIdx.y] +
                s_T_old[threadIdx.x + 1][threadIdx.y + 2] -
                4.0 * s_T_old[threadIdx.x + 1][threadIdx.y + 1]) /
            dx2;
    }
}

u32 main() {
    f64 start1 = measureTime();

    u32 size_xy = 4096;
    u32 n_steps = 10000;
    f32 eta = 1.0;
    f32 dx = 1.0 / size_xy;
    f32 dx2 = dx * dx;
    f32 dt = (dx2 * dx2) / (4.0 * eta * dx2);
    f64 pi = acos(-1);

    f32* T_old = (f32*)malloc(sizeof(*T_old) * size_xy * size_xy);
    for (u32 i = 0; i < size_xy * size_xy; i++) {
        T_old[i] = 0.0;
    }

    f32* T_old_d;
    f32* T_new_d;

    cudaMalloc((void**)(&T_old_d), size_xy * size_xy * sizeof(f32));
    cudaMalloc((void**)(&T_new_d), size_xy * size_xy * sizeof(f32));

    u8* image = (u8*)malloc(sizeof(*image) * size_xy * size_xy * 3);

    dim3 numBlocks(size_xy / BLOCK_SIZE_X + 1, size_xy / BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    for (u32 i = 0; i < size_xy; i++) {
        u32 k_left = i * size_xy;
        u32 k_right = i * size_xy + size_xy - 1;
        u32 k_top = i;
        u32 k_bottom = (size_xy - 1) * size_xy + i;

        T_old[k_left] = 255.0 * cos(i * pi / float(size_xy)) * cos(i * pi / float(size_xy));
        T_old[k_right] = 255.0 * sin(i * pi / float(size_xy)) * sin(i * pi / float(size_xy));
        T_old[k_top] = 255.0 - 255.0 * i / float(size_xy);
        T_old[k_bottom] = 255.0 - 255.0 * i / float(size_xy);
    }

    cudaMemcpy(T_old_d, T_old, size_xy * size_xy * sizeof(f32), cudaMemcpyHostToDevice);
    cudaMemcpy(T_new_d, T_old, size_xy * size_xy * sizeof(f32), cudaMemcpyHostToDevice);

    f64 end1 = measureTime();

    printf("Grid Creation and Initialization Time: %.9lfs\n", end1 - start1);

    if (image != NULL) {
        f64 start2 = measureTime();
        for (u32 n = 0; n < n_steps; n++) {
            heatFlowGlobal <<< numBlocks, threadsPerBlock >>> (T_old_d, T_new_d, size_xy, dx2, eta, dt);
            cudaDeviceSynchronize();

            if (n % 1000 == 0) {
                cudaMemcpy(T_old, T_old_d, size_xy * size_xy * sizeof(f32), cudaMemcpyDeviceToHost);
                cudaError_t errorCode = cudaGetLastError();
                if (errorCode != cudaSuccess) {
                    printf("Cuda Error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                    exit(0);
                }
                char filename[100] = "";
                sprintf(filename, "pics/heat_%05d.png", n);
                for (u32 i = 0; i < size_xy * size_xy; i++) {
                    image[i * 3] = (u8)T_old[i];
                    image[i * 3 + 1] = 0;
                    image[i * 3 + 2] = 0;
                }
                u32 error = lodepng_encode24_file(filename, image, size_xy, size_xy);
                if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
            }
            
            f32* temp = T_old_d;
            T_old_d = T_new_d;
            T_new_d = temp;
        }

        free(T_old);

        cudaFree(T_old_d);
        cudaFree(T_new_d);

        f64 end2 = measureTime();
        printf("Grid Creation and Initialization Time: %.9lfs\n", end2 - start2);

    }
    return 0;
}
