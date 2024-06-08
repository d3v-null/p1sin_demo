
#define HIP_ENABLE_PRINTF

#include "fee.h"

// replicate parts of fee.h:fee_kernel that call p1sin_device
__global__ void p1sin_kernel(int8_t n_max, const FLOAT *azs, const FLOAT *zas, const int num_directions)
{
    for (int i_direction = blockIdx.x * blockDim.x + threadIdx.x; i_direction < num_directions;
         i_direction += gridDim.x * blockDim.x)
    {
        const FLOAT za = zas[i_direction];

        // Set up our "P1sin" arrays. This is pretty expensive, but only depends
        // on the zenith angle and "n_max".
        FLOAT P1sin_arr[NMAX * NMAX + 2 * NMAX], P1_arr[NMAX * NMAX + 2 * NMAX];
        jones_p1sin_device(n_max, za, P1sin_arr, P1_arr);
    }
}

int main(int argc, char *argv[])
{
    int num_directions = atoi(argv[argc - 1]);
    printf("num_directions: %d\n", num_directions);
    FLOAT az_rad[num_directions];
    FLOAT za_rad[num_directions];
    FLOAT PI = 3.14159265358979323846;
    for (int i = 0; i < num_directions; i++)
    {
        az_rad[i] = 0.4 + 0.3 * PI * ((FLOAT)i / (FLOAT)num_directions);
        za_rad[i] = 0.3 + 0.4 * PI * ((FLOAT)i / (FLOAT)num_directions) / 2.0;
        // printf("[%d] az %f, za %f\n", i, az_rad[i], za_rad[i]);
    }
    FLOAT *d_azs, *d_zas;
    GPUCHECK(gpuMalloc(&d_azs, num_directions * sizeof(FLOAT)));
    GPUCHECK(gpuMalloc(&d_zas, num_directions * sizeof(FLOAT)));
    GPUCHECK(gpuMemcpy(d_azs, az_rad, num_directions * sizeof(FLOAT), gpuMemcpyHostToDevice));
    GPUCHECK(gpuMemcpy(d_zas, za_rad, num_directions * sizeof(FLOAT), gpuMemcpyHostToDevice));

    const int8_t n_max = 22;
    const int32_t num_coeffs = 1;

    dim3 gridDim, blockDim;
    blockDim.x = warpSize;
    gridDim.x = (int)ceil((double)num_directions / (double)blockDim.x);
    gridDim.y = num_coeffs;

    p1sin_kernel<<<gridDim, blockDim>>>(n_max, d_azs, d_zas, num_directions);
}