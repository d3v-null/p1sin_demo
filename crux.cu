#include "gpu_common.cuh"
#if __HIPCC__
#include "hip_helpers.cuh"
#endif

const int size = 2045;

inline __device__ void crux_p1sin(FLOAT *p1sin_out) { return; }

__global__ void crux_kernel()
{
    FLOAT P1sin_arr[size];
    crux_p1sin(P1sin_arr);
}

int main(int argc, char *argv[])
{
#if __HIPCC__
    h_report_on_device(0);
#endif
    int num_directions = atoi(argv[argc - 1]);
    debug_printf("num_directions: %d\n", num_directions);
    crux_kernel<<<num_directions, num_directions>>>();
    GPUCHECK(gpuDeviceSynchronize());
    return 0;
}