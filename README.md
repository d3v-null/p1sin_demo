# HIP segfault demo

This demonstrates a segfault that occurs in the `jones_p1sin_device` function of [mwa_hyperbeam](https://github.com/mwaTelescope/mwa_hyperbeam).

It only occurs in HIP, but not when the same code is run in CUDA.

## setup

- change `--offload-arch` in `GPUFLAGS` make variable for your hardware
- increase `NDIRS` until it segfaults.

## Run in debugger

### fee

full example of how code is actually used

```bash
make clean NDIRS=99 GPUFLAGS="-g -O0 --offload-arch=gfx1101" dbg_fee
```

on my hardware, this segfaults at 33+ directions, but not 32.

```txt
Thread 6 "fee" received signal SIGBUS, Bus error.
[Switching to thread 6, lane 0 (AMDGPU Lane 1:1:1:1/0 (0,0,0)[0,0,0])]
0x00007fffed269020 in jones_p1sin_device (nmax=<error reading variable: Cannot access memory at address private_lane#0x446c>, theta=<error reading variable: Cannot access memory at address private_lane#0x4470>,
    p1sin_out=<error reading variable: Cannot access memory at address private_lane#0x4478>, p1_out=<error reading variable: Cannot access memory at address private_lane#0x4480>) at ./fee.h:317
317                     p1sin_out[i] = Pm_sin_merged[modified];
```

### p1sin

only calls `jones_p1sin_device` where the segfault happens in the previous example.


```bash
make clean NDIRS=99 GPUFLAGS="-g -O0 --offload-arch=gfx1101" dbg_p1sin
```

Does not trigger the segfault at 99 directions

```txt
[Inferior 1 (process 7367) exited normally]
```

but does segfault (at a different location) at 9999 directions.

```txt
Thread 6 "fee" received signal SIGBUS, Bus error.
[Switching to thread 6, lane 0 (AMDGPU Lane 1:1:1:3/0 (1,0,0)[0,0,0])]
0x00007fffed26994c in jones_p1sin_device (nmax=<error reading variable: Cannot access memory at address private_lane#0x446c>, theta=<error reading variable: Cannot access memory at address private_lane#0x4470>,
    p1sin_out=<error reading variable: Cannot access memory at address private_lane#0x4478>, p1_out=<error reading variable: Cannot access memory at address private_lane#0x4480>) at ./fee.h:328
328                     p1_out[i] = Pm1_merged[modified];
```

### cuda

no issues with CUDA at all.

```bash
make clean CXX=nvcc GDB=cuda-gdb GPUFLAGS="-g -G -arch=sm_86" NDIRS=9999 dbg_fee
```

```txt
...
[9996] -0.002828 0.000503 0.000859 0.000044 0.000871 0.001035 0.001226 0.004104
[9997] -0.002818 0.000498 0.000856 0.000046 0.000872 0.001035 0.001233 0.004101
[9998] -0.002809 0.000492 0.000852 0.000048 0.000873 0.001034 0.001241 0.004099
...
[Inferior 1 (process 49383) exited normally]
```

```bash
make clean CXX=nvcc GDB=cuda-gdb GDBINIT=cudagdbinit GPUFLAGS="-g -G -arch=sm_86" NDIRS=9999 dbg_p1sin
```

```txt
[Inferior 1 (process 50965) exited normally]
```
