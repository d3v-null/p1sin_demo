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

### debug prints

enabling debug prints shows that `p1_out` at `0x2000000004998` is being overwritten in
thread 32, but not in threads 0 and 16.

```c
make clean NDIRS=33 GPUFLAGS="-g -O0 --offload-arch=gfx1101 -DDEBUG=1" dbg_fee
```

```txt
Device id: 0
        name:                                    AMD Radeon RX 7800 XT
        global memory size:                      17163 MB (0x3ff000000)
        available registers per block:           65536
        maximum shared memory size per block:    65 KB (0x10000)
        maximum pitch size for memory copies:    2147 MB (0x7fffffff)
        max block size:                          (1024,1024,1024)
        max threads in a block:                  1024
        max Grid size:                           (2147483647,65536,65536)
num_directions: 33
./fee.h:484:gpu_fee_calc_jones(): gridDim (  1,  1,  1) blockDim: [ 64,  1,  1]
[New Thread 0x7ffeddbff640 (LWP 73463)]
[Thread 0x7ffeddbff640 (LWP 73463) exited]
[New Thread 0x7fffed285640 (LWP 73464)]
./fee.h:423:fee_kernel(): start (  0,  0,  0)[  0,  0,  0]
./fee.h:423:fee_kernel(): start (  0,  0,  0)[ 16,  0,  0]
./fee.h:423:fee_kernel(): start (  0,  0,  0)[ 32,  0,  0]
./fee.h:423:fee_kernel(): start (  0,  0,  0)[ 48,  0,  0]
./fee.h:428:fee_kernel(): i direction: 32
./fee.h:428:fee_kernel(): i direction: 0
./fee.h:428:fee_kernel(): i direction: 16
./fee.h:437:fee_kernel(): (  0,  0,  0)[  0,  0,  0] 0x2000000000240+3ff <- P1sin_arr
./fee.h:437:fee_kernel(): (  0,  0,  0)[ 16,  0,  0] 0x2000000000240+3ff <- P1sin_arr
./fee.h:437:fee_kernel(): (  0,  0,  0)[ 32,  0,  0] 0x2000000000240+3ff <- P1sin_arr
./fee.h:439:fee_kernel(): (  0,  0,  0)[ 32,  0,  0] 0x2000000002240+3ff <- P1_arr
./fee.h:439:fee_kernel(): (  0,  0,  0)[  0,  0,  0] 0x2000000002240+3ff <- P1_arr
./fee.h:439:fee_kernel(): (  0,  0,  0)[ 16,  0,  0] 0x2000000002240+3ff <- P1_arr
./fee.h:255:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] start
./fee.h:255:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] start
./fee.h:255:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] start
./fee.h:265:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] 0x2000000004990    <- &p1sin_out
./fee.h:265:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] 0x2000000004990    <- &p1sin_out
./fee.h:265:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] 0x2000000004990    <- &p1sin_out
./fee.h:267:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] 0x16000000000+3ff <- p1sin_out
./fee.h:267:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] 0x2000000000240+3ff <- p1sin_out
./fee.h:267:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] 0x2000000000240+3ff <- p1sin_out
./fee.h:269:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] 0x2000000004998    <- &p1_out
./fee.h:269:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] 0x2000000004998    <- &p1_out
./fee.h:269:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] 0x2000000004998    <- &p1_out
./fee.h:271:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] 0x656e6f6a00000000+3ff <- p1_out
./fee.h:271:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] 0x2000000002240+3ff <- p1_out
./fee.h:271:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] 0x2000000002240+3ff <- p1_out
./fee.h:280:jones_p1sin_device(): (  0,  0,  0)[  0,  0,  0] 0x20000000051d0+210 <- legendret
./fee.h:280:jones_p1sin_device(): (  0,  0,  0)[ 16,  0,  0] 0x20000000051d0+210 <- legendret
./fee.h:280:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] 0x20000000051d0+210 <- legendret
./fee.h:353:jones_p1sin_device(): (  0,  0,  0)[ 32,  0,  0] end

Thread 6 "fee" received signal SIGSEGV, Segmentation fault.
[Switching to thread 6, lane 0 (AMDGPU Lane 1:1:1:2/0 (0,0,0)[32,0,0])]
0x00007fffec79a988 in fee_kernel (coeffs=..., azs=0x7fffe6e00000, zas=0x7fffe6e01000, num_directions=33, norm_jones=0x0, latitude_rad=0x7fffe6e1a000, iau_order=1, fee_jones=0x7fffe6e1b000) at ./fee.h:442
442                     debug_printf("i direction: %d (%3d,%3d,%3d)\n", i_direction, blockIdx.x, blockIdx.y, blockIdx.z);
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
