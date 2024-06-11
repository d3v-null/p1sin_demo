# HIP segfault demo

This demonstrates a segfault that occurs in the `jones_p1sin_device` function of [mwa_hyperbeam](https://github.com/mwaTelescope/mwa_hyperbeam).

It only occurs in HIP, but not when the same code is run in CUDA.

## setup

- change `--offload-arch` in `GPUFLAGS` make variable for your hardware
- increase `NDIRS` until it segfaults.

## Run in debugger

### crux

the simplest example that triggers the segfault

```bash
make clean NDIRS=33 GPUFLAGS="-ggdb -O0 --offload-arch=gfx1101" dbg_crux
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
:3:hip_platform.cpp         :219 : 301282370896 us: [pid:145680 tid:0x7fffed309180]  __hipPushCallConfiguration ( {33,1,1}, {33,1,1}, 0, stream:<null> )
:3:hip_platform.cpp         :223 : 301282370906 us: [pid:145680 tid:0x7fffed309180] __hipPushCallConfiguration: Returned hipSuccess :
:3:hip_platform.cpp         :228 : 301282370917 us: [pid:145680 tid:0x7fffed309180]  __hipPopCallConfiguration ( {2189384,0,4160737344}, {4294958584,32767,2183536}, 0x7fffffffdc48, 0x7fffffffdc40 )
:3:hip_platform.cpp         :237 : 301282370924 us: [pid:145680 tid:0x7fffed309180] __hipPopCallConfiguration: Returned hipSuccess :
:3:hip_module.cpp           :669 : 301282370943 us: [pid:145680 tid:0x7fffed309180]  hipLaunchKernel ( 0x201828, {33,1,1}, {33,1,1}, 0x7fffffffdc10, 0, stream:<null> )
:3:rocdevice.cpp            :2935: 301282370964 us: [pid:145680 tid:0x7fffed309180] number of allocated hardware queues with low priority: 0, with normal priority: 0, with high priority: 0, maximum per priority is: 4
:3:rocdevice.cpp            :3013: 301282375932 us: [pid:145680 tid:0x7fffed309180] created hardware queue 0x7fffed28e000 with size 16384 with priority 1, cooperative: 0
:3:rocdevice.cpp            :3105: 301282375951 us: [pid:145680 tid:0x7fffed309180] acquireQueue refCount: 0x7fffed28e000 (1)
:3:devprogram.cpp           :2679: 301282620849 us: [pid:145680 tid:0x7fffed309180] Using Code Object V5.
:3:rocvirtual.cpp           :3016: 301282651089 us: [pid:145680 tid:0x7fffed309180] ShaderName : _Z11crux_kernelv
:3:rocdevice.cpp            :3163: 301282654488 us: [pid:145680 tid:0x7fffed309180] Created hostcall buffer 0x7ffee4a00000 for hardware queue 0x7fffed28e000
[New Thread 0x7ffee49ff640 (LWP 145696)]
[Thread 0x7ffee49ff640 (LWP 145696) exited]
[New Thread 0x7fffed267640 (LWP 145697)]
:3:devhostcall.cpp          :404 : 301282655276 us: [pid:145680 tid:0x7fffed309180] Launched hostcall listener at 0xb55dd0
:3:devhostcall.cpp          :417 : 301282655284 us: [pid:145680 tid:0x7fffed309180] Registered hostcall buffer 0x7ffee4a00000 with listener 0xb55dd0
:3:rocvirtual.cpp           :798 : 301282655358 us: [pid:145680 tid:0x7fffed309180] Arg0: ulong heap_to_initialize = val:140732622962688
:3:rocvirtual.cpp           :798 : 301282655364 us: [pid:145680 tid:0x7fffed309180] Arg1: ulong initial_blocks = val:140732612476928
:3:rocvirtual.cpp           :798 : 301282655368 us: [pid:145680 tid:0x7fffed309180] Arg2: uint heap_size = val:131072
:3:rocvirtual.cpp           :798 : 301282655372 us: [pid:145680 tid:0x7fffed309180] Arg3: uint number_of_initial_blocks = val:4
:3:rocvirtual.cpp           :3016: 301282655376 us: [pid:145680 tid:0x7fffed309180] ShaderName : __amd_rocclr_initHeap
:3:hip_module.cpp           :670 : 301282655406 us: [pid:145680 tid:0x7fffed309180] hipLaunchKernel: Returned hipSuccess :
:3:hip_device_runtime.cpp   :608 : 301282655416 us: [pid:145680 tid:0x7fffed309180]  hipDeviceSynchronize (  )
:3:rocvirtual.hpp           :66  : 301282655426 us: [pid:145680 tid:0x7fffed309180] Host active wait for Signal = (0x7fffed1ff780) for -1 ns

Thread 6 "crux" received signal SIGSEGV, Segmentation fault.
[Switching to thread 6, lane 1 (AMDGPU Lane 1:2:1:1/1 (31,0,0)[0,0,1])]
0x00007fffed2b8b18 in crux_p1sin (p1sin_out=<error reading variable: Cannot access memory at address private_lane#0x4000>) at crux.cu:7
7       {
   0x00007fffed2b8b00 <_Z10crux_p1sinPd+0>:     00 00 89 bf     s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
   0x00007fffed2b8b04 <_Z10crux_p1sinPd+4>:     21 00 83 be     s_mov_b32 s3, s33
   0x00007fffed2b8b08 <_Z10crux_p1sinPd+8>:     20 00 a1 be     s_mov_b32 s33, s32
   0x00007fffed2b8b0c <_Z10crux_p1sinPd+12>:    c1 24 80 be     s_xor_saveexec_b32 s0, -1
   0x00007fffed2b8b10 <_Z10crux_p1sinPd+16>:    08 00 69 dc 00 05 21 00 scratch_store_b32 off, v5, s33 offset:8
=> 0x00007fffed2b8b18 <_Z10crux_p1sinPd+24>:    00 00 fe be     s_mov_b32 exec_lo, s0
   0x00007fffed2b8b1c <_Z10crux_p1sinPd+28>:    05 00 61 d7 7e 00 01 00 v_writelane_b32 v5, exec_lo, 0
   0x00007fffed2b8b24 <_Z10crux_p1sinPd+36>:    20 90 20 81     s_add_i32 s32, s32, 16
   0x00007fffed2b8b28 <_Z10crux_p1sinPd+40>:    00 03 04 7e     v_mov_b32_e32 v2, v0
   0x00007fffed2b8b2c <_Z10crux_p1sinPd+44>:    01 03 06 7e     v_mov_b32_e32 v3, v1
   0x00007fffed2b8b30 <_Z10crux_p1sinPd+48>:    ed 01 80 be     s_mov_b64 s[0:1], src_private_base
   0x00007fffed2b8b34 <_Z10crux_p1sinPd+52>:    a0 00 82 be     s_mov_b32 s2, 32
   0x00007fffed2b8b38 <_Z10crux_p1sinPd+56>:    00 02 80 85     s_lshr_b64 s[0:1], s[0:1], s2
   0x00007fffed2b8b3c <_Z10crux_p1sinPd+60>:    21 02 00 7e     v_mov_b32_e32 v0, s33
   0x00007fffed2b8b40 <_Z10crux_p1sinPd+64>:    00 02 08 7e     v_mov_b32_e32 v4, s0
   0x00007fffed2b8b44 <_Z10crux_p1sinPd+68>:    04 03 02 7e     v_mov_b32_e32 v1, v4
   0x00007fffed2b8b48 <_Z10crux_p1sinPd+72>:    00 00 6c dc 00 02 7c 00 flat_store_b64 v[0:1], v[2:3]
(gdb) info threads
  Id   Target Id                                 Frame
  1    Thread 0x7fffed309180 (LWP 145680) "crux" 0x00007fffed4996a2 in ?? () from /opt/rocm-6.1.2/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
  2    Thread 0x7fffecfff640 (LWP 145694) "crux" __GI___ioctl (fd=3, request=3222817548) at ../sysdeps/unix/sysv/linux/ioctl.c:36
  5    Thread 0x7fffed267640 (LWP 145697) "crux" __GI___ioctl (fd=3, request=3222817548) at ../sysdeps/unix/sysv/linux/ioctl.c:36
* 6    AMDGPU Wave 1:2:1:1 (31,0,0)/1 "crux"     0x00007fffed2b8b18 in crux_p1sin (p1sin_out=<error reading variable: Cannot access memory at address private_lane#0x4000>) at crux.cu:7
```

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

it even works when hip targets the cuda device

```bash
make clean CXX="HIP_PLATFORM=nvidia CUDA_PATH=/usr/lib/nvidia-cuda-toolkit/ hipcc" GDB=rocgdb GPUFLAGS="-g -O0 -DDEBUG=1" NDIRS=33 dbg_fee
```
