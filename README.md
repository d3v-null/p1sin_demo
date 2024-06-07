# hyperbeam demo

This demonstrates a segfault that occurs in hip, but not when the same code is run in cuda.

## setup

- change `--offload-arch` in `GPUFLAGS` make variable for your hardware
- increase `NDIRS` until it segfaults.

## Run in debugger

### fee

full example of how code is actually used

```bash
make clean NDIRS=99 GPUFLAGS=--offload-arch=gfx1101 fee_debug
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
make clean NDIRS=99 GPUFLAGS=--offload-arch=gfx1101 p1sin_debug
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