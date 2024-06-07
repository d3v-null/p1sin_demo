# hyperbeam demo

This demonstrates a segfault that occurs in hip, but not when the same code is run in cuda.

## setup

- change `--offload-arch` in Makefile
- change `num_directions` in `fee.cu:main`. on my hardware, this segfaults at 33+ directions, but not 32. If it doesn't segfault, try 999 directions.

## Run in debugger

### fee

full example of how code is actually used

```bash
make clean fee_debug
```

### p1sin

only calls `jones_p1sin_device` where the segfault happens in the previous example.

Does not trigger the segfault in HIP?

```bash
make clean p1sin_debug
```