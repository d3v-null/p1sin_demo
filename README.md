# hyperbeam demo

This demonstrates a segfault that occurs in hip, but not when the same code is run in cuda.

## setup

- change `--offload-arch` in Makefile
- change `num_directions` in `fee.cu:main`. on my hardware, this segfaults at 33+ directions, but not 32. If it doesn't segfault, try 999 directions.

## Run in debugger

```bash
make clean && make ./fee
rocgdb -x rocgdbinit --args ./fee
```