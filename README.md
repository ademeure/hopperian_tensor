# Insanely Fast H100 Matrix Multiplication

WIP experiments forked from the H100 matrix multiplication code of https://github.com/pranjalssh/fast.cu/

The most exciting optimisation is support for "hybrid cluster sizes": as only 120 SMs can be used with cluster sizes of 4 or 8, we launch the variants of the same kernel:
1) 120 SMs with a cluster size of 8 (maximum power efficiency)
2) 8(*) SMs with a cluster size of 2 (to complete our square 2048x2048 supertile)

*: Using all 132 SMs is slightly slower when power limited due to worse cache locality

A whole bunch of other crazy tricks have been implemented (and/or attempted in older versions of this repo), such as hardcoding the shared memory addresses in a way that significantly reduces the number of SASS instructions for address calculations. These allow us to get extremely close to the theoretical peak at low clock frequencies but aren't actually very useful when power limited or at small sizes (and at least several of them no longer help after other changes). Support for L2 side awareness of Matrix A was present at one point but didn't help much. But it only reduced cache duplication and not DRAM accesses, it's likely that "full" L2 side awareness would help much more.

I do not plan to develop this further and it is not production grade, but might be interested in building some crazy fast Blackwell kernels in the future... :)

---

GH200 96GiB 700W Power Limited

16384x16384
- cuBLAS: ~715 to ~710 TFLOPS
- fast.cu kernel 12: ~710 to ~705
- this repo (cluster 2): ~710 to ~705 TFLOPS
- this repo (hybrid 2+8): ~740 to ~725 TFLOPS (>3% Perf/W!)

8192x8192
- cuBLAS: ~700
- fast.cu kernel 11: >700+
- fast.cu kernel 12: ~710 to ~705
- this repo (cluster 2): ~720 to ~715
- this repo (hybrid 2+8): ~735 to ~730 (>3% Perf/W!)

4096x4096
- cuBLAS: ~690
- fast.cu kernel 12: ~678
- this repo (cluster 2): ~675
- this repo (hybrid 2+8): ~683 (might need better ordering for cache locality?)

(sudo nvidia-smi boost-slider --vboost 1, very high repeat count, results different from fast.cu baselines due to focus on sustained performance/power as opposed to short runs that don't have much thermal throttling with fast.cu defaults)
