# Insanely Fast H100 Matrix Multiplication

Experimental & personal playground for crazy H100 kernel ideas originally forked from https://github.com/pranjalssh/fast.cu/

The most successful optimisation is support for "hybrid cluster sizes": as only 120 SMs can be used with cluster sizes of 4 or 8, we launch the variants of the same kernel:
1) 120 SMs with a cluster size of 8 (maximum power efficiency)
2) 8(*) SMs with a cluster size of 2 (to complete our square 2048x2048 supertile)

*: Using all 132 SMs is slightly slower when power limited due to worse cache locality

A whole bunch of other crazy tricks have been implemented (and/or attempted in older versions of this repo), such as hardcoding the shared memory addresses in a way that significantly reduces the number of SASS instructions for address calculations. These allow us to get extremely close to the theoretical peak at low clock frequencies but aren't actually very useful when power limited or at small sizes (and at least several of them no longer help after other changes). Support for L2 side awareness of Matrix A was present at one point but didn't help much. But it only reduced cache duplication and not DRAM accesses, it's likely that "full" L2 side awareness would help much more but it would require storing data in a very complicated order (see: some of the experiments I did on my l2sideaware llm.c branch).

I do not plan to develop this further and it is not production grade, but might be interested in building some (production grade?) crazy fast Blackwell kernels in the future... :)

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
- this repo (hybrid 2+8): ~683 (might need better ordering for cache locality? or possibly 256B DRAM bursts?)

sudo nvidia-smi boost-slider --vboost 1, very high repeat count, results different from fast.cu baselines due to focus on sustained performance/power as opposed to short runs that don't have much thermal throttling with fast.cu defaults)

---
A few crazy things I played with & furure ideas
---
Something cool that I nearly got working in an earlier commit is "per 256x256 tile FP8 scaling" which would allow true drop-in FP8, and I managed to get the tile absmax calculation and extra multiplications (interleaved between the two producers) to run at close to peak performance which was interesting because of the synchronization requirements! But it's complex and not orthogonal with other features, so I decided I did not want to spend lots of time to make it production grade. There's still unscaled FP8 support in the latest version which "just works" though!

One "obvious" optimisation would be increasing BK from 64 to 128 so that we load 256 consecutive bytes which matches the H100 DRAM hashing and would reduce DRAM page open/close power. However a whole bunch of things have been changed from the fast.cu original making an efficient implementation of this difficult while maximising the number of tiles in flight etc...

This could also be solved at the same time as L2 side awareness by changing the memory layout so that 64x64 FP8 subtiles (or 64x32 BF16) are stored contiguously in a way that guarantees they are on the same side of the L2, then some crazy swizzling so that half of A & D are on one side (in a matching way) and the other half on the other, and having SMs of each side process "their part" of A(/D). But it would be extremely complex unless you could simply allocate memory in such a way that everything is on one side or the other at 2MiB granularity rather than 4KiB (nice & simple if we could do that!)

Sadly I really can't afford to spend the time refactoring things for the 3rd or 4th time on something that will never be production grade, as fun as these experiments were for the most part! Maybe for Blackwell or another GPU/AI accelerator, we'll see... :)
