// pointless overkill for producer before changing approach completely

// Deadlocks…


namespace M10 {

CUtensorMap d_tma_map_A, d_tma_map_B, d_tma_map_C;
int _prev_m=0, _prev_n=0, _prev_k=0;
__constant__ __int128 descAB[32];
__constant__ int next_desc_id[32];

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true>
__host__ static inline CUtensorMap create_tensor_map(bf16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64*sizeof(bf16), 0, 0, 0};
    uint32_t smem_box_shape[5] = {64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[16][8], uint64_t desc_a, uint64_t desc_b) {
    // On H100, this takes 128 cycles: 256N * 64M * 16K * 2 flops = 512K
    // 4096 BF16 flops per SM per clock ==> 512K / 4096 = 128 cycles
    // So doing 4 of those with BK=64 takes 512 cycles
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N/16][8], __int128 desc128) {
    uint64_t desc_a = ((uint64_t*)&desc128)[0]; // N.B.: ULDC is up to 64-bit reads, so can't coalesce into 128-bit anyway
    uint64_t desc_b = ((uint64_t*)&desc128)[1];

    // TODO: only using 256 right now, but 128 might be useful for the 1st & last iterations of each BM*BN tile
    // so we could parallelise some of the pre/post-processing with the matmuls of the other half of the tile
    static_assert(WGMMA_N == 256);
    wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a, desc_b);
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ static __forceinline__ void expect_bytes(uint32_t mbar_ptr, uint32_t bytes) {
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(mbar_ptr), "r"(bytes));
}

__device__ static __forceinline__ void load_async(uint32_t dst_ptr, void const* src_tma_map, uint32_t mbar_ptr, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
        : "memory"
    );
}

__device__ static __forceinline__ void wait(uint32_t mbar_ptr, int kPhaseBit) {
    // Call mbarrier.try_wait in a while loop till it returns true.
    // slight variants (e.g. branch to DONE on @P then unconditionally to LAB_WAIT) result in different code
    // not obvious what's best in the general case or why the compiler acts the way it does, but good enough for now
    asm volatile (
        "{\n"
        ".reg .pred                P;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P                      bra.uni LAB_WAIT;\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void wait_warpgroup_arrive(uint32_t mbar_ptr, int kPhaseBit) {
    wait(mbar_ptr, kPhaseBit);
    warpgroup_arrive();
}

__device__ static inline void load_async_multicast(uint32_t dst_ptr, void const* src_tma_map, uint32_t mbar_ptr, int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

    asm volatile (
    "{\n"
        //".reg .pred P;\n"
        //"elect.sync _|P, 0xFFFFFFFF;\n"
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        "}\n"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx/64), "h"(cluster_mask)
        : "memory"
    );
}

__device__ void arrive_cluster(uint32_t mbar_ptr, uint32_t cta_id, uint32_t count=1) {
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(mbar_ptr), "r"(cta_id), "r"(count));
}

// to help the compiler not be silly... (threadIdx.x == constant should be enough, come on guys!)
__device__ static __forceinline__ void elect_or_exit() {
    asm volatile (
        "{\n"
        ".reg .pred P;\n"
        "elect.sync _|P, 0xFFFFFFFF;\n"
        "@!P exit;"
        "}\n"
        :
    );
}




template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN, int CLUSTER_M>
struct Schedule;

template<int NUM_SM, int BM, int BN, int TM, int TN, int CLUSTER_M>
struct Schedule<1, NUM_SM, BM, BN, TM, TN, CLUSTER_M> {
    int block;
    int it;
    int total_blocks_m, total_blocks_n;
    int rank_m;

    __device__ __forceinline__ Schedule(int M, int N, int _rank_m, int _block) {
        it = 0, block = _block, rank_m = _rank_m;
        total_blocks_m = CEIL_DIV(M, BM);
        total_blocks_n = CEIL_DIV(N, BN);
        //assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0); // TODO: check on CPU
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        int num = it*NUM_SM + block;
        if (num >= total_blocks_m*total_blocks_n) return false;

        int cur_tile = num / (TM*TN);
        int cur_tile_pos = num % (TM*TN);
        block_m = TM*(cur_tile / (total_blocks_n/TN));
        block_n = TN*(cur_tile % (total_blocks_n/TN));
        block_m += cur_tile_pos / TN;
        block_n += cur_tile_pos % TN;
        block_n = block_n;
        block_m = block_m * CLUSTER_M + rank_m;
        it++;
        return true;
    }
};

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
    alignas(128) bf16 C[BN*BM/4];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
};

#define FULL_PTR(i) (full_start + i*8)
#define EMPTY_PTR(i) (empty_start + i*8)
#define SA_PTR(i) (sA_start + i*BK*BM*2)
#define SB_PTR(i) (sB_start + i*BK*BN*2)

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int CLUSTER_M, int DELAYED_WAIT=0, bool RELU=false, bool SQUARED=false>
__global__  __launch_bounds__(NUM_THREADS) void  __cluster_dims__(CLUSTER_M, 1, 1) matmulKernel10(int M, int N, int K, bf16* C, const __grid_constant__ CUtensorMap tensorMapC, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB) {
    constexpr int MINIMUM_ANY_DIMENSION = 256;
    constexpr int MINIMUM_K_ITERATIONS = MINIMUM_ANY_DIMENSION / BK;

    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    bf16 *sA = s.A, *sB = s.B, *sC = s.C;
    uint64_t *full = s.full, *empty = s.empty;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier(&full[i], 0, 1);
            init_barrier(&empty[i], 0, num_consumers*CLUSTERS);
        }
    }

    // ------------------------------------------------------------------------------------------------
    asm volatile("barrier.cluster.arrive;\n" : :);

    uint32_t cluster_id, cluster_rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_rank) :);
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    // trick to make sure compiler knows this is uniform for a given warp
    // TODO: check if this is necessary for these specific variables (you'd hope the compiler was smart enough but...)
    wg_idx = __shfl_sync(0xffffffff, wg_idx, 0);
    cluster_id = __shfl_sync(0xffffffff, cluster_id, 0);
    cluster_rank = __shfl_sync(0xffffffff, cluster_rank, 0);

    constexpr uint32_t sA_base = 0x0400;
    constexpr uint32_t sB_base = sA_base + sizeof(s.A);
    constexpr uint32_t sC_base = sB_base + sizeof(s.B);
    constexpr uint32_t full_base = sC_base + sizeof(s.C);
    constexpr uint32_t empty_base = full_base + sizeof(s.full);

    uint32_t sA_start = sA_base + cluster_rank * 0x1000000;
    uint32_t sB_start = sB_base + cluster_rank * 0x1000000;
    uint32_t full_start = full_base + cluster_rank * 0x1000000;
    uint32_t empty_start = empty_base + cluster_rank * 0x1000000;

    Schedule<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN, 16/CLUSTER_M, 8, CLUSTER_M> schedule(M, N, cluster_rank, cluster_id);

    int num_block_m, num_block_n;
    bool schedule_next = schedule.next(num_block_m, num_block_n);

    asm volatile("barrier.cluster.wait;\n" : :);
    // ------------------------------------------------------------------------------------------------

    const int num_blocks_k = K / BK;

    // Producer
    if (wg_idx == 0) {
        // 24 because it is impossible to allocate all 512 registers using 3 warps per sub-core...
        // 168*3 = 504, so we "lose" 8 registers per sub-core! *sigh*
        warpgroup_reg_dealloc<24>();
        elect_or_exit();

        if (threadIdx.x == 0) {
            int p = 0, qidx = 0;
            if (CLUSTER_M > 1 && cluster_rank == 0) {
                constexpr uint32_t multicast_mask = (1 << CLUSTER_M) - 1;
                while (schedule_next) {
                    #pragma unroll 2
                    for (int block_k_iter = 0; block_k_iter < num_blocks_k;) {
                        if constexpr (MINIMUM_K_ITERATIONS % QSIZE == 0) {
                            qidx = 0;
                        }
                        #pragma unroll MINIMUM_K_ITERATIONS
                        for (int j = 0; j < MINIMUM_K_ITERATIONS; j++, block_k_iter++) {
                            expect_bytes(FULL_PTR(qidx), (BK*BN+BK*BM)*sizeof(bf16));
                            load_async_multicast(SB_PTR(qidx), &tensorMapB, FULL_PTR(qidx), block_k_iter*BK, num_block_n*BN, multicast_mask);
                            load_async(SA_PTR(qidx), &tensorMapA, FULL_PTR(qidx), block_k_iter*BK, num_block_m*BM);

                            if (++qidx == QSIZE) { qidx = 0; p ^= 1; }
                            wait(EMPTY_PTR(qidx), p);
                        }
                    }

                    schedule_next = schedule.next(num_block_m, num_block_n);
                }
            } else {
                while (schedule_next) {
                    #pragma unroll 2
                    for (int block_k_iter = 0; block_k_iter < num_blocks_k;) {
                        if constexpr (MINIMUM_K_ITERATIONS % QSIZE == 0) {
                            qidx = 0;
                        }
                        #pragma unroll MINIMUM_K_ITERATIONS
                        for (int j = 0; j < MINIMUM_K_ITERATIONS; j++, block_k_iter++) {
                            expect_bytes(FULL_PTR(qidx), (BK*BN+BK*BM)*sizeof(bf16));
                            load_async(SA_PTR(qidx), &tensorMapA, FULL_PTR(qidx), block_k_iter*BK, num_block_m*BM);
                            if constexpr (CLUSTER_M == 1) {
                                load_async(SB_PTR(qidx), &tensorMapB, FULL_PTR(qidx), block_k_iter*BK, num_block_n*BN);
                            }

                            if (++qidx == QSIZE) { qidx = 0; p ^= 1;}
                            wait(EMPTY_PTR(qidx), p);
                        }
                    }
                    schedule_next = schedule.next(num_block_m, num_block_n);
                }
            }
        }
    } else {
        warpgroup_reg_alloc<240>();
        wg_idx -= 1;
        for (int qidx = 0; qidx < QSIZE; qidx++) {
            if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(qidx), tid);
        }

        float d[WGMMA_N/16][8];
        bf16 d_bf16[WGMMA_N/16][8];

        bf16* block_sC = sC + wg_idx*B_WG_M*BN/4;
        int4* block_sC_128b = (int4*)block_sC;
        int* block_sC_32b = (int*)block_sC;
        bf16 *block_C = C;

        int4* out0[2] = { &block_sC_128b[tid], &block_sC_128b[tid + B_WG_M*BN/(8*8)] };
        int4* out1[2] = { &block_sC_128b[tid + 128], &block_sC_128b[tid + B_WG_M*BN/(8*8) + 128] };

        int desc_id = wg_idx * 64/WGMMA_K * QSIZE;
        constexpr __int128 desc_multiplier = ((__int128)0x2 << (__int128)64) | (__int128)0x2;

        int x = ((threadIdx.x % 8) * 8) + (threadIdx.x / 128 - 1) * 64;
        int y_base = ((threadIdx.x % 128) / 8) * 2;
        int x_wg = x % 64;
        int idx_32b_x = ((x_wg % 16) / 8 + (x_wg / 16) * 32 * 4);
        int idx_32b_base = idx_32b_x + (y_base % 8) * 4 / 2 + ((y_base / 8) % 2) * 2;

        int p = 0, qidx = 0, old_qidx = 0;
        bool output_to_gmem = false;

        int num_blocks_k = K / BK;

        while (schedule_next) {
            float absmax = 0.0f;
            int4* block_C_thread = (int4*)&block_C[x + y_base*M];

            constexpr int post_process_iterations = 8;
            constexpr int write_iterations = 4;
            constexpr int unrolled_iterations = post_process_iterations + write_iterations;

            #pragma unroll
            for (int iter = 0; iter < unrolled_iterations; iter++) {
                __int128 desc128 = descAB[desc_id];
                desc_id = next_desc_id[desc_id];

                // TODO: is the implicit sync from wait_warpgroup_arrive() + WGMMA enough?
                // with double buffering, we only need to sync inside a warpgroup (not the full threadgroup)
                if (iter < 8) {
                    *out0[iter%2] = ((int4*)d_bf16)[iter*2];
                    *out1[iter%2] = ((int4*)d_bf16)[iter*2+1];
                }

                if (iter > DELAYED_WAIT) {
                    warpgroup_wait<DELAYED_WAIT>();
                    if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(old_qidx), tid);
                }

                wait_warpgroup_arrive(FULL_PTR(qidx), p);
                if (iter == 0) wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d, desc128);
                for (int k_it = iter ? 0 : 1; k_it < 64/WGMMA_K; k_it++) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d, desc128 | (k_it * desc_multiplier));
                }
                warpgroup_commit_batch();

                if (output_to_gmem) {
                    if (iter < post_process_iterations) {
                        bf16 data_bf16_col[2][8];
                        int idx_32b = idx_32b_base + (y_base / 16) * 4 * 128;
                        if (iter % 2) idx_32b += B_WG_M*BN/(8*2);

                        for(int k = 0; k < 8; k++) {
                            int data = block_sC_32b[idx_32b];
                            idx_32b += 4 * 4;

                            if constexpr (SQUARED) {
                                int zero = 0;
                                asm volatile("fma.rn.bf16x2 %0, %1, %1, %2;" : "=r"(data) : "r"(data), "r"(zero));
                            }
                            d_bf16[iter*2+0][k] = ((bf16*)&data)[0];
                            d_bf16[iter*2+1][k] = ((bf16*)&data)[1];

                            /*int data0 = data << 16;
                            int data1 = data & 0xFFFF0000;
                            float d0 = *(float*)&data0;
                            float d1 = *(float*)&data1;

                            // TODO: Elementwise processing here, e.g. GELU
                            // this will require an additional BF16 input in cases like GELU backwards
                            // and smart prefetching from memory with limited available register/smem/l2 space...
                            data_bf16_col[0][k] = (bf16)max(0.0f, d0);
                            data_bf16_col[1][k] = (bf16)max(0.0f, d1);
                            */
                        }
                    } else if (iter - post_process_iterations < write_iterations) {
                        constexpr int writes_per_iteration = post_process_iterations / write_iterations;
                        int i = (iter - post_process_iterations) * writes_per_iteration;

                        for (int batch = 0; batch < writes_per_iteration; batch++, i++) {
                            for (int col = 0; col < 2; col++) {
                                __stcs(&block_C_thread[col*M/8], ((int4*)d_bf16)[i*2+col]);
                            }
                            block_C_thread += 4*M;
                        }
                    }
                }

                old_qidx = qidx - DELAYED_WAIT;
                if (qidx < DELAYED_WAIT) old_qidx += QSIZE;
                if (++qidx == QSIZE) {qidx = 0; p ^= 1; };
            }

            for (int block_k_iter = unrolled_iterations; block_k_iter < num_blocks_k; block_k_iter++) {
                __int128 desc128 = descAB[desc_id];
                desc_id = next_desc_id[desc_id];

                warpgroup_wait<DELAYED_WAIT>();
                if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(old_qidx), tid);
                wait_warpgroup_arrive(FULL_PTR(qidx), p);

                for (int k_it = 0; k_it < 64/WGMMA_K; k_it++) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d, desc128 | (k_it * desc_multiplier));
                }
                warpgroup_commit_batch();

                old_qidx = qidx - DELAYED_WAIT;
                if (qidx < DELAYED_WAIT) old_qidx += QSIZE;
                if (++qidx == QSIZE) {qidx = 0; p ^= 1; };
            }

            output_to_gmem = true;
            block_C = C + num_block_n*BN*M + num_block_m*BM;
            schedule_next = schedule.next(num_block_m, num_block_n);

            warpgroup_wait<DELAYED_WAIT>();
            if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(old_qidx), tid);

            if constexpr (DELAYED_WAIT > 0) {
                warpgroup_wait<0>();
                for (int i = 1; i <= DELAYED_WAIT; i++) {
                    old_qidx = (old_qidx + 1) % QSIZE;
                    if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(old_qidx), tid);
                }
            }

            for(int n_tile = 0; n_tile < 16; n_tile++) {
                if constexpr (RELU) {
                    for (int k = 0; k < 8; k += 2) {
                        asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(*(int*)(&d_bf16[n_tile][k])) : "f"(d[n_tile][k+1]), "f"(d[n_tile][k]));
                    }
                } else {
                    for (int k = 0; k < 8; k++) {
                        d_bf16[n_tile][k] = (bf16)d[n_tile][k];
                    }
                }
            }

            // on the last iteration, we can't overlap the global memory writes with the matmuls, so do it immediately
            if (!schedule_next) {
                bf16* block_C_thread = &block_C[x + y_base*M];
                #pragma unroll
                for (int iter = 0; iter < 8; iter++) {
                    // double buffering means we don't need a barrier between the read and write, only write to read
                    *out0[iter%2] = ((int4*)d_bf16)[iter*2];
                    *out1[iter%2] = ((int4*)d_bf16)[iter*2+1];
                    // synchronise per-warpgroup (don't need to read shared memory written by the other warpgroup)
                    asm volatile("bar.sync %0, 128;\n" :: "r"(wg_idx));

                    bf16 data_bf16_col[2][8];
                    int idx_32b = idx_32b_base + (y_base / 16) * 4 * 128;
                    if (iter % 2) idx_32b += B_WG_M*BN/(8*2);

                    for(int k = 0; k < 8; k++) {
                        int data = block_sC_32b[idx_32b];
                        data_bf16_col[0][k] = ((bf16*)&data)[0];
                        data_bf16_col[1][k] = ((bf16*)&data)[1];
                        idx_32b += 4 * 4;
                    }

                    __stcs((int4*)&block_C_thread[0], *(int4*)data_bf16_col[0]);
                    __stcs((int4*)&block_C_thread[M], *(int4*)data_bf16_col[1]);
                    block_C_thread += 32*M;
                }
            }
        }
    }
}

void runKernel10(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128*3;
    constexpr int QSIZE = 4;
    constexpr int CLUSTER_M = 2;
    constexpr int NUM_SM = 114; // H100 PCIe :(
    static_assert(NUM_SM % (CLUSTER_M) == 0);
    assert(K >= 8 * BK);

    if (_prev_m != M || _prev_n != N || _prev_k != K) {
        d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
        d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);
        d_tma_map_C = create_tensor_map<BN, BM, false>(C, N, M);
        _prev_m = M, _prev_n = N, _prev_k = K;

        // TODO: make this slightly less hacky
        constexpr uint64_t desc_base = 0x4000004000010000;
        constexpr int K_iterations = BK/16;
        constexpr int uniqueA = K_iterations * QSIZE * CLUSTER_M;
        constexpr int uniqueB = K_iterations * QSIZE;
        constexpr int sizeA = (BM * BK * sizeof(bf16)) >> 4;
        constexpr int sizeB = (BN * BK * sizeof(bf16)) >> 4;
        constexpr int sizeA_warpgroup_offset = sizeA / (NUM_THREADS/128 - 1);
        constexpr int startA = 0x40; // 1024 bytes used by driver (not clear if that is *always* the case though...)
        constexpr int startB = startA + sizeA*QSIZE;
        constexpr int offsetK = 0x2;

        uint64_t hostA[uniqueA], hostB[uniqueB];
        for (int i = 0; i < uniqueA; i++) {
            hostA[i] = (startA + (i % K_iterations) * offsetK + ((i / K_iterations) % QSIZE) * sizeA + ((i / K_iterations) / QSIZE) * sizeA_warpgroup_offset) | desc_base;
        }
        for (int i = 0; i < uniqueB; i++) {
            hostB[i] = (startB + (i % K_iterations) * offsetK + ((i / K_iterations) % QSIZE) * sizeB) | desc_base;
        }

        __int128 hostAB[uniqueA];
        for (int i = 0; i < uniqueA; i++) {
            hostAB[i] = (__int128)hostA[i] | ((__int128)hostB[i % uniqueB] << (__int128)64);
        }
        cudaMemcpyToSymbol(descAB, hostAB, sizeof(hostAB));

        int next_desc_id_host[uniqueA];
        for (int i = 0; i < uniqueA; i++) {
            next_desc_id_host[i] = ((K_iterations + i) % uniqueB) + (i >= uniqueB ? uniqueB : 0);
        }
        cudaMemcpyToSymbol(next_desc_id, next_desc_id_host, sizeof(next_desc_id_host));
    }

    auto* kernel = matmulKernel10<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(M, N, K, C, d_tma_map_C, d_tma_map_A, d_tma_map_B);
}

} // namespace M10

using M10::runKernel10;
