namespace M10 {

constexpr bool hybrid_cluster_sizes = true; // hybrid of 8-wide + 2-wide clusters, see README.md
constexpr int supertile_sync_tolerance = -1; // to force supertile to be in sync (overhead vs cache locality)
constexpr int post_process_iterations = 8; // iterations used for transposing previous output (+elementwise?)
constexpr int write_iterations = 2; // iterations used for writing output (of previous tile) to global memory
constexpr int unrolled_iterations = post_process_iterations + write_iterations;

CUtensorMap d_tma_map_A, d_tma_map_B, d_tma_map_C, d_tma_map_I;
int _prev_m=0, _prev_n=0, _prev_k=0;
__constant__ __int128 descAB[32];

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

template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true, typename T=floatX>
__host__ static inline CUtensorMap create_tensor_map(void* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    constexpr size_t MAGIC_VALUE = 128/sizeof(T);
    static_assert(BlockMinorSize >= MAGIC_VALUE);
    assert(global_width % MAGIC_VALUE == 0);
    uint64_t gmem_prob_shape[5] = {MAGIC_VALUE, (uint64_t)global_height, (uint64_t)global_width/MAGIC_VALUE, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(T) * global_width, MAGIC_VALUE*sizeof(T), 0, 0, 0};
    uint32_t smem_box_shape[5] = {MAGIC_VALUE, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/MAGIC_VALUE), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, std::is_same<T, floatX>::value ? CU_TENSOR_FLOATX : CU_TENSOR_FLOATP, 3, gmem_address, gmem_prob_shape,
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
        WGMMA_INSTRUCTION
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
        " %128, %129,"
#ifdef FP8
        " %130, %131, %132;\n"
#else
        " %130, %131, %132, %133, %134;\n"
#endif
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
        : "l"(desc_a), "l"(desc_b),
#ifdef FP8
        "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
#else
        "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N/16][8], __int128 desc128) {
    uint64_t desc_a = ((uint64_t*)&desc128)[0]; // N.B.: ULDC is up to 64-bit reads, so HW won't coalesce into 128-bit
    uint64_t desc_b = ((uint64_t*)&desc128)[1];

    // TODO: only using 256 right now, but 128 might be useful for the 1st & last iterations of each BM*BN tile
    // so we could parallelise some of the pre/post-processing with the matmuls of the other half of the tile
    // (probably not worth the complexity though, compared to every other crazy trick I still want to try!)
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

__device__ static void init_barrier(uint64_t* bar, int count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar_ptr), "r"(count)
    );
}

__device__ static void expect_bytes(uint32_t mbar_ptr, uint32_t bytes) {
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mbar_ptr), "r"(bytes));
}

template<typename T=floatX>
__device__ static void load_async(uint32_t dst_ptr, void const* src_tma_map, uint32_t mbar_ptr, int global_col_idx, int global_row_idx) {
    // TODO: Support "streaming" cache policy (for C/I)
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);


    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx/(128/(int)sizeof(T))) : "memory"
    );
}

__device__ static inline void load_async_multicast(uint32_t dst_ptr, void const* src_tma_map, uint32_t mbar_ptr, int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;\n"
        :: "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx/(128/(int)sizeof(floatX))), "h"(cluster_mask) : "memory"
    );
}

__device__ static void wait(uint32_t mbar_ptr, int kPhaseBit) {
    // Call mbarrier.try_wait in a while loop till it returns true.
    // slight variants (e.g. branch to DONE on @P then unconditionally to LAB_WAIT) result in different code
    // not obvious what's best in the general case or why the compiler acts the way it does, but good enough (for now)
    asm volatile (
        "{\n\t"
        ".reg .pred P;\n\t"
        "RETRY:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra.uni RETRY;\n"
        "}\n" :: "r"(mbar_ptr), "r"(kPhaseBit));
}

__device__ static void arrive_cluster(uint32_t mbar_ptr, uint32_t cta_id, uint32_t count=1) {
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n"
        "}" :: "r"(mbar_ptr), "r"(cta_id), "r"(count));
}

__device__ static void elect_or_exit() {
    // to help the compiler not be silly... (threadIdx.x == constant should be enough, come on guys!)
    asm volatile (
        "{\n\t"
        ".reg .pred P;\n\t"
        "elect.sync _|P, 0xFFFFFFFF;\n\t"
        "@!P exit;\n\t"
        "}\n" :: );
}

template<int VERSION, int BM, int BN, int TM, int TN, int M_MULT>
struct Schedule;

template<int BM, int BN, int TM, int TN, int M_MULT>
struct Schedule<1, BM, BN, TM, TN, M_MULT> {
    int super_tile_id;
    int num_super_tiles, num_super_n;
    int local_pos_m, local_pos_n;
    int total_blocks_m, total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        super_tile_id = 0;
        local_pos_m = _block % TN;
        local_pos_n = _block / TN;
        total_blocks_m = CEIL_DIV(M, BM);
        total_blocks_n = CEIL_DIV(N, BN);

        num_super_n = total_blocks_n / TN;
        num_super_tiles = (total_blocks_m*total_blocks_n) / (TM*TN);
        if (_block > 256) {
            num_super_tiles = 0; // skip this cluster
        }
        assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        if (super_tile_id >= num_super_tiles) { return false; }

        int m_super_pos = super_tile_id / num_super_n;
        block_m = TM * m_super_pos + local_pos_m;
        block_n = TN*(super_tile_id % num_super_n) + local_pos_n;
        if (m_super_pos % 2 != 0) {
            // go in reverse to slightly improve cache locality
            // without the complexity / SM count dependence of hilbert curves
            // (useful for testing, but not actually as good for small sizes...)
            block_n = total_blocks_n - block_n - 1;
        }

        block_m *= M_MULT;
        ++super_tile_id;
        return true;
    }
};


template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(1024) floatX A[BM*BK*QSIZE];
    alignas(1024) floatX B[BK*BN*QSIZE];
    alignas(1024) floatP C[BN*BM/2];
    // mbarriers
    alignas(8) uint64_t full[QSIZE];
    alignas(8) uint64_t empty[QSIZE];
    alignas(8) uint64_t absmax_barrier;
    // metadata shared across cluster
    alignas(16) ushort4 tileinfo[4];
};

#define FULL_PTR(i) (full_start + (i)*8)
#define EMPTY_PTR(i) (empty_start + (i)*8)
#define SA_PTR(i) (sA_start + (i)*BK*BM*sizeof(floatX))
#define SB_PTR(i) (sB_start + (i)*BK*BN*sizeof(floatX))
#define SC_PTR(i) (sC_start + ((i)*BM*BN*sizeof(floatP))/8)

template<bool SQUARED=false>
__device__ void output_postprocess(int i, floatP d_x16[256/16][8], int idx_32b_base, int y_base, int WGMMA_M, int BN, int tid, int4* out0[4], int4* out1[4], int wg_idx, int *block_sC_32b) {
    int4 input0, input1;

    int idx_32b = idx_32b_base + (y_base / 16) * 4 * 128;
    idx_32b += (i%4) * 2*WGMMA_M*BN / (8*2);

    int4 out0_data = ((int4*)d_x16)[i*2], out1_data = ((int4*)d_x16)[i*2+1];
    if constexpr (REDUCE_SHARED_CONFLICTS) {
        if (tid & 32) {
            out0_data = make_int4(out0_data.z, out0_data.w, out0_data.x, out0_data.y);
            out1_data = make_int4(out1_data.z, out1_data.w, out1_data.x, out1_data.y);
        }
        idx_32b ^= (idx_32b & 128) ? 2 : 0;
    }

    *out0[i%4] = out0_data, *out1[i%4] = out1_data;
    asm volatile("bar.sync %0, 128;\n" :: "r"(wg_idx));

    #pragma unroll 8
    for(int k = 0; k < 8; k++) {
        int data = block_sC_32b[idx_32b + k*16];

        if constexpr (SQUARED) {
            asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(data) : "r"(data), "r"(data), "r"(0));
        }

        d_x16[i*2+0][k] = ((floatP*)&data)[0];
        d_x16[i*2+1][k] = ((floatP*)&data)[1];
    }
}

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int CLUSTERS, bool RELU=false, bool SQUARED=false>
__global__  __launch_bounds__(NUM_THREADS) void  __cluster_dims__(CLUSTERS, 1, 1) matmulKernel10(int M, int N, int K, floatP* C, floatP* D, const __grid_constant__ CUtensorMap tensorMapI, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB, unsigned int* counters_zeroed, int base_pos, int num_sm) {
    constexpr int MULTIPLE_EVERY_DIMENSON = 256;
    constexpr int MINIMUM_K_ITERATIONS = MULTIPLE_EVERY_DIMENSON / BK;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1; // == 2
    constexpr int WGMMA_M = 64, WGMMA_N=BN, WGMMA_K = (sizeof(floatX) == 1 ? 32 : 16);

    // ------------------------------------------------------------------------------------------------
    uint32_t cluster_id, cluster_rank, smid;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_rank) :);
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    // trick to make sure compiler knows this is uniform for a given warp
    // TODO: check if this is necessary for these specific variables (you'd hope the compiler was smart enough but...)
    wg_idx = __shfl_sync(0xffffffff, wg_idx, 0);
    cluster_id = __shfl_sync(0xffffffff, cluster_id, 0);
    cluster_rank = __shfl_sync(0xffffffff, cluster_rank, 0);

    // ------------------------------------------------------------------------------------------------
    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    floatX *sA = s.A, *sB = s.B;
    floatP *sC = s.C;
    uint64_t *full = s.full, *empty = s.empty, *absmax_barrier = &s.absmax_barrier;

    constexpr uint32_t sA_base = 0x0400;
    constexpr uint32_t sB_base = sA_base + sizeof(s.A);
    constexpr uint32_t sC_base = sB_base + sizeof(s.B);
    constexpr uint32_t full_base = sC_base + sizeof(s.C);
    constexpr uint32_t empty_base = full_base + sizeof(s.full);

    uint32_t sA_start = sA_base + cluster_rank * 0x1000000;
    uint32_t sB_start = sB_base + cluster_rank * 0x1000000;
    uint32_t sC_start = sC_base + cluster_rank * 0x1000000;
    uint32_t full_start = full_base + cluster_rank * 0x1000000;
    uint32_t empty_start = empty_base + cluster_rank * 0x1000000;

    const int num_blocks_k = K / BK;
    int block_m, block_n;
    int tileinfo_idx = 0;
    bool schedule_next;
    
    // ------------------------------------------------------------------------------------------------
    // Control Warpgroup
    // Warp 0: Dynamic Scheduling (legacy, doesn't really need to be split anymore...)
    // Warp 1: Producer via TMA (+specialised copy-paste for cluster rank 0 doing multicast)
    // Warp 2: Updates regular barriers based on mbarrier status when producer data is ready
    // (using regular barriers was slightly faster at one point, but it might no longer be true!)
    if (wg_idx == 0) {
        // 24 because it is impossible to allocate all 512 registers using 3 warps per sub-core...
        // 168*3 = 504, so we "lose" 8 registers per sub-core! *sigh*
        warpgroup_reg_dealloc<24>();
        elect_or_exit();
        int p = 0, qidx = 0;

        if (threadIdx.x == 0) {
            // dynamic scheduling, writing next tile m/n to shared memory for entire cluster
            // this is completely useless now because we no longer actually do dynamic scheduling...
            // but keeping it because it looks cool (aka: deadlocked when tried removing & I gave up)
            for (int i = 0; i < QSIZE; ++i) {
                init_barrier(&full[i], 1);
                init_barrier(&empty[i], num_consumers*CLUSTERS);
            }
            init_barrier(absmax_barrier, num_consumers*CLUSTERS);

            if (cluster_rank == 0) {
                Schedule<1, 256, 256, 8, 8, 2> schedule(M, N, cluster_id * (CLUSTERS / 2) + base_pos);
                schedule_next = schedule.next(block_m, block_n);

                ushort4* tileinfo_ptr[CLUSTERS];
                for (int c = 0; c < CLUSTERS; c++) {
                    asm volatile("mapa.shared::cluster.u64  %0, %1, %2;\n\t" : "=l"(tileinfo_ptr[c]) : "l"(s.tileinfo), "r"(c));
                    tileinfo_ptr[c][0] = make_ushort4(block_m, block_n, schedule_next, 0);
                }
                asm volatile("barrier.cluster.arrive;\n" ::);

                while (schedule_next) {
                    asm volatile("barrier.sync 8, 64;\n" ::);
                    schedule_next = schedule.next(block_m, block_n);
                    
                    tileinfo_idx = (tileinfo_idx + 1) % 4;
                    for (int c = 0; c < CLUSTERS; c++) {
                        tileinfo_ptr[c][tileinfo_idx] = make_ushort4(block_m, block_n, schedule_next, 0);
                    }
                }
            } else {
                asm volatile("barrier.cluster.arrive;\n" : :);
            }
        } else if (threadIdx.x == 32) {
            asm volatile("barrier.cluster.arrive; barrier.cluster.wait; \n" ::);
            ushort4 tileinfo = s.tileinfo[0];
            block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

            bool first = true;
            int previous_m = -1, previous_n = -1;

            // specialised so that one branch does multicast for B
            // and the other doesn't load B at all (without conditionals)
            // arguably overkill (but that's the name of the game?)
            if (cluster_rank == 0) {
                constexpr uint32_t multicast_mask = (1 << CLUSTERS) - 1;
                while (schedule_next) {
                    asm volatile("barrier.arrive 8, 64;\n" ::);
                    int block_k_iter = 0;

                    #pragma unroll 1
                    for (; block_k_iter < num_blocks_k;) {
                        #pragma unroll 1
                        for (int j = 0; j < MINIMUM_K_ITERATIONS; j++, block_k_iter++) {
                            wait(EMPTY_PTR(qidx), p);
                            expect_bytes(FULL_PTR(qidx), (BK*BN+BK*BM)*sizeof(floatX));
                            load_async_multicast(SB_PTR(qidx), &tensorMapB, FULL_PTR(qidx), block_k_iter*BK, block_n*BN, multicast_mask);
                            load_async(SA_PTR(qidx), &tensorMapA, FULL_PTR(qidx), block_k_iter*BK, block_m*BM);

                            if (++qidx == QSIZE) { qidx = 0; p ^= 1; }
                            __nanosleep(128); // TODO: check if/which nanosleep helps beyond noise (I think this one does)
                        }
                    }
                    ushort4 tileinfo = s.tileinfo[++tileinfo_idx % 4];
                    first = false, previous_m = block_m, previous_n = block_n;
                    block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

                    // supertile sync looks to be less beneficial with larger/hybrid cluster sizes
                    // keeping it because it feels safer, and necessary if we had e.g. another kernel running in parallel...
                    if constexpr (supertile_sync_tolerance >= 0) {
                        if (atomicInc(counters_zeroed, num_sm-1) < num_sm - supertile_sync_tolerance - 1) {
                            do {
                                __nanosleep(32);
                                int val = atomicCAS(counters_zeroed, 1<<20, 0);
                                if (val == 0 || val >= num_sm - supertile_sync_tolerance) break;
                            } while (true);
                        }
                    }
                }
            } else {
                while (schedule_next) {
                    int block_k_iter = 0;

                    #pragma unroll 1
                    for (; block_k_iter < num_blocks_k;) {
                        #pragma unroll 1
                        for (int j = 0; j < MINIMUM_K_ITERATIONS; j++, block_k_iter++) {
                            wait(EMPTY_PTR(qidx), p);
                            expect_bytes(FULL_PTR(qidx), (BK*BN+BK*BM)*sizeof(floatX));
                            // no multicast! amazing... kinda.
                            load_async(SA_PTR(qidx), &tensorMapA, FULL_PTR(qidx), block_k_iter*BK, block_m*BM);
                            if (++qidx == QSIZE) { qidx = 0; p ^= 1;}
                            __nanosleep(128);
                        }
                    }
                    ushort4 tileinfo = s.tileinfo[++tileinfo_idx % 4];
                    first = false, previous_m = block_m, previous_n = block_n;
                    block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

                    if constexpr (supertile_sync_tolerance >= 0) {
                        if (atomicInc(counters_zeroed, num_sm-1) < num_sm - supertile_sync_tolerance - 1) {
                            do {
                                __nanosleep(32);
                                int val = atomicCAS(counters_zeroed, 1<<20, 0);
                                if (val == 0 || val >= num_sm - supertile_sync_tolerance) break;
                            } while (true);
                        }
                    }
                }
            }
        } else if (threadIdx.x == 64) {
            // turns wait(FULL_PTR(qidx)) into regular barriers for consumers to wait on
            // slightly more efficient(???) and fewer SASS instructions in the consumers
            // no longer relevant on Blackwell since MMA is issued from single thread...
            asm volatile("barrier.cluster.arrive; barrier.cluster.wait; \n" ::);
            ushort4 tileinfo = s.tileinfo[0];
            block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

            while (schedule_next) {
                #pragma unroll 1
                for (int block_k_iter = 0; block_k_iter < num_blocks_k;) {
                    if constexpr (MINIMUM_K_ITERATIONS % QSIZE == 0) {
                        qidx = 0;
                    }
                    #pragma unroll MINIMUM_K_ITERATIONS
                    for (int j = 0; j < MINIMUM_K_ITERATIONS; j++, block_k_iter++) {
                        wait(FULL_PTR(qidx), p);
                        asm volatile("barrier.sync 5, 160;\n" ::);
                        asm volatile("barrier.sync 6, 160;\n" ::);
                        if (++qidx == QSIZE) { qidx = 0; p ^= 1;}
                    }
                }
                ushort4 tileinfo = s.tileinfo[++tileinfo_idx % 4];
                block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;
            }
        } else if (threadIdx.x == 96) {
            // ...
        }
    } else {
        asm volatile("barrier.cluster.arrive; barrier.cluster.wait; \n" ::);
        for (int qidx = 0; qidx < QSIZE; qidx++) {
            if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(qidx), tid);
        }
        warpgroup_reg_alloc<240>();
        wg_idx -= 1;

        float d[WGMMA_N/16][8];
        floatP d_x16[WGMMA_N/16][8];

        floatP* block_sC = sC + wg_idx*64*32; //wg_idx*WGMMA_M*BN/2;
        int4* block_sC_128b = (int4*)block_sC;
        int* block_sC_32b = (int*)block_sC;
        int4 *block_C_thread;

        int idx = tid;
        int set_4 = idx & 4;
        int set_32 = idx & 32;
        int4* out0[4] = { &block_sC_128b[idx], &block_sC_128b[idx + 512], &block_sC_128b[idx + 1024], &block_sC_128b[idx + 1536] };
        int4* out1[4] = { &block_sC_128b[idx + 128], &block_sC_128b[idx + 640], &block_sC_128b[idx + 1152], &block_sC_128b[idx + 1664] };
        constexpr __int128 desc_multiplier = ((__int128)2 << (__int128)64) | (__int128)2;

        int x = ((threadIdx.x % 8) * 8) + (threadIdx.x / 128 - 1) * 64;
        int y_base = ((threadIdx.x % 128) / 8) * 2;
        int x_wg = x % 64;
        int idx_32b_x = ((x_wg % 16) / 8 + (x_wg / 16) * 32 * 4);
        int idx_32b_base = idx_32b_x + (y_base % 8) * 4 / 2 + ((y_base / 8) % 2) * 2;

        bool output_to_gmem = false;
        ushort4 tileinfo = s.tileinfo[0];
        block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

        int desc_id = 0;
        __int128 *descAB_base = &descAB[wg_idx * QSIZE];
        __int128 desc128 = descAB_base[desc_id];
        int barrier_idx = 5 + wg_idx;

        while (schedule_next) {
            #pragma unroll
            for (int iter = 0; iter < unrolled_iterations; iter++) {
                if (iter > 0) {
                    warpgroup_wait<0>();
                    if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(((iter+(QSIZE-1)) & (QSIZE-1))), tid);
                }

                warpgroup_arrive();
                asm volatile("barrier.sync %0, 160;\n" :: "r"(barrier_idx));
                if (iter == 0) wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d, desc128);
                else wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d, desc128);
                for (int k_it = 1; k_it < BK/WGMMA_K; k_it++) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d, desc128 | ((k_it % 4) * desc_multiplier));
                }
                warpgroup_commit_batch();
                desc_id = (desc_id + 1) & (QSIZE-1);
                desc128 = descAB_base[desc_id];

                if (output_to_gmem) {
                    if (iter < post_process_iterations) {
                        int i = iter * (8 / post_process_iterations);
                        for (int j = 0; j < 8 / post_process_iterations; i++, j++) {
                            output_postprocess<SQUARED>(i, d_x16, idx_32b_base, y_base, WGMMA_M, BN, tid, out0, out1, wg_idx, block_sC_32b);
                        }
                    } else if (iter < post_process_iterations + write_iterations) {
                        int i = (iter - post_process_iterations) * (8 / write_iterations);
                        for (int j = 0; j < 8 / write_iterations; i++, j++) {
                            __stcs(&block_C_thread[0],   ((int4*)d_x16)[i*2]);
                            __stcs(&block_C_thread[M/8], ((int4*)d_x16)[i*2+1]);
                            block_C_thread += (32*M)/8;
                        }
                    }
                }
            }

            output_to_gmem = true;
            block_C_thread = (int4*)(C + block_n*BN*M + block_m*BM + x + y_base*M);
            assert(block_m < M/BM && block_n < N/BN);

            #pragma unroll 1
            for (int block_k_iter = unrolled_iterations, x = 0; block_k_iter < num_blocks_k; block_k_iter++, x++) {
                warpgroup_wait<0>();
                if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR(((block_k_iter+(QSIZE-1)) & (QSIZE-1))), tid);
                asm volatile("barrier.sync %0, 160;\n" :: "r"(barrier_idx));

                warpgroup_arrive();
                for (int k_it = 0; k_it < BK/WGMMA_K; k_it++) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d, desc128 | ((k_it % 4) * desc_multiplier));
                }
                warpgroup_commit_batch();
                desc_id = (desc_id + 1) & (QSIZE-1);
                desc128 = descAB_base[desc_id];
            }

            warpgroup_wait<0>();
            if (tid < CLUSTERS) arrive_cluster(EMPTY_PTR((QSIZE-1)), tid);
            for(int n_tile = 0; n_tile < 16; n_tile++) {
                for (int k = 0; k < 8; k += 2) {
                    if constexpr (RELU) {
                        asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(*(int*)(&d_x16[n_tile][k])) : "f"(d[n_tile][k+1]), "f"(d[n_tile][k]));
                    } else {
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(*(int*)(&d_x16[n_tile][k])) : "f"(d[n_tile][k+1]), "f"(d[n_tile][k]));
                    }
                }
            }

            ushort4 tileinfo = s.tileinfo[++tileinfo_idx % 4];
            block_m = tileinfo.x + cluster_rank, block_n = tileinfo.y, schedule_next = tileinfo.z;

            // on the last iteration, we can't overlap the global memory writes with the matmuls, so do it immediately
            if (!schedule_next) {
                for (int iter = 0; iter < post_process_iterations + write_iterations; iter++) {
                    if (iter < post_process_iterations) {
                        int i = iter * (8 / post_process_iterations);
                        for (int j = 0; j < 8 / post_process_iterations; i++, j++) {
                            output_postprocess<SQUARED>(i, d_x16, idx_32b_base, y_base, WGMMA_M, BN, tid, out0, out1, wg_idx, block_sC_32b);
                        }
                    } else if (iter < post_process_iterations + write_iterations) {
                        int i = (iter - post_process_iterations) * (8 / write_iterations);
                        for (int j = 0; j < 8 / write_iterations; i++, j++) {
                            __stcs(&block_C_thread[0],   ((int4*)d_x16)[i*2]);
                            __stcs(&block_C_thread[M/8], ((int4*)d_x16)[i*2+1]);
                            block_C_thread += (32*M)/8;
                        }
                    }
                }
                return;
            }
        }
    }

    // this made me waste 3++ hours of debugging and miss a party... :(
    asm volatile("barrier.cluster.arrive; barrier.cluster.wait; \n" ::);
}

// not currently used as it didn't give any benefit with larger clusters :(
// in theory could help improve L2 cache hit rate for A or B matrix
template<int MAX_SM=256>
__global__ void l2_side_per_sm(unsigned int* data) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int address = 0;

        long long int start_clock = clock64();
        for (int i = 0; i < 10; i++) {
            int value = atomicAdd(&data[address], 1);
            address += (value > MAX_SM*1000) ? 1 : 0; // impossible condition to make the compiler play along
        }
        int total_latecy = clock64() - start_clock;
        data[MAX_SM + smid] = total_latecy;
        atomicAdd(&data[1], total_latecy);
        __threadfence_block();

        int num_done = atomicAdd(&data[2], 1);
        if (num_done == gridDim.x - 1) {
            int average_latency = data[1] / gridDim.x;
            for (int i = 0; i < MAX_SM; i++) {
                data[i] = data[MAX_SM + i] > average_latency ? 1 : 0;
            }
        }
    }
}

void runKernel10(int M, int N, int K, floatX *A, floatX *B, floatP *C, floatP *I, unsigned int* zeroed_metadata_gpu) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = sizeof(floatX) == 1 ? 128 : 64;
    constexpr int NUM_THREADS = 128*3;
    constexpr int QSIZE = 4; // now only works with 2 or 4

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    static unsigned int* gpu_l2_sides = nullptr;
    static cudaStream_t stream_2x, stream_8x;
    static bool init_misc = true;
    if (init_misc) {
        init_misc = false;
        cudaStreamCreate(&stream_2x);
        cudaStreamCreate(&stream_8x);

        // not currently used, no benefit when combined with larger/hybrid cluster sizes? :(
        //cudaMalloc(&gpu_l2_sides, 2 * 256 * sizeof(int));
        //cudaMemset(gpu_l2_sides, 0, 2 * 256 * sizeof(int));
        //l2_side_per_sm<256><<<num_sm, 128>>>(gpu_l2_sides);
    }

    if (_prev_m != M || _prev_n != N || _prev_k != K) {
        d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
        d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);
        d_tma_map_I = create_tensor_map<BN/8, BM, false, floatP>(I, N, M);
        _prev_m = M, _prev_n = N, _prev_k = K;

        // TODO: make this slightly less hacky
        constexpr uint64_t desc_base = 0x4000004000010000;
        constexpr int uniqueA = QSIZE * 2; // 2 consumers
        constexpr int uniqueB = QSIZE;
        constexpr int sizeA = (BM * BK * sizeof(floatX)) >> 4;
        constexpr int sizeB = (BN * BK * sizeof(floatX)) >> 4;
        constexpr int sizeA_warpgroup_offset = sizeA / (NUM_THREADS/128 - 1);
        constexpr int startA = 0x40; // 1024 bytes used by driver (not clear if that is *always* the case though...)
        constexpr int startB = startA + sizeA*QSIZE;

        uint64_t hostA[uniqueA], hostB[uniqueB];
        for (int i = 0; i < uniqueA; i++) {
            hostA[i] = (startA + (i % QSIZE) * sizeA + (i / QSIZE) * sizeA_warpgroup_offset) | desc_base;
        }
        for (int i = 0; i < uniqueB; i++) {
            hostB[i] = (startB + (i % QSIZE) * sizeB) | desc_base;
        }
        __int128 hostAB[uniqueA];
        for (int i = 0; i < uniqueA; i++) {
            hostAB[i] = (__int128)hostA[i] | ((__int128)hostB[i % uniqueB] << (__int128)64);
        }
        cudaMemcpyToSymbol(descAB, hostAB, sizeof(hostAB));
    }

    auto* kernel_2x = matmulKernel10<BM, BN, BK, NUM_THREADS, QSIZE, 2>;
    auto* kernel_8x = matmulKernel10<BM, BN, BK, NUM_THREADS, QSIZE, 8>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
    static_assert(sMemSize < 256 * 1024);
    cudaCheck(cudaFuncSetAttribute(kernel_2x, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
    cudaCheck(cudaFuncSetAttribute(kernel_8x, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    assert(num_sm >= 128);
    if constexpr (hybrid_cluster_sizes) {
        kernel_8x<<<120, NUM_THREADS, sMemSize, stream_8x>>>(M, N, K, C, I, d_tma_map_I, d_tma_map_A, d_tma_map_B, zeroed_metadata_gpu, 0, 128);
        kernel_2x<<<8, NUM_THREADS, sMemSize, stream_2x>>>(M, N, K, C, I, d_tma_map_I, d_tma_map_A, d_tma_map_B, zeroed_metadata_gpu,  60, 128);
    } else {
        kernel_2x<<<128, NUM_THREADS, sMemSize, stream_2x>>>(M, N, K, C, I, d_tma_map_I, d_tma_map_A, d_tma_map_B, zeroed_metadata_gpu,  0, 128);
    }

    cudaCheck(cudaDeviceSynchronize());
}

} // namespace M10

using M10::runKernel10;
