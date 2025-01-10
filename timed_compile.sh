#!/bin/bash

# Save original environment variables
ORIGINAL_NVVM_BRANCH=$(_NVVM_BRANCH_)
ORIGINAL_SPACE=$(_SPACE_)
ORIGINAL_CUDART=$(_CUDART_)
ORIGINAL_HERE=$(_HERE_)
ORIGINAL_THERE=$(_THERE_)
ORIGINAL_TARGET_SIZE=$(_TARGET_SIZE_)
ORIGINAL_TARGET_DIR=$(_TARGET_DIR_)
ORIGINAL_TOP=$TOP
ORIGINAL_CICC_PATH=$CICC_PATH
ORIGINAL_NVVMIR_LIBRARY_DIR=$NVVMIR_LIBRARY_DIR
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ORIGINAL_PATH=$PATH
ORIGINAL_INCLUDES=$INCLUDES
ORIGINAL_LIBRARIES=$LIBRARIES
ORIGINAL_CUDAFE_FLAGS=$CUDAFE_FLAGS
ORIGINAL_PTXAS_FLAGS=$PTXAS_FLAGS

# Set environment variables
export _NVVM_BRANCH_=nvvm
export _SPACE_=
export _CUDART_=cudart
export _HERE_=/usr/local/cuda-12.6/bin
export _THERE_=/usr/local/cuda-12.6/bin
export _TARGET_SIZE_=
export _TARGET_DIR_=targets/x86_64-linux
export TOP=/usr/local/cuda-12.6/bin/..
export CICC_PATH=/usr/local/cuda-12.6/bin/../nvvm/bin
export NVVMIR_LIBRARY_DIR=/usr/local/cuda-12.6/bin/../nvvm/libdevice
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/bin/../lib:/usr/local/cuda-12.6/lib64
export PATH=/usr/local/cuda-12.6/bin/../nvvm/bin:/usr/local/cuda-12.6/bin:$PATH
export INCLUDES="-I/usr/local/cuda-12.6/bin/../targets/x86_64-linux/include"
export LIBRARIES="-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib"
export CUDAFE_FLAGS=
export PTXAS_FLAGS=

# Function to time commands
time_command() {
    echo "Running: $1"
    start_time=$(date +%s%3N)
    eval $1
    end_time=$(date +%s%3N)
    elapsed_time=$((end_time - start_time))
    echo "Elapsed time: ${elapsed_time}ms"
    echo
}

# Run commands and time them
time_command 'gcc -std=c++17 -D__CUDA_ARCH_LIST__=900 -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ -D__NVCC__ -D__CUDACC_RELAXED_CONSTEXPR__ -fPIE -Wno-psabi -fno-strict-aliasing -O1 -w -I"/usr/local/cuda-12.6/include" "-I/usr/local/cuda-12.6/bin/../targets/x86_64-linux/include" -D "NDEBUG" -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=68 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "matmul.cu" -o "/tmp/tmpxft_000759b9_00000000-5_matmul.cpp4.ii"'

time_command 'cudafe++ --c++17 --gnu_version=120300 -w --display_error_number --orig_src_file_name "matmul.cu" --orig_src_path_name "/home/ubuntu/2025/fast.cu/matmul.cu" --allow_managed --relaxed_constexpr --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_000759b9_00000000-6_matmul.cudafe1.cpp" --stub_file_name "tmpxft_000759b9_00000000-6_matmul.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_000759b9_00000000-4_matmul.module_id" "/tmp/tmpxft_000759b9_00000000-5_matmul.cpp4.ii"'

time_command 'gcc -std=c++17 -D__CUDA_ARCH__=900 -D__CUDA_ARCH_FEAT_SM90_ALL -D__CUDA_ARCH_LIST__=900 -D__NV_LEGACY_LAUNCH -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -D__CUDACC_RELAXED_CONSTEXPR__ -fPIE -Wno-psabi -fno-strict-aliasing -O1 -w -I"/usr/local/cuda-12.6/include" "-I/usr/local/cuda-12.6/bin/../targets/x86_64-linux/include" -D "NDEBUG" -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=68 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "matmul.cu" -o "/tmp/tmpxft_000759b9_00000000-9_matmul.cpp1.ii"'

time_command '"$CICC_PATH/cicc" --c++17 --gnu_version=120300 -w --display_error_number --orig_src_file_name "matmul.cu" --orig_src_path_name "/home/ubuntu/2025/fast.cu/matmul.cu" --allow_managed --relaxed_constexpr -arch compute_90a -m64 --no-version-ident -ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1 -fast-math --gen_div_approx_ftz --include_file_name "tmpxft_000759b9_00000000-3_matmul.fatbin.c" -tused --module_id_file_name "/tmp/tmpxft_000759b9_00000000-4_matmul.module_id" --gen_c_file_name "/tmp/tmpxft_000759b9_00000000-6_matmul.cudafe1.c" --stub_file_name "/tmp/tmpxft_000759b9_00000000-6_matmul.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_000759b9_00000000-6_matmul.cudafe1.gpu" "/tmp/tmpxft_000759b9_00000000-9_matmul.cpp1.ii" -o "/tmp/tmpxft_000759b9_00000000-6_matmul.ptx"'

time_command 'ptxas -w -arch=sm_90a -m64 "/tmp/tmpxft_000759b9_00000000-6_matmul.ptx" -o "/tmp/tmpxft_000759b9_00000000-10_matmul.cubin"'

time_command 'fatbinary -64 --cicc-cmdline="-ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1" "--image3=kind=elf,sm=90a,file=/tmp/tmpxft_000759b9_00000000-10_matmul.cubin" --embedded-fatbin="/tmp/tmpxft_000759b9_00000000-3_matmul.fatbin.c"'

time_command 'rm /tmp/tmpxft_000759b9_00000000-3_matmul.fatbin'

time_command 'gcc -std=c++17 -D__CUDA_ARCH__=900 -D__CUDA_ARCH_FEAT_SM90_ALL -D__CUDA_ARCH_LIST__=900 -D__NV_LEGACY_LAUNCH -c -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS -fPIE -Wno-psabi -fno-strict-aliasing -O1 -w -Wno-psabi -I"/usr/local/cuda-12.6/include" "-I/usr/local/cuda-12.6/bin/../targets/x86_64-linux/include" -m64 "/tmp/tmpxft_000759b9_00000000-6_matmul.cudafe1.cpp" -o "/tmp/tmpxft_000759b9_00000000-11_matmul.o"'

time_command 'nvlink -m64 --arch=sm_90 --register-link-binaries="/tmp/tmpxft_000759b9_00000000-7_matmul_dlink.reg.c" -w -lcuda -lcublas "-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_000759b9_00000000-11_matmul.o" -lcudadevrt -o "/tmp/tmpxft_000759b9_00000000-12_matmul_dlink.cubin" --host-ccbin "gcc"'

time_command 'fatbinary -64 --cicc-cmdline="-ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1" -link "--image3=kind=elf,sm=90a,file=/tmp/tmpxft_000759b9_00000000-12_matmul_dlink.cubin" --embedded-fatbin="/tmp/tmpxft_000759b9_00000000-8_matmul_dlink.fatbin.c"'

time_command 'rm /tmp/tmpxft_000759b9_00000000-8_matmul_dlink.fatbin'

time_command 'gcc -std=c++17 -D__CUDA_ARCH_LIST__=900 -D__NV_LEGACY_LAUNCH -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_000759b9_00000000-8_matmul_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_000759b9_00000000-7_matmul_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ -fPIE -Wno-psabi -fno-strict-aliasing -O1 -w -Wno-psabi -I"/usr/local/cuda-12.6/include" "-I/usr/local/cuda-12.6/bin/../targets/x86_64-linux/include" -D "NDEBUG" -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=68 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda-12.6/bin/crt/link.stub" -o "/tmp/tmpxft_000759b9_00000000-13_matmul_dlink.o"'

time_command 'g++ -D__CUDA_ARCH_LIST__=900 -D__NV_LEGACY_LAUNCH -fPIE -Wno-psabi -fno-strict-aliasing -O1 -w -m64 -std=c++17 -Wl,--start-group "/tmp/tmpxft_000759b9_00000000-13_matmul_dlink.o" "/tmp/tmpxft_000759b9_00000000-11_matmul.o" -lcuda -lcublas "-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.6/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -Wl,--end-group -o "out/matmul"'

# Restore original environment variables
export _NVVM_BRANCH_=$ORIGINAL_NVVM_BRANCH
export _SPACE_=$ORIGINAL_SPACE
export _CUDART_=$ORIGINAL_CUDART
export _HERE_=$ORIGINAL_HERE
export _THERE_=$ORIGINAL_THERE
export _TARGET_SIZE_=$ORIGINAL_TARGET_SIZE
export _TARGET_DIR_=$ORIGINAL_TARGET_DIR
export TOP=$ORIGINAL_TOP
export CICC_PATH=$ORIGINAL_CICC_PATH
export NVVMIR_LIBRARY_DIR=$ORIGINAL_NVVMIR_LIBRARY_DIR
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
export PATH=$ORIGINAL_PATH
export INCLUDES=$ORIGINAL_INCLUDES
export LIBRARIES=$ORIGINAL_LIBRARIES
export CUDAFE_FLAGS=$ORIGINAL_CUDAFE_FLAGS
export PTXAS_FLAGS=$ORIGINAL_PTXAS_FLAGS
