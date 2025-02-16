NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
NVCC_INCLUDES = -I/usr/local/cuda-12.8/include
NVCC_LDLIBS =
NCLL_INCUDES =
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo $(NVCC_INCLUDES) $(NVCC_LDLIBS)

matmul: matmul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmulprofile: matmul
	$(NCU_COMMAND) -c 7 -o $@ -f $(OUT_DIR)/$^

clean:
	rm $(OUT_DIR)/*
