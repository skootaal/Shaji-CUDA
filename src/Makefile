# Makefile for CUDA Projects
# Compiles all CUDA sample programs

# Compiler settings
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CUDA_ARCH = sm_90  # H100 compute capability

# Compiler flags
NVCC_FLAGS = -O2 -arch=$(CUDA_ARCH) -std=c++17
INCLUDE_DIRS = -I$(CUDA_PATH)/include
LIB_DIRS = -L$(CUDA_PATH)/lib64
LIBRARIES = -lcudart -lcublas

# Directories
SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj

# Source files
SOURCES = $(SRC_DIR)/vector_add.cu \
          $(SRC_DIR)/matrix_multiply.cu \
          $(SRC_DIR)/h100_test.cu

# Target executables
TARGETS = $(BIN_DIR)/vector_add \
          $(BIN_DIR)/matrix_multiply \
          $(BIN_DIR)/h100_test

# Default target
all: $(BIN_DIR) $(TARGETS)
	@echo "Build complete!"
	@echo "Executables are in $(BIN_DIR)/"

# Create bin directory
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

# Vector addition target
$(BIN_DIR)/vector_add: $(SRC_DIR)/vector_add.cu
	@echo "Compiling vector_add..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) \
		-o $@ $< $(LIBRARIES)
	@echo "  ✓ $@ compiled successfully"

# Matrix multiplication target
$(BIN_DIR)/matrix_multiply: $(SRC_DIR)/matrix_multiply.cu
	@echo "Compiling matrix_multiply..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) \
		-o $@ $< $(LIBRARIES)
	@echo "  ✓ $@ compiled successfully"

# H100 performance test target
$(BIN_DIR)/h100_test: $(SRC_DIR)/h100_test.cu
	@echo "Compiling h100_test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) \
		-o $@ $< $(LIBRARIES)
	@echo "  ✓ $@ compiled successfully"

# Run targets
run_vector_add: $(BIN_DIR)/vector_add
	@echo "Running vector_add..."
	@$(BIN_DIR)/vector_add 100000

run_matrix: $(BIN_DIR)/matrix_multiply
	@echo "Running matrix_multiply..."
	@$(BIN_DIR)/matrix_multiply 512 512 512

run_h100: $(BIN_DIR)/h100_test
	@echo "Running h100_test..."
	@$(BIN_DIR)/h100_test

run_all: run_vector_add run_matrix run_h100

# Debug build
debug: NVCC_FLAGS = -O0 -g -G -arch=$(CUDA_ARCH) -std=c++17
debug: all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BIN_DIR) $(OBJ_DIR)
	@echo "Clean complete"

# Help target
help:
	@echo "CUDA Build System"
	@echo "================="
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Build all programs (default)"
	@echo "  make run_vector_add - Build and run vector addition"
	@echo "  make run_matrix   - Build and run matrix multiplication"
	@echo "  make run_h100     - Build and run H100 performance test"
	@echo "  make run_all      - Build and run all programs"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make help         - Display this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  NVCC:     $(NVCC)"
	@echo "  ARCH:     $(CUDA_ARCH)"
	@echo "  FLAGS:    $(NVCC_FLAGS)"
	@echo ""

.PHONY: all run_vector_add run_matrix run_h100 run_all debug clean help
