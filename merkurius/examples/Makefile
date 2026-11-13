# Merkurius demo: build, run and look inside the example programs.
# `make` (or `make help`) lists the commands.

PROGRAMS := hello factorial fib count

# A dump fails if merkc does, though it is piped.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

DEMO_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# merkc from the build tree beside llvm-project, else from the PATH.
MERKC ?= $(firstword $(abspath $(wildcard $(DEMO_DIR)../../../build-merk/bin/merkc)) merkc)
# The optimization level, 0 to 3, for the programs and the dumps.
O ?= 0
BUILD := $(DEMO_DIR)build
# The MLIR dumps show the module's target attributes in full with ATTRS=1.
ATTRS ?= 0
ifeq ($(ATTRS),1)
SHORTEN := cat
else
SHORTEN := sed -E '1s/^(module @[^ ]+) attributes \{.*\} \{$$/\1 attributes {...} {/'
endif

.DEFAULT_GOAL := help
.PHONY: help all run-all dumps clean check-merkc \
	gpu-to-gpu run-gpu-to-gpu gpu-to-gpu-cpu run-gpu-to-gpu-cpu \
	work-stealing run-work-stealing run-work-stealing-no-steal \
	$(PROGRAMS) $(addprefix run-,$(PROGRAMS)) $(addprefix merk-,$(PROGRAMS)) \
	$(addprefix mlir-,$(PROGRAMS)) $(addprefix llvm-,$(PROGRAMS))

help:
	@echo "Merkurius demo: Mercury compiled through MLIR and LLVM"
	@echo ""
	@echo "Programs: $(PROGRAMS)"
	@echo "  hello      Hello, world: a string and a foreign_proc call into C"
	@echo "  factorial  0! to 20!: arithmetic, if-then-else, recursion"
	@echo "  fib        Fibonacci numbers: double recursion and a tail-call loop"
	@echo "  count      Count to 10: a musttail self-call, constant stack"
	@echo ""
	@echo "For each program P:"
	@echo "  make P          Compile P.m to the executable build/P"
	@echo "  make run-P      Compile P and run it"
	@echo "  make merk-P     Print the merk dialect: Mercury procedures in MLIR"
	@echo "  make mlir-P     Print the MLIR LLVM dialect that merk is lowered to"
	@echo "  make llvm-P     Print the LLVM IR, with the C library code linked in"
	@echo ""
	@echo "Two GPUs playing ping-pong (gpu-to-gpu/README.md):"
	@echo "  make gpu-to-gpu          Compile it for the GPU"
	@echo "  make run-gpu-to-gpu      Compile it and run both nodes on this machine"
	@echo "  make run-gpu-to-gpu-cpu  The same with the GPU regions on the CPU"
	@echo ""
	@echo "Work stealing between GPU processes (work-stealing/README.md):"
	@echo "  make work-stealing                Compile it for the GPU, at -O2"
	@echo "  make run-work-stealing            Compile it and run it with stealing"
	@echo "  make run-work-stealing-no-steal   The same with stealing off"
	@echo ""
	@echo "For all programs:"
	@echo "  make all        Compile every program"
	@echo "  make run-all    Compile and run every program"
	@echo "  make dumps      Write every dump to build/P.merk.mlir, build/P.llvm.mlir"
	@echo "                  and build/P.ll"
	@echo "  make clean      Remove build/"
	@echo ""
	@echo "Options:"
	@echo "  O=N             Optimization level 0 to 3 (now $(O)); try"
	@echo "                  'make llvm-count O=2' to see the loop unrolled"
	@echo "  ATTRS=1         Show the module's target attributes in the MLIR dumps"
	@echo "                  (the triple, data layout and address spaces)"
	@echo "  MERKC=path      The compiler (now $(MERKC))"
	@echo "  ARCH=name       The GPU's architecture (now $(ARCH))"
	@echo "  PORT=N          The port node B listens on (now $(PORT))"
	@echo "  WHERE=host      Run the work-stealing places on the CPU"

check-merkc:
	@command -v $(MERKC) > /dev/null || { \
	  echo "merkc not found: build it, or set MERKC=/path/to/merkc"; exit 1; }

$(BUILD):
	@mkdir -p $@

# The executable, rebuilt when the source or the optimization level changes.
$(addprefix $(BUILD)/,$(PROGRAMS)): $(BUILD)/%: $(DEMO_DIR)%.m $(BUILD)/.O$(O) | check-merkc $(BUILD)
	@echo "merkc -O$(O) $*.m -o build/$*"
	@$(MERKC) -O$(O) $< -o $@

$(BUILD)/.O$(O): | $(BUILD)
	@rm -f $(BUILD)/.O*
	@touch $@

$(PROGRAMS): %: $(BUILD)/%

$(addprefix run-,$(PROGRAMS)): run-%: $(BUILD)/%
	@echo "--- $* ---"
	@$<

$(addprefix merk-,$(PROGRAMS)): merk-%: $(DEMO_DIR)%.m | check-merkc
	@$(MERKC) -fc1 -O$(O) -emit-merk $< -o - | $(SHORTEN)

$(addprefix mlir-,$(PROGRAMS)): mlir-%: $(DEMO_DIR)%.m | check-merkc
	@$(MERKC) -fc1 -O$(O) -emit-mlir $< -o - | $(SHORTEN)

$(addprefix llvm-,$(PROGRAMS)): llvm-%: $(DEMO_DIR)%.m | check-merkc
	@$(MERKC) -fc1 -O$(O) -emit-llvm $< -o -

# GPU to GPU: two modules, and two nodes to run.
GPU_DIR := $(DEMO_DIR)gpu-to-gpu/
GPU_SOURCES := $(GPU_DIR)gpu_to_gpu.m $(GPU_DIR)demo_node.m
ARCH ?= gfx1030
PORT ?= 47100
# Where libomp is in the build tree that merkc is in.
LLVM_BUILD := $(abspath $(dir $(MERKC))..)
LIBOMP_DIRS := $(LLVM_BUILD)/runtimes/runtimes-bins/openmp/runtime/src \
	$(LLVM_BUILD)/lib $(LLVM_BUILD)/lib/x86_64-unknown-linux-gnu
LIBOMP_FLAGS ?= $(addprefix -L,$(LIBOMP_DIRS))
LIBOMP_PATH := $(subst $(eval) ,:,$(strip $(LIBOMP_DIRS)))

$(BUILD)/gpu_to_gpu: $(GPU_SOURCES) $(BUILD)/.O$(O) | check-merkc $(BUILD)
	@echo "merkc -O$(O) -fopenmp --offload-arch=$(ARCH) gpu_to_gpu.m demo_node.m -o build/gpu_to_gpu"
	@cd $(BUILD) && $(MERKC) -O$(O) -fdo-loops -fopenmp --offload-arch=$(ARCH) $(LIBOMP_FLAGS) $(GPU_SOURCES) -o $@

$(BUILD)/gpu_to_gpu_cpu: $(GPU_SOURCES) $(BUILD)/.O$(O) | check-merkc $(BUILD)
	@echo "merkc -O$(O) gpu_to_gpu.m demo_node.m -o build/gpu_to_gpu_cpu"
	@cd $(BUILD) && $(MERKC) -O$(O) -fdo-loops $(GPU_SOURCES) -o $@

gpu-to-gpu: $(BUILD)/gpu_to_gpu
gpu-to-gpu-cpu: $(BUILD)/gpu_to_gpu_cpu

# Node B in the background and node A here; fails if either does.
define run_two_nodes
	@export MERK_COOKIE=demo LD_LIBRARY_PATH="$(LIBOMP_PATH):$$LD_LIBRARY_PATH"; \
	$(1) $< b 127.0.0.1:$(PORT) & b=$$!; \
	$(1) $< a 127.0.0.1:$(PORT); a=$$?; \
	wait $$b; exit $$((a | $$?))
endef

run-gpu-to-gpu: $(BUILD)/gpu_to_gpu
	$(call run_two_nodes,OMP_TARGET_OFFLOAD=MANDATORY)

run-gpu-to-gpu-cpu: $(BUILD)/gpu_to_gpu_cpu
	$(call run_two_nodes,)

# Work stealing: one module, at -O2 whatever O is, as it compares times and
# a GPU process's stack at -O0 is too big.
WS_SOURCE := $(DEMO_DIR)work-stealing/work_stealing.m
# `host` runs the places on the host's CPUs.
WHERE ?=

$(BUILD)/work_stealing: $(WS_SOURCE) | check-merkc $(BUILD)
	@echo "merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=$(ARCH) work_stealing.m -o build/work_stealing"
	@cd $(BUILD) && $(MERKC) -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=$(ARCH) $(LIBOMP_FLAGS) $< -o $@

work-stealing: $(BUILD)/work_stealing

run-work-stealing: $(BUILD)/work_stealing
	@LD_LIBRARY_PATH="$(LIBOMP_PATH):$$LD_LIBRARY_PATH" $< $(WHERE)

run-work-stealing-no-steal: $(BUILD)/work_stealing
	@LD_LIBRARY_PATH="$(LIBOMP_PATH):$$LD_LIBRARY_PATH" $< no-steal $(WHERE)

all: $(addprefix $(BUILD)/,$(PROGRAMS))

run-all: $(addprefix run-,$(PROGRAMS))

dumps: | check-merkc $(BUILD)
	@for p in $(PROGRAMS); do \
	  $(MERKC) -fc1 -O$(O) -emit-merk $(DEMO_DIR)$$p.m -o $(BUILD)/$$p.merk.mlir && \
	  $(MERKC) -fc1 -O$(O) -emit-mlir $(DEMO_DIR)$$p.m -o $(BUILD)/$$p.llvm.mlir && \
	  $(MERKC) -fc1 -O$(O) -emit-llvm $(DEMO_DIR)$$p.m -o $(BUILD)/$$p.ll && \
	  echo "build/$$p.merk.mlir build/$$p.llvm.mlir build/$$p.ll"; \
	done

clean:
	rm -rf $(BUILD)
