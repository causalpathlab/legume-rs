# =============================================================================
# Binaries
# =============================================================================
# PATH_BINS: built from this workspace.
# CRATES_BINS: cargo install from crates.io (see resolve_bin).
PATH_BINS   := senna chickpea gene-text lupin
CRATES_BINS := pinto cocoa mung data-beans data-beans-sim
BINARIES    := $(PATH_BINS) $(CRATES_BINS)

# $$bin -> pkg / extra_feat / from_crates  (must run inside a shell recipe)
resolve_bin = case $$bin in \
	mung)           pkg=mung-cnv;   extra_feat=;   from_crates=1;; \
	cocoa)          pkg=cocoa-rs;   extra_feat=;   from_crates=1;; \
	pinto)          pkg=pinto-rs;   extra_feat=;   from_crates=1;; \
	data-beans)     pkg=data-beans; extra_feat=;   from_crates=1;; \
	data-beans-sim) pkg=data-beans; extra_feat=sim; from_crates=1;; \
	*)              pkg=$$bin;      extra_feat=;   from_crates=;; \
	esac

# =============================================================================
# Backend / HDF5 / CUDA capability
# =============================================================================
UNAME_S  := $(shell uname -s)
HAS_NVCC := $(shell command -v nvcc 2>/dev/null)

ifeq ($(UNAME_S),Darwin)
DEFAULT_BACKEND := metal
else ifneq ($(HAS_NVCC),)
DEFAULT_BACKEND := cuda
else
DEFAULT_BACKEND := cpu
endif
BACKEND ?= $(DEFAULT_BACKEND)

H5CC_PATH   := $(shell command -v h5cc 2>/dev/null)
H5CC_PREFIX := $(if $(H5CC_PATH),$(shell dirname $$(dirname $(H5CC_PATH))))
HDF5_CANDIDATES := $(HDF5_DIR) $(CONDA_PREFIX) $(H5CC_PREFIX) /opt/homebrew /usr/local /usr
HDF5_DETECTED_DIR := $(firstword $(foreach d,$(HDF5_CANDIDATES), \
	$(if $(strip $(d)),$(if $(wildcard $(d)/include/hdf5.h), \
	$(if $(or $(wildcard $(d)/lib/libhdf5.so),$(wildcard $(d)/lib/libhdf5.a), \
	          $(wildcard $(d)/lib/libhdf5.dylib),$(wildcard $(d)/lib64/libhdf5.so), \
	          $(wildcard $(d)/lib64/libhdf5.a)),$(d))))))

ifeq ($(HDF5_DETECTED_DIR),)
HDF5_OK := $(shell pkg-config --exists hdf5 2>/dev/null || pkg-config --exists hdf5-serial 2>/dev/null && echo yes)
else
HDF5_OK := yes
endif
DEFAULT_HDF5 := $(if $(filter yes,$(HDF5_OK)),on,off)
HDF5 ?= $(DEFAULT_HDF5)

ifeq ($(HDF5),on)
ifneq ($(HDF5_DETECTED_DIR),)
export HDF5_DIR := $(HDF5_DETECTED_DIR)
endif
endif

ifeq (,$(filter $(BACKEND),cpu cuda metal))
$(error Unknown BACKEND='$(BACKEND)'; use cpu, cuda, or metal)
endif
ifeq (,$(filter $(HDF5),on off))
$(error Unknown HDF5='$(HDF5)'; use on or off)
endif

# CUDA_COMPUTE_CAP: caller > nvidia-smi > cache > /proc GPU model
CUDA_CAP_CACHE := $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/legume-rs/cuda-compute-cap
cuda_cap_from_model = $(strip \
	$(if $(or $(findstring RTX 50,$(1)),$(findstring Blackwell,$(1))),120, \
	$(if $(or $(findstring B200,$(1)),$(findstring B100,$(1)),$(findstring GB200,$(1))),100, \
	$(if $(or $(findstring H100,$(1)),$(findstring H200,$(1)),$(findstring H800,$(1)),$(findstring H20,$(1)),$(findstring GH200,$(1))),90, \
	$(if $(or $(findstring RTX 40,$(1)),$(findstring L40,$(1)),$(findstring L4,$(1)),$(findstring Ada,$(1))),89, \
	$(if $(or $(findstring A100,$(1)),$(findstring A800,$(1)),$(findstring A30,$(1))),80, \
	$(if $(or $(findstring RTX 30,$(1)),$(findstring RTX A,$(1)),$(findstring A10,$(1)),$(findstring A40,$(1)),$(findstring A16,$(1)),$(findstring A2,$(1))),86, \
	$(if $(or $(findstring RTX 20,$(1)),$(findstring GTX 16,$(1)),$(findstring Quadro RTX,$(1)),$(findstring TITAN RTX,$(1)),$(findstring T4,$(1))),75, \
	$(if $(or $(findstring V100,$(1)),$(findstring TITAN V,$(1))),70, \
	$(if $(or $(findstring GTX 10,$(1)),$(findstring P100,$(1)),$(findstring P40,$(1)),$(findstring TITAN X,$(1))),60,))))))))))

ifeq ($(BACKEND),cuda)
ifneq ($(strip $(CUDA_COMPUTE_CAP)),)
CUDA_CAP_SOURCE := caller
else
CUDA_CAP_SMI := $(shell timeout 15 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
	| head -n1 | tr -d ' ' | grep -E -x '[0-9]+\.[0-9]+' | tr -d .)
ifneq ($(CUDA_CAP_SMI),)
CUDA_COMPUTE_CAP := $(CUDA_CAP_SMI)
CUDA_CAP_SOURCE := nvidia-smi
$(shell mkdir -p $(dir $(CUDA_CAP_CACHE)) && echo $(CUDA_CAP_SMI) > $(CUDA_CAP_CACHE))
else ifneq ($(wildcard $(CUDA_CAP_CACHE)),)
CUDA_COMPUTE_CAP := $(shell cat $(CUDA_CAP_CACHE))
CUDA_CAP_SOURCE := cache
else
CUDA_GPU_MODEL := $(shell sed -n 's/^Model:[[:space:]]*//p' /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -n1)
CUDA_COMPUTE_CAP := $(call cuda_cap_from_model,$(CUDA_GPU_MODEL))
CUDA_CAP_SOURCE := $(if $(CUDA_COMPUTE_CAP),model,none)
endif
endif
ifneq ($(strip $(CUDA_COMPUTE_CAP)),)
export CUDA_COMPUTE_CAP
endif
endif

# =============================================================================
# Cargo --features
# =============================================================================
FEATS :=
ifneq ($(BACKEND),cpu)
FEATS += $(BACKEND)
endif
ifeq ($(HDF5),on)
FEATS += hdf5
endif

empty :=
space := $(empty) $(empty)
comma := ,
CARGO_FEATURES := $(if $(strip $(FEATS)),--features $(subst $(space),$(comma),$(strip $(FEATS))))
CARGO_FEATURES_CPU := $(if $(filter on,$(HDF5)),--features hdf5)
CUDA_CAP_HINT := $(if $(filter cuda,$(BACKEND)$(if $(CUDA_COMPUTE_CAP),x)), \
	echo "  hint: pass CUDA_COMPUTE_CAP=<cap> (e.g. 86)";)

INSTALL_STATUS := $(CURDIR)/.make-install-status

# =============================================================================
# Targets
# =============================================================================
.PHONY: all help install install-cpu install-cuda install-metal \
	$(addprefix install-,$(BINARIES)) \
	uninstall $(addprefix uninstall-,$(BINARIES)) \
	build build-cuda build-metal test clean \
	_status_init _status_report

all: install

help:
	@echo "backend=$(DEFAULT_BACKEND)  hdf5=$(DEFAULT_HDF5)$(if $(HDF5_DETECTED_DIR), dir=$(HDF5_DETECTED_DIR))"
ifeq ($(BACKEND),cuda)
	@echo "cuda_cap=$(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif
	@echo "bins: $(BINARIES)"
	@echo "targets: install[-cpu|-cuda|-metal|-<bin>]  uninstall[-<bin>]  build  test  clean"
	@echo "vars: BACKEND=  HDF5=  HDF5_DIR=  CUDA_COMPUTE_CAP="

install: _status_init $(addprefix install-,$(BINARIES)) _status_report
install-cpu:  ; @$(MAKE) install BACKEND=cpu
install-cuda: ; @$(MAKE) install BACKEND=cuda
install-metal:; @$(MAKE) install BACKEND=metal

_status_init:
	@rm -f $(INSTALL_STATUS)
ifeq ($(BACKEND),cuda)
	@echo "CUDA compute capability: $(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif

_status_report:
	@echo ""; echo "Install summary (backend=$(BACKEND) hdf5=$(HDF5)):"
	@if [ -f $(INSTALL_STATUS) ]; then \
		awk '{ printf "  %-18s -> %s\n", $$1, $$2 }' $(INSTALL_STATUS); \
		if grep -q ' cpu$$' $(INSTALL_STATUS) && [ "$(BACKEND)" != cpu ]; then \
			echo "  (some binaries fell back to CPU)"; \
		fi; \
	fi
	@rm -f $(INSTALL_STATUS)

$(addprefix install-,$(BINARIES)):
	@bin=$(@:install-%=%); $(resolve_bin); \
	feats="$(CARGO_FEATURES)"; feats_cpu="$(CARGO_FEATURES_CPU)"; \
	if [ -n "$$extra_feat" ]; then \
		if [ -n "$$feats" ]; then feats="$$feats,$$extra_feat"; else feats="--features $$extra_feat"; fi; \
		if [ -n "$$feats_cpu" ]; then feats_cpu="$$feats_cpu,$$extra_feat"; else feats_cpu="--features $$extra_feat"; fi; \
	fi; \
	if [ -n "$$from_crates" ]; then cmd="cargo install --locked --force $$pkg"; \
	else cmd="cargo install --locked --path $$pkg"; fi; \
	if [ -n "$$feats" ]; then \
		echo "Installing $$bin ($(BACKEND))..."; \
		if $$cmd $$feats; then echo "$$bin $(BACKEND)" >> $(INSTALL_STATUS); \
		else \
			echo "  $(BACKEND) failed for $$bin; retrying CPU"; $(CUDA_CAP_HINT) \
			$$cmd $$feats_cpu; echo "$$bin cpu" >> $(INSTALL_STATUS); \
		fi; \
	else \
		echo "Installing $$bin (cpu)..."; \
		$$cmd $$feats_cpu; echo "$$bin cpu" >> $(INSTALL_STATUS); \
	fi

uninstall: $(addprefix uninstall-,$(BINARIES))
	@echo "done"

$(addprefix uninstall-,$(BINARIES)):
	@bin=$(@:uninstall-%=%); $(resolve_bin); \
	echo "Uninstalling $$bin..."; \
	cargo uninstall $$pkg --bin $$bin 2>/dev/null || rm -f "$$HOME/.cargo/bin/$$bin" || true

build:
ifeq ($(BACKEND),cuda)
	@echo "CUDA compute capability: $(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif
	@for bin in $(PATH_BINS); do \
		$(resolve_bin); \
		echo "Building $$bin ($(BACKEND))..."; \
		if [ -n "$(CARGO_FEATURES)" ]; then \
			cargo build --release -p $$pkg $(CARGO_FEATURES) \
			|| { echo "  $(BACKEND) failed; retrying CPU"; $(CUDA_CAP_HINT) \
			     cargo build --release -p $$pkg $(CARGO_FEATURES_CPU) || exit $$?; }; \
		else \
			cargo build --release -p $$pkg $(CARGO_FEATURES_CPU) || exit $$?; \
		fi; \
	done

build-cuda:  ; @$(MAKE) build BACKEND=cuda
build-metal: ; @$(MAKE) build BACKEND=metal
test:        ; cargo test --workspace
clean:       ; cargo clean; rm -f $(INSTALL_STATUS)
