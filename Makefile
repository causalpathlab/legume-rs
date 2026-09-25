# =============================================================================
# Binaries (all from crates.io)
# =============================================================================
BINARIES := senna lupin chickpea pinto cocoa mung faba fqtl data-beans

# $$bin -> pkg / extra_feat / supported (must run inside a shell recipe).
# `supported` lists the backend/HDF5 features the crate actually has, so we
# never pass e.g. `--features metal` to faba or `--features hdf5` to fqtl.
resolve_bin = case $$bin in \
	senna)      pkg=senna-rs;    extra_feat=;    supported="cuda metal hdf5";; \
	lupin)      pkg=lupin-rs;    extra_feat=;    supported="cuda metal hdf5";; \
	chickpea)   pkg=chickpea-rs; extra_feat=;    supported="cuda metal hdf5";; \
	pinto)      pkg=pinto-rs;    extra_feat=;    supported="cuda metal hdf5";; \
	cocoa)      pkg=cocoa-rs;    extra_feat=;    supported="cuda metal hdf5";; \
	mung)       pkg=mung-cnv;    extra_feat=;    supported="cuda metal hdf5";; \
	faba)       pkg=faba;        extra_feat=;    supported="hdf5";; \
	fqtl)       pkg=fqtl-rs;     extra_feat=;    supported="cuda metal";; \
	data-beans) pkg=data-beans;  extra_feat=sim; supported="cuda metal hdf5";; \
	*)          pkg=$$bin;       extra_feat=;    supported="";; \
	esac

# $$1 = requested features; sets $$feat to "--features a,b" (or empty),
# keeping only those in $$supported plus $$extra_feat.
pick_feats = feat=; for f in "$$@" $$extra_feat; do \
	case " $$supported $$extra_feat " in *" $$f "*) feat="$$feat,$$f";; esac; \
	done; feat=$${feat\#,}; [ -n "$$feat" ] && feat="--features $$feat"

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

FEATS :=
ifneq ($(BACKEND),cpu)
FEATS += $(BACKEND)
endif
ifeq ($(HDF5),on)
FEATS += hdf5
endif

CPU_FEATS := $(if $(filter on,$(HDF5)),hdf5)
CUDA_CAP_HINT := $(if $(filter cuda,$(BACKEND)$(if $(CUDA_COMPUTE_CAP),x)), \
	echo "  hint: pass CUDA_COMPUTE_CAP=<cap> (e.g. 86)";)

.PHONY: help install install-cpu install-cuda install-metal uninstall

help:
	@echo "legume-rs meta installer (crates.io only)"
	@echo "  make install [BACKEND=cpu|cuda|metal] [HDF5=on|off]"
	@echo "  make uninstall"
	@echo "Binaries: $(BINARIES)"

install: install-$(BACKEND)

install-cpu:
	@$(MAKE) _install BACKEND=cpu

install-cuda:
	@$(MAKE) _install BACKEND=cuda

install-metal:
	@$(MAKE) _install BACKEND=metal

_install:
	@echo "Installing from crates.io (BACKEND=$(BACKEND) HDF5=$(HDF5))"
	@ok=0; failed=; \
	for bin in $(BINARIES); do \
		$(resolve_bin); \
		set -- $(FEATS); $(pick_feats); \
		echo "==> $$bin ($$pkg) $$feat"; \
		if cargo install --locked --force $$pkg $$feat; then \
			ok=$$((ok+1)); continue; \
		fi; \
		set -- $(CPU_FEATS); $(pick_feats); \
		echo "  retry CPU for $$bin $$feat"; \
		$(CUDA_CAP_HINT) \
		if cargo install --locked --force $$pkg $$feat; then \
			ok=$$((ok+1)); \
		else \
			failed="$$failed $$bin"; \
		fi; \
	done; \
	echo "done: $$ok ok"; \
	if [ -n "$$failed" ]; then echo "failed:$$failed"; exit 1; fi

uninstall:
	@for bin in $(BINARIES); do \
		$(resolve_bin); \
		echo "cargo uninstall $$pkg"; \
		cargo uninstall $$pkg 2>/dev/null || true; \
	done
