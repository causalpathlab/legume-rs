BINARIES := senna pinto cocoa faba chickpea data-beans data-beans-sim fagioli gene-text canna

# Packages whose crate directory / Cargo package name differs from the
# installed binary name. `canna` is the CLI; the crate stays `cnv`.
#
# The mapping has to happen in the shell, not with a make function: the loops
# below iterate over `$$bin` inside a recipe, so make would only ever see the
# literal text `$$bin` and never match it.
crate_pkg_case = case $$bin in canna) pkg=cnv;; *) pkg=$$bin;; esac

# Binaries with no `cuda` / `metal` feature to pass. `faba` reads BAM files and
# writes sparse matrices; nothing on that path touches a GPU, and the
# model-fitting subcommands that once did now live in senna. Passing a backend
# feature these crates do not declare makes cargo fail, which the loops below
# would then "recover" from by retrying on CPU -- a wasted compile and a
# summary line that reads like a GPU failure. Build them as CPU directly.
CPU_ONLY_BINARIES := faba

# Backend selection.
#
# `make install` auto-detects: CUDA on Linux if `nvcc` is on PATH, Metal on
# macOS, otherwise CPU. The per-binary recipe also falls back to CPU at runtime
# if a GPU build fails (e.g. the CUDA toolkit is present but broken).
#
# Force a specific backend:
#   make install-cpu     # CPU only (no GPU features)
#   make install-cuda    # NVIDIA CUDA + cuDNN
#   make install-metal   # Apple Metal + Accelerate
#   make install BACKEND={cpu|cuda|metal}
#
# HDF5 (.h5/.h5ad I/O) is opt-in — libhdf5 isn't on every host:
#   make install HDF5=on
UNAME_S := $(shell uname -s)
HAS_NVCC := $(shell command -v nvcc 2>/dev/null)

ifeq ($(UNAME_S),Darwin)
DEFAULT_BACKEND := metal
else ifneq ($(HAS_NVCC),)
DEFAULT_BACKEND := cuda
else
DEFAULT_BACKEND := cpu
endif

BACKEND ?= $(DEFAULT_BACKEND)

# libhdf5 discovery (called only when HDF5=on or auto-detection runs).
#
# Why we can't just trust h5cc: on some HPC module systems h5cc is on PATH
# but its -showconfig output points at unreachable paths, so hdf5-metno-sys's
# own discovery fails with "Unable to locate HDF5 root directory and/or
# headers". We bypass that by setting HDF5_DIR ourselves to a prefix where
# we've verified both `include/hdf5.h` and a `libhdf5` library exist.
#
# Probes, in order of preference:
#   1. $HDF5_DIR if the caller already set it
#   2. $CONDA_PREFIX (caller is in an active conda env shipping HDF5)
#   3. h5cc on PATH → derive prefix via `dirname $(dirname $(which h5cc))`
#      (h5cc lives at <prefix>/bin/h5cc, so two dirnames gives the prefix)
#   4. Common system prefixes: /opt/homebrew, /usr/local, /usr
#
# Override with `HDF5_DIR=<prefix>` if your install lives somewhere unusual.
H5CC_PATH := $(shell command -v h5cc 2>/dev/null)
# Linux + macOS: avoid `readlink -f` (BSD readlink on macOS doesn't have it).
# `dirname $(dirname …)` is fine here because h5cc always lives at
# `<prefix>/bin/h5cc` whether the entry on PATH is the real file or a symlink.
H5CC_PREFIX := $(if $(H5CC_PATH),$(shell dirname $$(dirname $(H5CC_PATH))))

HDF5_CANDIDATES := $(HDF5_DIR) $(CONDA_PREFIX) $(H5CC_PREFIX) /opt/homebrew /usr/local /usr

# Stage 1: pick the first candidate that has BOTH `<prefix>/include/hdf5.h`
# and `<prefix>/lib*/libhdf5.{so,a,dylib}`. This handles conda, homebrew,
# `/usr/local`, and the HPC-Anaconda case where h5cc -showconfig is unreliable
# but the install is intact under a clean prefix.
HDF5_DETECTED_DIR := $(firstword $(foreach d,$(HDF5_CANDIDATES), \
    $(if $(strip $(d)), \
    $(if $(wildcard $(d)/include/hdf5.h), \
    $(if $(or $(wildcard $(d)/lib/libhdf5.so), \
              $(wildcard $(d)/lib/libhdf5.a), \
              $(wildcard $(d)/lib/libhdf5.dylib), \
              $(wildcard $(d)/lib64/libhdf5.so), \
              $(wildcard $(d)/lib64/libhdf5.a)), \
    $(d))))))

# Stage 2 (Debian/Ubuntu split layout): header lives at
# `/usr/include/hdf5/serial/hdf5.h` and library at
# `/usr/lib/<arch>-linux-gnu/hdf5/serial/libhdf5.so` — no single prefix has
# both, so we can't set HDF5_DIR. But `libhdf5-dev` ships pkg-config files
# (`hdf5` and/or `hdf5-serial`) and hdf5-metno-sys's own discovery succeeds
# there. Detecting this just flips HDF5 on; HDF5_DIR stays unset.
ifeq ($(HDF5_DETECTED_DIR),)
HDF5_PKGCFG_OK := $(shell if pkg-config --exists hdf5 2>/dev/null || pkg-config --exists hdf5-serial 2>/dev/null; then echo yes; fi)
else
HDF5_PKGCFG_OK := yes
endif

ifeq ($(HDF5_PKGCFG_OK),yes)
DEFAULT_HDF5 := on
else
DEFAULT_HDF5 := off
endif

HDF5 ?= $(DEFAULT_HDF5)

# Only export HDF5_DIR when stage 1 found a clean prefix. In the stage-2
# (Debian split) case we deliberately leave HDF5_DIR unset and let
# hdf5-metno-sys's internal probe handle the discovery.
ifeq ($(HDF5),on)
ifneq ($(HDF5_DETECTED_DIR),)
export HDF5_DIR := $(HDF5_DETECTED_DIR)
endif
endif

# Validate so a typo (e.g. `BACKEND=gpu`, `HDF5=yes`) fails fast instead of
# silently producing an unknown --features flag.
ifeq (,$(filter $(BACKEND),cpu cuda metal))
$(error Unknown BACKEND='$(BACKEND)'; valid values are: cpu, cuda, metal)
endif
ifeq (,$(filter $(HDF5),on off))
$(error Unknown HDF5='$(HDF5)'; valid values are: on, off)
endif

# CUDA compute capability (consulted only when BACKEND=cuda).
#
# candle-kernels resolves the GPU architecture at build time through
# cudaforge, whose own detection is: $CUDA_COMPUTE_CAP, else `nvidia-smi`.
# nvidia-smi goes through NVML, which refuses to run whenever the userspace
# driver library and the loaded kernel module disagree -- the normal state
# after a driver package update until the next reboot -- and it is absent in
# containers and on GPU-less build hosts. The build then dies with
# `ComputeCapDetectionFailed`, and the per-binary fallback below "recovers"
# by shipping a CPU-only binary on a machine that has a perfectly good GPU.
#
# So resolve the capability here and export it, in order of preference:
#   1. CUDA_COMPUTE_CAP set by the caller (`make install CUDA_COMPUTE_CAP=86`)
#   2. nvidia-smi, when it works; the answer is cached in $(CUDA_CAP_CACHE)
#      for the days when it doesn't
#   3. that cache
#   4. the kernel module's own record of the GPU model in
#      /proc/driver/nvidia/gpus/*/information (readable even when NVML is
#      broken), mapped to its architecture family below. Where a family
#      spans several capabilities the table rounds DOWN: PTX built for a
#      lower compute_XX still JIT-compiles on a newer GPU of the same or a
#      later generation, PTX built for a higher one does not load at all.
# If all four come up empty the variable stays unset and the CUDA build
# fails exactly as before, with the hint to pass CUDA_COMPUTE_CAP.
CUDA_CAP_CACHE := $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/legume-rs/cuda-compute-cap

# GPU model name -> lowest compute capability of that family. Consumer parts
# match on the series prefix ("RTX 30"), data-centre parts on the model.
# Order matters where one pattern is a substring of another ("A10" is in
# "A100"): the more specific family is tested first.
cuda_cap_from_model = $(strip \
  $(if $(or $(findstring RTX 50,$(1)),$(findstring Blackwell,$(1))),120, \
  $(if $(or $(findstring B200,$(1)),$(findstring B100,$(1)),$(findstring GB200,$(1))),100, \
  $(if $(or $(findstring H100,$(1)),$(findstring H200,$(1)),$(findstring H800,$(1)),$(findstring H20,$(1)),$(findstring GH200,$(1))),90, \
  $(if $(or $(findstring RTX 40,$(1)),$(findstring L40,$(1)),$(findstring L4,$(1)),$(findstring Ada,$(1))),89, \
  $(if $(or $(findstring A100,$(1)),$(findstring A800,$(1)),$(findstring A30,$(1))),80, \
  $(if $(or $(findstring RTX 30,$(1)),$(findstring RTX A,$(1)),$(findstring A10,$(1)),$(findstring A40,$(1)),$(findstring A16,$(1)),$(findstring A2,$(1))),86, \
  $(if $(or $(findstring RTX 20,$(1)),$(findstring GTX 16,$(1)),$(findstring Quadro RTX,$(1)),$(findstring TITAN RTX,$(1)),$(findstring T4,$(1))),75, \
  $(if $(or $(findstring V100,$(1)),$(findstring TITAN V,$(1))),70, \
  $(if $(or $(findstring GTX 10,$(1)),$(findstring P100,$(1)),$(findstring P40,$(1)),$(findstring TITAN X,$(1))),60, \
  ))))))))))

ifeq ($(BACKEND),cuda)
ifneq ($(strip $(CUDA_COMPUTE_CAP)),)
CUDA_CAP_SOURCE := set by caller
else
# nvidia-smi prints its NVML complaint on stdout, so accept only a bare
# "major.minor" line, and normalise "8.6" -> "86" (cudaforge takes either).
CUDA_CAP_SMI := $(shell timeout 15 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -n1 | tr -d ' ' | grep -E -x '[0-9]+\.[0-9]+' | tr -d .)
ifneq ($(CUDA_CAP_SMI),)
CUDA_COMPUTE_CAP := $(CUDA_CAP_SMI)
CUDA_CAP_SOURCE := nvidia-smi
$(shell mkdir -p $(dir $(CUDA_CAP_CACHE)) && echo $(CUDA_CAP_SMI) > $(CUDA_CAP_CACHE))
else ifneq ($(wildcard $(CUDA_CAP_CACHE)),)
CUDA_COMPUTE_CAP := $(shell cat $(CUDA_CAP_CACHE))
CUDA_CAP_SOURCE := cached from an earlier nvidia-smi run ($(CUDA_CAP_CACHE))
else
CUDA_GPU_MODEL := $(shell sed -n 's/^Model:[[:space:]]*//p' /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -n1)
CUDA_COMPUTE_CAP := $(call cuda_cap_from_model,$(CUDA_GPU_MODEL))
CUDA_CAP_SOURCE := $(if $(CUDA_COMPUTE_CAP),from the GPU model '$(CUDA_GPU_MODEL)'; nvidia-smi is not working,not detected$(if $(CUDA_GPU_MODEL),; nvidia-smi is not working and '$(CUDA_GPU_MODEL)' is not in the family table))
endif
endif
ifneq ($(strip $(CUDA_COMPUTE_CAP)),)
export CUDA_COMPUTE_CAP
endif
endif

# Compose --features = $(BACKEND),hdf5 depending on toggles. Empty when both
# off so we don't pass an empty --features to cargo.
CARGO_FEATURE_LIST :=
ifneq ($(BACKEND),cpu)
CARGO_FEATURE_LIST += $(BACKEND)
endif
ifeq ($(HDF5),on)
CARGO_FEATURE_LIST += hdf5
endif

empty :=
space := $(empty) $(empty)
comma := ,
ifeq ($(strip $(CARGO_FEATURE_LIST)),)
CARGO_FEATURES :=
else
CARGO_FEATURES := --features $(subst $(space),$(comma),$(strip $(CARGO_FEATURE_LIST)))
endif

# CPU-only fallback feature string. Drops the GPU backend on retry but keeps
# HDF5 if it was requested — so a GPU build that fails because of missing
# CUDA libraries doesn't also strip HDF5 support along the way.
ifeq ($(HDF5),on)
CARGO_FEATURES_CPU_FALLBACK := --features hdf5
else
CARGO_FEATURES_CPU_FALLBACK :=
endif

# Printed after a failed CUDA build when no compute capability could be
# resolved: that is the one failure the CPU retry silently papers over.
ifeq ($(BACKEND)$(strip $(CUDA_COMPUTE_CAP)),cuda)
CUDA_CAP_HINT := echo "  (no CUDA compute capability could be detected; if the GPU is fine, rerun with CUDA_COMPUTE_CAP=<cap>, e.g. 86)";
else
CUDA_CAP_HINT :=
endif

# Per-binary fallback status is written here so the aggregate `install`
# target can report what each binary was actually built with.
INSTALL_STATUS_FILE := $(CURDIR)/.make-install-status

.PHONY: all install install-cpu install-cuda install-metal \
        $(addprefix install-,$(BINARIES)) \
        uninstall $(addprefix uninstall-,$(BINARIES)) \
        build build-cuda build-metal test clean help \
        _install_status_init _install_status_report

help:
	@echo "Legume-rs Makefile"
	@echo ""
	@echo "Auto-detected backend on this host: $(DEFAULT_BACKEND)"
	@echo "Auto-detected HDF5 support:         $(DEFAULT_HDF5)$(if $(HDF5_DETECTED_DIR), (HDF5_DIR=$(HDF5_DETECTED_DIR)))"
ifeq ($(BACKEND),cuda)
	@echo "CUDA compute capability:            $(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif
	@echo ""
	@echo "Install targets:"
	@echo "  install              - Install all binaries with auto-detected backend"
	@echo "                         (falls back to CPU if a GPU build fails)"
	@echo "  install-cpu          - Force CPU-only install"
	@echo "  install-cuda         - Force CUDA + cuDNN install (Linux)"
	@echo "  install-metal        - Force Metal + Accelerate install (macOS)"
	@echo "  install-<binary>     - Install one binary ($(BINARIES))"
	@echo ""
	@echo "Other targets:"
	@echo "  uninstall            - Uninstall all binaries"
	@echo "  uninstall-<binary>   - Uninstall one binary"
	@echo "  build                - Build the workspace with auto-detected backend"
	@echo "                         (falls back to CPU per-binary if a GPU build fails)"
	@echo "  build-cuda           - Build with CUDA"
	@echo "  build-metal          - Build with Metal"
	@echo "  test                 - Run all tests"
	@echo "  clean                - Clean build artifacts"
	@echo ""
	@echo "Overrides:"
	@echo "  make <target> BACKEND={cpu|cuda|metal}"
	@echo "  make <target> HDF5={on|off}      # default = auto-detected above"
	@echo "  HDF5_DIR=<prefix> make ...       # override the detected prefix"
	@echo "  make <target> CUDA_COMPUTE_CAP=86 # GPU architecture when nvidia-smi can't say"

all: install

# Reset the per-binary status file before the install loop, then print a
# per-binary summary table at the end so users can see which binaries
# actually got the requested backend versus fell back to CPU.
install: _install_status_init $(addprefix install-,$(BINARIES)) _install_status_report

_install_status_init:
	@rm -f $(INSTALL_STATUS_FILE)
ifeq ($(BACKEND),cuda)
	@echo "CUDA compute capability: $(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif

_install_status_report:
	@echo ""
	@echo "Install summary (requested backend: $(BACKEND), HDF5: $(HDF5)$(if $(and $(filter on,$(HDF5)),$(HDF5_DIR)), [HDF5_DIR=$(HDF5_DIR)])):"
	@if [ -f $(INSTALL_STATUS_FILE) ]; then \
	    awk '{ printf "  %-18s -> %s\n", $$1, $$2 }' $(INSTALL_STATUS_FILE); \
	    if grep -q ' cpu$$' $(INSTALL_STATUS_FILE) && [ "$(BACKEND)" != "cpu" ]; then \
	        echo ""; \
	        echo "  Note: one or more binaries fell back to CPU."; \
	    fi; \
	    if grep -q ' n/a$$' $(INSTALL_STATUS_FILE); then \
	        echo ""; \
	        echo "  n/a = no GPU code path; built CPU-only by design, not a fallback."; \
	    fi; \
	else \
	    echo "  (no per-binary status was recorded)"; \
	fi
	@rm -f $(INSTALL_STATUS_FILE)

install-cpu:
	@$(MAKE) install BACKEND=cpu

install-cuda:
	@$(MAKE) install BACKEND=cuda

install-metal:
	@$(MAKE) install BACKEND=metal

# Per-binary install. When a GPU backend is requested, try it first and fall
# back to a CPU install if the GPU build fails — so a default `make install`
# on a Linux box without CUDA libraries still succeeds. Each binary's actual
# backend is appended to INSTALL_STATUS_FILE for the summary.
$(addprefix install-,$(BINARIES)):
	@bin=$(@:install-%=%); \
	$(crate_pkg_case); \
	if echo " $(CPU_ONLY_BINARIES) " | grep -q " $$bin "; then \
	    echo "Installing $$bin (no GPU backend; CPU-only by design)..."; \
	    cargo install --locked --path $$pkg $(CARGO_FEATURES_CPU_FALLBACK); \
	    echo "$$bin n/a" >> $(INSTALL_STATUS_FILE); \
	elif [ -n "$(CARGO_FEATURES)" ]; then \
	    echo "Installing $$bin (backend: $(BACKEND))..."; \
	    if cargo install --locked --path $$pkg $(CARGO_FEATURES); then \
	        echo "$$bin $(BACKEND)" >> $(INSTALL_STATUS_FILE); \
	    else \
	        echo ""; \
	        echo "  $(BACKEND) build of $$bin failed; retrying with CPU"; \
	        $(CUDA_CAP_HINT) \
	        echo ""; \
	        cargo install --locked --path $$pkg $(CARGO_FEATURES_CPU_FALLBACK); \
	        echo "$$bin cpu" >> $(INSTALL_STATUS_FILE); \
	    fi; \
	else \
	    echo "Installing $$bin (backend: cpu)..."; \
	    cargo install --locked --path $$pkg $(CARGO_FEATURES_CPU_FALLBACK); \
	    echo "$$bin cpu" >> $(INSTALL_STATUS_FILE); \
	fi

uninstall: $(addprefix uninstall-,$(BINARIES))
	@echo "All binaries uninstalled successfully"

$(addprefix uninstall-,$(BINARIES)):
	@bin=$(@:uninstall-%=%); \
	$(crate_pkg_case); \
	echo "Uninstalling $$bin..."; \
	cargo uninstall $$pkg

# Mirror the install per-binary fallback: a single GPU-build failure does
# NOT abort the whole loop. Each binary that can't be built with the
# requested backend is retried with CPU features.
build:
ifeq ($(BACKEND),cpu)
ifeq ($(HDF5),on)
	@for bin in $(BINARIES); do \
	    $(crate_pkg_case); \
	    cargo build --release -p $$pkg --features hdf5 || exit $$?; \
	done
else
	cargo build --release --workspace
endif
else
ifeq ($(BACKEND),cuda)
	@echo "CUDA compute capability: $(or $(CUDA_COMPUTE_CAP),none) ($(CUDA_CAP_SOURCE))"
endif
	@for bin in $(BINARIES); do \
	    $(crate_pkg_case); \
	    if echo " $(CPU_ONLY_BINARIES) " | grep -q " $$bin "; then \
	        echo "Building $$bin (no GPU backend; CPU-only by design)..."; \
	        cargo build --release -p $$pkg $(CARGO_FEATURES_CPU_FALLBACK) || exit $$?; \
	        continue; \
	    fi; \
	    echo "Building $$bin (backend: $(BACKEND))..."; \
	    if ! cargo build --release -p $$pkg $(CARGO_FEATURES); then \
	        echo ""; \
	        echo "  $(BACKEND) build of $$bin failed; retrying with CPU"; \
	        $(CUDA_CAP_HINT) \
	        echo ""; \
	        cargo build --release -p $$pkg $(CARGO_FEATURES_CPU_FALLBACK) || exit $$?; \
	    fi; \
	done
endif

build-cuda:
	@$(MAKE) build BACKEND=cuda

build-metal:
	@$(MAKE) build BACKEND=metal

test:
	cargo test --workspace

clean:
	cargo clean
	@rm -f $(INSTALL_STATUS_FILE)
