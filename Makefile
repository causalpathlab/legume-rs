BINARIES := senna pinto cocoa faba chickpea data-beans data-beans-sim fagioli

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

all: install

# Reset the per-binary status file before the install loop, then print a
# per-binary summary table at the end so users can see which binaries
# actually got the requested backend versus fell back to CPU.
install: _install_status_init $(addprefix install-,$(BINARIES)) _install_status_report

_install_status_init:
	@rm -f $(INSTALL_STATUS_FILE)

_install_status_report:
	@echo ""
	@echo "Install summary (requested backend: $(BACKEND), HDF5: $(HDF5)$(if $(and $(filter on,$(HDF5)),$(HDF5_DIR)), [HDF5_DIR=$(HDF5_DIR)])):"
	@if [ -f $(INSTALL_STATUS_FILE) ]; then \
	    awk '{ printf "  %-18s -> %s\n", $$1, $$2 }' $(INSTALL_STATUS_FILE); \
	    if grep -q ' cpu$$' $(INSTALL_STATUS_FILE) && [ "$(BACKEND)" != "cpu" ]; then \
	        echo ""; \
	        echo "  Note: one or more binaries fell back to CPU."; \
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
	if [ -n "$(CARGO_FEATURES)" ]; then \
	    echo "Installing $$bin (backend: $(BACKEND))..."; \
	    if cargo install --locked --path $$bin $(CARGO_FEATURES); then \
	        echo "$$bin $(BACKEND)" >> $(INSTALL_STATUS_FILE); \
	    else \
	        echo ""; \
	        echo "  $(BACKEND) build of $$bin failed; retrying with CPU"; \
	        echo ""; \
	        cargo install --locked --path $$bin $(CARGO_FEATURES_CPU_FALLBACK); \
	        echo "$$bin cpu" >> $(INSTALL_STATUS_FILE); \
	    fi; \
	else \
	    echo "Installing $$bin (backend: cpu)..."; \
	    cargo install --locked --path $$bin $(CARGO_FEATURES_CPU_FALLBACK); \
	    echo "$$bin cpu" >> $(INSTALL_STATUS_FILE); \
	fi

uninstall: $(addprefix uninstall-,$(BINARIES))
	@echo "All binaries uninstalled successfully"

$(addprefix uninstall-,$(BINARIES)):
	@echo "Uninstalling $(@:uninstall-%=%)..."
	cargo uninstall $(@:uninstall-%=%)

# Mirror the install per-binary fallback: a single GPU-build failure does
# NOT abort the whole loop. Each binary that can't be built with the
# requested backend is retried with CPU features.
build:
ifeq ($(BACKEND),cpu)
ifeq ($(HDF5),on)
	@for bin in $(BINARIES); do \
	    cargo build --release -p $$bin --features hdf5 || exit $$?; \
	done
else
	cargo build --release --workspace
endif
else
	@for bin in $(BINARIES); do \
	    echo "Building $$bin (backend: $(BACKEND))..."; \
	    if ! cargo build --release -p $$bin $(CARGO_FEATURES); then \
	        echo ""; \
	        echo "  $(BACKEND) build of $$bin failed; retrying with CPU"; \
	        echo ""; \
	        cargo build --release -p $$bin $(CARGO_FEATURES_CPU_FALLBACK) || exit $$?; \
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
