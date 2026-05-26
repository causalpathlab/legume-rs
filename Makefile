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

# Validate BACKEND so a typo (e.g. `BACKEND=gpu`) fails fast instead of
# silently producing `--features gpu` and degrading to CPU for every binary.
ifeq (,$(filter $(BACKEND),cpu cuda metal))
$(error Unknown BACKEND='$(BACKEND)'; valid values are: cpu, cuda, metal)
endif

# HDF5 support is opt-in: enabled automatically when libhdf5 headers are
# detectable on the build host, otherwise compiled out so installs on
# HDF5-less systems (e.g. cluster login nodes) still succeed.
# Detection probes, in order:
#   1. `h5cc -showconfig` (the canonical HDF5 wrapper script)
#   2. `pkg-config --exists hdf5`
#   3. brew --prefix hdf5 (macOS Homebrew)
# Override with `HDF5=on` to force-enable, or `HDF5=off` to force-disable.
HAS_H5CC := $(shell command -v h5cc 2>/dev/null)
HAS_HDF5_PKGCFG := $(shell pkg-config --exists hdf5 2>/dev/null && echo yes)
HAS_HDF5_BREW := $(shell brew --prefix hdf5 >/dev/null 2>&1 && echo yes)

ifeq ($(or $(HAS_H5CC),$(HAS_HDF5_PKGCFG),$(HAS_HDF5_BREW)),)
DEFAULT_HDF5 := off
else
DEFAULT_HDF5 := on
endif

HDF5 ?= $(DEFAULT_HDF5)

ifeq (,$(filter $(HDF5),on off))
$(error Unknown HDF5='$(HDF5)'; valid values are: on, off)
endif

# Compose --features = $(BACKEND),hdf5 depending on toggles.
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
	@echo "Auto-detected HDF5 support:         $(DEFAULT_HDF5)"
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
	@echo "Override backend explicitly:  make <target> BACKEND={cpu|cuda|metal}"
	@echo "Override HDF5 detection:      make <target> HDF5={on|off}"
	@echo "    on  pulls in libhdf5 (.h5/.h5ad readers + HDF5 sparse backend)"
	@echo "    off compiles those code paths out — useful on hosts missing"
	@echo "        libhdf5 development headers (cluster nodes, minimal Linux)."

all: install

# Reset the per-binary status file before the install loop, then print a
# per-binary summary table at the end so users can see which binaries
# actually got the requested backend versus fell back to CPU.
install: _install_status_init $(addprefix install-,$(BINARIES)) _install_status_report

_install_status_init:
	@rm -f $(INSTALL_STATUS_FILE)

_install_status_report:
	@echo ""
	@echo "Install summary (requested backend: $(BACKEND), HDF5: $(HDF5)):"
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

# CPU-only fallback feature string. Drops the GPU backend on retry but keeps
# HDF5 if it was requested — so a GPU build that fails because of missing
# CUDA libraries doesn't also strip HDF5 support along the way.
ifeq ($(HDF5),on)
CARGO_FEATURES_CPU_FALLBACK := --features hdf5
else
CARGO_FEATURES_CPU_FALLBACK :=
endif

# Per-binary install. When a GPU backend is requested, try it first and fall
# back to a CPU install if the GPU build fails — so a default `make install`
# on a Linux box without CUDA libraries still succeeds. Each binary's actual
# backend is appended to INSTALL_STATUS_FILE for the summary.
$(addprefix install-,$(BINARIES)):
	@bin=$(@:install-%=%); \
	if [ -n "$(CARGO_FEATURES)" ]; then \
	    echo "Installing $$bin (backend: $(BACKEND), HDF5: $(HDF5))..."; \
	    if cargo install --path $$bin $(CARGO_FEATURES); then \
	        echo "$$bin $(BACKEND)" >> $(INSTALL_STATUS_FILE); \
	    else \
	        echo ""; \
	        echo "  $(BACKEND) build of $$bin failed; retrying with CPU"; \
	        echo ""; \
	        cargo install --path $$bin $(CARGO_FEATURES_CPU_FALLBACK); \
	        echo "$$bin cpu" >> $(INSTALL_STATUS_FILE); \
	    fi; \
	else \
	    echo "Installing $$bin (backend: cpu, HDF5: $(HDF5))..."; \
	    cargo install --path $$bin $(CARGO_FEATURES_CPU_FALLBACK); \
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
	    echo "Building $$bin (backend: $(BACKEND), HDF5: $(HDF5))..."; \
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
