.PHONY: all install install-senna install-pinto install-cocoa install-faba install-chickpea install-candle-util clean build test help

BINARIES = senna pinto cocoa faba chickpea candle-util

help:
	@echo "Legume-rs Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install              - Install all binaries"
	@echo "  install-<binary>     - Install a specific binary (senna, pinto, cocoa, faba, chickpea, candle-util)"
	@echo "  build                - Build all workspace members"
	@echo "  test                 - Run all tests"
	@echo "  clean                - Clean build artifacts"
	@echo ""

all: install

install: install-senna install-pinto install-cocoa install-faba install-chickpea install-candle-util
	@echo "All binaries installed successfully"

install-senna:
	@echo "Installing senna..."
	cargo install --path senna

install-pinto:
	@echo "Installing pinto..."
	cargo install --path pinto

install-cocoa:
	@echo "Installing cocoa..."
	cargo install --path cocoa

install-faba:
	@echo "Installing faba..."
	cargo install --path faba

install-chickpea:
	@echo "Installing chickpea..."
	cargo install --path chickpea

install-candle-util:
	@echo "Installing candle-util..."
	cargo install --path candle-util

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean
