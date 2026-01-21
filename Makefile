.PHONY: all install install-senna install-pinto install-cocoa install-faba install-chickpea clean build test help

BINARIES = senna pinto cocoa faba chickpea

help:
	@echo "Legume-rs Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install              - Install all binaries"
	@echo "  install-<binary>     - Install a specific binary (senna, pinto, cocoa, faba, chickpea)"
	@echo "  build                - Build all workspace members"
	@echo "  test                 - Run all tests"
	@echo "  clean                - Clean build artifacts"
	@echo ""

all: install

install: install-senna install-pinto install-cocoa install-faba install-chickpea
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

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean
