BINARIES := senna pinto cocoa faba chickpea candle-util data-beans

.PHONY: all install $(addprefix install-,$(BINARIES)) build test clean help

help:
	@echo "Legume-rs Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install              - Install all binaries"
	@echo "  install-<binary>     - Install a specific binary ($(BINARIES))"
	@echo "  build                - Build all workspace members"
	@echo "  test                 - Run all tests"
	@echo "  clean                - Clean build artifacts"

all: install

install: $(addprefix install-,$(BINARIES))
	@echo "All binaries installed successfully"

$(addprefix install-,$(BINARIES)):
	@echo "Installing $(@:install-%=%)..."
	cargo install --path $(@:install-%=%)

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean
