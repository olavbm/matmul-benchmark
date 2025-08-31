# Matrix Multiplication Benchmark Analysis Pipeline
# ================================================

# Configuration
TRIALS ?= 20
BINARY = target/release/matmul
RUST_FLAGS = --release

# Data files
STATS_DATA = scaling_stats_$(TRIALS).txt
SIMPLE_DATA = scaling_data.txt

# Generated plots  
PERFORMANCE_PLOT = distribution_performance_$(TRIALS).png
ANALYSIS_PLOT = distribution_analysis_$(TRIALS).png
SIMPLE_PLOTS = scaling_analysis.png speedup_analysis.png

# Default target
.PHONY: all
all: analyze

# Help target
.PHONY: help
help:
	@echo "Matrix Multiplication Benchmark Analysis"
	@echo "========================================"
	@echo ""
	@echo "Targets:"
	@echo "  help              Show this help message"
	@echo "  build            Build the release binary"
	@echo "  quick            Quick benchmark (10 trials) + analysis + plots"
	@echo "  analyze          Full benchmark ($(TRIALS) trials) + analysis + plots"
	@echo "  benchmark        Run statistical benchmark only"
	@echo "  simple           Simple single-trial benchmark + gnuplot"
	@echo "  plots            Generate distribution plots from existing data"
	@echo "  view             Open distribution plots in image viewer"
	@echo "  clean            Remove generated files"
	@echo "  stats            Show statistical summary from existing data"
	@echo ""
	@echo "Configuration:"
	@echo "  TRIALS=$(TRIALS)          Number of trials per matrix size"
	@echo ""
	@echo "Usage examples:"
	@echo "  make analyze              # Full analysis with $(TRIALS) trials"
	@echo "  make analyze TRIALS=100   # Custom trial count"
	@echo "  make quick                # Fast 10-trial analysis"
	@echo "  make simple               # Single-trial + gnuplot"

# Build the release binary
.PHONY: build
build: $(BINARY)

$(BINARY): src/*.rs Cargo.toml
	@echo "Building release binary..."
	cargo build $(RUST_FLAGS)

# Quick benchmark (10 trials)
.PHONY: quick
quick: 
	@$(MAKE) analyze TRIALS=10

# Full statistical benchmark and analysis
.PHONY: analyze
analyze: build $(STATS_DATA) plots stats view

# Run statistical benchmark
.PHONY: benchmark
benchmark: $(STATS_DATA)

$(STATS_DATA): $(BINARY)
	@echo "Running statistical benchmark ($(TRIALS) trials per matrix size)..."
	@echo "This may take several minutes for larger trial counts..."
	./$(BINARY) --scaling $(TRIALS) > $@
	@echo "Benchmark complete: $@"

# Generate distribution plots
.PHONY: plots
plots: $(PERFORMANCE_PLOT) $(ANALYSIS_PLOT)

$(PERFORMANCE_PLOT) $(ANALYSIS_PLOT): $(STATS_DATA) analyze_stats.py
	@echo "Generating statistical distribution plots..."
	python3 analyze_stats.py $< 
	@# Rename files to include trial count
	@if [ -f "distribution_performance.png" ]; then \
		mv distribution_performance.png $(PERFORMANCE_PLOT); \
	fi
	@if [ -f "distribution_analysis.png" ]; then \
		mv distribution_analysis.png $(ANALYSIS_PLOT); \
	fi
	@echo "Generated: $(PERFORMANCE_PLOT)"
	@echo "Generated: $(ANALYSIS_PLOT)"

# Show statistical summary
.PHONY: stats
stats: $(STATS_DATA)
	@echo "Statistical Analysis Summary:"
	@echo "============================="
	python3 analyze_stats.py $< | head -30

# Simple single-trial benchmark with gnuplot
.PHONY: simple
simple: $(SIMPLE_DATA) $(SIMPLE_PLOTS) view-simple

$(SIMPLE_DATA): $(BINARY)
	@echo "Running simple benchmark (1 trial per matrix size)..."
	./$(BINARY) --scaling 1 > $@

$(SIMPLE_PLOTS): $(SIMPLE_DATA) scaling.plt
	@echo "Generating plots with gnuplot..."
	gnuplot scaling.plt
	@echo "Generated: $(SIMPLE_PLOTS)"

# View distribution plots
.PHONY: view
view: $(PERFORMANCE_PLOT) $(ANALYSIS_PLOT)
	@echo "Opening distribution plots..."
	@# Try different image viewers in order of preference
	@if command -v eog >/dev/null 2>&1; then \
		eog $(PERFORMANCE_PLOT) $(ANALYSIS_PLOT) & \
	elif command -v feh >/dev/null 2>&1; then \
		feh $(PERFORMANCE_PLOT) $(ANALYSIS_PLOT) & \
	elif command -v display >/dev/null 2>&1; then \
		display $(PERFORMANCE_PLOT) & \
		display $(ANALYSIS_PLOT) & \
	elif command -v open >/dev/null 2>&1; then \
		open $(PERFORMANCE_PLOT) $(ANALYSIS_PLOT) \
	else \
		echo "No image viewer found. Please open these files manually:"; \
		echo "  - $(PERFORMANCE_PLOT)"; \
		echo "  - $(ANALYSIS_PLOT)"; \
	fi

# View simple gnuplot results
.PHONY: view-simple
view-simple: $(SIMPLE_PLOTS)
	@echo "Opening gnuplot results..."
	@if command -v eog >/dev/null 2>&1; then \
		eog scaling_analysis.png speedup_analysis.png & \
	elif command -v feh >/dev/null 2>&1; then \
		feh scaling_analysis.png speedup_analysis.png & \
	elif command -v display >/dev/null 2>&1; then \
		display scaling_analysis.png & \
		display speedup_analysis.png & \
	elif command -v open >/dev/null 2>&1; then \
		open scaling_analysis.png speedup_analysis.png \
	else \
		echo "No image viewer found. Please open these files manually:"; \
		echo "  - scaling_analysis.png"; \
		echo "  - speedup_analysis.png"; \
	fi

# Data validation
.PHONY: validate
validate: $(STATS_DATA)
	@echo "Validating benchmark data..."
	@lines=$$(wc -l < $(STATS_DATA)); \
	expected=$$(($(TRIALS) * 2 * 12 + 1)); \
	if [ $$lines -eq $$expected ]; then \
		echo "✓ Data validation passed ($$lines lines, expected $$expected)"; \
	else \
		echo "✗ Data validation failed ($$lines lines, expected $$expected)"; \
		exit 1; \
	fi

# Comprehensive benchmark comparison
.PHONY: compare
compare: build
	@echo "Running comprehensive comparison..."
	@echo "1. Quick benchmark (10 trials):"
	@$(MAKE) --no-print-directory analyze TRIALS=10
	@echo ""
	@echo "2. Standard benchmark (50 trials):"  
	@$(MAKE) --no-print-directory analyze TRIALS=50
	@echo ""
	@echo "Comparison complete!"

# Performance test
.PHONY: test
test: build
	@echo "Running correctness tests..."
	cargo test
	@echo "Running basic benchmark test..."
	./$(BINARY) --scaling 3 > test_data.txt
	@if [ -s test_data.txt ]; then \
		echo "✓ Benchmark test passed"; \
		rm test_data.txt; \
	else \
		echo "✗ Benchmark test failed"; \
		exit 1; \
	fi

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f scaling_stats_*.txt
	rm -f scaling_data.txt
	rm -f distribution_*.png
	rm -f scaling_analysis.png speedup_analysis.png
	rm -f *.tmp
	rm -f test_data.txt
	@echo "Clean complete"

# Deep clean (including Rust build artifacts)
.PHONY: distclean
distclean: clean
	cargo clean
	@echo "Deep clean complete"

# Install dependencies
.PHONY: deps
deps:
	@echo "Checking dependencies..."
	@echo "Rust toolchain:"
	@rustc --version || (echo "Install Rust from https://rustup.rs/" && exit 1)
	@echo "Python dependencies:"
	@python3 -c "import matplotlib, numpy" 2>/dev/null || \
		(echo "Install with: pip3 install matplotlib numpy" && exit 1)
	@echo "Gnuplot:"
	@gnuplot --version >/dev/null || (echo "Install gnuplot package" && exit 1)
	@echo "✓ All dependencies satisfied"

# Show project info
.PHONY: info
info:
	@echo "Matrix Multiplication Benchmark Project"
	@echo "======================================"
	@echo "Project structure:"
	@find . -name "*.rs" -o -name "*.py" -o -name "*.plt" -o -name "Makefile" | \
		grep -v target | sort
	@echo ""
	@echo "Generated files:"
	@ls -la *.txt *.png 2>/dev/null || echo "  (none yet - run 'make analyze')"
	@echo ""
	@echo "Binary info:"
	@ls -la $(BINARY) 2>/dev/null || echo "  (not built yet - run 'make build')"

# Phony targets for make tab completion
.PHONY: analyze benchmark plots view stats simple quick clean distclean test compare validate deps info