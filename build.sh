#!/usr/bin/env bash
#
# Build and install fast-forward development environment
#
# Usage:
#   ./build.sh              # Build and install only
#   ./build.sh --test       # Build, install, and run tests
#   ./build.sh --clean      # Clean build artifacts first
#   ./build.sh --with-s3    # Install S3 test dependencies
#

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="${PROJECT_ROOT}/rust"
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Parse arguments
RUN_TESTS=false
CLEAN_BUILD=false
INSTALL_S3=false

for arg in "$@"; do
    case $arg in
        --test)
            RUN_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --with-s3)
            INSTALL_S3=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test      Run tests after building"
            echo "  --clean     Clean build artifacts before building"
            echo "  --with-s3   Install S3 test dependencies (moto, s3fs, boto3)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Change to project root
cd "$PROJECT_ROOT"

# Step 0: Check for Rust source code
if [ ! -d "$RUST_DIR" ]; then
    error "Rust source directory not found at $RUST_DIR"
    error "Please clone the private Rust repository:"
    error "  git clone https://github.com/UnravelSports/fast-forward-rs.git rust"
    exit 1
fi

if [ ! -f "$RUST_DIR/Cargo.toml" ]; then
    error "Cargo.toml not found in $RUST_DIR"
    exit 1
fi

# Step 1: Clean build artifacts if requested
if [ "$CLEAN_BUILD" = true ]; then
    info "Cleaning build artifacts..."
    rm -rf "$RUST_DIR/target/release" "$RUST_DIR/target/debug"
    rm -rf "$PROJECT_ROOT"/python/fastforward/*.so
    rm -rf "$PROJECT_ROOT"/python/fastforward/*.pyd
    info "Clean complete."
fi

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    info "Virtual environment created at $VENV_DIR"
fi

# Step 3: Unset CONDA_PREFIX first to avoid conflicts with maturin
# This must be done before setting VIRTUAL_ENV
if [ -n "$CONDA_PREFIX" ]; then
    info "Unsetting CONDA_PREFIX to avoid maturin conflicts..."
    unset CONDA_PREFIX
fi

# Step 4: Configure PyO3 and maturin to use venv's Python
info "Using virtual environment at $VENV_DIR"
export VIRTUAL_ENV="${VENV_DIR}"
export PYO3_PYTHON="${PYTHON}"

# Get the actual library path from Python (needed for linking)
LIBDIR=$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export DYLD_LIBRARY_PATH="$LIBDIR:$DYLD_LIBRARY_PATH"
info "Python: $PYO3_PYTHON"
info "Library: $LIBDIR"

# Step 5: Upgrade pip
info "Upgrading pip..."
"$PIP" install --upgrade pip --quiet

# Step 6: Install build dependencies
info "Installing build dependencies..."
"$PIP" install maturin --quiet

# Step 7: Install dev dependencies (pytest, polars, memory-profiler)
info "Installing development dependencies..."
"$PIP" install pytest polars memory-profiler --quiet

# Step 7b: Install S3 test dependencies if requested
if [ "$INSTALL_S3" = true ]; then
    info "Installing S3 test dependencies..."
    "$PIP" install ".[test-s3]" --quiet
    info "S3 test dependencies installed (moto, s3fs, boto3)"
fi

# Step 8: Build Rust extension with maturin (targeting venv Python)
info "Building Rust extension with maturin..."
info "Using Cargo.toml from: $RUST_DIR/Cargo.toml"
"${VENV_DIR}/bin/maturin" develop --release

# Step 9: Verify installation
info "Verifying installation..."
"$PYTHON" -c "from fastforward import secondspectrum; print('fastforward imported successfully')"

# Step 10: Run Python tests if requested
if [ "$RUN_TESTS" = true ]; then
    info "Running Python integration tests..."
    "$PYTHON" -m pytest tests/ -v
fi

# Success message
echo ""
info "Build complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v              # Python integration tests"
if [ "$INSTALL_S3" = true ]; then
    echo "  pytest tests/test_io.py::TestS3Adapter -v  # S3 integration tests"
fi
echo ""
if [ "$INSTALL_S3" = false ]; then
    echo "To install S3 test dependencies:"
    echo "  ./build.sh --with-s3"
    echo ""
fi
