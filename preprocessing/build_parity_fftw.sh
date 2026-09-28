#!/bin/sh
set -eu
prefix=${1:?Usage: sh build_parity_fftw.sh ABSOLUTE_INSTALL_DIRECTORY}
case "$prefix" in /*) ;; *) echo 'Install directory must be absolute' >&2; exit 1;; esac
test ! -e "$prefix" || { echo 'Install directory already exists' >&2; exit 1; }
stage=$(mktemp -d)
archive="$stage/fftw.tar.gz"
curl -L --fail https://www.fftw.org/fftw-3.3.8.tar.gz -o "$archive"
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$archive")
else
    actual=$(shasum -a 256 "$archive")
fi
test "${actual%% *}" = 6113262f6e92c5bd474f2875fa1b01054c4ad5040f6b0da7c03c98821d9ae303
tar -xzf "$archive" -C "$stage"
cd "$stage/fftw-3.3.8"
# The Linux x86-64 MATLAB references use the SSE2 codelets. The scalar build
# follows a different floating-point operation order on these inputs.
# Keep other platforms unchanged and allow explicit selection for validation.
simd=${AET_FFTW_SIMD:-auto}
if [ "$simd" = auto ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) simd=sse2 ;;
        *) simd=none ;;
    esac
fi
case "$simd" in
    sse2) set -- --enable-sse2 ;;
    none) set -- ;;
    *) echo 'AET_FFTW_SIMD must be auto, sse2, or none' >&2; exit 1 ;;
esac
./configure --prefix="$prefix" --enable-float --enable-shared --disable-static \
    --disable-fortran "$@" CC="${CC:-clang}" CFLAGS='-O3 -ffp-contract=off'
make -j "${AET_BUILD_JOBS:-4}"
make install
echo "FFTW installed in $prefix; build and source retained at $stage"
