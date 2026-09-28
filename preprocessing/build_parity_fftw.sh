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
# Select SIMD instructions for the target architecture.
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
for precision in double single; do
    if [ "$precision" = single ]; then float_option=--enable-float; else float_option=; fi
    ./configure --prefix="$prefix" $float_option --enable-shared --disable-static \
        --disable-fortran "$@" CC="${CC:-cc}" CFLAGS='-O3 -ffp-contract=off'
    make -j "${AET_BUILD_JOBS:-4}"
    make install
    make distclean
done
echo "FFTW single and double libraries installed in $prefix"
