#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-examples}"

JOBS="${JOBS:-2}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"
EXAMPLE_TARGET="${EXAMPLE_TARGET:-00_basic_gemm}"
USE_BEAR="${USE_BEAR:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-t target] [-a cuda_archs] [-j jobs]

Examples:
  $(basename "$0") -t 00_basic_gemm -a 86 -j 4
  $(basename "$0") -t cute_tutorial_sgemm_2 -a 89 -j 4

Environment:
  BUILD_DIR       Build directory, default: ${ROOT_DIR}/build-examples
  EXAMPLE_TARGET  CMake example target, default: 00_basic_gemm
  CUDA_ARCHS      CUTLASS_NVCC_ARCHS value, default: 86
  JOBS            Ninja parallel jobs, default: 2
  USE_BEAR        Use bear for compile_commands.json, default: 1
EOF
}

while getopts ":t:a:j:h" opt; do
  case "${opt}" in
    t)
      EXAMPLE_TARGET="${OPTARG}"
      ;;
    a)
      CUDA_ARCHS="${OPTARG}"
      ;;
    j)
      JOBS="${OPTARG}"
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "option -${OPTARG} requires an argument" >&2
      exit 1
      ;;
    \?)
      echo "unknown option: -${OPTARG}" >&2
      exit 1
      ;;
  esac
done
echo "jobs=${JOBS} arch=${CUDA_ARCHS} target=${EXAMPLE_TARGET}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_NVCC_ARCHS="${CUDA_ARCHS}" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CUDA_HOST_COMPILER=clang++ \
  -G Ninja
  # -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  # -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}" \

if [[ "${USE_BEAR}" != "0" ]] && command -v bear >/dev/null 2>&1; then
  # using bear generates compile_commands.json
  bear -- ninja -C "${BUILD_DIR}" "${EXAMPLE_TARGET}" -j"${JOBS}" -v
else
  ninja -C "${BUILD_DIR}" "${EXAMPLE_TARGET}" -j"${JOBS}" -v
fi

EXAMPLE_BINARY="$(find "${BUILD_DIR}/examples" -type f -name "${EXAMPLE_TARGET}" -perm /111 -print -quit)"
if [[ -z "${EXAMPLE_BINARY}" ]]; then
  echo "failed to locate built executable for target ${EXAMPLE_TARGET}" >&2
  exit 1
fi

cp "${EXAMPLE_BINARY}" "${ROOT_DIR}"
echo "copied ${EXAMPLE_BINARY} -> ${ROOT_DIR}/$(basename "${EXAMPLE_BINARY}")"
