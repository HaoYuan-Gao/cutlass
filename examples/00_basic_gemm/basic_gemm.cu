/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

enum class MatrixLayout {
  kColumnMajor,
  kRowMajor
};

char const *to_string(MatrixLayout layout) {
  switch (layout) {
    case MatrixLayout::kColumnMajor:
      return "ColumnMajor";
    case MatrixLayout::kRowMajor:
      return "RowMajor";
    default:
      return "Unknown";
  }
}

__host__ __device__ int matrix_offset(
  int row,
  int column,
  int rows,
  int columns,
  MatrixLayout layout) {
  return (layout == MatrixLayout::kColumnMajor) ? row + column * rows
                                                : row * columns + column;
}

cudaError_t InitializeMatrix(
  float *matrix,
  int rows,
  int columns,
  MatrixLayout layout,
  int seed = 0);

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

cudaError_t CutlassSgemmNN_RowMajor(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,     // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,     // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,     // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  MatrixLayout layout,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = matrix_offset(i, j, rows, columns, layout);

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {
  return InitializeMatrix(matrix, rows, columns, MatrixLayout::kColumnMajor, seed);
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(
  float *matrix,
  int rows,
  int columns,
  MatrixLayout layout,
  int seed) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, layout, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(
  float **matrix,
  int rows,
  int columns,
  MatrixLayout layout = MatrixLayout::kColumnMajor,
  int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, layout, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  MatrixLayout layout_a,
  float const *B,
  int ldb,
  MatrixLayout layout_b,
  float beta,
  float *C,
  int ldc,
  MatrixLayout layout_c) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      int offset_a = (layout_a == MatrixLayout::kColumnMajor) ? i + k * lda : i * lda + k;
      int offset_b = (layout_b == MatrixLayout::kColumnMajor) ? k + j * ldb : k * ldb + j;
      accumulator += A[offset_a] * B[offset_b];
    }

    int offset_c = (layout_c == MatrixLayout::kColumnMajor) ? i + j * ldc : i * ldc + j;
    C[offset_c] = alpha * accumulator + beta * C[offset_c];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  MatrixLayout layout_a,
  float const *B,
  int ldb,
  MatrixLayout layout_b,
  float beta,
  float *C,
  int ldc,
  MatrixLayout layout_c) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(
    M, N, K, alpha, A, lda, layout_a, B, ldb, layout_b, beta, C, ldc, layout_c);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct PerformanceResult {
  float average_runtime_ms = 0;
  float gflops = 0;
};

cudaError_t BenchmarkGemm(
  char const *name,
  cudaError_t (*gemm_fn)(int, int, int, float, float const *, int, float const *, int, float, float *, int),
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  int warmups,
  int iterations,
  PerformanceResult &performance) {

  cudaError_t result = cudaSuccess;

  for (int iter = 0; iter < warmups; ++iter) {
    result = gemm_fn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    if (result != cudaSuccess) {
      return result;
    }
  }

  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    return result;
  }

  cudaEvent_t start;
  cudaEvent_t stop;

  result = cudaEventCreate(&start);
  if (result != cudaSuccess) {
    return result;
  }

  result = cudaEventCreate(&stop);
  if (result != cudaSuccess) {
    cudaEventDestroy(start);
    return result;
  }

  result = cudaEventRecord(start);
  if (result == cudaSuccess) {
    for (int iter = 0; iter < iterations; ++iter) {
      result = gemm_fn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      if (result != cudaSuccess) {
        break;
      }
    }
  }

  if (result == cudaSuccess) {
    result = cudaEventRecord(stop);
  }
  if (result == cudaSuccess) {
    result = cudaEventSynchronize(stop);
  }

  if (result == cudaSuccess) {
    float elapsed_ms = 0;
    result = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (result == cudaSuccess) {
      performance.average_runtime_ms = elapsed_ms / static_cast<float>(iterations);
      double flops = 2.0 * double(M) * double(N) * double(K);
      performance.gflops =
        static_cast<float>(flops / (performance.average_runtime_ms * 1.0e6));
      std::cout << name << " average runtime: " << std::fixed << std::setprecision(3)
                << performance.average_runtime_ms << " ms, throughput: "
                << std::setprecision(2) << performance.gflops << " GFLOP/s" << std::endl;
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestCutlassGemm(
  int M,
  int N,
  int K,
  float alpha,
  float beta,
  MatrixLayout layout,
  int warmups,
  int iterations) {
  cudaError_t result;

  int lda = (layout == MatrixLayout::kColumnMajor) ? M : K;
  int ldb = (layout == MatrixLayout::kColumnMajor) ? K : N;
  int ldc = (layout == MatrixLayout::kColumnMajor) ? M : N;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * M * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, layout, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, layout, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, layout, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, layout, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  auto gemm_fn = (layout == MatrixLayout::kColumnMajor) ? &CutlassSgemmNN
                                                        : &CutlassSgemmNN_RowMajor;

  PerformanceResult performance;
  result = BenchmarkGemm(to_string(layout),
                         gemm_fn,
                         M,
                         N,
                         K,
                         alpha,
                         A,
                         lda,
                         B,
                         ldb,
                         beta,
                         C_cutlass,
                         ldc,
                         warmups,
                         iterations,
                         performance);

  if (result != cudaSuccess) {
    std::cerr << to_string(layout) << " CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //

  // Launch reference GEMM
  result = ReferenceGemm(M, N, K, alpha, A, lda, layout, B, ldb, layout, beta, C_reference, ldc, layout);

  if (result != cudaSuccess) {
    std::cerr << to_string(layout) << " reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(M * N, 0);
  std::vector<float> host_reference(M * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy " << to_string(layout) << " CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy " << to_string(layout) << " reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << to_string(layout) << " CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  std::cout << to_string(layout) << " verification passed." << std::endl;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta> <warmups> <iterations>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  int warmups = 10;
  int iterations = 50;

  if (argc > 6) {
    std::stringstream ss(arg[6]);
    ss >> warmups;
  }

  if (argc > 7) {
    std::stringstream ss(arg[7]);
    ss >> iterations;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  std::cout << "Problem size: M=" << problem[0] << ", N=" << problem[1]
            << ", K=" << problem[2] << ", alpha=" << scalars[0]
            << ", beta=" << scalars[1] << ", warmups=" << warmups
            << ", iterations=" << iterations << std::endl;

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],     // beta
    MatrixLayout::kColumnMajor,
    warmups,
    iterations
  );

  if (result == cudaSuccess) {
    result = TestCutlassGemm(
      problem[0],
      problem[1],
      problem[2],
      scalars[0],
      scalars[1],
      MatrixLayout::kRowMajor,
      warmups,
      iterations
    );
  }

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
