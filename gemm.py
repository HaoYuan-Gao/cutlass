import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group

    first_pid_m = group_id * GROUP_M

    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M)
            & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )

        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_K)
            & (offs_n[None, :] < N),
            other=0.0,
        )

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"])
        * triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


def benchmark(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeat):
        fn()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat


if __name__ == "__main__":
    torch.manual_seed(0)

    M = 4096
    N = 4096
    K = 4096

    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)

    c_triton = triton_matmul(a, b)
    c_torch = torch.matmul(a, b)

    print(torch.allclose(c_triton, c_torch, atol=1e-1, rtol=1e-1))

    triton_ms = benchmark(lambda: triton_matmul(a, b))
    torch_ms = benchmark(lambda: torch.matmul(a, b))

    flops = 2 * M * N * K

    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    print(f"Triton : {triton_ms:.3f} ms  {triton_tflops:.2f} TFLOPS")
    print(f"Torch  : {torch_ms:.3f} ms  {torch_tflops:.2f} TFLOPS")