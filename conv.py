import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv1x1_kernel(
    X, W, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    C: tl.constexpr, H: tl.constexpr, W_IN: tl.constexpr,
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

    # M = batch * H * W
    n_idx = offs_m // (H * W_IN)
    hw = offs_m % (H * W_IN)
    h_idx = hw // W_IN
    w_idx = hw % W_IN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        c_idx = k0 + offs_k

        x_ptrs = (
            X
            + n_idx[:, None] * C * H * W_IN
            + c_idx[None, :] * H * W_IN
            + h_idx[:, None] * W_IN
            + w_idx[:, None]
        )

        w_ptrs = (
            W
            + offs_n[None, :] * C
            + c_idx[:, None]
        )

        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (c_idx[None, :] < K),
            other=0.0,
        )

        weight = tl.load(
            w_ptrs,
            mask=(offs_n[None, :] < N) & (c_idx[:, None] < K),
            other=0.0,
        )

        acc += tl.dot(x, weight)

    y_ptrs = (
        Y
        + n_idx[:, None] * N * H * W_IN
        + offs_n[None, :] * H * W_IN
        + h_idx[:, None] * W_IN
        + w_idx[:, None]
    )

    tl.store(
        y_ptrs,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_conv1x1(x, weight):
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16

    B, C, H, W_IN = x.shape
    K_OUT, Cw, R, S = weight.shape

    assert C == Cw
    assert R == 1 and S == 1

    y = torch.empty((B, K_OUT, H, W_IN), device=x.device, dtype=torch.bfloat16)

    M = B * H * W_IN
    N = K_OUT
    K = C

    weight_2d = weight.reshape(K_OUT, C)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    conv1x1_kernel[grid](
        x,
        weight_2d,
        y,
        M,
        N,
        K,
        C,
        H,
        W_IN,
    )

    return y


def benchmark(fn, warmup=30, repeat=100):
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

    B = 32
    C = 256
    H = 56
    W_IN = 56
    K_OUT = 512

    x = torch.randn((B, C, H, W_IN), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((K_OUT, C, 1, 1), device="cuda", dtype=torch.bfloat16)

    y_triton = triton_conv1x1(x, weight)
    y_torch = F.conv2d(x, weight)

    print(torch.allclose(y_triton, y_torch, atol=1e-1, rtol=1e-1))

    triton_ms = benchmark(lambda: triton_conv1x1(x, weight))
    torch_ms = benchmark(lambda: F.conv2d(x, weight))

    flops = 2 * B * H * W_IN * K_OUT * C

    print(f"Triton Conv1x1: {triton_ms:.3f} ms, {flops / (triton_ms * 1e-3) / 1e12:.2f} TFLOPS")
    print(f"Torch  Conv1x1: {torch_ms:.3f} ms, {flops / (torch_ms * 1e-3) / 1e12:.2f} TFLOPS")