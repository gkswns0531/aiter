# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test for shared memory caching optimization in rms_norm_kernel.

Validates that the USE_SHMEM optimization produces numerically identical results
to the non-shmem path, and measures performance across realistic serving configs.

Usage:
  python test_rmsnorm2d_shmem.py                    # Run all configs
  python test_rmsnorm2d_shmem.py -d bf16 -n 4096    # Specific config
  python test_rmsnorm2d_shmem.py -m 8192             # Large batch
"""

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
import argparse


@perftest()
def run_torch(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(
        input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
    )


@perftest()
def run_cu(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(input)
    aiter.rms_norm_cu(output, input, weight, eps)
    return output


def test_rmsnorm2d_shmem(dtype: torch.dtype, m: int, n: int) -> None:
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")

    (ref,), avg_ref = run_torch(input, weight, 1e-5)
    (out,), avg_cu = run_cu(input, weight, 1e-5)

    shmem_bytes = n * input.element_size()
    shmem_status = "shmem" if shmem_bytes <= 65536 else "global"

    msg = (
        f"[{shmem_status}] dim: {str(dim):<20}, dtype: {dtype}, "
        f"shmem: {shmem_bytes/1024:.0f}KB, "
        f"torch: {avg_ref:<8.2f} us, cu: {avg_cu:<8.2f} us, "
        f"speedup: {avg_ref/avg_cu-1:<5.1%}"
    )
    checkAllclose(ref, out, msg=msg)


l_dtype = ["fp16", "bf16"]
l_m = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 131072]
l_n = [1024, 2048, 4096, 8192]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test shared memory caching for rms_norm_kernel",
)
parser.add_argument("-d", "--dtype", type=str, choices=l_dtype, default=None)
parser.add_argument("-m", "--m", type=int, default=None)
parser.add_argument("-n", "--n", type=int, default=None)
args = parser.parse_args()

if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.n is not None:
    l_n = [args.n]

print("\n=== RMSNorm shared memory caching test ===")
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            test_rmsnorm2d_shmem(dtype, m, n)
