"""
Minimal reproducer: torch.compile produces NaN for bf16 subnormal values on gfx1151

Hardware: AMD Radeon 8060S (gfx1151, Strix Halo)
Software: PyTorch 2.11.0a0+rocm7.11.0a20260106, ROCm 7.11.0 nightlies

The compiled kernel for (x / scalar).sqrt() + eps produces NaN when x
contains bf16 subnormal values (~1e-38, just below bf16 min normal 1.175e-38).
The eager (uncompiled) version handles the same values correctly.

This is the root cause of systematic NaN in compiled Adam optimizer steps
on gfx1151, where cold embedding rows' exp_avg_sq values decay to bf16
subnormals after hundreds of training steps.

Usage: python minimal_reproducer_subnormal_nan.py
"""

import torch

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda")

# bf16 min normal = 2^-126 ≈ 1.175e-38
# Values below this are subnormals (denormals)
subnormal_values = [1.14e-38, 1.0e-38, 5.0e-39, 1.0e-39]
normal_values = [1.2e-38, 1.0e-37, 1.0e-30, 1.0e-5, 1.0]

@torch.compile(dynamic=False, fullgraph=True)
def compute_denom_compiled(x, bias2, eps):
    """Adam-style denom: (x / bias2).sqrt() + eps"""
    return (x / bias2).sqrt() + eps

def compute_denom_eager(x, bias2, eps):
    """Same computation, no compile"""
    return (x / bias2).sqrt() + eps

print("\n=== Test 1: Individual bf16 subnormal values ===")
eps = torch.tensor(1e-10, dtype=torch.float32, device=device)
bias2 = torch.tensor(1.0, dtype=torch.float32, device=device)

for val in subnormal_values + normal_values:
    x = torch.full((1024,), val, dtype=torch.bfloat16, device=device)
    result_compiled = compute_denom_compiled(x, bias2, eps)
    result_eager = compute_denom_eager(x, bias2, eps)
    nan_c = torch.isnan(result_compiled).sum().item()
    nan_e = torch.isnan(result_eager).sum().item()
    label = "SUBNORMAL" if val < 1.175e-38 else "normal"
    status = "BUG!" if nan_c > 0 and nan_e == 0 else "ok"
    print(f"  val={val:.2e} ({label:9s}): compiled_nan={nan_c}, eager_nan={nan_e}  [{status}]")
    if nan_c > 0 and nan_e == 0:
        print(f"    compiled[0]={result_compiled[0].item()}, eager[0]={result_eager[0].item():.8e}")

print("\n=== Test 2: Mixed tensor (realistic scenario) ===")
# Simulate value_embeds exp_avg_sq: most values normal, some cold rows subnormal
x = torch.rand(8192, 320, dtype=torch.bfloat16, device=device) * 1e-5
# Make row 529 col 269 subnormal (the actual index from our training runs)
x[529, 269] = torch.tensor(1.14e-38, dtype=torch.bfloat16)

result_compiled = compute_denom_compiled(x, bias2, eps)
result_eager = compute_denom_eager(x, bias2, eps)
nan_c = torch.isnan(result_compiled).sum().item()
nan_e = torch.isnan(result_eager).sum().item()
print(f"  Mixed tensor [8192, 320]: compiled_nan={nan_c}, eager_nan={nan_e}")
if nan_c > 0:
    nan_mask = torch.isnan(result_compiled)
    nan_idx = torch.nonzero(nan_mask, as_tuple=False)[:5]
    for idx in nan_idx:
        r, c = idx[0].item(), idx[1].item()
        print(f"    NaN at [{r}, {c}]: x={x[r, c].item():.8e}, eager={result_eager[r, c].item():.8e}")

print("\n=== Test 3: Just sqrt on bf16 subnormals ===")

@torch.compile(dynamic=False, fullgraph=True)
def just_sqrt_compiled(x):
    return x.sqrt()

def just_sqrt_eager(x):
    return x.sqrt()

for val in subnormal_values + normal_values[:2]:
    x = torch.full((1024,), val, dtype=torch.bfloat16, device=device)
    result_compiled = just_sqrt_compiled(x)
    result_eager = just_sqrt_eager(x)
    nan_c = torch.isnan(result_compiled).sum().item()
    nan_e = torch.isnan(result_eager).sum().item()
    label = "SUBNORMAL" if val < 1.175e-38 else "normal"
    status = "BUG!" if nan_c > 0 and nan_e == 0 else "ok"
    print(f"  sqrt({val:.2e}) ({label:9s}): compiled_nan={nan_c}, eager_nan={nan_e}  [{status}]")

print("\n=== Test 4: Division of bf16 subnormals by fp32 scalar ===")

@torch.compile(dynamic=False, fullgraph=True)
def just_div_compiled(x, scalar):
    return x / scalar

def just_div_eager(x, scalar):
    return x / scalar

for val in subnormal_values + normal_values[:2]:
    x = torch.full((1024,), val, dtype=torch.bfloat16, device=device)
    result_compiled = just_div_compiled(x, bias2)
    result_eager = just_div_eager(x, bias2)
    nan_c = torch.isnan(result_compiled).sum().item()
    nan_e = torch.isnan(result_eager).sum().item()
    label = "SUBNORMAL" if val < 1.175e-38 else "normal"
    status = "BUG!" if nan_c > 0 and nan_e == 0 else "ok"
    print(f"  div({val:.2e}, 1.0) ({label:9s}): compiled_nan={nan_c}, eager_nan={nan_e}  [{status}]")

print("\n=== Test 5: fp32 tensors with same subnormal values (control) ===")
for val in subnormal_values:
    x = torch.full((1024,), val, dtype=torch.float32, device=device)
    result_compiled = compute_denom_compiled(x, bias2, eps)
    result_eager = compute_denom_eager(x, bias2, eps)
    nan_c = torch.isnan(result_compiled).sum().item()
    nan_e = torch.isnan(result_eager).sum().item()
    status = "BUG!" if nan_c > 0 and nan_e == 0 else "ok"
    print(f"  fp32 val={val:.2e}: compiled_nan={nan_c}, eager_nan={nan_e}  [{status}]")

print("\nDone.")
