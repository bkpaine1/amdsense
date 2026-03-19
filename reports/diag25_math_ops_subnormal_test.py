"""
DIAG 25: Test compiled math operations on subnormal values (gfx1151)

Tests: sqrt, rsqrt, log, exp, reciprocal, abs, neg on subnormal bf16/fp32 inputs.
Compares compiled vs eager results to characterize the scope of the sqrt bug.
"""

import torch

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda")

# bf16 min normal = 2^-126 ≈ 1.175e-38
subnormal_vals = [1.14e-38, 1.0e-38, 5.0e-39, 1.0e-39]
normal_vals = [1.2e-38, 1.0e-30, 1.0e-5, 1.0]

ops = {
    'sqrt': (lambda x: x.sqrt(), "sqrt(x)"),
    'rsqrt': (lambda x: x.rsqrt(), "1/sqrt(x)"),
    'reciprocal': (lambda x: x.reciprocal(), "1/x"),
    'log': (lambda x: x.log(), "log(x)"),
    'exp': (lambda x: x.exp(), "exp(x)"),
    'abs': (lambda x: x.abs(), "abs(x)"),
    'neg': (lambda x: -x, "-x"),
    'square': (lambda x: x * x, "x*x"),
    'add_eps': (lambda x: x + 1e-10, "x + 1e-10"),
    'div_by_1': (lambda x: x / 1.0, "x / 1.0"),
    'mul_by_2': (lambda x: x * 2.0, "x * 2.0"),
}

# Create compiled versions
compiled_ops = {}
for name, (fn, _) in ops.items():
    compiled_ops[name] = torch.compile(fn, dynamic=False, fullgraph=True)

print("\n=== bf16 subnormal tests ===")
print(f"{'Op':<15} {'Value':<12} {'Compiled NaN':<14} {'Eager NaN':<12} {'Status'}")
print("-" * 65)

for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
    print(f"\n--- {dtype_name} ---")
    for val in subnormal_vals + normal_vals:
        is_subnormal = val < 1.175e-38
        for name, (fn, desc) in ops.items():
            x = torch.full((1024,), val, dtype=dtype, device=device)
            try:
                result_c = compiled_ops[name](x)
                nan_c = torch.isnan(result_c).sum().item()
                inf_c = torch.isinf(result_c).sum().item()
            except Exception as e:
                nan_c = f"ERR:{e}"
                inf_c = 0
            try:
                result_e = fn(x)
                nan_e = torch.isnan(result_e).sum().item()
                inf_e = torch.isinf(result_e).sum().item()
            except Exception as e:
                nan_e = f"ERR:{e}"
                inf_e = 0

            # Only print if there's a discrepancy or it's a subnormal test
            if is_subnormal:
                if isinstance(nan_c, int) and isinstance(nan_e, int):
                    status = "BUG!" if nan_c > 0 and nan_e == 0 else ("ok" if nan_c == nan_e else "DIFF")
                else:
                    status = "ERR"
                print(f"{name:<15} {val:<12.2e} {str(nan_c):<14} {str(nan_e):<12} {status}")

print("\n=== Summary: Which compiled ops produce NaN for subnormals? ===")
print(f"{'Op':<15} {'bf16 bug?':<12} {'fp32 bug?':<12}")
print("-" * 40)

for name, (fn, desc) in ops.items():
    results = {}
    for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        x = torch.full((1024,), 1.0e-38, dtype=dtype, device=device)
        try:
            nan_c = torch.isnan(compiled_ops[name](x)).sum().item()
            nan_e = torch.isnan(fn(x)).sum().item()
            results[dtype_name] = "BUG" if nan_c > 0 and nan_e == 0 else "ok"
        except Exception:
            results[dtype_name] = "err"
    print(f"{name:<15} {results['bf16']:<12} {results['fp32']:<12}")

print("\nDone.")
