#!/usr/bin/env python3
"""
Step1: TabICL MPS最小テスト
- mps/cpu で最小動作確認
"""
import numpy as np
import torch

print("=" * 70)
print("Step1: TabICL MPS Minimal Test")
print("=" * 70)
print(f"torch={torch.__version__}, mps_available={torch.backends.mps.is_available()}")

# Import & test
print("\n[Phase 1] Testing TabICL with minimal data...")
from tabicl import TabICLRegressor

X = np.random.randn(200, 5).astype(np.float32)
y = np.random.randn(200).astype(np.float32)
Xt = np.random.randn(50, 5).astype(np.float32)

results = {}
for dev in ["mps", "cpu"]:
    print(f"\n  Testing device='{dev}'...")
    try:
        # Try device parameter
        reg = TabICLRegressor(n_estimators=1, device=dev, random_state=42)
        reg.fit(X, y)
        pred = reg.predict(Xt)
        results[dev] = ("OK", pred.shape)
        print(f"    ✓ OK: pred shape {pred.shape}, pred[0]={pred[0]:.4f}")
    except TypeError as e:
        # Device parameter not supported; try torch default_device
        if "device" in str(e):
            print(f"    Device param not supported, trying torch.set_default_device...")
            try:
                torch.set_default_device(dev)
                reg = TabICLRegressor(n_estimators=1, random_state=42)
                reg.fit(X, y)
                pred = reg.predict(Xt)
                results[dev] = ("OK_torch_default", pred.shape)
                print(f"    ✓ OK (via torch default): pred shape {pred.shape}")
                torch.set_default_device("cpu")  # Reset
            except Exception as e2:
                results[dev] = ("FAIL", str(e2)[:150])
                print(f"    ✗ FAIL: {type(e2).__name__}: {str(e2)[:150]}")
        else:
            results[dev] = ("FAIL", str(e)[:150])
            print(f"    ✗ FAIL: {type(e).__name__}: {str(e)[:150]}")
    except Exception as e:
        results[dev] = ("FAIL", str(e)[:150])
        print(f"    ✗ FAIL: {type(e).__name__}: {str(e)[:150]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for dev, (status, detail) in results.items():
    if isinstance(detail, tuple):
        print(f"  {dev:5s}: {status:15s} shape={detail}")
    else:
        print(f"  {dev:5s}: {status:15s} {detail}")

mps_ok = results.get("mps", ("FAIL", ""))[0].startswith("OK")
cpu_ok = results.get("cpu", ("FAIL", ""))[0].startswith("OK")

if mps_ok:
    print("\n✓ MPS is available! Can proceed to Step2: v10 full pipeline on MPS")
elif cpu_ok:
    print("\n⚠ MPS FAILED but CPU OK. Will fallback to CPU in Step2.")
else:
    print("\n✗ Both MPS and CPU failed. Cannot proceed.")
