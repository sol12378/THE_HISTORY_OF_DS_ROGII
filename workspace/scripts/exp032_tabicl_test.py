#!/usr/bin/env python3
"""TabICL動作確認 — 100x10 ダミーデータで fit+predict 成功するか."""
import time, numpy as np

print("Testing TabICL import + first-call (will download model)...")
t0 = time.time()
from tabicl import TabICLRegressor
print(f"  import OK: {time.time()-t0:.1f}s")

rng = np.random.default_rng(42)
X_tr = rng.standard_normal((200, 8))
y_tr = X_tr[:, 0] + 0.5 * X_tr[:, 1] + 0.1 * rng.standard_normal(200)
X_te = rng.standard_normal((50, 8))
y_te = X_te[:, 0] + 0.5 * X_te[:, 1] + 0.1 * rng.standard_normal(50)

t1 = time.time()
model = TabICLRegressor(n_estimators=1, batch_size=1, verbose=True, n_jobs=1)
print(f"  init: {time.time()-t1:.1f}s")
t2 = time.time()
model.fit(X_tr, y_tr)
print(f"  fit: {time.time()-t2:.1f}s")
t3 = time.time()
p = model.predict(X_te, output_type="mean")
print(f"  predict: {time.time()-t3:.1f}s, RMSE={np.sqrt(((p-y_te)**2).mean()):.4f}, baseline std={y_te.std():.4f}")
print(f"\nTotal: {time.time()-t0:.1f}s, OK")
