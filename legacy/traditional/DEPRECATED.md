# Deprecated

Use `ampbench.models.traditional`, `ampbench.metrics.errors`, and `scripts/train_traditional.py`.

Notable cleanups vs this folder:

- No `sys.path` hacks
- No forced `plt.show()` during training/grid search
- MPM design-matrix column index fixed to `m * (N + 1) + k`
- `Hammerstein or 'Wiener'` bug removed
