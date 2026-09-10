# Orukeet Metal kernel numerical audit

The synthetic kernel audit compares the preserved original, first-patch and
optimized Metal runtimes using identical quantized input bytes. It covers 411
graph fixtures in each of four profiles: uniform values, cancellation,
eightfold-scaled values, and F16 rounding boundaries. Metal shader validation
is enabled. The [full receipt](orukeet-metal-numerics.json) contains the
unmodified results, runtime identities, input checks and source hashes.

Across 1,644 cases and 214,041,624 elements per reference comparison, every
group except Q8 matrix-vector multiplication is numerically identical between
the optimized and earlier Metal implementations. All 864 deliberate exact-zero
checks pass. No optimized/reference comparison contains nonfinite values.

The short Q8 matrix-vector change alters accumulation order. Of its 56 cases,
eight scaled-input cases fail the elementwise `atol=rtol=1e-5` criterion;
34 elements also fail `atol=rtol=1e-4`. Maximum absolute difference is
`0.00048828125`, so every audited Metal comparison passes `atol=rtol=1e-3`.
The uniform profile passes the stricter `1e-5` test in all 411 cases.

The independent FP64 accumulation reference dequantizes the same Q8 weights
and retains the original F32 inputs. On 28 batch-one Q8 cases, both original
and optimized Metal have maximum absolute error `0.000244140625`; both fail
four scaled cases at `1e-5` and pass every case at `1e-4`. The GGML CPU Q8
path additionally quantizes activations, so its differences from both Metal
versions are reported separately. Existing short-dot/CPU fallback differences
are also preserved in the receipt.

These synthetic results are separate from the
[full-model audit](orukeet-model-numerics.md), which passes `atol=rtol=1e-5`
for every final score on the 24 benchmark clips, each repeated twice.

```sh
python3 scripts/verify-orukeet-metal-numerics.py \
  --original-runtime ../openwhispr/.cache/orukeet-optimization/baseline-runtime \
  --previous-runtime ../openwhispr/.cache/orukeet-2x/previous-runtime \
  --output-dir .cache/orukeet-metal-numerics/full \
  --seed 20260910 \
  --profile uniform --profile cancellation \
  --profile saturation --profile half-boundaries
```

This command exits 1 for the documented strict-tolerance failures; it is not
a blanket passing test. Its source hashes identify the version executed
before the repository formatter was applied. The checked-in audit received
formatting and include-order changes afterward; the fixture calculations
and comparison thresholds were not changed. The full-model receipt separately
matches its checked-in audit source hashes.
