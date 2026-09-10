# Orukeet Q8: full-model numerical correctness on M4 Pro

**Pass: every captured model output meets `atol=rtol=1e-5`.** This also
passes `1e-4` and `1e-3`. The original and optimized Metal runtimes independently
recognize the same audio; the optimized model receives no intermediate values
from the reference run.

Measured on 2026-09-10, Apple M4 Pro, macOS 26.5, device `MTL0`. The reference
is the preserved runtime from `a5b6953`; the optimized runtime contains the
changes committed in `a6217cc`. Both load the same Orukeet Q8 GGUF, SHA-256
`93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`.
Loaded library paths and hashes are recorded in the
[machine-readable receipt](orukeet-model-numerics.json).

All 24 LibriSpeech benchmark clips, spanning 5–30 seconds and 364.285 seconds
of unique audio, run twice in each runtime: 48 recognitions per runtime,
96 total. The second pass exercises warmed graph and positional caches.
The input is the same cached mono 16 kHz F32 PCM used by the benchmark.

| Tensor | Elements compared, including repeats | Maximum absolute difference | Failures at `atol=rtol=1e-5` |
| --- | ---: | ---: | ---: |
| Frontend features | 9,330,944 | 0 | 0 |
| Final encoder output | 9,351,168 | 0 | 0 |
| Encoder joint projection | 5,844,480 | 0 | 0 |
| Final token logits | 32,903,088 | 0.000030517578125 | 0 |
| Final duration logits | 20,080 | 0.0000152587890625 | 0 |

The test applies the elementwise criterion, with the original runtime as
reference:

```text
abs(optimized - reference) <= atol + rtol * abs(reference)
```

Orukeet's TDT head produces 8,198 scores per decoder step: 8,193 token/blank
scores and five duration scores. **All 32,923,168 final scores across 4,016
decoder steps were compared**, including scores for tokens that were not
selected. The maximum absolute difference is below `1e-4`. At the stricter
combined `1e-5` tolerance, the worst error uses only 14.22% of its allowed
bound. No captured values are NaN or infinite.

All 8,032 token/duration decisions and all 48 transcript comparisons match
exactly, so corresponding decoder steps stay aligned. The audit additionally
checks that each captured score vector reproduces the GPU's actual argmax
decisions. Both runtimes reproduce every captured tensor bit-for-bit between
their first and second passes.

The test-only macOS interposer retains the final encoder tensor and joint
logits, including their backing views, until synchronized readback. It calls
the preserved runtime libraries for all model arithmetic. Kernel fusion,
graph optimization and the optimized positional cache remain enabled.
A separate one-clip diagnostic confirms **27 fused convolution dispatches**
and **116 fused LSTM dispatches** in the optimized runtime. Its captured
tensor bytes exactly match the corresponding main-run capture. Instrumented
latency is not used as performance evidence.

This establishes agreement with the original **Q8 Metal implementation** on
the selected workload. It does not establish equivalence to an unquantized
source model or a bound for arbitrary audio. The separate kernel audit's
synthetic scaled Q8 cases still fail `1e-4` and pass `1e-3`; that result covers
different inputs and does not contradict these full-model measurements.
The [kernel audit and its full receipt](orukeet-metal-numerics.md) preserve
those failures explicitly.

Run from the NeMo-Speech.cpp checkout, with the existing benchmark assets:

```sh
python3 scripts/verify-orukeet-model-numerics.py \
  --original-runtime ../openwhispr/.cache/orukeet-optimization/baseline-runtime \
  --output-dir .cache/orukeet-model-numerics/full \
  --repeats 2
```

Use `--limit 1 --repeats 1 --kernel-debug` with a separate output directory
to reproduce the fusion diagnostic. The default candidate is OpenWhispr's
staged Metal runtime; `--runtime` selects another runtime. Model, manifest
and PCM paths can also be overridden. The runner compiles the C++ capture
driver with `clang++`, verifies the loaded libraries, compares every tensor,
and writes binary snapshots, per-clip reports and an aggregate `report.json`.
Exit codes are 0 for a valid pass at `1e-4`, 1 for numerical or decision
failure, and 2 for an invalid or incomplete audit.
