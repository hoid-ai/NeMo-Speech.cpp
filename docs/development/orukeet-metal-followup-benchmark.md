# Orukeet Q8: second Metal optimization pass on M4 Pro

The second pass reaches **1.74× the original native speed** and **1.65× the
original OpenWhispr backend speed**. The 2× median-latency target was **not
reached**. Native median is now **88.9 ms**, versus 154.2 ms originally and
111.4 ms after the first patch. Backend median is **100.3 ms**, versus
166.0 ms originally and 122.6 ms after the first patch.

This combines [`0022-metal-asr-fusions.patch`](../../ggml-patches/0022-metal-asr-fusions.patch)
with the accompanying relative-position attention changes. The model remains
Q8 on Metal. Word error rate remains **2.40%**, with every measured transcript
matching the original baseline exactly.

## Results

Measured on 2026-09-10 UTC: Apple M4 Pro, 12 CPU cores, 24 GiB RAM,
macOS 26.5, Metal device MTL0. All receipts report battery power, with
identical power settings (`powermode 0`).

| Metric | Original | First patch | Second pass | Reduction vs original |
| --- | ---: | ---: | ---: | ---: |
| Native median | 154.2 ms | 111.4 ms | **88.9 ms** | **42.4%** |
| Native p95 | 280.8 ms | 198.6 ms | **156.1 ms** | **44.4%** |
| Native p99 | 283.7 ms | 200.9 ms | 159.1 ms | 43.9% |
| Backend median | 166.0 ms | 122.6 ms | **100.3 ms** | **39.6%** |
| Backend p95 | 295.0 ms | 212.1 ms | **170.1 ms** | **42.3%** |
| Backend p99 | 303.2 ms | 216.3 ms | 172.6 ms | 43.1% |
| Native audio seconds per second | 94.2 | 130.5 | **162.6** | — |
| Backend audio seconds per second | 87.4 | 118.6 | **144.0** | — |

Speedup is `original latency / new latency`; latency reduction is
`100 × (1 − new / original)`. Both use unrounded measurements. Relative to
the first patch, this pass cuts native median by **20.2%** and backend median
by **18.2%**. A 2× improvement over the original requires native median at
or below **77.1 ms** and backend median at or below **83.0 ms**.

Fresh controls check machine drift. Before measuring the second pass, the
preserved first-patch runtime returned **110.7 ms native / 122.5 ms backend**.
Afterward, the preserved original runtime returned **154.6 ms / 166.3 ms**.
Both controls are within 1% of their earlier medians. Against the fresh
first-patch control, the reductions are 19.7% native and 18.1% backend.
The optimized runtime was restored after the control run.

Each run has a 120-second measurement window after startup and one warmup
request per clip. The second pass completed **1,139 measured requests**.
The same 24 public LibriSpeech test-clean clips span 5–30 seconds,
24 speakers, 364.285 seconds of unique audio and 916 reference words.
Requests run sequentially with batch size one. The benchmark source,
decoding configuration, model and clip manifest are unchanged.

Native timing covers the recognition call, including the audio frontend,
encoder and decoder. Backend timing covers that same call plus OpenWhispr's
audio conversion, temporary PCM file and worker IPC. Every request makes
one native recognition call. Microphone capture, UI and paste are outside
the benchmark's scope.

Across the 24 clips, native latency reductions range from **33.2–44.9%**,
with a median of **41.5%**. Backend reductions range from **29.3–42.9%**,
with a median of **39.3%**. WER is 22/916 words: 16 substitutions,
4 deletions and 2 insertions. Repetitions increase the timing sample;
accuracy is scored once per unique clip.
All **4,400 measured requests** across the five compared receipts have
exactly the same transcript as the original for their respective clip.

## Implementation and tradeoffs

- **Short convolution fusion:** combines F16 input lowering and its dot
  product. The shader retains the original F32-to-F16 rounding before the
  F32 accumulation. If the graph allocator reused the input for the result,
  the original lowering allocation holds the compact result until a safe
  copy can complete. Unsupported layouts, extra consumers and exposed
  intermediates retain the original operations.
- **LSTM gate fusion:** combines thirteen gate/state operations into one
  shader for a single recurrent stream. The matcher supports both the
  original dependency order and the order produced by Metal's graph
  optimizer. It checks consumers, outputs, layouts and memory aliases;
  unsupported batches retain the original operations.
- **Short Q8 matrix-vector products:** use one SIMD group when the reduction
  dimension is at most 1,024, avoiding a shared-memory reduction for the
  recurrent decoder's small projections. Longer reductions are unchanged.
- **Attention views:** remove four staging copies and express the relative
  shift as a strided view, retaining head and batch strides. Tests compare
  both versions exactly on CPU and Metal, including batch size two.
- **Constant positional projection:** precomputes the projection of fixed
  positional embeddings once during weight setup, using the same Q8 Metal
  matrix multiplication. It caches no audio, activations or transcripts.

The positional cache adds **95.9 MiB of persistent memory** for this model.
The total cache budget is capped at 128 MiB across encoder layers. Cached
projections are used for 5–512 encoder frames and the expected positional
embedding layout. Smaller or larger inputs, other layouts and ineligible
devices or weight types retain the original projection. The lower limit
preserves the existing Metal vector-kernel rounding on very short inputs.
For Orukeet's 8× subsampling, 512 frames cover about 41 seconds.

Set `NEMO_SPEECH_METAL_POS_CACHE_DISABLE=1` before starting the worker to
disable this cache. The timed OpenWhispr workload stays within its existing
30-second per-request cap; the native API was tested beyond the cache limit
separately.

The optimized receipt reports **55.7 ms** for native model creation and
**817.6 ms** for OpenWhispr startup, compared with 57.8 ms and 805.9 ms in
the first-patch receipt. Startup uses a fresh worker with existing filesystem
and shader caches, including the backend's built-in warmup. These numbers
do not characterize a clean first launch or isolate the cache's setup cost.

## Validation

**249 new CPU-reference tests passed with Metal shader validation:**

| Test group | Cases |
| --- | ---: |
| Short convolution graphs | 128 |
| LSTM gate graphs | 60 |
| Relative-shift views | 24 |
| Attention input views | 24 |
| Cached positional projections | 6 |
| Short Q8 matrix-vector products | 7 |

The tests cover allocator reuse, shifted aliases, additional consumers,
exposed intermediates, odd sizes, padding, stride, dilation, optimizer
ordering, unsupported-batch fallback and both sides of dispatch boundaries.
Convolution and LSTM comparisons use a `1e-7` NMSE limit. Attention views,
relative shifts and cached projections match their original implementations
exactly in the tested shapes. Q8 matrix-vector comparisons use the existing
matrix test tolerance.

An additional **54 direct C API checks** compare the first-patch runtime,
the new runtime and the new runtime with the positional cache disabled.
Each repeats nine audio lengths twice: 0.04, 0.08, 0.24, 0.32, 1.0,
40.8, 40.88, 41.0 and 50.0 seconds. These produce 1, 2, 4, 5, 13,
511, 512, 513 and 626 encoder frames. Every transcript matches across all
variants and repeats. This is correctness evidence, not a timing or WER
measurement for long audio.

The OpenWhispr JFK regression passes all **seven checks**: verified model
installation, worker reuse, serialized concurrent calls, silence, early
cancellation, in-flight cancellation/recovery and explicit shutdown.
The normal OpenWhispr Metal build, runtime verification and all pre-commit
checks pass. Applying all 22 patches to the pinned ggml revision reproduces
the live ggml source exactly; the submodule pin is unchanged.

## Reproduce

Use the sibling OpenWhispr checkout on `hoid-orukeet`, with benchmark commit
`2437b3db8fbe1ba5917b8bd80dbc1d68c8efd79c` or a compatible revision.
Its `docs/ORUKEET_BENCHMARK.md` describes dependency and public-asset setup.
From OpenWhispr:

```sh
OPENWHISPR_ORUKEET_BUILD_DEVICE=metal npm run build:orukeet
npm run benchmark:orukeet -- --seconds 120
```

For separate baseline builds, use `OPENWHISPR_NEMO_SPEECH_SOURCE` to select
a checkout of NeMo commit `a5b6953c4a579a2bbd1c0913ad8a85c2a4d99953`
for the original, or `90f5e58d2a32aea85399fdecf2e7406651abad59` for the
first patch. Keep model, clip manifest, benchmark source and power mode
identical. The [first report](orukeet-metal-benchmark.md) gives the ggml
test-build configuration. After building `test-backend-ops`, run:

```sh
MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 \
  -o SHORT_CONV_FUSION,LSTM_GATES_FUSION,RELATIVE_SHIFT_VIEW,ATTENTION_INPUT_VIEWS
MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 -o POS_PROJECTION_CACHE
MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 -o MUL_MAT \
  -p '^type_a=q8_0,type_b=f32,m=(640|2560|8198|129),n=1,k=(32|128|640|1024|1056),'
```

The [machine-readable evidence](orukeet-metal-followup-benchmark.json)
includes exact metrics, per-clip medians, transcript hashes, model and
source hashes, runtime provenance, receipt hashes and validation details.
The measured runtime was built from the first-patch commit plus the source
changes in this commit; its receipt records the dirty source hash.
Full new receipts remain in the sibling OpenWhispr checkout:

- `benchmark-results/orukeet/m4-pro-metal-followup-previous-120s.json`
- `benchmark-results/orukeet/m4-pro-metal-followup-optimized-120s.json`
- `benchmark-results/orukeet/m4-pro-metal-followup-original-control-120s.json`

These results apply to this M4 Pro and the selected small corpus of clean
read English. Other devices, broader speech accuracy and live app usage
were not measured.
