# Orukeet Q8: Metal kernel optimization on M4 Pro

Patch [`0021-metal-asr-kernels.patch`](../../ggml-patches/0021-metal-asr-kernels.patch)
reduces native median transcription latency by **27.8%** and OpenWhispr backend
median latency by **26.1%** on this Mac. Word error rate stays at **2.40%**,
and every measured transcript matches the baseline exactly.

## Measured results

Measured on 2026-09-10 UTC: Apple M4 Pro, 12 CPU cores, 24 GiB RAM,
macOS 26.5, Metal device MTL0. All three benchmark receipts report **battery
power** and identical power settings (`powermode 0`). The earlier setup-shell
power readout is not used to classify the measured runs.

Each run has a 120-second measurement window after model startup and a full
corpus warmup. The same 24 public English LibriSpeech test-clean clips cover
5–30 seconds, 24 speakers, 364.285 seconds of unique audio and 916 reference
words. Requests run sequentially with batch size one. Every request makes one
native recognition call; native and backend timing come from that same call.

| Metric | Original | Patched | Original repeated afterward | Reduction vs original |
| --- | ---: | ---: | ---: | ---: |
| Native median | 154.2 ms | **111.4 ms** | 155.4 ms | **27.8%** |
| Native p95 | 280.8 ms | **198.6 ms** | 281.3 ms | **29.3%** |
| Native p99 | 283.7 ms | 200.9 ms | 284.0 ms | 29.2% |
| Backend median | 166.0 ms | **122.6 ms** | 166.7 ms | **26.1%** |
| Backend p95 | 295.0 ms | **212.1 ms** | 294.5 ms | **28.1%** |
| Backend p99 | 303.2 ms | 216.3 ms | 297.5 ms | 28.7% |

Latency reduction is `100 × (1 − patched / original)`, calculated from the
unrounded values. The original runtime was preserved before editing and
repeated after the patched run: its median latencies returned within 1% of the
initial baseline. The optimized runtime was restored afterward.

Native throughput rises from **94.2× to 130.5× audio seconds per second**;
backend throughput rises from **87.4× to 118.6×**. The runs completed 691,
938 and 693 measured requests, respectively. Across the 24 individual clips,
the median reduction in each clip's native median is 27.0%; its range is
20.9–29.7%. Backend reductions per clip range from 19.2–28.5%, with a median
of 25.7%. The aggregate median target exceeds 20% for both scopes.

WER is 22/916 words in every run: 16 substitutions, 4 deletions and 2
insertions. Accuracy is scored once per unique clip; repetitions increase
the timing sample only. Native model creation was 55.7 ms before and 57.8 ms
after; OpenWhispr startup was 829.0 ms before and 805.9 ms after. These startup
measurements use a fresh worker in the current OS session, with existing
filesystem and shader caches.

## What changed

The patch optimizes ordinary ggml Metal operations used by the model:

- **Convolution input preparation (`IM2COL`):** a flat grid fills consecutive
  output elements with full SIMD groups. The previous batch-one, 1×1
  convolution path launched one active thread per threadgroup. F16 conversion,
  F32 output, padding, dilation and batch indexing retain their original meaning.
- **F32 tensor copies:** aligned contiguous copies use `float4` loads/stores;
  strided input uses a flat gather. Logical indexing uses 32-bit arithmetic
  when the element count fits; byte strides remain 64-bit. Other copies use
  the existing path.
- **Short F16 depthwise dot products:** exchanging the operands for
  `[K,1,C] × [K,N,C]` makes SIMD lanes process separate audio positions.
  The output memory order and per-dot-product arithmetic remain the same.
  Eligibility requires `K < 32`, contiguous K dimensions and equal batch
  dimensions; asymmetric broadcasting uses the existing path.

The Q8 model file, benchmark source, selected audio and decoding configuration
are identical across the runs. Temporary per-operation profiling code was
removed before measurement. The ggml submodule remains pinned to upstream;
the fork's setup script applies the new patch after patches 0001–0020.

## Reproduce

Use the sibling `openwhispr` checkout on `hoid-orukeet`, with the performance
benchmark at OpenWhispr commit
`2437b3db8fbe1ba5917b8bd80dbc1d68c8efd79c` or a compatible revision. Follow its
`docs/ORUKEET_BENCHMARK.md` for dependency and public-asset setup.

From OpenWhispr, rebuild the adjacent Hoid NeMo fork and run:

```sh
OPENWHISPR_ORUKEET_BUILD_DEVICE=metal npm run build:orukeet
npm run benchmark:orukeet -- --seconds 120
```

For the original baseline, build from NeMo commit
`a5b6953c4a579a2bbd1c0913ad8a85c2a4d99953`, which has patches 0001–0020.
`OPENWHISPR_NEMO_SPEECH_SOURCE` selects a separate baseline checkout without
changing the benchmark's model or audio cache. Use the same power mode,
clip manifest and benchmark source for each run.

The measured model is Orukeet r3 Q8, SHA-256
`93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd`.
The pinned upstream ggml commit is
`c03b4e2bcece5134827881af90242086daf75be5`.

## Validation and evidence

**155 CPU-reference operation tests passed with Metal shader validation:**
98 IM2COL cases, 40 F32 copy/CONT/DUP cases and 17 short F16 matrix cases.
The new short-K tests use a `1e-7` NMSE limit; the unchanged K=32 boundary
uses the existing matrix test tolerance. Tests include vector-copy tails,
non-aligned views, batch dimensions, strided rows, asymmetric broadcasting,
1D/2D lowering, padding and dilation.

```sh
cmake -S ggml -B .cache/orukeet-metal-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DGGML_CCACHE=OFF
cmake --build .cache/orukeet-metal-tests --target test-backend-ops --parallel 4

MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 -o IM2COL
MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 -o CONT,CPY,DUP \
  -p '(^type=f32,|^type_src=f32,type_dst=f32,)'
MTL_SHADER_VALIDATION=1 .cache/orukeet-metal-tests/bin/test-backend-ops \
  test -b MTL0 -o MUL_MAT \
  -p 'type_a=f16,type_b=f16,m=1,.*k=(1|9|31|32),'
```

The complete 21-patch series applies to the pinned upstream tree and reproduces
the live ggml source exactly. OpenWhispr's build verification passes. Its
separate JFK native regression also passes all seven checks, including worker
reuse, serialized concurrent calls, silence, cancellation, recovery and shutdown.

The [machine-readable evidence](orukeet-metal-benchmark.json) records exact
metrics, per-clip medians, transcript hashes, runtime hashes, patch hash and
receipt hashes. Full receipts remain in the sibling OpenWhispr checkout:

- `benchmark-results/orukeet/m4-pro-metal-kernel-baseline-120s.json`
- `benchmark-results/orukeet/m4-pro-metal-kernel-optimized-120s.json`
- `benchmark-results/orukeet/m4-pro-metal-kernel-baseline-control-120s.json`

These are warm, sustained processing results for this M4 Pro and a small sample
of clean read English speech. The benchmark covers the native recognition call
and OpenWhispr backend; microphone capture, UI, live previews and paste are
outside its scope. Other devices and broader speech accuracy were not measured.
