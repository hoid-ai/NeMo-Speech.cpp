# Orukeet on x86 CPU

Notes from optimizing CPU inference of the Orukeet ASR model in this runtime
on an AVX2 Intel desktop part. Orukeet is a stock Parakeet TDT 0.6B v3
FastConformer graph - 24 layers, `d_model` 1024, depthwise conv kernel 9, 8x
subsampling, TDT decoder - so nothing here is specific to it; it applies to any
FastConformer the CPU backend runs.

The work is on branch `cpu-x86-orukeet`. Every change is a separate commit with
its own measurement.

- [Machine](#machine)
- [How things were measured](#how-things-were-measured)
- [Results](#results)
- [Where the time went](#where-the-time-went)
- [The changes](#the-changes)
- [Tried and rejected](#tried-and-rejected)
- [Building and packaging](#building-and-packaging)
- [Open questions](#open-questions)

## Machine

| | |
|---|---|
| CPU | Intel Core i7-13700 (Raptor Lake), 8 P-cores + 8 E-cores, 24 threads |
| Max frequency | 5.2 GHz P-core, 4.1 GHz E-core |
| Cache | 640 KiB L1d, 24 MiB L2, 30 MiB L3 |
| SIMD | AVX2, FMA, F16C, **AVX-VNNI**; no AVX-512 |
| Memory | 62 GiB |
| OS / toolchain | Ubuntu 24.04, Linux 6.8.0, GCC 13.3.0, CMake 3.28.3, Ninja |
| Container | unprivileged; `perf` unusable (see below) |

P-cores are CPUs 0-15 (SMT pairs), E-cores are CPUs 16-23 (no SMT).

Build, exactly as specified for this work:

```sh
scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TOOLS=ON \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON
cmake --build --preset cpu-asr --parallel
```

`GGML_NATIVE=OFF` with `GGML_AVX2=ON` means the binary must run on any AVX2
x86. The one AVX-VNNI kernel added here is therefore compiled through a
function-level `target` attribute and selected by a CPUID check at run time,
never by compiling the backend for VNNI.

## How things were measured

```sh
build/cpu-asr/bin/nemo-speech bench asr <dir> --model <model>.gguf \
    --device cpu -c 1 -n 30 --json
```

- **Clips.** `test_files/asr/wav/test/jfk.wav` (11 s), and the same clip
  concatenated 6x (66 s) for the long case.
- **Models.** `orukeet-v0.1.0-q8.gguf` (Q8_0, 714 MB) and
  `orukeet-v0.1.0-f16.gguf` (F16, 1.3 GB).
- **Statistic.** Median of 5 process invocations, each of 30 utterances (5 for
  the 66 s clip). Medians, not minima: this machine's run-to-run spread is real
  (sustained 16-thread AVX-VNNI work hits power limits, and the box is shared),
  and a single best-of-N run is not reproducible. Min and max are reported
  alongside so the spread is visible.
- **Correctness gate.** Every change had to reproduce the baseline transcript
  **byte-identically** on Q8 and F16, for both clip lengths, before being kept.
  `bench --json` also reports `transcript_mismatches`, which stayed 0
  throughout.

### Profiling

`perf` does not work in this container: it is unprivileged, and both hardware
and software events are refused (`No permission to enable cpu-clock event`,
`perf_event_paranoid` is 4). Installing `linux-tools-6.8.0-111-generic` makes
the binary match the kernel but does not grant the events. So all profiles here
come from **gperftools**' sampling profiler:

```sh
CPUPROFILE=/tmp/p.prof CPUPROFILE_FREQUENCY=1000 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so \
  build/cpu-asr/bin/nemo-speech bench asr ...
google-pprof --text build/cpu-asr/bin/nemo-speech /tmp/p.prof
```

Two caveats when reading the tables below. `__nss_database_lookup` and
`omp_get_num_procs` are **misattributions** - pprof resolves to the nearest
exported symbol, and these are libc's `memcpy` family and libgomp's team
setup/barrier code respectively. And the bench process loads the model once, so
one-time costs (`repack_q8_0_to_q8_0_8_bl`) appear in the profile.

Per-graph timing came from the runtime's own `NEMO_SPEECH_TIMING=1`, which
prints node count and compute time for each `Session` graph - that is what
separated encoder cost from decoder cost.

## Results

Baseline is `main` (a5b6953) with stock ggml, built with the same flags.
Median of 5 invocations; min-max shows the spread.

| model | clip | baseline | final | speedup | RTFx baseline -> final |
|---|---|---|---|---|---|
| Q8_0 | 11 s | 863.6 ms | **235.6 ms** | **3.67x** | 12.74 -> 46.69 |
| Q8_0 | 66 s | 5942.6 ms | **1796.2 ms** | **3.31x** | 11.11 -> 36.74 |
| F16 | 11 s | 1187.9 ms | **427.9 ms** | **2.78x** | 9.26 -> 25.70 |
| F16 | 66 s | 7869.5 ms | **2833.5 ms** | **2.78x** | 8.39 -> 23.29 |

Spread (min-max over the 5 invocations): baseline Q8 11 s 798-871 ms, final
Q8 11 s 231-239 ms; final F16 11 s 420-433 ms. Transcripts are
byte-identical to the baseline in every cell, and `transcript_mismatches` is 0.

For reference, NVIDIA's prebuilt CPU backend runs the 11 s clip at ~325 ms on
an Apple M5 (RTFx ~34). This branch does it in 236 ms on the i7-13700, from
864 ms before.

### Thread-count sweep

Final build, Q8, median of 3, `--asr.backend.cpu_threads` (or
`NEMO_SPEECH_CPU_THREADS`):

| threads | 11 s ms/utt | 66 s ms/utt |
|---|---|---|
| 4 (the old hardcoded value) | 303.2 | 2832.2 |
| 8 | 242.7 | 1944.7 |
| 16 (the new default: one per physical core) | 231.8 | 1804.1 |
| 24 (all logical CPUs) | 225.2 | 1729.6 |

24 threads reads ~3% better than 16 here, but that gap does not survive
repetition - a second measurement of the same two configurations at
concurrency 1 gave 175.5 ms for 16 threads and 229.2 ms for 24. Run-to-run
variance on this machine is larger than the difference, so the default stays at
one thread per physical core, which is also the safer choice when several
recognizers share the box. Concurrency does separate them from the rejected
`GGML_OPENMP=OFF` build, which is a real effect (see below).

### Concurrency

Q8, 11 s clip, final build, 16 threads:

| concurrency | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms/utterance | 175.5 | 159.0 | 190.7 | 188.5 |
| utterances/s | 5.70 | 6.29 | 5.24 | 5.31 |

## Where the time went

Profile of the Q8 model on the 11 s clip, before and after. Percentages are of
total samples in the bench process.

| symbol | baseline | final | note |
|---|---|---|---|
| `ggml_vec_dot_q8_0_q8_0` | 66.2% | - | one call per output element; gone |
| `ggml_gemm_q8_0_8x4_q8_0_vnni` | - | 47.6% | its replacement |
| `ggml_compute_forward_im2col` | 10.1% | 1.8% | depthwise convs no longer lowered |
| memcpy (as `__nss_database_lookup`) | 5.3% | 5.7% | the attention `cont` copies |
| `ggml_vec_dot_f16` | 4.1% | 2.5% | |
| `ggml_vec_dot_f32` | 2.4% | 4.3% | attention |
| libgomp (as `omp_get_num_procs`) | 0.9% | 9.0% | per-graph fork/join + barriers |
| `MelSpectrogramExtractor::compute_padded` | 1.6% | 5.7% | frontend, not ggml |
| `quantize_row_q8_0` / `ggml_quantize_mat_q8_0_4x4` | 0.9% | 1.5% | activation quantization |
| `ggml_compute_forward_conv_2d_dw` | - | 1.8% | the new direct kernel |

The baseline column is at the shipped 4 threads; the final column at 16. Shares
that "grew" mostly did not - the denominator shrank by 4.5x.

The encoder is where essentially all of this lives. `NEMO_SPEECH_TIMING` on the
11 s clip, baseline:

| graph | nodes | calls/utterance | compute |
|---|---|---|---|
| offline transducer encoder | 2240 | 2 | ~355 ms (dominates) |
| TDT decoder stages | 76 | 40 | 30 ms |
| decoder joint tail | 14 | 19 | 2 ms |
| mel frontend | - | 2 | 13 ms |

## The changes

Numbers in this section are the incremental effect of each commit, measured
with an earlier and slightly noisier protocol (median of 5 x 10 utterances for
the 11 s clip, 5 x 3 for the 66 s clip). They are for attribution between
changes; the [Results](#results) table is the authoritative before/after.

### 1. `check_backend_coverage` crashed on this model

Not a performance change, but it blocked the first diagnostic step. The tool
segfaulted on Orukeet, exactly as reported on macOS. Two independent bugs:

- `is_rnnt` tested only `HeadKind::Rnnt`, but `RnntModel::head_kind()` returns
  `HeadKind::Tdt` whenever the RNNT config carries duration bins. TDT models
  were therefore `static_cast` to `CtcModel` and crashed inside
  `MelSpectrogramExtractor::compute_padded` on the first `infer_ctc` call.
- With that fixed it aborted again: `CacheStreamRunner` was constructed
  unconditionally, and Orukeet's encoder is full-context, so
  `set_cache_right_ctx` threw *"model encoder does not support cache-aware
  streaming"*.

Gating the runner on `supports_cache_streaming()` leaves a full-context model
building no encoder graph at all, so the tool now also calls `infer_offline`.
It exits 0 and reports the 2240-node encoder graph.

### 2. Thread pool sized from the machine (`792 -> 575 ms`)

`Session::compute` passed a literal `4` to every `ggml_graph_compute`. On a
24-thread part that left 20 threads idle.

Sweep, Q8 / 11 s clip:

| threads | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|
| ms/utterance | 792 | 625 | 602 | **575** | 581 | 582 |

`default_compute_threads()` (new `src/runtime/ggml/cpu_topology.{h,cpp}`) picks
one thread per **physical core** - SMT siblings collapsed - capped by the
process CPU-affinity mask and any cgroup CPU quota, so a container CPU-set or a
`taskset` wrapper is never overrun. `NEMO_SPEECH_CPU_THREADS` overrides it and
`asr.backend.cpu_threads` / `--cpu-threads` sets it per recognizer.

The hybrid-core hazard did not bind. The obvious worry is that ggml's equal
per-op split hands an E-core the same slice as a P-core and stalls the barrier
behind it. Measured, restricting the pool to the 8 P-cores cost **~8%** against
one thread per physical core, because MUL_MAT - ~78% of encoder time - hands
out its chunks through an atomic counter, so P-cores simply claim more of them.
Pinning experiments (`taskset`, `OMP_PROC_BIND`, `GOMP_CPU_AFFINITY`) were
within noise of the unpinned result and were dropped as not worth the
fragility.

### 3. Per-graph thread count (`575 -> 567 ms`, and 1.5x on the decoder)

ggml joins every worker on a barrier after each node, so a graph costs roughly
`work/threads + nodes * barrier * threads`. Orukeet's graphs span four orders of
magnitude of work, and one global thread count cannot serve both:

| threads | encoder (2240 nodes) | decoder (76 nodes, 40x/utterance) |
|---|---|---|
| 2 | 1892 ms | 13.9 ms |
| 4 | 1050 ms | 10.5 ms |
| 8 | 637 ms | 10.9 ms |
| 16 | **572 ms** | 20.0 ms |

Minimizing that model over the thread count gives `sqrt(work / (nodes * c))`.
One fitted constant reproduces the measured optimum for all three graph shapes
the model runs. It is resolved once per cached graph shape, so there is no
per-call cost. Decoder-stage time per utterance: 20.0 -> 13.1 ms.

### 4. A Q8_0 repack GEMM for x86 (`562 -> 397 ms`)

This is the main result. `ggml_repack_get_optimal_repack_type` has **no x86
branch for Q8_0** - upstream repacks it only for NEON and RISC-V. An x86 Q8_0
matmul therefore fell through to the generic path, which calls
`ggml_vec_dot_q8_0_q8_0` once per output element: no register blocking, and the
activation column re-read from memory for every weight row. That one function
was 56% of encoder time, and it is most of why an Apple M5 - whose NEON path
*does* have a repacked int8 GEMM - beat this machine on the same clip.

`ggml-patches/0021-cpu-x86-q8-repack-gemm.patch` adds the `q8_0_8x4` layout:
eight weight columns interleaved at 4-byte granularity, so one 32-byte load
spans all eight columns and `vpmaddwd` reduces exactly four bytes into each of
eight int32 lanes - one lane per column, no horizontal shuffle in the inner
loop. Four activation rows are held in registers per pass, so each weight load
feeds four rows.

Two kernels share that inner loop through a macro so they cannot drift apart:

- **AVX2**: `vpmaddubsw` + `vpmaddwd`.
- **AVX-VNNI**: `vpdpbusd`, which folds all three into one instruction.

Signedness follows ggml's own scalar Q8_0 dot product. Both instructions want
an unsigned left operand, so the weight's sign is folded into the activation
and `|w|` used. Q8_0 quantizes with `d = amax/127`, so `|w| <= 127` and
`|a| <= 127`, and a `vpmaddubsw` pair sums to at most 32258 - int16 cannot
saturate. (The alternative, biasing weights by +128 to get an unsigned operand
directly, *would* saturate `vpmaddubsw`, so it was not used.)

AVX-VNNI is reached through `__attribute__((target("avx2,avxvnni")))` and
selected by `CPUID.(EAX=7,ECX=1):EAX[4]` at run time. ggml's own
`ggml_cpu_has_avx_vnni()` is compile-time only (`#if defined(__AVXVNNI__)`) and
so is useless in a `GGML_NATIVE=OFF` build - hence the local CPUID check. MSVC
has no target attribute but emits intrinsics regardless of switches, so it
compiles the kernel directly; every other non-GNU/Clang toolchain stays on
AVX2. `arch-fallback.h` aliases the generic kernels for every non-x86
architecture, so nothing outside x86 changes.

`scripts/configure.sh` now applies the ggml patch series for `cpu-*` presets,
which it previously did only for CUDA and Metal. Without that a CPU build would
not contain the patch at all.

### 5. Actually reaching it: weight buffer types (the kernel was dead code)

Adding the kernel changed nothing - 562 -> 562 ms. The profile still showed
`ggml_vec_dot_q8_0_q8_0` at 58%.

`TensorContainer::m_create_tensor` placed **every** declared tensor on
`device_main_buft()`, the device's *main* buffer type. ggml exposes
weight-layout buffer types (`CPU_REPACK`) only as a device's *extra* buffer
types. `BackendManager::init_backends` was already collecting them into
`buft_list`; nothing ever selected one. The entire repack machinery was
unreachable from this runtime.

`TensorRole::MatmulWeight` marks a declaration as "only ever a `ggml_mul_mat`
src[0]". Such a tensor goes to the first buffer type that accepts it for
MUL_MAT, probed the way llama.cpp's `weight_buft_supported()` does it: build
the op, attach a zero-length buffer of the candidate type to a throwaway
weight, and ask `ggml_backend_dev_supports_op`, which routes to the extra
buffer type's own predicate.

The opt-in is **explicit per declaration site**, not inferred from dtype or
shape, because these layouts permute the weight bytes - a quantized embedding
under `GET_ROWS` would silently produce garbage. It is also restricted to the
device `device_main_buft()` already chose, so an accelerator build cannot pull
a weight back to host memory just because the CPU offers a repacking buffer
type.

Isolated per-matmul speedup at FastConformer shapes (16 threads, T=137):

| shape | plain | repack | |
|---|---|---|---|
| 1024x1024 attention projection | 2.042 ms | 0.244 ms | 8.4x |
| 1024x4096 feed-forward up | 3.838 ms | 0.789 ms | 4.9x |
| 4096x1024 feed-forward down | 3.313 ms | 0.756 ms | 4.4x |

### 6. The fused QKV weight, and tinyBLAS (`397 -> 393 ms`, F16 `783 -> 534 ms`)

24 Q8_0 weights still missed the opt-in. The fused Q/K/V projection is not
declared by `Linear` - the encoder allocates one `linear_qkv.weight`
[1024, 3072] per layer and stacks the three GGUF tensors into it. Those are the
largest single matmul in each layer.

They could not simply be marked: `set_data` filled them with three
`ggml_backend_tensor_set` calls at thirds offsets, and a weight-layout buffer
type rewrites the whole tensor on `set_tensor` and asserts on a partial write.
Staging the three thirds in host memory and uploading once fixes that, at the
cost of one transient buffer per layer at load time.

That also settled `GGML_LLAMAFILE`. ggml's tinyBLAS is off by default in this
vendored ggml. Turning it on makes F16 matmuls ~1.4x faster, but *before* this
fix it cost the Q8 model ~17%, because the unrepacked QKV weights fell into
tinyBLAS's Q8_0 kernel, which is slower here than ggml's own vec_dot path. With
them repacked, Q8 no longer cares and F16 keeps the win:

| | `LLAMAFILE=OFF` | `LLAMAFILE=ON` |
|---|---|---|
| Q8 11 s | 392.8 ms | 393.2 ms |
| Q8 66 s | 2575.0 ms | 2607.5 ms |
| F16 11 s | 777.0 ms | **533.7 ms** |
| F16 66 s | 5035.6 ms | **3430.0 ms** |

So it is on for the `cpu-*` presets.

### 7. AVX2 activation quantizer (`393 -> 355 ms`)

`ggml_quantize_mat_q8_0_4x4` converts src1 into the layout the new GEMM reads.
x86 had no vectorized version - `arch-fallback.h` aliased it to the scalar C
implementation - and with the weight matmuls repacked, that loop and its
per-element `roundf` had become ~8% of encoder time.

The interleave falls out of the pack sequence for free: after `packs_epi32` /
`packs_epi16` the 32 bytes sit as int32 lanes
`[r0e0-3, r1e0-3, r2e0-3, r3e0-3, r0e4-7, ...]`, and the layout wants
`qs[k*16 + row*4 + t]` - lane `(half*4 + row)`, the same order. Unlike the
existing 8-byte variant it needs no closing `permutevar8x32`.

Matching the vectorized round-to-nearest also made the repacked result
**bit-identical** to the unrepacked path: the reference comparison went from
12% of a float-summation-order tolerance to an exact match, because
`quantize_row_q8_0` on the plain path already rounds this way and only the
scalar `roundf` differed.

### 8. Direct depthwise convolution (`355 -> 182 ms`)

im2col was 18.7% of encoder time, split roughly evenly between the 24
FastConformer conv modules (depthwise conv1d, kernel 9, 1024 channels) and the
pre-encode subsampling convs. Per encoder graph, im2col materialized 81 M
elements, 52% of them for the depthwise convs.

For a depthwise convolution that lowering is nearly all overhead: it writes
`knl_w` values per output element in order to take a `knl_w`-long dot product -
9 F16 values written and read back for 9 MACs of real work.

ggml already has `GGML_OP_CONV_2D_DW`, and `nn.cpp` already knew how to emit it
- but only for CUDA, because the CPU implementation cast `kernel->data`
straight to `float` and these weights are F16. The existing comment said so: a
CPU session *"silently misreads the F16 kernel as F32 (garbage subsampling
output)"*.

`ggml-patches/0022-cpu-direct-depthwise-conv.patch`:

- stages the per-channel kernel through F32 once, so F16 weights work at all;
- adds a vectorized path for the 1-D unit-stride case - the conv module - which
  is `knl_w` `ggml_vec_mad_f32` passes over the output row;
- makes `ggml_backend_cpu_device_supports_op` report CONV_2D_DW type support
  instead of falling through to its `default: return true`.

`NEMO_SPEECH_CPU_DIRECT_DW_CONV` (default OFF, enabled by the `cpu-*` presets)
mirrors the existing `NEMO_SPEECH_DIRECT_DW_CONV`. **It must stay OFF against
stock ggml**; `nn.cpp` then keeps the portable lowering, which is F16-safe
everywhere. Both call sites `||` in the CPU case rather than replacing the CUDA
gate, so accelerator builds are untouched.

The direct path is also the more *accurate* of the two: the im2col lowering
rounds activations to F16, and this does not.

## Tried and rejected

| tried | result | kept? |
|---|---|---|
| One thread per **performance** core (8) instead of per physical core (16) | ~8% slower; MUL_MAT's atomic chunk counter already rebalances across P/E | no |
| `taskset` / `OMP_PROC_BIND=close` / `GOMP_CPU_AFFINITY` pinning to P-cores | within noise of unpinned once measured properly; adds real fragility | no |
| `OMP_WAIT_POLICY=active` | catastrophic - 3834 ms vs 485 ms; spinning OMP threads starve the runtime's own workers | no |
| `GGML_LLAMAFILE=ON` **before** the QKV weights were repacked | F16 1.4x faster but Q8 ~17% slower | deferred, then adopted |
| Restricting tinyBLAS to unbatched matmuls (theory: its per-slice blocking loses on per-head attention slices) | worth nothing once QKV was repacked, and cost F16 ~2% | no |
| `GGML_OPENMP=OFF` (ggml's own spin threadpool) | ~2-6% better at concurrency 1, but clearly worse under load: 6.88 -> 5.35 utt/s at concurrency 8 | no |
| Biasing weights +128 for a native unsigned `vpmaddubsw`/`vpdpbusd` operand | `vpmaddubsw` saturates int16 at 255*127*2; the sign trick is exact | no |

## Building and packaging

```sh
git submodule update --init --depth 1 ggml llama.cpp
scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TOOLS=ON \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON
cmake --build --preset cpu-asr --parallel
cmake --install build/cpu-asr --prefix out/nemo-speech
tar czf nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz -C out nemo-speech
```

`scripts/configure.sh` applies the ggml patch series for `cpu-*` presets. A raw
`cmake --preset cpu-asr` does **not**, and would otherwise build with
`NEMO_SPEECH_CPU_DIRECT_DW_CONV=ON` against an unpatched ggml, which misreads
the F16 depthwise kernels - a silent wrong-output failure, not a build failure.
CMake therefore checks the vendored ggml for patch 0022 whenever that option is
on and stops with instructions if it is missing. Use the script, or pass
`-DNEMO_SPEECH_CPU_DIRECT_DW_CONV=OFF`.

Building against pristine upstream ggml works and is tested: with
`-DNEMO_SPEECH_GGML_PATCHED=OFF -DNEMO_SPEECH_CPU_DIRECT_DW_CONV=OFF` and no
patches applied, the branch builds and transcribes correctly - it just loses the
two CPU patches' gains.

The tarball built from this branch:

| | |
|---|---|
| file | `nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz` |
| size | 2,036,213 bytes |
| sha256 | `b790c9b4df44ef1bd5bd91b01d055c11657d090fc5d656087045dad2024fe45b` |

Verified by extracting it elsewhere and running it against the Q8 model with
`LD_LIBRARY_PATH=<prefix>/lib`: correct transcript, 180 ms/utterance.

## Open questions

- **The attention `cont` copies are untouched.** The unfused relative-position
  attention path still does ~8 `ggml_cont` copies per layer plus a
  concat/reshape/view shift; that memcpy traffic is 5.7% of the final profile,
  and `ggml_vec_dot_f32` on the attention matmuls another 4.3%. A CPU
  equivalent of the fused rel-pos op (patch 0001, CUDA-only) is the obvious next
  step and was not attempted here.
- **libgomp fork/join is now 9%** of the profile - ~61 `omp parallel` regions
  per utterance plus a barrier per graph node. `GGML_OPENMP=OFF` trades it for
  worse concurrent throughput (above). A persistent threadpool via
  `ggml_backend_cpu_set_threadpool`, which this runtime never sets, might get
  both; untested.
- **The mel frontend is 5.7%** and is hand-written C++ outside ggml, so none of
  this touched it. It is now a bigger share than im2col.
- **The 11 s clip scales better than the 66 s clip** (RTFx ~52 vs ~37). Some of
  that is attention's O(T^2), but it was not investigated.
- **`joint.joint_net.2.weight` [640, 8198] cannot be repacked** - 8198 % 8 != 0 -
  so the vocabulary projection stays on the generic path. Padding the
  allocation to 8200 rows and slicing would fix it; it is a small share of
  runtime and was left alone.
- **`test-backend-ops` cannot validate any of this.** It compares a backend
  against the CPU reference, so with only the CPU backend present it reports
  `Skipping CPU backend` for every op. The CPU-side equivalents used here were
  purpose-built reference comparisons (repack buffer type vs plain buffer type;
  direct depthwise vs im2col lowering) plus ggml's own `test-conv1d-dw-c1`,
  `test-conv1d-dw-c2`, `test-conv2d-dw`, `test-conv1d`, `test-conv2d` and
  `test-quantize-fns`, which all pass.
- **`perf` was unavailable**, so there is no cycle-level or cache-miss data
  here - only gperftools sampling. A privileged run would be worth doing before
  the next round of kernel tuning.
- **Only one machine.** Everything here was measured on a single Raptor Lake
  desktop part. The AVX2 (non-VNNI) kernel path is correctness-tested but its
  *performance* is unmeasured, as is anything pre-Alder-Lake or AMD.
