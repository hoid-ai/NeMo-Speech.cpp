# Prepare the Hoid Orukeet Metal SDK release

The optimized SDK can use OpenWhispr's existing archive-download path. Its
public C ABI and CMake SDK version remain `0.1.0`; Hoid release tags identify
the optimized build independently. A versioned SDK archive removes the need
for an adjacent NeMo source checkout when building OpenWhispr with that entry.

`scripts/package-orukeet-metal-sdk.py` prepares the release from a built SDK
and its correctness receipts. It checks the SDK fingerprint, every library
loaded by the full-model test, the ARM64 binaries, relocatable dynamic-library
paths, CMake package, and redistributed license files. It preserves the
original build provenance, including any recorded dirty source state; it
does not relabel an old binary as a build of the current branch tip.

For the already-built and qualified SDK in the adjacent OpenWhispr checkout:

```sh
python3 scripts/package-orukeet-metal-sdk.py \
  --sdk ../openwhispr/.cache/orukeet-metal/arm64-bc68d647ce82/sdk \
  --runtime-receipt ../openwhispr/resources/bin/orukeet-darwin-arm64/metal/runtime.json \
  --validation-report docs/development/orukeet-model-numerics.json \
  --tag v0.1.0-hoid-orukeet.1 \
  --output-dir .cache/orukeet-sdk-release/v0.1.0-hoid-orukeet.1 \
  --openwhispr-catalog ../openwhispr/resources/orukeet/native-runtimes.json
```

The SDK path is machine-specific: select the install prefix associated with
the runtime receipt being packaged. The example path is the current M4 Pro
build. Build instructions are in [the SDK guide](../sdk.md) and the sibling
OpenWhispr benchmark guide. Changed SDK binaries require a corresponding
numerical validation receipt before packaging.

The output directory contains:

- `nemo-speech-0.1.0-hoid-orukeet.1-macos-aarch64-metal.tar.gz` and its `.sha256`.
- A provenance JSON with the SDK fingerprint and qualification details.
- `openwhispr-metal-sdk.json`, the complete replacement Metal platform entry.
- `openwhispr-metal-sdk.patch`, when `--openwhispr-catalog` is provided.

The tarball contains the installed SDK, license files, build provenance and
full-model validation receipt. No model weights or audio are bundled. File
order, ownership and archive timestamps are normalized, so identical inputs
produce the same archive checksum. Existing archives are never overwritten.
The script writes only its output directory; it does not edit OpenWhispr,
create a Git tag or publish a GitHub release.

The [M4 Pro package validation receipt](orukeet-metal-sdk.json) records a
1,840,339-byte candidate archive with SHA-256
`26c0377dcffa031a1216b97350ab8c9c4687920c1842c9e6cc7e3e9d283ce10f`.
Two packaging runs produce identical archives. Extraction preserves the
qualified SDK fingerprint. OpenWhispr's worker builds against the extracted
SDK through its existing CMake configuration and runs from a separate install
directory without dynamic-loader path overrides. All 24 benchmark transcripts
match the original runtime, and the worker shuts down cleanly. The generated
catalog patch passes `git apply --check` against `hoid-orukeet`.

After the PR is reviewed, publish the exact prepared archive, checksum and
JSON assets to the `hoid-ai/NeMo-Speech.cpp` release tagged
`v0.1.0-hoid-orukeet.1`. Point the release tag at the reviewed commit containing
the optimization and qualification reports. A later rebuild must use a new
tag and regenerated checksum if its bytes differ. Signing or modifying the
SDK after qualification also changes the bytes and requires requalification.

After that release exists, the prepared OpenWhispr change can be applied in
one command from this checkout:

```sh
git -C ../openwhispr apply ../NeMo-Speech.cpp/.cache/orukeet-sdk-release/v0.1.0-hoid-orukeet.1/openwhispr-metal-sdk.patch
```

This changes only the `asr-nvidia-metal.platforms.macos_arm64` dependency
entry in `resources/orukeet/native-runtimes.json`. It replaces `source` with
`archives`, so the existing OpenWhispr builder downloads the SDK, verifies
its SHA-256 and size, and links its native worker against the installed C ABI.
CPU, CUDA and Vulkan catalog entries are preserved.

A literal URL-only edit is insufficient: the checksum and archive size must
describe the new SDK too. The generated entry and patch keep those values
together; no OpenWhispr build-script changes are needed for this switch.
Rebuild with `OPENWHISPR_ORUKEET_BUILD_DEVICE=metal npm run build:orukeet`
after applying the patch. Until publication, the generated release URL is
only the planned destination and cannot be downloaded.
