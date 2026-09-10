#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Hoid AI
# SPDX-License-Identifier: Apache-2.0
"""Package a qualified Apple Silicon Metal SDK for a versioned Hoid release.

Reads the existing SDK and validation receipts without rebuilding or publishing.
Emits a deterministic tarball, checksum, build provenance, OpenWhispr catalog
entry, and optionally a catalog patch that can be applied after publication.
"""

import argparse
import difflib
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sdk_fingerprint(sdk):
    # Same bin/lib/include fingerprint used by OpenWhispr's Metal source builder.
    result = hashlib.sha256()

    def visit(directory):
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            result.update(f"{path.relative_to(sdk).as_posix()}\0{info.st_mode & 0o777}\0".encode())
            if path.is_symlink():
                result.update(f"link:{os.readlink(path)}\0".encode())
            elif path.is_dir():
                visit(path)
            elif path.is_file():
                result.update(f"file:{info.st_size}\0".encode())
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        result.update(block)
            else:
                raise RuntimeError(f"Unsupported SDK file: {path}")

    for group in ("bin", "lib", "include"):
        visit(sdk / group)
    return result.hexdigest()


def validate(sdk, runtime, validation):
    require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "Qualify and package this SDK on Apple Silicon",
    )
    require(runtime.get("device") == "metal", "Runtime receipt must describe Metal")
    require(
        runtime.get("spec", {}).get("platform") == "macos_arm64",
        "Runtime receipt must describe macOS ARM64",
    )
    require(
        runtime.get("source", {}).get("remote", "").removesuffix(".git")
        == "https://github.com/hoid-ai/NeMo-Speech.cpp",
        "Expected the Hoid fork build",
    )
    fingerprint = sdk_fingerprint(sdk)
    require(
        fingerprint == runtime["source"]["sdk_sha256"],
        "SDK differs from the recorded build; rebuild and qualify it before packaging",
    )
    for key in (
        "valid_aligned_comparison",
        "input_features_exact",
        "all_decisions_exact",
        "all_transcripts_exact",
        "pass_at_1e-4",
    ):
        require(validation.get(key) is True, f"Full-model validation did not pass: {key}")
    require(validation.get("recognitions_per_runtime", 0) > 0, "Empty validation receipt")
    required = (
        "bin/nemo-speech",
        "include/nemo_speech/asr.h",
        "lib/libnemo_speech_asr_c.dylib",
        "lib/libnemo_speech_asr.dylib",
        "lib/libggml-metal.dylib",
        "lib/cmake/NeMoSpeech/NeMoSpeechConfig.cmake",
        "lib/cmake/NeMoSpeech/NeMoSpeechConfigVersion.cmake",
        "share/licenses/nemo-speech/LICENSE",
        "share/licenses/nemo-speech/NOTICE",
        "share/licenses/nemo-speech/THIRD_PARTY_NOTICES.md",
        "share/licenses/nemo-speech/third_party/ggml/LICENSE",
        "share/licenses/nemo-speech/third_party/sentencepiece/LICENSE",
        "share/licenses/nemo-speech/third_party/sentencepiece/absl-LICENSE",
        "share/licenses/nemo-speech/third_party/sentencepiece/darts-clone-LICENSE",
        "share/licenses/nemo-speech/third_party/sentencepiece/protobuf-lite-LICENSE",
    )
    for relative in required:
        require((sdk / relative).is_file(), f"SDK is missing {relative}")
    version_config = (sdk / "lib/cmake/NeMoSpeech/NeMoSpeechConfigVersion.cmake").read_text()
    require(
        re.search(r'set\(PACKAGE_VERSION "0\.1\.0"\)', version_config),
        "OpenWhispr's current native worker requires SDK ABI version 0.1.0",
    )
    libraries = validation["variants"]["optimized"]["libraries"]
    require(any("ggml-metal" in name for name in libraries), "Metal was not validated")
    require(any("nemo_speech_asr_c" in name for name in libraries), "ASR C ABI was not validated")
    for name, record in libraries.items():
        require(Path(name).name == name, "Invalid library name in validation receipt")
        path = sdk / "lib" / name
        require(
            path.is_file() and digest(path) == record["sha256"],
            f"SDK library is not the numerically validated binary: {name}",
        )
    binaries = [sdk / "bin/nemo-speech"] + sorted(
        path for path in (sdk / "lib").glob("*.dylib") if not path.is_symlink()
    )
    for binary in binaries:
        arch = subprocess.check_output(["lipo", "-archs", str(binary)], text=True).strip()
        require(arch == "arm64", f"Expected arm64 in {binary.name}, found {arch}")
        dependencies = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        for line in dependencies.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if dependency.startswith("@rpath/"):
                require(
                    (sdk / "lib" / dependency[len("@rpath/") :]).is_file(),
                    f"Unbundled dependency: {dependency}",
                )
            else:
                require(
                    dependency.startswith(("/usr/lib/", "/System/Library/")),
                    f"Non-portable dependency in {binary.name}: {dependency}",
                )
        load_commands = subprocess.check_output(["otool", "-l", str(binary)], text=True)
        rpaths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset", load_commands)
        require(
            rpaths and all(item.startswith("@loader_path") for item in rpaths),
            f"Non-relocatable runtime search path in {binary.name}: {rpaths}",
        )
    return fingerprint


def archive_sdk(sdk, output, prefix, extras):
    # Preserve executable modes and symlinks; normalize ownership and timestamps.
    # No gzip filename or wall-clock timestamp enters the archive checksum.
    paths = sorted(sdk.rglob("*"))
    for path in paths:
        require(
            path.relative_to(sdk).as_posix() not in extras,
            f"SDK already contains reserved release metadata: {path}",
        )
        if path.is_symlink():
            target = os.readlink(path)
            require(
                not os.path.isabs(target) and path.resolve().is_relative_to(sdk),
                f"SDK symlink escapes the package: {path}",
            )
        else:
            require(path.is_file() or path.is_dir(), f"Unsupported SDK entry: {path}")
    with output.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
    ) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT, dereference=False
        ) as archive:
            top = tarfile.TarInfo(prefix)
            top.type = tarfile.DIRTYPE
            top.mode = 0o755
            archive.addfile(top)
            for path in paths:
                info = archive.gettarinfo(str(path), f"{prefix}/{path.relative_to(sdk).as_posix()}")
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.pax_headers = {}
                if info.isfile():
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                else:
                    archive.addfile(info)
            for name, content in sorted(extras.items()):
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size = len(content)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=ROOT / "docs/development/orukeet-model-numerics.json",
    )
    parser.add_argument("--tag", required=True, help="For example v0.1.0-hoid-orukeet.1")
    parser.add_argument("--repository", default="hoid-ai/NeMo-Speech.cpp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--openwhispr-catalog",
        type=Path,
        help="Read this catalog and emit its future patch; never edit it",
    )
    args = parser.parse_args()
    require(
        re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9._-]+)?", args.tag),
        "Use a versioned release tag; mutable latest/nightly tags are unsupported",
    )
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository),
        "Invalid GitHub repository",
    )
    sdk = args.sdk.resolve(strict=True)
    output = args.output_dir.resolve()
    require(output != sdk and not output.is_relative_to(sdk), "Output must be outside the SDK")
    output.mkdir(parents=True, exist_ok=True)
    runtime = json.loads(args.runtime_receipt.read_text())
    validation = json.loads(args.validation_report.read_text())
    fingerprint = validate(sdk, runtime, validation)
    version = args.tag[1:]
    prefix = f"nemo-speech-{version}-macos-aarch64-metal"
    name = prefix + ".tar.gz"
    provenance = {
        "schema_version": 1,
        "repository": args.repository,
        "release_tag": args.tag,
        "platform": "macos_arm64",
        "backend": "metal",
        "component_api": 1,
        "sdk_abi_version": "0.1.0",
        "sdk_sha256": fingerprint,
        "build_source": runtime["source"],
        "runtime_receipt_sha256": digest(args.runtime_receipt),
        "validation_report_sha256": digest(args.validation_report),
        "qualified_source_commit": validation.get("versions", {}).get("optimized"),
        "recognitions_per_runtime": validation["recognitions_per_runtime"],
        "final_logit_elements": sum(
            validation["groups"][head]["elements"] for head in ("token_logits", "duration_logits")
        ),
        "atol_rtol_1e_4_passed": validation["pass_at_1e-4"],
        "all_transcripts_exact": validation["all_transcripts_exact"],
        "publication": "Prepared locally; this tool does not create tags or upload releases.",
    }
    extras = {
        "share/nemo-speech/hoid-build-provenance.json": json_bytes(provenance),
        "share/nemo-speech/orukeet-model-validation.json": args.validation_report.read_bytes(),
    }
    archive = output / name
    require(not archive.exists(), f"Refusing to replace an existing release archive: {archive}")
    with tempfile.TemporaryDirectory(prefix="orukeet-package-", dir=output) as scratch:
        temporary = Path(scratch) / name
        archive_sdk(sdk, temporary, prefix, extras)
        temporary.replace(archive)
    checksum = digest(archive)
    (output / (name + ".sha256")).write_text(f"{checksum}  {name}\n")
    installed_bytes = sum(
        path.stat().st_size for path in sdk.rglob("*") if path.is_file() and not path.is_symlink()
    ) + sum(map(len, extras.values()))
    spec = {
        "version": version,
        "component_api": 1,
        "install_bytes": installed_bytes,
        "platform": "macos_arm64",
        "archives": [
            {
                "name": name,
                "url": f"https://github.com/{args.repository}/releases/download/{args.tag}/{name}",
                "sha256": checksum,
                "size_bytes": archive.stat().st_size,
                "extract": "native-tar",
            }
        ],
    }
    (output / "openwhispr-metal-sdk.json").write_bytes(json_bytes(spec))
    (output / (prefix + ".provenance.json")).write_bytes(json_bytes(provenance))
    if args.openwhispr_catalog:
        original = args.openwhispr_catalog.read_text()
        catalog = json.loads(original)
        catalog["asr-nvidia-metal"]["platforms"]["macos_arm64"] = spec
        updated = json_bytes(catalog).decode()
        patch = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile="a/resources/orukeet/native-runtimes.json",
                tofile="b/resources/orukeet/native-runtimes.json",
            )
        )
        (output / "openwhispr-metal-sdk.patch").write_text(patch)
    print(f"Prepared SDK: {archive}")
    print(f"SHA-256: {checksum}")
    print(f"Archive bytes: {archive.stat().st_size}; SDK fingerprint: {fingerprint}")
    print(f"OpenWhispr catalog entry: {output / 'openwhispr-metal-sdk.json'}")
    print("The release URL becomes usable after these assets are published on GitHub.")


if __name__ == "__main__":
    main()
