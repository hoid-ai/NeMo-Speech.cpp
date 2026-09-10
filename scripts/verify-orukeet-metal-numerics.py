#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Hoid AI
# SPDX-License-Identifier: Apache-2.0
"""Audit actual Orukeet Metal runtimes with elementwise atol/rtol checks."""

import argparse
import collections
import datetime
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP_COUNTS = {
    "im2col": 98,
    "f32_copy": 40,
    "short_f16_dot": 17,
    "convolution_fusion": 128,
    "lstm_fusion": 60,
    "relative_shift": 24,
    "attention_views": 24,
    "position_projection": 6,
    "q8_matvec": 14,
}
SCOPES = (
    "vs_original_metal",
    "vs_previous_metal",
    "vs_cpu",
    "original_metal_vs_cpu",
    "previous_metal_vs_cpu",
    "vs_fp64",
    "original_metal_vs_fp64",
    "previous_metal_vs_fp64",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_info(path):
    path = path.resolve(strict=True)
    libraries = path / "lib"
    hashes = {
        library.name: sha256(library)
        for library in sorted(libraries.glob("libggml*.dylib"))
        if not library.is_symlink()
    }
    if not hashes or not (libraries / "libggml-metal.dylib").is_file():
        raise RuntimeError(f"Not a Metal runtime: {path}")
    receipt = path / "runtime.json"
    return {
        "directory": str(path),
        "library_sha256": hashes,
        "receipt_sha256": sha256(receipt) if receipt.is_file() else None,
    }


def aggregate(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row["group"], {})
        for scope in SCOPES:
            if scope not in row:
                continue
            value = row[scope]
            stats = group.setdefault(
                scope,
                {
                    "cases": 0,
                    "failed_cases": 0,
                    "elements": 0,
                    "failed_elements": 0,
                    "nonfinite_elements": 0,
                    "failed_elements_at_1e_4": 0,
                    "max_absolute_error": 0,
                    "max_relative_error_nonzero_reference": 0,
                    "max_absolute_error_at_zero_reference": 0,
                    "max_error_over_bound": 0,
                    "worst_case": None,
                },
            )
            stats["cases"] += 1
            stats["failed_cases"] += not value["passed"]
            for key in (
                "elements",
                "failed_elements",
                "nonfinite_elements",
                "failed_elements_at_1e_4",
            ):
                stats[key] += int(value[key])
            for key in (
                "max_absolute_error",
                "max_relative_error_nonzero_reference",
                "max_absolute_error_at_zero_reference",
            ):
                stats[key] = max(stats[key], value[key])
            if value["max_error_over_bound"] > stats["max_error_over_bound"]:
                stats["max_error_over_bound"] = value["max_error_over_bound"]
                stats["worst_case"] = {
                    "id": row["id"],
                    "scenario": row["scenario"],
                    "element": value["worst"],
                }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-runtime", type=Path, required=True)
    parser.add_argument("--previous-runtime", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT.parent / "openwhispr/resources/bin/orukeet-darwin-arm64/metal",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/orukeet-metal-numerics")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--profile",
        choices=("uniform", "cancellation", "saturation", "half-boundaries"),
        action="append",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This audit requires macOS and Metal")
    seeds = args.seed or [20260910]
    profiles = args.profile or ["uniform"]
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in seeds):
        parser.error("Seeds must fit uint32")
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    runtime_paths = {
        "original": args.original_runtime.resolve(),
        "previous": args.previous_runtime.resolve(),
        "optimized": args.runtime.resolve(),
    }
    runtimes = {name: runtime_info(path) for name, path in runtime_paths.items()}
    source_files = (
        "tests/cpp/orukeet_metal_numerics.cpp",
        "ggml/tests/test-backend-ops.cpp",
        "src/common/json.cpp",
        "src/common/json.h",
    )
    binary = work / "audit"
    build = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-UNDEBUG",
        "-I" + str(ROOT / "ggml/include"),
        str(ROOT / source_files[0]),
        str(ROOT / source_files[2]),
        "-L" + str(args.runtime.resolve() / "lib"),
        "-lggml",
        "-lggml-base",
        "-lggml-metal",
        "-lggml-cpu",
        "-Wl,-rpath," + str(args.runtime.resolve() / "lib"),
        "-o",
        str(binary),
    ]
    print("Building numerical auditor; runtime kernels remain unchanged.", flush=True)
    with (work / "build.log").open("w") as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    all_rows, scenarios = [], []
    for profile in profiles:
        for seed in seeds:
            scenario = f"{profile}-{seed}"
            snapshots = {name: work / f"{name}-{scenario}.bin" for name in ("original", "previous")}
            runs = {}
            for name, runtime in runtime_paths.items():
                report = work / f"{name}-{scenario}.jsonl"
                if name == "optimized":
                    command = [
                        str(binary),
                        "compare",
                        str(snapshots["original"]),
                        str(snapshots["previous"]),
                        str(report),
                        str(seed),
                        profile,
                        str(args.atol),
                        str(args.rtol),
                    ]
                else:
                    command = [
                        str(binary),
                        "record",
                        str(snapshots[name]),
                        str(report),
                        str(seed),
                        profile,
                    ]
                env = dict(os.environ)
                for key in (
                    "GGML_METAL_FUSION_DISABLE",
                    "GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
                    "GGML_METAL_GRAPH_DEBUG",
                    "GGML_METAL_FUSION_DEBUG",
                ):
                    env.pop(key, None)
                env.update(
                    DYLD_LIBRARY_PATH=str(runtime / "lib"),
                    MTL_SHADER_VALIDATION="1",
                    GGML_METAL_GRAPH_DEBUG="1",
                )
                print(f"{scenario}: {name}", flush=True)
                with (work / f"{name}-{scenario}.log").open("w") as log:
                    result = subprocess.run(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode not in (0, 1):
                    raise RuntimeError(f"{name} audit crashed; see {log.name}")
                rows = [json.loads(line) for line in report.read_text().splitlines()]
                if rows[-1].get("kind") != "summary":
                    raise RuntimeError(f"Incomplete numerical audit: {report}")
                metadata = rows[0]
                for key in ("metal_library", "cpu_library", "base_library"):
                    loaded = Path(metadata[key]).resolve(strict=True)
                    if loaded.parent != (runtime / "lib").resolve():
                        raise RuntimeError(f"Wrong {key} was loaded: {loaded}")
                    if sha256(loaded) != runtimes[name]["library_sha256"][loaded.name]:
                        raise RuntimeError(f"Library changed during audit: {loaded}")
                cases = [row for row in rows if row["kind"] == "case"]
                if dict(collections.Counter(row["group"] for row in cases)) != GROUP_COUNTS:
                    raise RuntimeError(
                        "Fixture coverage changed; inspect the audit's group selection"
                    )
                runs[name] = {"report": report.name, "sha256": sha256(report), "summary": rows[-1]}
                if name != "optimized":
                    runs[name]["snapshot_sha256"] = sha256(snapshots[name])
                else:
                    for row in cases:
                        row["scenario"] = scenario
                    all_rows.extend(cases)
            scenarios.append({"id": scenario, "seed": seed, "profile": profile, "runs": runs})

    groups = aggregate(all_rows)
    passed = all(row[scope]["passed"] for row in all_rows for scope in SCOPES[:2])
    exact_checks = [
        row[key]
        for row in all_rows
        for key in (
            "metal_exact_zero",
            "cpu_exact_zero",
            "original_exact_zero",
            "previous_exact_zero",
        )
        if key in row
    ]
    passed &= all(exact_checks)
    evidence = {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "success_vs_metal_references": passed,
        "success_vs_ggml_cpu": all(row["vs_cpu"]["passed"] for row in all_rows),
        "criterion": "abs(actual-reference) <= atol + rtol*abs(reference), for every element",
        "atol": args.atol,
        "rtol": args.rtol,
        "secondary_diagnostic_atol_rtol": 1e-4,
        "metal_shader_validation": True,
        "fixtures_per_scenario": GROUP_COUNTS,
        "case_count": len(all_rows),
        "exact_zero_checks": len(exact_checks),
        "exact_zero_checks_passed": all(exact_checks),
        "inputs_identical": True,
        "input_identity_check": "SHA-256 of every leaf's type, shape and actual quantized bytes",
        "source_sha256": {name: sha256(ROOT / name) for name in source_files},
        "audit_binary_sha256": sha256(binary),
        "runtimes": runtimes,
        "scenarios": scenarios,
        "groups": groups,
    }
    output = work / "report.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    for name, scopes in groups.items():
        for scope in ("vs_original_metal", "vs_previous_metal", "vs_cpu", "vs_fp64"):
            if scope not in scopes:
                continue
            value = scopes[scope]
            print(
                f"{name} {scope}: {value['cases'] - value['failed_cases']}/{value['cases']} passed; "
                f"max abs {value['max_absolute_error']:.9g}; "
                f"worst error/bound {value['max_error_over_bound']:.6g}"
            )
    print(f"Report: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
