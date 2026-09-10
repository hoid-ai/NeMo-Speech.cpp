#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Hoid AI
# SPDX-License-Identifier: Apache-2.0
"""Compare full Orukeet encoder outputs and every TDT logit on two Metal runtimes.

Uses the actual preserved dylibs, identical PCM, and independent recognition
calls. It does not replace model intermediates with reference values. On macOS
the test-only interposer retains the last encoder tensor and joint scores until
readback. Kernel fusion, graph optimization, and the positional cache stay on.
The resulting latency is deliberately NOT a benchmark measurement.

Exit 0: all model outputs pass atol=rtol=1e-4, with identical decoding decisions.
Exit 1: numerical/decision failure. Exit 2: invalid or incomplete audit.
All reports also include 1e-5 and 1e-3, with the original runtime as reference.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENWHISPR = ROOT.parent / "openwhispr"
SOURCE = ROOT / "tests/cpp/orukeet_model_numerics.cpp"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate(comparisons):
    groups = {}
    for comparison in comparisons:
        for name, stats in comparison["groups"].items():
            if name not in groups:
                groups[name] = {
                    "elements": 0,
                    "nonfinite": 0,
                    "exact_differences": 0,
                    "max_absolute_error": 0,
                    "worst_absolute": None,
                    "tolerances": [
                        {
                            "atol": t,
                            "rtol": t,
                            "failed_elements": 0,
                            "failed_recognitions": 0,
                            "max_error_over_bound": 0,
                            "worst": None,
                        }
                        for t in (1e-5, 1e-4, 1e-3)
                    ],
                }
            target = groups[name]
            for key in ("elements", "nonfinite", "exact_differences"):
                target[key] += stats[key]
            context = {"clip": comparison["id"], "repeat": comparison["repeat"]}
            if stats["max_absolute_error"] > target["max_absolute_error"]:
                target["max_absolute_error"] = stats["max_absolute_error"]
                target["worst_absolute"] = dict(stats["worst_absolute"], **context)
            for out, current in zip(target["tolerances"], stats["tolerances"]):
                out["failed_elements"] += current["failed_elements"]
                out["failed_recognitions"] += not current["pass"]
                if current["max_error_over_bound"] > out["max_error_over_bound"]:
                    out["max_error_over_bound"] = current["max_error_over_bound"]
                    out["worst"] = dict(current["worst"], **context)
    for stats in groups.values():
        for threshold in stats["tolerances"]:
            threshold["pass"] = threshold["failed_elements"] == 0
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-runtime", type=Path, required=True)
    parser.add_argument(
        "--runtime", type=Path, default=OPENWHISPR / "resources/bin/orukeet-darwin-arm64/metal"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=OPENWHISPR / ".cache/orukeet-benchmark/model-cache/parakeet-models/"
        "orukeet-v0.1.0-q8/orukeet-r3-93ce19c6-q8.gguf",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OPENWHISPR / ".cache/orukeet-benchmark/clips-orukeet-quick-v1-24.json",
    )
    parser.add_argument(
        "--pcm-dir", type=Path, default=OPENWHISPR / ".cache/orukeet-optimization/pcm"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/orukeet-model-numerics")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--kernel-debug",
        action="store_true",
        help="Log actual Metal graph fusions (use --limit 1 for a small receipt)",
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This audit requires Apple Silicon with Metal")
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise RuntimeError("Repeats and limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clips = read_json(args.manifest)["clips"][: args.limit]
    if not clips:
        raise RuntimeError("No clips selected")
    jobs = []
    for clip in clips:
        pcm = (args.pcm_dir / (clip["id"] + ".f32")).resolve()
        if not pcm.is_file() or pcm.stat().st_size % 4:
            raise RuntimeError("Missing or invalid F32 PCM: " + str(pcm))
        jobs.append(
            {
                "id": clip["id"],
                "pcm": str(pcm),
                "pcm_sha256": sha256(pcm),
                "seconds": pcm.stat().st_size / 4 / 16000,
            }
        )
    jobs_path = output / "jobs.json"
    write_json(jobs_path, jobs)
    runtime = args.runtime.resolve()
    runtime_lib = runtime / "lib"
    driver = output / "orukeet-model-audit"
    tap = output / "liborukeet-model-tap.dylib"
    common = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-I" + str(ROOT / "include"),
        "-I" + str(ROOT / "ggml/include"),
        str(SOURCE),
        str(ROOT / "src/common/json.cpp"),
        "-L" + str(runtime_lib),
        "-lnemo_speech_asr_c",
        "-lggml",
        "-lggml-base",
        "-Wl,-rpath," + str(runtime_lib),
    ]
    with (output / "build.log").open("w") as log:
        for command in (
            common + ["-o", str(driver)],
            common + ["-DORUKEET_MODEL_TAP", "-dynamiclib", "-o", str(tap)],
        ):
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    variants = {}
    for name, runtime in (
        ("original", args.original_runtime.resolve()),
        ("optimized", args.runtime.resolve()),
    ):
        variant_dir = output / name
        variant_dir.mkdir(exist_ok=True)
        env = dict(os.environ)
        for key in (
            "GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
            "GGML_METAL_FUSION_DISABLE",
            "NEMO_SPEECH_METAL_POS_CACHE_DISABLE",
            "GGML_METAL_GRAPH_DEBUG",
        ):
            env.pop(key, None)
        env["DYLD_LIBRARY_PATH"] = str(runtime / "lib")
        env["DYLD_INSERT_LIBRARIES"] = str(tap)
        env["GGML_BACKEND_DL_PATH"] = str(runtime / "lib")
        if args.kernel_debug:
            env["GGML_METAL_GRAPH_DEBUG"] = "1"
            env["GGML_METAL_FUSION_DEBUG"] = "1"
            env["ORUKEET_AUDIT_KERNEL_LOG"] = "1"
        print(
            "Capturing",
            name,
            "full-model tensors:",
            len(jobs),
            "clips x",
            args.repeats,
            "passes",
            flush=True,
        )
        with (variant_dir / "runtime.log").open("w") as log, (
            variant_dir / "recognitions.jsonl"
        ).open("w") as records:
            command = [
                str(driver),
                str(args.model.resolve()),
                str(jobs_path),
                str(variant_dir),
                str(args.repeats),
            ]
            with subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=log, text=True
            ) as process:
                rows = []
                for line in process.stdout:
                    records.write(line)
                    records.flush()
                    row = json.loads(line)
                    if "libraries" in row:
                        loaded = {}
                        for library in row["libraries"]:
                            path = Path(library).resolve()
                            if path.parent != (runtime / "lib").resolve():
                                raise RuntimeError(
                                    "Unexpected loaded runtime library: " + str(path)
                                )
                            loaded[path.name] = {"path": str(path), "sha256": sha256(path)}
                        if not any("ggml-metal" in p for p in loaded):
                            raise RuntimeError("Metal library was not loaded")
                    else:
                        rows.append(row)
                        print(name, row["id"], "pass", row["repeat"] + 1, flush=True)
                if process.wait() != 0:
                    raise RuntimeError("Capture failed; see " + str(variant_dir / "runtime.log"))
        if len(rows) != len(jobs) * args.repeats:
            raise RuntimeError("Incomplete recognition set: " + name)
        variants[name] = {"runtime": str(runtime), "libraries": loaded, "rows": rows}

    comparisons = []
    for reference, candidate in zip(variants["original"]["rows"], variants["optimized"]["rows"]):
        for field in ("id", "repeat", "samples"):
            if reference[field] != candidate[field]:
                raise RuntimeError("Recognition ordering mismatch in " + field)
        path = output / (reference["id"] + "-r" + str(reference["repeat"]) + ".comparison.json")
        subprocess.run(
            [str(driver), "--compare", reference["prefix"], candidate["prefix"], str(path)],
            check=True,
        )
        comparison = read_json(path)
        comparison.update(
            {
                "id": reference["id"],
                "repeat": reference["repeat"],
                "transcripts_equal": reference["transcript"] == candidate["transcript"],
            }
        )
        comparisons.append(comparison)
        print("Compared", comparison["id"], "pass", comparison["repeat"] + 1, flush=True)
    groups = aggregate(comparisons)
    repeatability = {}
    for name, variant in variants.items():
        repeat_checks = []
        first = {row["id"]: row for row in variant["rows"] if row["repeat"] == 0}
        for row in variant["rows"]:
            if row["repeat"] == 0:
                continue
            reference = first[row["id"]]
            path = output / name / (row["id"] + "-r" + str(row["repeat"]) + ".repeat.json")
            subprocess.run(
                [str(driver), "--compare", reference["prefix"], row["prefix"], str(path)],
                check=True,
            )
            check = read_json(path)
            check.update(
                {
                    "id": row["id"],
                    "repeat": row["repeat"],
                    "transcripts_equal": row["transcript"] == reference["transcript"],
                }
            )
            repeat_checks.append(check)
        repeatability[name] = {
            "comparisons": len(repeat_checks),
            "all_tensors_exact": bool(repeat_checks)
            and all(
                stats["exact_differences"] == 0 and stats["nonfinite"] == 0
                for row in repeat_checks
                for stats in row["groups"].values()
            ),
            "all_decisions_exact": all(row["decision_differences"] == 0 for row in repeat_checks),
            "all_transcripts_exact": all(row["transcripts_equal"] for row in repeat_checks),
            "groups": aggregate(repeat_checks),
        }
    decisions_equal = all(row["decision_differences"] == 0 for row in comparisons)
    transcripts_equal = all(row["transcripts_equal"] for row in comparisons)
    inputs_equal = (
        groups["input_features"]["exact_differences"] == 0
        and groups["input_features"]["nonfinite"] == 0
    )
    valid = inputs_equal and decisions_equal
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent full-model recognition from identical PCM: frontend features, "
        "final encoder activations, encoder projection, and every token/duration "
        "logit before greedy selection; original and optimized Q8 Metal runtimes.",
        "criterion": "abs(actual-reference) <= atol + rtol*abs(reference)",
        "instrumentation": "Test-only dyld interposition retains encoder output and full joint "
        "logits through their views, then synchronizes for readback. No model "
        "or kernel arithmetic is replaced; production fusion/optimization "
        "and positional caching remain enabled. Not timing evidence.",
        "source_sha256": {
            str(SOURCE.relative_to(ROOT)): sha256(SOURCE),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
        },
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
        "clips": jobs,
        "repeats": args.repeats,
        "recognitions_per_runtime": len(comparisons),
        "variants": variants,
        "valid_aligned_comparison": valid,
        "input_features_exact": inputs_equal,
        "all_decisions_exact": decisions_equal,
        "all_transcripts_exact": transcripts_equal,
        "total_decisions": sum(row["decisions"] for row in comparisons),
        "total_decision_differences": sum(row["decision_differences"] for row in comparisons),
        "groups": groups,
        "comparisons": comparisons,
        "repeatability": repeatability,
        "kernel_debug": args.kernel_debug,
    }
    report["pass_at_1e-4"] = (
        valid
        and transcripts_equal
        and all(stats["tolerances"][1]["pass"] for stats in groups.values())
    )
    write_json(output / "report.json", report)
    print(
        "Identical inputs:",
        inputs_equal,
        "decisions:",
        decisions_equal,
        "transcripts:",
        transcripts_equal,
    )
    for name, stats in groups.items():
        print(
            name,
            "elements:",
            stats["elements"],
            "max abs:",
            stats["max_absolute_error"],
            "failed elements [1e-5, 1e-4, 1e-3]:",
            [t["failed_elements"] for t in stats["tolerances"]],
        )
    print("Report:", output / "report.json")
    return 0 if report["pass_at_1e-4"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Full-model audit error:", error, file=sys.stderr)
        sys.exit(2)
