// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ggml_runtime {

// Threads to hand ggml's CPU backend when the caller has no preference: one
// per physical core, with SMT siblings collapsed.
//
// This is neither the logical-CPU count nor a performance-core count. On a
// hybrid part (Intel P/E, ARM big.LITTLE) the obvious worry is that ggml's
// equal per-op split hands an efficiency core the same slice as a
// performance core and stalls the barrier behind it. Measured on an i7-13700
// (8 P-cores + 8 E-cores) that worry is misplaced for this workload: MUL_MAT
// carries ~78% of encoder time and distributes its chunks through an atomic
// counter, so fast cores simply claim more of them. Restricting the pool to
// the 8 performance cores cost ~8% against one thread per physical core, and
// oversubscribing to all 24 logical CPUs bought nothing back.
//
// Resolution order:
//   1. one thread per physical core, when sysfs topology is legible;
//   2. the CPU count reported by the C++ runtime otherwise.
// Any cgroup CPU quota and the caller's CPU-affinity mask cap the result, so
// a container CPU-set or a `taskset` wrapper is never overrun.
//
// Computed once and cached; always >= 1.
int default_compute_threads();

}  // namespace ggml_runtime
