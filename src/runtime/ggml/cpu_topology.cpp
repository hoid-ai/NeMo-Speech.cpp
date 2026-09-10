// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "cpu_topology.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace ggml_runtime {

namespace {

int
clamp_positive(long value, int fallback) {
    if (value <= 0)
        return fallback;
    if (value > 1024)
        return 1024;
    return static_cast<int>(value);
}

#if defined(__linux__)

bool
read_long_file(const std::string& path, long& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
        return false;
    long v = 0;
    const bool ok = std::fscanf(f, "%ld", &v) == 1;
    std::fclose(f);
    if (ok)
        out = v;
    return ok;
}

// CPUs this process may actually run on. A container CPU-set or a `taskset`
// wrapper must not be overrun, and neither must be counted as absent cores.
std::vector<int>
allowed_cpus() {
    std::vector<int> cpus;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set))
                cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (long cpu = 0; cpu < n; ++cpu) cpus.push_back(static_cast<int>(cpu));
    }
    return cpus;
}

// Whole CPUs a cgroup CPU-bandwidth quota permits, or 0 when unlimited. A
// quota below one CPU still gets one thread; fractional parallelism only
// buys context switches.
int
cgroup_cpu_quota() {
    // cgroup v2: "<quota> <period>", or "max <period>" when unrestricted.
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char quota[32] = {0};
        long period = 0;
        const bool ok = std::fscanf(f, "%31s %ld", quota, &period) == 2;
        std::fclose(f);
        if (ok && period > 0 && std::string(quota) != "max") {
            const long q = std::strtol(quota, nullptr, 10);
            if (q > 0)
                return std::max(1L, q / period);
        }
        return 0;
    }
    // cgroup v1.
    long quota = 0, period = 0;
    if (read_long_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", quota) &&
        read_long_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", period) && quota > 0 &&
        period > 0) {
        return std::max(1L, quota / period);
    }
    return 0;
}

// Physical cores among `cpus`, counted by distinct (package, core) identity
// so SMT siblings collapse into one. Returns 0 when sysfs carries no usable
// topology.
int
count_physical_cores(const std::vector<int>& cpus) {
    std::set<std::pair<long, long>> cores;
    for (int cpu : cpus) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        long core_id = 0;
        if (!read_long_file(base + "/topology/core_id", core_id))
            return 0;
        long package = 0;
        if (!read_long_file(base + "/topology/physical_package_id", package))
            package = 0;
        cores.insert({package, core_id});
    }
    return static_cast<int>(cores.size());
}

int
detect_threads() {
    const std::vector<int> cpus = allowed_cpus();
    const int n_logical = clamp_positive(static_cast<long>(cpus.size()),
                                         clamp_positive(std::thread::hardware_concurrency(), 4));

    int threads = count_physical_cores(cpus);
    if (threads <= 0)
        threads = n_logical;

    if (const int quota = cgroup_cpu_quota())
        threads = std::min(threads, quota);
    return std::clamp(threads, 1, n_logical);
}

#else  // !__linux__

// No portable physical-core query outside Linux sysfs.
// hardware_concurrency() counts SMT siblings, so halve it when it looks like
// an SMT count and clamp to something sane.
int
detect_threads() {
    const int n_logical = clamp_positive(std::thread::hardware_concurrency(), 4);
    return std::max(1, n_logical >= 4 ? n_logical / 2 : n_logical);
}

#endif  // __linux__

}  // namespace

int
default_compute_threads() {
    static const int n = [] {
        // Escape hatch for benchmarking and for operators who know their
        // placement better than this heuristic does.
        if (const char* env = std::getenv("NEMO_SPEECH_CPU_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0)
                return std::min(v, 1024);
        }
        return detect_threads();
    }();
    return n;
}

}  // namespace ggml_runtime
