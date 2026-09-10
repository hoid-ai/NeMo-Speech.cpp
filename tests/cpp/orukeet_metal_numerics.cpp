// SPDX-FileCopyrightText: Copyright (c) 2026 Hoid AI
// SPDX-License-Identifier: Apache-2.0

// Reuse the pinned ggml graph fixtures, including their deliberate aliases and
// scheduler orderings. This executable never changes a production kernel.
#define main ggml_backend_ops_original_main
#include "../../ggml/tests/test-backend-ops.cpp"
#undef main

#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <ggml-cpu.h>
#include <ggml-metal.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "../../src/common/json.h"

namespace {
using Json = nemo_speech::json::Value;

void
require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}

std::string
library_path(void* symbol) {
    Dl_info info{};
    require(dladdr(symbol, &info) != 0 && info.dli_fname, "Cannot identify loaded library");
    return info.dli_fname;
}

std::string
hex_digest(CC_SHA256_CTX& digest) {
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(bytes, &digest);
    std::ostringstream text;
    for (auto byte : bytes) text << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return text.str();
}

// Use integer PRNG output rather than implementation-dependent distributions.
// Override every leaf after the fixture has installed its aliases. All graph
// inputs are hashed after quantization, so the reference must receive identical
// bytes, shapes and types. Intermediate buffers are not used as test inputs.
std::string
initialize_leaves(ggml_context* ctx, uint32_t seed, const std::string& profile) {
    std::mt19937 generator(seed);
    CC_SHA256_CTX digest;
    CC_SHA256_Init(&digest);
    size_t leaf = 0;
    for (auto t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE || t->view_src)
            continue;
        require(ggml_is_contiguous(t), "Non-contiguous input leaf");
        std::vector<float> values(ggml_nelements(t));
        for (size_t i = 0; i < values.size(); ++i) {
            const float u = float(generator() >> 8) * (1.0f / 16777216.0f);
            float value = 2.0f * u - 1.0f;
            if (profile == "saturation")
                value *= 8.0f;
            if (profile == "cancellation") {
                // Alternating signs against almost constant positive inputs.
                // Half/Q8 weights round the small perturbation away.
                value = (leaf % 2 == 0 && i % 2 ? -1.0f : 1.0f) * (1.0f + value * 1e-5f);
            }
            if (profile == "half-boundaries") {
                const float edges[] = {0.0f,     -0.0f,     0x1p-24f,        -0x1p-24f,
                                       0x1p-14f, -0x1p-14f, 1.0f + 0x1p-11f, -1.0f - 0x1p-11f};
                value = edges[i % 8];
                if (i % 3 == 1)
                    value = std::nextafter(value, INFINITY);
                if (i % 3 == 2)
                    value = std::nextafter(value, -INFINITY);
            }
            values[i] = value;
        }
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F32) {
            require(bytes.size() == values.size() * sizeof(float), "Bad F32 input size");
            std::memcpy(bytes.data(), values.data(), bytes.size());
        } else {
            require(
                t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0,
                "Unexpected audit input type");
            ggml_quantize_chunk(
                t->type, values.data(), bytes.data(), 0, ggml_nelements(t) / t->ne[0], t->ne[0],
                nullptr);
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        CC_SHA256_Update(&digest, &t->type, sizeof(t->type));
        CC_SHA256_Update(&digest, t->ne, sizeof(t->ne));
        CC_SHA256_Update(&digest, bytes.data(), static_cast<CC_LONG>(bytes.size()));
        ++leaf;
    }
    require(leaf > 0, "No audit inputs");
    return hex_digest(digest);
}

std::string
group_for(test_case* tc) {
    if (dynamic_cast<test_pos_projection_cache*>(tc))
        return "position_projection";
    if (dynamic_cast<test_attention_input_views*>(tc))
        return "attention_views";
    if (dynamic_cast<test_relative_shift_view*>(tc))
        return "relative_shift";
    if (dynamic_cast<test_short_conv_fusion*>(tc))
        return "convolution_fusion";
    if (dynamic_cast<test_lstm_gates_fusion*>(tc))
        return "lstm_fusion";
    if (dynamic_cast<test_im2col*>(tc))
        return "im2col";
    if (auto p = dynamic_cast<test_cpy*>(tc)) {
        if (p->type_src == GGML_TYPE_F32 && p->type_dst == GGML_TYPE_F32)
            return "f32_copy";
    }
    if (auto p = dynamic_cast<test_dup*>(tc)) {
        if (p->type == GGML_TYPE_F32)
            return "f32_copy";
    }
    if (auto p = dynamic_cast<test_cont*>(tc)) {
        if (p->type == GGML_TYPE_F32)
            return "f32_copy";
    }
    if (dynamic_cast<test_mul_mat_short_f16*>(tc))
        return "short_f16_dot";
    if (auto p = dynamic_cast<test_mul_mat*>(tc)) {
        if (p->type_a == GGML_TYPE_Q8_0 && p->type_b == GGML_TYPE_F32 && p->n == 1 &&
            (((p->m == 640 || p->m == 2560 || p->m == 8198) && p->k == 640) ||
             (p->m == 129 && (p->k == 32 || p->k == 128 || p->k == 1024 || p->k == 1056)))) {
            return "q8_matvec";
        }
    }
    return "";
}

struct Snapshot {
    std::string id;
    std::string input_sha256;
    std::vector<std::vector<float>> outputs;
};

template <typename T>
void
put(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template <typename T>
T
get(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(stream), "Truncated reference snapshot");
    return value;
}
void
put_string(std::ostream& stream, const std::string& value) {
    put(stream, uint64_t(value.size()));
    stream.write(value.data(), value.size());
}
std::string
get_string(std::istream& stream) {
    const auto size = get<uint64_t>(stream);
    require(size < 65536, "Invalid reference string size");
    std::string value(size, '\0');
    stream.read(value.data(), size);
    require(bool(stream), "Truncated reference string");
    return value;
}
void
write_snapshot(std::ostream& stream, const Snapshot& value) {
    put_string(stream, value.id);
    put_string(stream, value.input_sha256);
    put(stream, uint64_t(value.outputs.size()));
    for (const auto& output : value.outputs) {
        put(stream, uint64_t(output.size()));
        stream.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    require(bool(stream), "Cannot write reference snapshot");
}
Snapshot
read_snapshot(std::istream& stream) {
    Snapshot value;
    value.id = get_string(stream);
    value.input_sha256 = get_string(stream);
    const auto count = get<uint64_t>(stream);
    require(count > 0 && count < 16, "Invalid reference output count");
    for (size_t i = 0; i < count; ++i) {
        const auto size = get<uint64_t>(stream);
        require(size > 0 && size <= 100000000, "Invalid reference output size");
        std::vector<float> output(size);
        stream.read(reinterpret_cast<char*>(output.data()), size * sizeof(float));
        require(bool(stream), "Truncated reference output");
        value.outputs.push_back(std::move(output));
    }
    return value;
}

Json
compare(
    const std::vector<std::vector<float>>& actual, const std::vector<std::vector<float>>& reference,
    double atol, double rtol) {
    require(actual.size() == reference.size(), "Reference output count changed");
    uint64_t elements = 0, failures = 0, nonfinite = 0, relaxed_failures = 0;
    double max_abs = 0, max_relative = 0, max_ratio = 0;
    double max_abs_at_zero = 0;
    Json worst;
    for (size_t output = 0; output < actual.size(); ++output) {
        require(
            actual[output].size() == reference[output].size(), "Reference output shape changed");
        for (size_t i = 0; i < actual[output].size(); ++i) {
            const double a = actual[output][i], b = reference[output][i];
            ++elements;
            if (!std::isfinite(a) || !std::isfinite(b)) {
                ++nonfinite;
                ++failures;
                ++relaxed_failures;
                continue;
            }
            const double error = std::abs(a - b);
            const double tolerance = atol + rtol * std::abs(b);
            const double ratio = tolerance > 0 ? error / tolerance : (error == 0 ? 0 : DBL_MAX);
            failures += error > tolerance;
            relaxed_failures += error > 1e-4 + 1e-4 * std::abs(b);
            max_abs = std::max(max_abs, error);
            if (b != 0)
                max_relative = std::max(max_relative, error / std::abs(b));
            else
                max_abs_at_zero = std::max(max_abs_at_zero, error);
            if (ratio > max_ratio) {
                max_ratio = ratio;
                worst = Json::Object{
                    {"output", int(output)}, {"index", double(i)},      {"actual", a},
                    {"reference", b},        {"absolute_error", error}, {"bound", tolerance}};
            }
        }
    }
    return Json::Object{
        {"elements", double(elements)},
        {"failed_elements", double(failures)},
        {"nonfinite_elements", double(nonfinite)},
        {"max_absolute_error", max_abs},
        {"max_relative_error_nonzero_reference", max_relative},
        {"max_absolute_error_at_zero_reference", max_abs_at_zero},
        {"max_error_over_bound", max_ratio},
        {"worst", worst},
        {"failed_elements_at_1e_4", double(relaxed_failures)},
        {"passed", failures == 0}};
}

bool
exact_zero(const std::vector<std::vector<float>>& outputs) {
    for (const auto& output : outputs)
        for (float value : output)
            if (value != 0)
                return false;
    return true;
}

void
check_comparator() {
    require(
        compare({{0.0f, 1.0f}}, {{0.0f, 1.0f}}, 0, 0).at("passed").boolean(),
        "Exact comparison self-check failed");
    require(
        compare({{5e-6f}}, {{0.0f}}, 1e-5, 1e-5).at("passed").boolean(),
        "Absolute tolerance near zero self-check failed");
    require(
        !compare({{3e-5f}}, {{0.0f}}, 1e-5, 1e-5).at("passed").boolean(),
        "Absolute tolerance rejection self-check failed");
    require(
        !compare({{1.25f}}, {{1.0f}}, 0, 0.2).at("passed").boolean(),
        "Reference-relative tolerance self-check failed");
    require(
        !compare({{NAN}}, {{0.0f}}, 1e-5, 1e-5).at("passed").boolean(),
        "NaN rejection self-check failed");
    require(
        !compare({{INFINITY}}, {{INFINITY}}, 1e-5, 1e-5).at("passed").boolean(),
        "Non-finite rejection self-check failed");
}

// ggml's Q8 CPU matmul additionally quantizes its F32 activations to Q8.
// For the actual batch-one decoder shapes, also check a reference with the
// same inputs as Metal: dequantized Q8 weights times unquantized F32 inputs,
// accumulated in FP64 and rounded to the F32 output type.
std::vector<std::vector<float>>
q8_fp64_reference(test_case* tc, ggml_tensor* out) {
    auto mm = dynamic_cast<test_mul_mat*>(tc);
    if (!mm || mm->type_a != GGML_TYPE_Q8_0 || mm->type_b != GGML_TYPE_F32 || mm->n != 1 ||
        mm->bs != std::array<int64_t, 2>{1, 1} || mm->nr != std::array<int64_t, 2>{1, 1})
        return {};
    auto weights = tensor_to_float(out->src[0]);
    auto input = tensor_to_float(out->src[1]);
    require(
        input.size() == size_t(mm->k) && weights.size() == size_t(mm->m * mm->k),
        "Unexpected Q8 reference layout");
    std::vector<float> expected(mm->m);
    for (int64_t row = 0; row < mm->m; ++row) {
        double sum = 0;
        for (int64_t k = 0; k < mm->k; ++k)
            sum += double(weights[row * mm->k + k]) * double(input[k]);
        expected[row] = float(sum);
    }
    return {std::move(expected)};
}

int
audit(int argc, char** argv) {
    check_comparator();
    // record SNAPSHOT REPORT SEED PROFILE
    // compare ORIGINAL_SNAPSHOT PREVIOUS_SNAPSHOT REPORT SEED PROFILE ATOL RTOL
    const bool recording = argc > 1 && std::string(argv[1]) == "record";
    require(
        (recording && argc == 6) || (!recording && argc == 9 && std::string(argv[1]) == "compare"),
        "Usage: record SNAPSHOT REPORT SEED PROFILE | compare ORIGINAL PREVIOUS REPORT SEED "
        "PROFILE ATOL RTOL");
    const int report_at = recording ? 3 : 4;
    const uint32_t seed = std::stoul(argv[report_at + 1]);
    const std::string profile = argv[report_at + 2];
    require(
        profile == "uniform" || profile == "cancellation" || profile == "saturation" ||
            profile == "half-boundaries",
        "Unknown fixture profile");
    const double atol = recording ? 1e-5 : std::stod(argv[7]);
    const double rtol = recording ? 1e-5 : std::stod(argv[8]);
    require(
        std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0, "Invalid tolerance");
    std::ofstream report(argv[report_at]);
    require(bool(report), "Cannot open report");
    std::ofstream sink;
    std::ifstream original, previous;
    constexpr uint64_t magic = 0x4f52554b45544531;
    if (recording) {
        sink.open(argv[2], std::ios::binary);
        require(bool(sink), "Cannot open snapshot output");
        put(sink, magic);
    } else {
        original.open(argv[2], std::ios::binary);
        previous.open(argv[3], std::ios::binary);
        require(
            get<uint64_t>(original) == magic && get<uint64_t>(previous) == magic,
            "Wrong reference snapshot format");
    }

    ggml_backend_ptr metal(ggml_backend_metal_init());
    ggml_backend_ptr cpu(recording ? nullptr : ggml_backend_cpu_init());
    require(bool(metal) && (recording || bool(cpu)), "Backend initialization failed");
    if (cpu) {
        ggml_backend_cpu_set_use_ref(cpu.get(), true);
        ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    }
    report << Json(
                  Json::Object{
                      {"kind", "metadata"},
                      {"seed", double(seed)},
                      {"profile", profile},
                      {"atol", atol},
                      {"rtol", rtol},
                      {"device", ggml_backend_name(metal.get())},
                      {"metal_library",
                       library_path(reinterpret_cast<void*>(ggml_backend_metal_init))},
                      {"cpu_library", library_path(reinterpret_cast<void*>(ggml_backend_cpu_init))},
                      {"base_library", library_path(reinterpret_cast<void*>(ggml_new_tensor))}})
                  .dump()
           << '\n';

    auto cases = make_test_cases_eval();
    // Existing Q8 fixtures also cover broadcast batches. Include the actual
    // batch-one decoder shapes explicitly.
    for (int64_t m : {640, 2560, 8198})
        cases.emplace_back(
            new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, m, 1, 640, {1, 1}, {1, 1}));
    for (int64_t k : {32, 128, 1024, 1056})
        cases.emplace_back(
            new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 129, 1, k, {1, 1}, {1, 1}));

    size_t count = 0;
    bool passed = true;
    for (auto& tc : cases) {
        const auto group = group_for(tc.get());
        if (group.empty())
            continue;
        const std::string id = group + ":" + tc->vars();
        ggml_context_ptr ctx(
            ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true}));
        require(bool(ctx), "Context allocation failed");
        tc->mode = MODE_TEST;
        tc->gf = ggml_new_graph(ctx.get());
        auto out = tc->build_graph(ctx.get());
        ggml_build_forward_expand(tc->gf, out);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), metal.get()));
        require(bool(buffer), "Tensor allocation failed for " + id);
        tc->initialize_tensors(ctx.get());
        const auto input_sha =
            initialize_leaves(ctx.get(), seed + uint32_t(count * 65537), profile);
        auto outputs = tc->fusion_test_nodes();
        if (outputs.empty())
            outputs.push_back(out);
        std::vector<int> indices;
        auto nodes = ggml_graph_nodes(tc->gf);
        for (auto t : outputs) {
            auto found = std::find(nodes, nodes + ggml_graph_n_nodes(tc->gf), t);
            require(found < nodes + ggml_graph_n_nodes(tc->gf), "Missing graph output");
            indices.push_back(int(found - nodes));
        }
        struct ggml_backend_graph_copy cpu_copy {};
        if (cpu) {
            cpu_copy = ggml_backend_graph_copy(cpu.get(), tc->gf);
            require(cpu_copy.buffer != nullptr, "CPU graph copy failed");
        }
        require(
            ggml_backend_graph_compute(metal.get(), tc->gf) == GGML_STATUS_SUCCESS,
            "Metal graph failed for " + id);
        Snapshot current{id, input_sha, {}};
        for (auto t : outputs) current.outputs.push_back(tensor_to_float(t));
        Json::Object row{
            {"kind", "case"}, {"id", id}, {"group", group}, {"input_sha256", input_sha}};
        const bool expect_zero = dynamic_cast<test_relative_shift_view*>(tc.get()) != nullptr;
        if (expect_zero)
            row["metal_exact_zero"] = exact_zero(current.outputs);
        if (recording) {
            write_snapshot(sink, current);
        } else {
            auto old = read_snapshot(original);
            auto prev = read_snapshot(previous);
            require(
                old.id == id && prev.id == id && old.input_sha256 == input_sha &&
                    prev.input_sha256 == input_sha,
                "Reference graph/input mismatch for " + id);
            require(
                ggml_backend_graph_compute(cpu.get(), cpu_copy.graph) == GGML_STATUS_SUCCESS,
                "CPU graph failed for " + id);
            std::vector<std::vector<float>> cpu_outputs;
            auto cpu_nodes = ggml_graph_nodes(cpu_copy.graph);
            for (int i : indices) cpu_outputs.push_back(tensor_to_float(cpu_nodes[i]));
            ggml_backend_graph_copy_free(cpu_copy);
            row["vs_original_metal"] = compare(current.outputs, old.outputs, atol, rtol);
            row["vs_previous_metal"] = compare(current.outputs, prev.outputs, atol, rtol);
            row["vs_cpu"] = compare(current.outputs, cpu_outputs, atol, rtol);
            row["original_metal_vs_cpu"] = compare(old.outputs, cpu_outputs, atol, rtol);
            row["previous_metal_vs_cpu"] = compare(prev.outputs, cpu_outputs, atol, rtol);
            const auto precise = q8_fp64_reference(tc.get(), out);
            if (!precise.empty()) {
                row["vs_fp64"] = compare(current.outputs, precise, atol, rtol);
                row["original_metal_vs_fp64"] = compare(old.outputs, precise, atol, rtol);
                row["previous_metal_vs_fp64"] = compare(prev.outputs, precise, atol, rtol);
            }
            if (expect_zero) {
                row["cpu_exact_zero"] = exact_zero(cpu_outputs);
                row["original_exact_zero"] = exact_zero(old.outputs);
                row["previous_exact_zero"] = exact_zero(prev.outputs);
                passed &=
                    exact_zero(cpu_outputs) && exact_zero(old.outputs) && exact_zero(prev.outputs);
            }
            passed &= row["vs_original_metal"].at("passed").boolean();
            passed &= row["vs_previous_metal"].at("passed").boolean();
        }
        if (expect_zero)
            passed &= exact_zero(current.outputs);
        report << Json(std::move(row)).dump() << '\n';
        report.flush();
        ++count;
        if (count % 50 == 0)
            std::cerr << count << " numerical cases completed\n";
    }
    require(count > 0, "No numerical cases selected");
    if (!recording)
        require(original.peek() == EOF && previous.peek() == EOF, "Unused reference cases");
    report << Json(Json::Object{
                       {"kind", "summary"},
                       {"cases", double(count)},
                       {"metal_comparison_passed", passed}})
                  .dump()
           << '\n';
    return passed ? 0 : 1;
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        return audit(argc, argv);
    }
    catch (const std::exception& error) {
        std::cerr << "Numerical audit failed: " << error.what() << '\n';
        return 2;
    }
}
