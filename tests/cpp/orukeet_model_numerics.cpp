// SPDX-FileCopyrightText: Copyright (c) 2026 Hoid AI
// SPDX-License-Identifier: Apache-2.0

// macOS-only full-model numerical audit. Build this file twice: once as a
// DYLD interposer, once as a C ABI driver/comparator. The preserved runtime
// libraries execute all model arithmetic; this code only retains final
// tensors, synchronizes, copies them to disk, and compares the copies.
#include <dlfcn.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <mach-o/dyld.h>
#include <nemo_speech/asr.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/common/json.h"

namespace {
using Json = nemo_speech::json::Value;

void
require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::string
read_text(const std::string& path) {
    std::ifstream file(path);
    require(bool(file), "Cannot open " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
}  // namespace

#ifdef ORUKEET_MODEL_TAP
namespace {
std::mutex trace_mutex;
std::ofstream trace_index;
std::ofstream trace_data;
size_t trace_offset = 0;
size_t trace_sequence = 0;
bool trace_active = false;

// As in Session::expand_graph_into, a retained view must retain its base too.
// Nothing is recomputed or fed from one runtime into the other runtime.
void
retain(ggml_tensor* tensor) {
    for (auto* t = tensor; t; t = t->view_src) ggml_set_output(t);
}

ggml_tensor*
joint_root(ggml_tensor* tensor) {
    auto* t = tensor;
    while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_CONT))
        t = t->src[0];
    if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != 8198 || t->op != GGML_OP_ADD || !t->src[1] ||
        std::strcmp(t->src[1]->name, "joint.joint_net.2.bias") != 0)
        return nullptr;
    return t;
}

void
dump_tensor(const char* role, ggml_tensor* tensor, const char* backend) {
    require(tensor && ggml_is_contiguous(tensor), std::string("Non-contiguous ") + role);
    require(
        tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32,
        std::string("Unexpected type for ") + role);
    const auto bytes = ggml_nbytes(tensor);
    std::vector<char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    Json::Array shape;
    for (auto n : tensor->ne) shape.emplace_back(double(n));
    Json row(Json::Object{
        {"role", role},
        {"sequence", double(trace_sequence++)},
        {"backend", backend},
        {"shape", std::move(shape)},
        {"type", ggml_type_name(tensor->type)},
        {"offset", double(trace_offset)},
        {"bytes", double(bytes)},
        {"name", tensor->name}});
    trace_index << row.dump() << '\n';
    trace_data.write(data.data(), data.size());
    require(bool(trace_index) && bool(trace_data), "Writing tensor trace failed");
    trace_offset += bytes;
}

ggml_tensor*
tap_argmax(ggml_context* context, ggml_tensor* source) {
    if (auto* logits = joint_root(source))
        retain(logits);
    // A call from this interposing image binds to the original implementation.
    return ggml_argmax(context, source);
}

ggml_tensor*
tap_mul_mat(ggml_context* context, ggml_tensor* weights, ggml_tensor* input) {
    if (std::strcmp(weights->name, "joint.enc.weight") == 0)
        retain(input);
    return ggml_mul_mat(context, weights, input);
}

ggml_status
capture_compute(ggml_backend_t backend, ggml_backend_sched_t scheduler, ggml_cgraph* graph) {
    // Capture graph endpoints before Metal's normal graph optimizer runs.
    ggml_tensor* encoder = nullptr;
    ggml_tensor* projection = nullptr;
    ggml_tensor* logits = nullptr;
    ggml_tensor* tokens = nullptr;
    ggml_tensor* durations = nullptr;
    std::set<ggml_tensor*> features;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_ADD && node->src[0] && node->src[1] &&
            std::strcmp(node->src[1]->name, "joint.enc.bias") == 0) {
            auto* matmul = node->src[0];
            require(
                matmul->op == GGML_OP_MUL_MAT && matmul->src[0] && matmul->src[1] &&
                    std::strcmp(matmul->src[0]->name, "joint.enc.weight") == 0,
                "Unexpected encoder projection topology");
            require(!projection, "More than one encoder projection per graph");
            encoder = matmul->src[1];
            projection = node;
        }
        if (node->op == GGML_OP_ARGMAX) {
            if (auto* root = joint_root(node->src[0])) {
                require(!logits || logits == root, "Multiple joint vectors per graph");
                logits = root;
                if (node->src[0]->ne[0] == 8193)
                    tokens = node;
                else if (node->src[0]->ne[0] == 5)
                    durations = node;
                else
                    throw std::runtime_error("Unexpected Orukeet argmax width");
            }
        }
        for (auto* source : node->src)
            if (source && std::strcmp(source->name, "input.features") == 0)
                features.insert(source);
    }
    std::lock_guard<std::mutex> lock(trace_mutex);
    auto synchronize = [&]() {
        if (scheduler)
            ggml_backend_sched_synchronize(scheduler);
        else
            ggml_backend_synchronize(backend);
    };
    auto backend_name = [&](ggml_tensor* tensor) {
        auto selected =
            scheduler ? ggml_backend_sched_get_tensor_backend(scheduler, tensor) : backend;
        require(selected != nullptr, "Cannot identify tensor execution backend");
        return ggml_backend_name(selected);
    };
    if (trace_active && !features.empty()) {
        synchronize();
        for (auto* feature : features)
            dump_tensor("input_features", feature, backend_name(feature));
    }
    auto status = scheduler ? ggml_backend_sched_graph_compute_async(scheduler, graph)
                            : ggml_backend_graph_compute_async(backend, graph);
    if (status != GGML_STATUS_SUCCESS || !trace_active || (!projection && !logits))
        return status;
    synchronize();
    if (projection) {
        dump_tensor("encoder_output", encoder, backend_name(encoder));
        dump_tensor("encoder_projection", projection, backend_name(projection));
    }
    if (logits) {
        require(tokens && durations, "Missing TDT argmax head");
        require(
            (logits->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "Final joint scores were not retained");
        dump_tensor("joint_logits", logits, backend_name(logits));
        dump_tensor("token_argmax", tokens, backend_name(tokens));
        dump_tensor("duration_argmax", durations, backend_name(durations));
    }
    return status;
}

ggml_status
tap_compute(ggml_backend_t backend, ggml_cgraph* graph) {
    return capture_compute(backend, nullptr, graph);
}

ggml_status
tap_sched_compute(ggml_backend_sched_t scheduler, ggml_cgraph* graph) {
    return capture_compute(nullptr, scheduler, graph);
}
}  // namespace

extern "C" void
orukeet_trace_begin(const char* prefix) {
    std::lock_guard<std::mutex> lock(trace_mutex);
    require(!trace_active, "Nested recognition is unsupported by the audit");
    trace_index.open(std::string(prefix) + ".jsonl");
    trace_data.open(std::string(prefix) + ".bin", std::ios::binary);
    require(bool(trace_index) && bool(trace_data), "Cannot create tensor trace");
    trace_offset = 0;
    trace_sequence = 0;
    trace_active = true;
}

extern "C" void
orukeet_trace_end() {
    std::lock_guard<std::mutex> lock(trace_mutex);
    trace_active = false;
    trace_index.close();
    trace_data.close();
    require(trace_sequence > 0, "No tensors captured: interposition did not work");
}

#define ORUKEET_INTERPOSE(replacement, original)                              \
    __attribute__((used)) static struct {                                     \
        const void* replacement_pointer;                                      \
        const void* original_pointer;                                         \
    } interpose_##original __attribute__((section("__DATA,__interpose"))) = { \
        reinterpret_cast<const void*>(&replacement), reinterpret_cast<const void*>(&original)}

ORUKEET_INTERPOSE(tap_argmax, ggml_argmax);
ORUKEET_INTERPOSE(tap_mul_mat, ggml_mul_mat);
ORUKEET_INTERPOSE(tap_compute, ggml_backend_graph_compute_async);
ORUKEET_INTERPOSE(tap_sched_compute, ggml_backend_sched_graph_compute_async);

#else
namespace {
struct Stats {
    size_t elements = 0;
    size_t nonfinite = 0;
    size_t exact_differences = 0;
    double max_absolute_error = 0;
    Json worst_absolute;
    std::array<size_t, 3> failed{};
    std::array<double, 3> max_ratio{};
    std::array<Json, 3> worst_ratio{};

    void add(float expected, float actual, const Json& location) {
        ++elements;
        if (!std::isfinite(expected) || !std::isfinite(actual)) {
            ++nonfinite;
            for (auto& n : failed) ++n;
            return;
        }
        const double error = std::abs(double(actual) - expected);
        exact_differences += error != 0;
        Json detail = location;
        detail["reference"] = double(expected);
        detail["actual"] = double(actual);
        detail["absolute_error"] = error;
        if (error > max_absolute_error) {
            max_absolute_error = error;
            worst_absolute = detail;
        }
        const double tolerance[] = {1e-5, 1e-4, 1e-3};
        for (size_t i = 0; i < 3; ++i) {
            const double bound = tolerance[i] + tolerance[i] * std::abs(double(expected));
            const double ratio = error / bound;
            failed[i] += error > bound;
            if (ratio > max_ratio[i]) {
                max_ratio[i] = ratio;
                worst_ratio[i] = detail;
            }
        }
    }

    Json json() const {
        Json::Array tolerances;
        const double tolerance[] = {1e-5, 1e-4, 1e-3};
        for (size_t i = 0; i < 3; ++i)
            tolerances.emplace_back(Json::Object{
                {"atol", tolerance[i]},
                {"rtol", tolerance[i]},
                {"failed_elements", double(failed[i])},
                {"pass", failed[i] == 0},
                {"max_error_over_bound", max_ratio[i]},
                {"worst", worst_ratio[i]}});
        return Json::Object{
            {"elements", double(elements)},
            {"nonfinite", double(nonfinite)},
            {"exact_differences", double(exact_differences)},
            {"max_absolute_error", max_absolute_error},
            {"worst_absolute", worst_absolute},
            {"tolerances", std::move(tolerances)}};
    }
};

Json
compare(const std::string& reference, const std::string& candidate) {
    std::ifstream ref_index(reference + ".jsonl"), new_index(candidate + ".jsonl");
    std::ifstream ref_data(reference + ".bin", std::ios::binary);
    std::ifstream new_data(candidate + ".bin", std::ios::binary);
    require(ref_index && new_index && ref_data && new_data, "Cannot open comparison inputs");
    std::map<std::string, Stats> stats;
    std::map<std::string, size_t> counts;
    std::array<std::vector<int32_t>, 2> expected_tokens, expected_durations;
    size_t decisions = 0, decision_differences = 0, rows = 0;
    Json first_decision_difference;
    std::string ref_line, new_line;
    while (std::getline(ref_index, ref_line)) {
        require(bool(std::getline(new_index, new_line)), "Candidate trace is shorter");
        auto ref = Json::parse(ref_line), actual = Json::parse(new_line);
        for (const char* field : {"role", "shape", "type", "bytes", "sequence"})
            require(
                ref.at(field).dump() == actual.at(field).dump(),
                std::string("Trace mismatch in ") + field + " at " + std::to_string(rows));
        const auto role = ref.at("role").string();
        if (role != "input_features") {
            for (const auto* row : {&ref, &actual})
                require(
                    row->at("backend").string().rfind("MTL", 0) == 0 ||
                        row->at("backend").string().find("Metal") != std::string::npos,
                    "Model output did not execute on Metal");
        }
        const size_t bytes = size_t(ref.at("bytes").number());
        require(bytes % sizeof(float) == 0, "Invalid tensor byte size");
        std::vector<float> old_values(bytes / 4), new_values(bytes / 4);
        ref_data.seekg(size_t(ref.at("offset").number()));
        new_data.seekg(size_t(actual.at("offset").number()));
        ref_data.read(reinterpret_cast<char*>(old_values.data()), bytes);
        new_data.read(reinterpret_cast<char*>(new_values.data()), bytes);
        require(ref_data && new_data, "Truncated tensor data");
        if (role == "joint_logits") {
            require(
                old_values.size() % 8198 == 0 && ref.at("shape").array()[0].number() == 8198,
                "Unexpected Orukeet joint shape");
            const std::vector<float>* values[] = {&old_values, &new_values};
            for (size_t variant = 0; variant < 2; ++variant) {
                expected_tokens[variant].clear();
                expected_durations[variant].clear();
                for (size_t start = 0; start < old_values.size(); start += 8198) {
                    auto first = values[variant]->begin() + start;
                    auto duration = first + 8193;
                    expected_tokens[variant].push_back(
                        int32_t(std::max_element(first, duration) - first));
                    expected_durations[variant].push_back(
                        int32_t(std::max_element(duration, first + 8198) - duration));
                }
            }
        }
        const auto invocation = counts[role]++;
        for (size_t i = 0; i < old_values.size(); ++i) {
            Json location(Json::Object{{"invocation", double(invocation)}, {"index", double(i)}});
            if (ref.at("type").string() == "i32") {
                int32_t a, b;
                std::memcpy(&a, &old_values[i], 4);
                std::memcpy(&b, &new_values[i], 4);
                auto& expected = role == "token_argmax" ? expected_tokens : expected_durations;
                require(
                    expected[0].size() == old_values.size() &&
                        expected[1].size() == new_values.size() && a == expected[0][i] &&
                        b == expected[1][i],
                    "Captured scores do not reproduce the runtime argmax: " + role);
                ++decisions;
                if (a != b) {
                    ++decision_differences;
                    if (first_decision_difference.is_null()) {
                        location["role"] = role;
                        location["reference"] = int(a);
                        location["actual"] = int(b);
                        first_decision_difference = location;
                    }
                }
            } else {
                std::string category = role;
                if (role == "joint_logits")
                    category = i % 8198 < 8193 ? "token_logits" : "duration_logits";
                stats[category].add(old_values[i], new_values[i], location);
            }
        }
        ++rows;
    }
    require(!std::getline(new_index, new_line), "Candidate trace is longer");
    require(
        counts["input_features"] > 0 && counts["encoder_output"] > 0 &&
            counts["encoder_projection"] == counts["encoder_output"] &&
            counts["joint_logits"] > 0 && counts["joint_logits"] == counts["token_argmax"] &&
            counts["joint_logits"] == counts["duration_argmax"],
        "Missing full-model trace endpoints");
    Json::Object groups;
    for (const auto& [name, value] : stats) groups[name] = value.json();
    Json::Object role_counts;
    for (const auto& [name, value] : counts) role_counts[name] = double(value);
    return Json::Object{
        {"groups", std::move(groups)},
        {"rows", double(rows)},
        {"counts", std::move(role_counts)},
        {"decisions", double(decisions)},
        {"decision_differences", double(decision_differences)},
        {"first_decision_difference", first_decision_difference}};
}

void
check(nemo_speech_asr_status status) {
    require(
        status == NEMO_SPEECH_ASR_OK,
        nemo_speech_asr_last_error() ? nemo_speech_asr_last_error() : "ASR error");
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--compare") {
            auto result = compare(argv[2], argv[3]);
            std::ofstream out(argv[4]);
            out << result.dump(2) << '\n';
            require(bool(out), "Cannot write comparison report");
            return 0;
        }
        require(
            argc == 5,
            "Usage: audit MODEL JOBS_JSON OUTPUT_DIR REPEATS\n"
            "       audit --compare REFERENCE_PREFIX CANDIDATE_PREFIX REPORT_JSON");
        auto begin =
            reinterpret_cast<void (*)(const char*)>(dlsym(RTLD_DEFAULT, "orukeet_trace_begin"));
        auto end = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "orukeet_trace_end"));
        require(begin && end, "The tensor interposer must be loaded with DYLD_INSERT_LIBRARIES");
        Json::Array libraries;
        for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
            std::string path = _dyld_get_image_name(i);
            if (path.find("libggml") != std::string::npos ||
                path.find("libnemo_speech") != std::string::npos)
                libraries.emplace_back(path);
        }
        std::cout << Json(Json::Object{{"libraries", std::move(libraries)}}).dump() << std::endl;
        nemo_speech_asr_backend_config backend{};
        backend.size = sizeof(backend);
        backend.gpu = 0;
        nemo_speech_asr_model_config model{};
        model.size = sizeof(model);
        model.path = argv[1];
        nemo_speech_asr_recognizer_config config{};
        config.size = sizeof(config);
        config.backend = &backend;
        config.model = &model;
        nemo_speech_asr_recognizer* raw_recognizer = nullptr;
        check(nemo_speech_asr_create(&config, &raw_recognizer));
        std::unique_ptr<nemo_speech_asr_recognizer, decltype(&nemo_speech_asr_destroy)> recognizer(
            raw_recognizer, nemo_speech_asr_destroy);
        if (std::getenv("ORUKEET_AUDIT_KERNEL_LOG"))
            ggml_log_set(
                [](ggml_log_level, const char* message, void*) {
                    if (message)
                        std::fputs(message, stderr);
                },
                nullptr);
        auto options = nemo_speech_asr_recognition_options_default();
        options.enable_word_time_offsets = true;
        options.enable_automatic_punctuation = true;
        auto jobs = Json::parse(read_text(argv[2]));
        const int repeats = std::stoi(argv[4]);
        require(repeats > 0, "REPEATS must be positive");
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (const auto& job : jobs.array()) {
                const auto id = job.at("id").string();
                std::ifstream pcm(job.at("pcm").string(), std::ios::binary | std::ios::ate);
                require(bool(pcm), "Cannot read PCM for " + id);
                const auto bytes = size_t(pcm.tellg());
                require(bytes > 0 && bytes % sizeof(float) == 0, "Bad PCM size for " + id);
                std::vector<float> samples(bytes / sizeof(float));
                pcm.seekg(0);
                pcm.read(reinterpret_cast<char*>(samples.data()), bytes);
                require(bool(pcm), "Truncated PCM for " + id);
                const auto prefix = std::string(argv[3]) + "/" + id + "-r" + std::to_string(repeat);
                begin(prefix.c_str());
                nemo_speech_asr_result* raw_result = nullptr;
                check(nemo_speech_asr_recognize_f32(
                    recognizer.get(), &options, samples.data(), samples.size(), 16000,
                    &raw_result));
                std::unique_ptr<nemo_speech_asr_result, decltype(&nemo_speech_asr_result_destroy)>
                    result(raw_result, nemo_speech_asr_result_destroy);
                end();
                const auto* transcript = nemo_speech_asr_result_transcript(result.get(), 0);
                std::cout << Json(Json::Object{
                                      {"id", id},
                                      {"repeat", repeat},
                                      {"prefix", prefix},
                                      {"samples", double(samples.size())},
                                      {"transcript", transcript ? transcript : ""}})
                                 .dump()
                          << std::endl;
            }
        }
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Full-model numerical audit failed: " << error.what() << std::endl;
        return 2;
    }
}
#endif
