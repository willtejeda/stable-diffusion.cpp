#undef NDEBUG
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    // Full CTest runs must not acquire a GPU implicitly. Run explicitly under
    // the workstation's GPU lease, with exclusive access for memory assertions.
    const char* enabled = std::getenv("SD_RUN_GPU_TESTS");
    if (!enabled || std::strcmp(enabled, "1") != 0) return 77;
    auto* backend = ggml_backend_cuda_init(0);
    assert(backend);
    ggml_backend_cuda_set_buffer_cache_enabled(backend, true);

    constexpr int k = 512, m = 512, n = 32;
    ggml_init_params params = {ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(16, false), nullptr, true};
    auto* ctx = ggml_init(params);
    assert(ctx);
    auto* weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    auto* output = ggml_mul_mat(ctx, weights, input);
    auto* graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    std::vector<ggml_fp16_t> w(k * m, ggml_fp32_to_fp16(1.0f));
    std::vector<float> x(k * n), y(m * n);
    ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(w[0]));

    bool reclaimed = false;
    for (float value : {1.0f, 2.0f, -1.0f, 0.5f}) {
        std::fill(x.begin(), x.end(), value);
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(x[0]));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        size_t before, after, total;
        ggml_backend_cuda_get_device_memory(0, &before, &total);
        ggml_backend_cuda_clear_buffer_cache(backend);
        ggml_backend_cuda_get_device_memory(0, &after, &total);
        reclaimed |= after > before;
        ggml_backend_tensor_get(output, y.data(), 0, y.size() * sizeof(y[0]));
        for (float actual : y) assert(std::fabs(actual - k * value) < 0.01f);
        // Repeated clear must be idempotent and leave live weights/results valid.
        ggml_backend_cuda_clear_buffer_cache(backend);
    }
    assert(reclaimed && "cache clear must return idle operation scratch to the device");
    std::puts("CUDA cache clear reclaims scratch; repeated matmul preserves live tensors");
    ggml_backend_buffer_free(buffer);
    ggml_backend_cuda_clear_buffer_cache(backend);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
