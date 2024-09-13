#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <functional>
#include <cstddef>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <utility>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, float* a, float* b, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N * K) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        //fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, M);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.a->data, a, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a)); // cuda requires copy the data directly to device
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, b, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));  // cuda requires copy the data directly to device
    }
}

struct ggml_cgraph * build_graph(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // zT = x @ yT
    struct ggml_tensor * result = ggml_mul_mat(ctx0, model.a, ggml_cont(ctx0, model.b));

    // z = (zT)T
    ggml_build_forward_expand(gf, ggml_cont(ctx0, ggml_transpose(ctx0, result)));

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute(const test_model & model, ggml_gallocr_t allocr, int n_threads) {
    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}


static void ggml_vec_dot_f16(const int n, float * s, float * x, float * y) {
    float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    *s = sumf;
}

static void gemm_f16_out_f32(int m, int n, int k,
                             float * A,
                             float * B,
                             float * C,
                             const int ith, const int nth) {
    // does not seem to make a difference
    int m0, m1, n0, n1;
    // patches per thread
    if (m > n) {
        n0 = 0;
        n1 = n;

        // total patches in dst
        const int np = m;

        // patches per thread
        const int dp = (np + nth - 1)/nth;

        // patch range for this thread
        m0 = dp*ith;
        m1 = std::min(m0 + dp, np);
    } else {
        m0 = 0;
        m1 = m;

        // total patches in dst
        const int np = n;

        // patches per thread
        const int dp = (np + nth - 1)/nth;

        // patch range for this thread
        n0 = dp*ith;
        n1 = std::min(n0 + dp, np);
    }

    // block-tiling attempt
    int64_t blck_n = 16;
    int64_t blck_m = 16;

    for (int j = n0; j < n1; j+=blck_n) {
        for (int i = m0; i < m1; i+=blck_m) {
            // printf("i j k => %d %d %d\n", i, j, K);
            for (int ii = i; ii < i + blck_m && ii < m1; ii++) {
                for (int jj = j; jj < j + blck_n && jj < n1; jj++) {
                    ggml_vec_dot_f16(k,
                                    C + ii*n + jj,
                                    A + ii * k,
                                    B + jj * k);
                }
            }
        }
    }
}

void perform_gemm_test(float* a, float* b, float* expected, int M, int N, int K) {
    printf("\nPerforming gemm_f16_out_f32 test:\n");

    std::vector<float> gemm_out(M * N);
    gemm_f16_out_f32(M, N, K, a, b, gemm_out.data(), 0, 1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1ff,", gemm_out[i * N + j]);
        }
        printf("\n");
    }

    bool passed = true;

    for(int i = 0; i < M * N; i++) {
        if(gemm_out[i] != expected[i]) {
            passed = false;
            break;
        }
    }

    printf("gemm_mult (%i): %s\n", (M * N), passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
}

std::vector<float> perform_ggml_mul_mat(float* matrixA, float* matrixB, int M, int N, int K, int n_threads) {
    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, true);
    
    ggml_gallocr_t allocr = NULL;
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    //create the worst case graph for memory usage estimation
    struct ggml_cgraph * gf = build_graph(model);

    // compute the required memory
    ggml_gallocr_reserve(allocr, gf);    
    struct ggml_tensor * result = compute(model, allocr, n_threads);
    std::vector<float> out_data(ggml_nelements(result));
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // free memory
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return out_data;
}

bool are_vectors_bit_exact(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    // compare float vectors by bits
    if (vec1.size() != vec2.size()) {
        return false;
    }
    return std::memcmp(vec1.data(), vec2.data(), vec1.size() * sizeof(float)) == 0;
}

std::pair<float*, float*> generate_random_matrices(int M, int N, int K, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    float* matrixA = new float[M * K];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            matrixA[i * K + j] = dis(gen);
        }
    }

    float* matrixB = new float[N * K];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            matrixB[i * K + j] = dis(gen);
        }
    }
    return std::make_pair(matrixA, matrixB);
}

std::string hash_vector_float_to_string(const std::vector<float>& vec) {
    std::hash<float> hasher;
    std::stringstream ss;
    for (const auto& elem : vec) {
        ss << std::hex << std::setw(8) << std::setfill('0') << hasher(elem);
    }
    std::string hash_str = ss.str();
    hash_str = hash_str.substr(0, 32);
    return hash_str;
}

int main(void)
{
    printf("====================== APUS Network Matrix Multiplication Determinism GPU Test ======================\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    ggml_time_init();
    
    // generate two matrix
    const int M = 100, N = 100, K = 100, n_threads = 10;
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [matrixA, matrixB] = generate_random_matrices(M, N, K, seed);

    // first compute
    std::vector<float> base_data = perform_ggml_mul_mat(matrixA, matrixB, M, N, K, n_threads);
    std::string base_hash = hash_vector_float_to_string(base_data);

    // print param
    printf("Number of threads: %d\n", n_threads);
    printf("Seed to generate matrix: %d\n", seed);
    printf("Matrix A (%d x %d):\n", M, K);
    printf("Matrix B (%d x %d):\n", N, K);
    printf("Result Matrix (%d x %d):\n", M, N);

    printf("[============================================Test Results============================================]\n");
    // test 10 rounds
    bool all_passed = true;
    for (int i = 1; i <= 20; i++) {
        std::vector<float> comp_data = perform_ggml_mul_mat(matrixA, matrixB, M, N, K, n_threads);
        std::string comp_hash = hash_vector_float_to_string(comp_data);
        if (are_vectors_bit_exact(base_data, comp_data)) {
            printf("%s Round%d\t\033[32mPASSED\033[0m\t|\033[32m%s\033[0m|\033[32m%s\033[0m|\t(Binary MATCH: 100%%)\n", u8"\U00002705", i, base_hash.c_str(), comp_hash.c_str());
        } else {
            printf("%s Round%d\t\033[31mFAILED\033[0m\t|%s|%s|\n", u8"\U0000274E", i, base_hash.c_str(), comp_hash.c_str());
            all_passed = false;
        }
    }
    printf("[====================================================================================================]\n");

    // free mem
    delete[] matrixA;
    delete[] matrixB;

    if (all_passed) {
        printf("Final Result: \033[32mALL TESTS PASSED\033[0m\n");
        printf("Accuracy: \033[32m100%%\033[0m\n");
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Time taken: \033[32m%ldms\033[0m\n", duration);

    printf("============================================Test Complete============================================\n");
    return 0;
}
