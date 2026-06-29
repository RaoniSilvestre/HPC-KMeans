#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Checagem de erro: aborta com mensagem clara em qualquer falha de CUDA.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "Erro CUDA em " << __FILE__ << ":" << __LINE__ << " -> "    \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

static constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Estrutura de leitura (host). O parsing fica em double e só convertemos para
// float ao montar os arrays achatados que vão para a GPU.
// ---------------------------------------------------------------------------
struct Point {
  std::vector<double> features;
};

std::vector<std::string> splitByWhitespace(const std::string &str) {
  std::vector<std::string> tokens;
  std::istringstream iss(str);
  std::string token;
  while (iss >> token)
    tokens.push_back(token);
  return tokens;
}

void normalize_points(std::vector<Point> &points) {
  size_t num_features = points[0].features.size();
  for (size_t f = 0; f < num_features; f++) {
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    for (const auto &p : points) {
      min_val = std::min(min_val, p.features[f]);
      max_val = std::max(max_val, p.features[f]);
    }
    double range = max_val - min_val;
    for (auto &p : points) {
      p.features[f] = (p.features[f] - min_val) / range;
      assert(p.features[f] >= 0.0 && p.features[f] <= 1.0);
    }
  }
}

// Centroides iniciais: k pontos aleatorios com seed fixa (mesmo criterio das
// versoes seq/omp de src/, para reprodutibilidade). Difere do init k-means++
// da versao c/ -- o resultado final converge igual; so muda o ponto de partida.
std::vector<float> init_centroid(const std::vector<Point> &points, int k,
                                 int num_features) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
  std::vector<float> centroids(k * num_features);
  for (int c = 0; c < k; ++c) {
    const auto &chosen = points[dist(rng)].features;
    for (int f = 0; f < num_features; ++f)
      centroids[c * num_features + f] = static_cast<float>(chosen[f]);
  }
  return centroids;
}

// ---------------------------------------------------------------------------
// Kernel 1 - Assignment: uma thread por ponto. Carrega os centroides em shared
// memory uma vez por bloco e atribui cada ponto ao centroide mais proximo.
// Usa distancia ao quadrado (evita sqrt; nao altera o argmin). Sinaliza em
// *changed se algum rotulo mudou (criterio de convergencia, igual ao c/).
// ---------------------------------------------------------------------------
__global__ void assign_kernel(const float *features, const float *centroids,
                              int *labels, int *changed, int n, int k,
                              int num_features) {
  extern __shared__ float s_centroids[]; // k * num_features
  for (int idx = threadIdx.x; idx < k * num_features; idx += blockDim.x)
    s_centroids[idx] = centroids[idx];
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float min_dist = 3.402823466e+38f; // FLT_MAX
  int best = -1;
  for (int c = 0; c < k; ++c) {
    float sum = 0.0f;
    for (int f = 0; f < num_features; ++f) {
      float diff = features[i * num_features + f] - s_centroids[c * num_features + f];
      sum += diff * diff;
    }
    if (sum < min_dist) {
      min_dist = sum;
      best = c;
    }
  }
  if (labels[i] != best) {
    labels[i] = best;
    *changed = 1; // corrida benigna: todas as threads escrevem o mesmo valor
  }
}

// ---------------------------------------------------------------------------
// Kernel 2 - Accumulate: uma thread por ponto. Acumula somas por feature e
// contagem por cluster via atomicAdd (espelha a reducao local->global da
// versao OpenMP; com k*F acumuladores a contencao e baixa).
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(const float *features, const int *labels,
                                  float *sums, int *counts, int n,
                                  int num_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int c = labels[i];
  for (int f = 0; f < num_features; ++f)
    atomicAdd(&sums[c * num_features + f], features[i * num_features + f]);
  atomicAdd(&counts[c], 1);
}

// ---------------------------------------------------------------------------
// Kernel 3 - Update: uma thread por componente de centroide (k * num_features).
// Novo centroide = soma / contagem (mantem cluster vazio inalterado).
// ---------------------------------------------------------------------------
__global__ void update_kernel(const float *sums, const int *counts,
                              float *centroids, int k, int num_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * num_features)
    return;

  int c = idx / num_features;
  if (counts[c] > 0)
    centroids[idx] = sums[idx] / counts[c];
}

// Retorna o tempo total da fase de fit (H2D + loop + D2H), em segundos.
double kmeans_cuda(const std::vector<float> &h_features,
                   std::vector<float> &h_centroids, std::vector<int> &h_labels,
                   int n, int k, int num_features, int max_iterations) {
  size_t feat_bytes = static_cast<size_t>(n) * num_features * sizeof(float);
  size_t cent_bytes = static_cast<size_t>(k) * num_features * sizeof(float);
  size_t count_bytes = static_cast<size_t>(k) * sizeof(int);
  size_t label_bytes = static_cast<size_t>(n) * sizeof(int);

  float *d_features = nullptr, *d_centroids = nullptr, *d_sums = nullptr;
  int *d_labels = nullptr, *d_counts = nullptr, *d_changed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
  CUDA_CHECK(cudaMalloc(&d_centroids, cent_bytes));
  CUDA_CHECK(cudaMalloc(&d_sums, cent_bytes));
  CUDA_CHECK(cudaMalloc(&d_labels, label_bytes));
  CUDA_CHECK(cudaMalloc(&d_counts, count_bytes));
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

  // Eventos para o breakdown H2D / kernels / D2H (info secundaria do relatorio).
  cudaEvent_t e0, e1, e2, e3;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventCreate(&e3));

  int grid_points = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_update = (k * num_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t shared_bytes = cent_bytes;

  // Inicializa rotulos com -1 no device para a deteccao de mudanca da 1a iter.
  CUDA_CHECK(cudaMemset(d_labels, 0xFF, label_bytes));

  // Wall clock da fase de fit (comparavel ao omp_get_wtime do c/).
  auto wall_start = std::chrono::high_resolution_clock::now();

  // --- H2D: dados ficam no device por todas as iteracoes ---
  CUDA_CHECK(cudaEventRecord(e0));
  CUDA_CHECK(cudaMemcpy(d_features, h_features.data(), feat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), cent_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(e1));

  // --- Loop de kernels com parada por convergencia (igual ao c/) ---
  int iters_run = 0;
  for (int iter = 0; iter < max_iterations; ++iter) {
    CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int)));
    assign_kernel<<<grid_points, BLOCK_SIZE, shared_bytes>>>(
        d_features, d_centroids, d_labels, d_changed, n, k, num_features);
    CUDA_CHECK(cudaMemsetAsync(d_sums, 0, cent_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, count_bytes));
    accumulate_kernel<<<grid_points, BLOCK_SIZE>>>(
        d_features, d_labels, d_sums, d_counts, n, num_features);
    update_kernel<<<grid_update, BLOCK_SIZE>>>(d_sums, d_counts, d_centroids, k,
                                               num_features);

    int h_changed = 0;
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                          cudaMemcpyDeviceToHost));
    ++iters_run;
    if (!h_changed)
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(e2));

  // --- D2H: traz labels e centroides finais ---
  CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, label_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids, cent_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e3));
  CUDA_CHECK(cudaEventSynchronize(e3));

  auto wall_end = std::chrono::high_resolution_clock::now();
  double elapsed_s =
      std::chrono::duration<double>(wall_end - wall_start).count();

  float t_h2d = 0, t_kernels = 0, t_d2h = 0;
  CUDA_CHECK(cudaEventElapsedTime(&t_h2d, e0, e1));
  CUDA_CHECK(cudaEventElapsedTime(&t_kernels, e1, e2));
  CUDA_CHECK(cudaEventElapsedTime(&t_d2h, e2, e3));

  std::cerr << "[breakdown] iters=" << iters_run << " H2D=" << t_h2d
            << "ms Kernels=" << t_kernels << "ms D2H=" << t_d2h << "ms\n";

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  cudaFree(d_features);
  cudaFree(d_centroids);
  cudaFree(d_sums);
  cudaFree(d_labels);
  cudaFree(d_counts);
  cudaFree(d_changed);

  return elapsed_s;
}

int main(int argc, char **argv) {
  const int k = 3;
  const char *path = (argc > 1) ? argv[1] : "data/seeds_dataset.txt";
  const int max_iterations = (argc > 2) ? std::stoi(argv[2]) : 300;

  std::ifstream dataset(path);
  if (!dataset.is_open()) {
    std::cerr << "Não foi possível abrir o arquivo: " << path << std::endl;
    return 1;
  }

  std::vector<Point> points;
  std::string line;
  while (std::getline(dataset, line)) {
    auto words = splitByWhitespace(line);
    if (words.empty())
      continue;
    std::vector<double> features;
    for (size_t i = 0; i < 7 && i < words.size(); ++i)
      features.push_back(std::stod(words[i]));
    points.push_back(Point{features});
  }

  normalize_points(points);

  int n = static_cast<int>(points.size());
  int num_features = static_cast<int>(points[0].features.size());

  // Achata os pontos num array contiguo (layout i*num_features + f) para
  // leitura coalescida na GPU.
  std::vector<float> h_features(static_cast<size_t>(n) * num_features);
  for (int i = 0; i < n; ++i)
    for (int f = 0; f < num_features; ++f)
      h_features[i * num_features + f] =
          static_cast<float>(points[i].features[f]);

  std::vector<float> h_centroids = init_centroid(points, k, num_features);
  std::vector<int> h_labels(n, -1);

  double elapsed_s = kmeans_cuda(h_features, h_centroids, h_labels, n, k,
                                 num_features, max_iterations);

  // Linha primaria, no mesmo formato/unidade da versao c/ (segundos).
  std::cout << "Elapsed time: " << elapsed_s << " seconds" << std::endl;

  std::vector<int> cluster_sizes(k, 0);
  for (int lbl : h_labels)
    if (lbl >= 0 && lbl < k)
      cluster_sizes[lbl]++;

  std::cerr << "Tamanho dos Clusters:\n";
  for (int i = 0; i < k; i++)
    std::cerr << "Cluster " << i << ": " << cluster_sizes[i] << " pontos\n";

  return 0;
}
