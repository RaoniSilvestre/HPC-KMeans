#include <omp.h>
#include <algorithm>
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

struct Point {
    std::vector<double> features;
    int cluster_id{-1};
};

double euclidean_distance(const Point& p, const std::vector<double>& centroid) {
    double sum = 0.0;
    for (size_t i = 0; i < p.features.size(); ++i) {
        double diff = p.features[i] - centroid[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<std::vector<double>> init_centroid(const std::vector<Point>& points, int k) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
    std::vector<std::vector<double>> centroids(k);
    for (int i = 0; i < k; ++i)
        centroids[i] = points[dist(rng)].features;
    return centroids;
}

void kmeans_openmp(std::vector<Point>& points, int k, int max_iterations) {
    int n = (int)points.size();
    int num_features = (int)points[0].features.size();
    auto centroids = init_centroid(points, k);

    for (int iter = 0; iter < max_iterations; ++iter) {

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best = -1;

            for (int c = 0; c < k; ++c) {
                double d = euclidean_distance(points[i], centroids[c]);
                if (d < min_dist) { min_dist = d; best = c; }
            }

            points[i].cluster_id = best;
        }

        std::vector<double> sum_features(k * num_features, 0.0);
        std::vector<int>    counts(k, 0);

        #pragma omp parallel
        {
            std::vector<double> local_sum(k * num_features, 0.0);
            std::vector<int>    local_count(k, 0);

            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; ++i) {
                int c = points[i].cluster_id;
                for (int f = 0; f < num_features; ++f)
                    local_sum[c * num_features + f] += points[i].features[f];
                local_count[c]++;
            }

            #pragma omp critical
            {
                for (int c = 0; c < k; ++c) {
                    counts[c] += local_count[c];
                    for (int f = 0; f < num_features; ++f)
                        sum_features[c * num_features + f] += local_sum[c * num_features + f];
                }
            }
        }


        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int f = 0; f < num_features; ++f)
                    centroids[c][f] = sum_features[c * num_features + f] / counts[c];
            }
        }
    }
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

std::vector<std::string> splitByWhitespace(const std::string &str) {
  std::vector<std::string> tokens;
  std::istringstream iss(str);
  std::string token;
  while (iss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

int main(int argc, char** argv) {
    if (argc > 1)
        omp_set_num_threads(std::stoi(argv[1]));

    std::ifstream dataset("data/seeds_dataset.txt");
    if (!dataset.is_open()) {
        std::cout << "Não foi possível abrir o arquivo.\n";
        return 1;
    }
    std::cout << "Arquivo aberto com sucesso!\n";
    std::cout << "Threads disponíveis: " << omp_get_max_threads() << "\n";

    std::vector<Point> points;
    std::string line;
    while (std::getline(dataset, line)) {
        auto words = splitByWhitespace(line);
        if (words.empty()) continue;
        std::vector<double> features;
        for (size_t i = 0; i < 7 && i < words.size(); ++i)
            features.push_back(std::stod(words[i]));
        points.push_back(Point{features, -1});
    }

    normalize_points(points);

    auto start = std::chrono::high_resolution_clock::now();
    kmeans_openmp(points, 3, 50000);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " ms\n";

    std::vector<int> cluster_sizes(3, 0);
    for (const auto& p : points)
        if (p.cluster_id >= 0 && p.cluster_id < 3)
            cluster_sizes[p.cluster_id]++;

    std::cout << "Tamanho dos Clusters:\n";
    for (int i = 0; i < 3; i++)
        std::cout << "Cluster " << i << ": " << cluster_sizes[i] << " pontos\n";

    return 0;
}
