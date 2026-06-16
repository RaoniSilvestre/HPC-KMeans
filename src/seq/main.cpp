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

double euclidean_distance(const Point &p, const std::vector<double> &centroid) {
  double sum = 0.0;
  for (size_t i = 0; i < p.features.size(); ++i) {
    double diff = p.features[i] - centroid[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

std::vector<std::vector<double>> init_centroid(const std::vector<Point> &points,
                                               int k) {
  std::vector<std::vector<double>> centroids(k);
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist(0, points.size() - 1);

  for (int i = 0; i < k; ++i) {
    size_t idx = dist(rng);
    centroids[i] = points[idx].features;
  }
  return centroids;
}

void kmeans_sequential(std::vector<Point> &points, int k, int max_iterations) {
  size_t num_features = points[0].features.size();
  std::vector<std::vector<double>> centroids = init_centroid(points, k);

  for (int iter = 0; iter < max_iterations; iter++) {

    for (auto &p : points) {
      double min_dist = std::numeric_limits<double>::max();
      int best_cluster = -1;

      for (int i = 0; i < k; ++i) {
        double d = euclidean_distance(p, centroids[i]);
        if (d < min_dist) {
          min_dist = d;
          best_cluster = i;
        }
      }

      if (p.cluster_id != best_cluster) {
        p.cluster_id = best_cluster;
      }
    }

    std::vector<std::vector<double>> sum_features(
        k, std::vector<double>(num_features, 0.0));
    std::vector<int> counts(k, 0);

    for (const auto &p : points) {
      if (p.cluster_id != -1) {
        for (size_t f = 0; f < num_features; ++f) {
          sum_features[p.cluster_id][f] += p.features[f];
        }
        counts[p.cluster_id]++;
      }
    }

    for (int i = 0; i < k; ++i) {
      if (counts[i] > 0) {
        for (size_t f = 0; f < num_features; ++f) {
          centroids[i][f] = sum_features[i][f] / counts[i];
        }
      }
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

int main(int argc, char **argv) {
  std::ifstream dataset("data/seeds_dataset.txt");

  if (!dataset.is_open()) {
    std::cout << "Não foi possível abrir o arquivo." << std::endl;
    return 1;
  }

  std::cout << "Arquivo aberto com sucesso!" << std::endl;

  std::vector<Point> points;
  std::string line;

  while (std::getline(dataset, line)) {
    auto words = splitByWhitespace(line);
    if (words.empty())
      continue;

    std::vector<double> features;
    // existem 7 features, tentei colocar todas pra ver se o bixo demora mais
    for (size_t i = 0; i < 7 && i < words.size(); ++i) {
      features.push_back(std::stod(words[i]));
    }

    points.push_back(Point{features, -1});
  }

  normalize_points(points);

  auto start = std::chrono::high_resolution_clock::now();
  // Tanto faz o max_iterations o bixo converge em 5.
  kmeans_sequential(points, 3, 50000);
  // Record end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate duration in milliseconds
  std::chrono::duration<double, std::milli> duration = end - start;

  // Output the result
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

  std::vector<int> cluster_sizes(3, 0);
  for (const auto &p : points) {
    if (p.cluster_id >= 0 && p.cluster_id < 3) {
      cluster_sizes[p.cluster_id]++;
    }
  }

  std::cout << "Tamanho dos Clusters:\n";
  for (int i = 0; i < 3; i++) {
    std::cout << "Cluster " << i << ": " << cluster_sizes[i] << " pontos\n";
  }

  return 0;
}
