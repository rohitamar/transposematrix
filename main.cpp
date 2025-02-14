#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <chrono>

#include "matrix.hpp"

using namespace std::chrono;

const int R = 1LL << 13;
const int C = 1LL << 13;

int main() {
    Matrix<int> mat(R, C);
    auto start = high_resolution_clock::now();
    mat.bf();
    auto end = high_resolution_clock::now();
    double normal_duration = duration_cast<microseconds>(end - start).count();
    std::cout << "Normal Time taken: " << normal_duration << "\n";

    start = high_resolution_clock::now(); 
    mat.sse();
    end = high_resolution_clock::now();
    double sse_duration = duration_cast<microseconds>(end - start).count();
    std::cout << "SSE Time taken: " << sse_duration << "\n";
    std::cout << "Ratio: " << normal_duration / sse_duration << "\n";
}