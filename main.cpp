#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <chrono>

#include "matrix.hpp"

using namespace std::chrono;

const int R = 1LL << 5;
const int C = 1LL << 5;

int main() {
    Matrix<int> mat(R, C);
    // mat.print();
    auto start = high_resolution_clock::now();
    Matrix<int> bf_trans = mat.bf();
    auto end = high_resolution_clock::now();
    // bf_trans.print();
    double normal_duration = duration_cast<seconds>(end - start).count();
    std::cout << "Normal Time taken: " << normal_duration << "\n";

    start = high_resolution_clock::now(); 
    // Matrix<int> trans = mat.sse44();
    // trans.print();
    end = high_resolution_clock::now();
    double sse_duration = duration_cast<microseconds>(end - start).count();
    std::cout << "SSE Time taken: " << sse_duration << "\n";

    // std::cout << "Ratio: " << normal_duration / sse_duration << "\n";
}