#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <chrono>
#include "matrix.hpp"

using namespace std::chrono;

const int R = 1LL << 12;
const int C = 1LL << 12;

int main() {
    Matrix<int> mat(R, C);
    auto start = high_resolution_clock::now();
    Matrix<int> bf_trans = mat.bf();
    auto end = high_resolution_clock::now();
    double normal_duration = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now(); 
    Matrix<int> sse44_trans = mat.sse44();
    end = high_resolution_clock::now();
    double sse_duration = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now(); 
    Matrix<int> avx88_trans = mat.avx88();
    end = high_resolution_clock::now();
    double avx88_duration = duration_cast<microseconds>(end - start).count();

    std::cout << "Normal Time taken: " << normal_duration << "\n";

    if(bf_trans == sse44_trans) {
        std::cout << "SSE Time taken: " << sse_duration << "\n";
    } else {
        std::cout << "sse44 computation is incorrect.\n"; 
        sse44_trans.print();
    }

    if(bf_trans == avx88_trans) {
        std::cout << "SSE Time taken: " << avx88_duration << "\n";
    } else {
        std::cout << "avx88 computation is incorrect.\n"; 
        sse44_trans.print();
    }



    
    // std::cout << "Ratio: " << normal_duration / sse_duration << "\n";
}