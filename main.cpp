#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include "matrix.hpp" 

using namespace std::chrono;

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cerr << "Early termination" << std::endl; 
        return 1;
    }    

    int N = std::atoi(argv[1]);
    std::cout << "Performing operations on a 2^" << N << " square matrix.\n";
    const int R = 1LL << N;
    const int C = 1LL << N;
    
    Matrix<int> mat(R, C);
    auto start = high_resolution_clock::now();
    Matrix<int> bf_trans = mat.bf();
    auto end = high_resolution_clock::now();
    double normal_duration = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now(); 
    Matrix<int> sse44_trans = mat.sse44();
    end = high_resolution_clock::now();
    double sse_duration = duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now(); 
    Matrix<int> avx88_trans = mat.avx88();
    end = high_resolution_clock::now();
    double avx88_duration = duration_cast<milliseconds>(end - start).count();

    std::cout << "Normal Time taken: " << normal_duration << "\n";

    if(bf_trans == sse44_trans) {
        std::cout << "SSE Time taken: " << sse_duration << "\n";
    } else {
        std::cout << "sse44 computation is incorrect.\n"; 
        sse44_trans.print();
    }

    if(bf_trans == avx88_trans) {
        std::cout << "avx88 Time taken: " << avx88_duration << "\n";
    } else {
        std::cout << "avx88 computation is incorrect.\n"; 
        sse44_trans.print();
    }
}