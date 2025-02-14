#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

const int R = 1LL << 8;
const int C = 1LL << 8;

alignas(32) int mat[R][C];
alignas(32) int transposed[C][R];

void fill() {
    for(int i = 0; i < C; i++) {
        std::fill_n(transposed[i], R, 0LL);
    }
    for(int i = 0; i < R; i++) {
        std::fill_n(mat[i], C, 0LL);
        std::iota(mat[i] + i, mat[i] + C, 1LL);
    }
}

void bf() {
    for(int i = 0; i < R; i++) {
        for(int j = 0; j < C; j++) {
            transposed[j][i] = mat[i][j];
        }
    }
}

void avx() {
    for(int i = 0; i < R / 8; i++) {
        for(int j = 0; j < R / 8; j++) {
            __m256i row0 = _mm256_load_si256(reinterpret_cast<__m256i*>(&mat[i][j]));
            __m256i row1 = _mm256_load_si256(reinterpret_cast<__m256i*>(&mat[i + 1][j]));
            __m256i row2 = _mm256_load_si256(reinterpret_cast<__m256i*>(&mat[i + 2][j]));
            __m256i row3 = _mm256_load_si256(reinterpret_cast<__m256i*>(&mat[i + 3][j]));
        }
    }
}

int main() {

}