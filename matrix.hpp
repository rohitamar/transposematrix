#include <memory>

template<typename T>
class Matrix {
    private:
    std::unique_ptr<T[]> data_;
    size_t rows_, cols_;

    public:
    Matrix(size_t R, size_t C) : data_(std::make_unique<T[]>(R * C)), rows_(R), cols_(C) {
        size_t size = rows_ * cols_ * sizeof(T);
        std::align(32, sizeof(T), reinterpret_cast<void*&>(data_), size);
        // fill with an upper-triangular pattern  
        for(size_t i = 0; i < R; i++) {
            T *ptr = data_.get() + i * cols_;
            std::iota(ptr, ptr + R, 1LL);
        }
    }
    
    T& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }

    size_t rows() const { return rows_; } 
    size_t cols() const { return cols_; } 

    inline void print() const {
        for(size_t i = 0; i < rows_; i++) {
            for(size_t j = 0; j < cols_; j++) {
                std::cout << (this)(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    Matrix<T> bf();
    Matrix<T> sse44();
    Matrix<T> avx88();
};

template<typename T>
Matrix<T> Matrix<T>::bf() {
    Matrix<T> transposed(cols_, rows_);
    for(size_t i = 0; i < rows_; i++) {
        for(size_t j = 0; j < cols_; j++) {
            transposed(j, i) = (*this)(i, j);
        }
    }
    return transposed;
}

template<typename T>
Matrix<T> Matrix<T>::sse44() {
    Matrix<T> transposed(cols_, rows_);
    
    for(size_t i = 0; i < rows_ / 4; i++) {
        for(size_t j = 0; j < cols_ / 4; j++) {
            __m128i row0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(i    , j)))); // (a0 a1 a2 a3)
            __m128i row1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(i + 1, j)))); // (b0 b1 b2 b3)
            __m128i row2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(i + 2, j)))); // (c0 c1 c2 c3)
            __m128i row3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(i + 2, j)))); // (d0 d1 d2 d3)

            __m128i tmp0 = _mm_unpacklo_epi32(row0, row1); // (a0 b0 a1 b1) 
            __m128i tmp1 = _mm_unpackhi_epi32(row0, row1); // (a2 b2 a3 b3)
            __m128i tmp2 = _mm_unpacklo_epi32(row2, row3); // (c0 d0 c1 d1)
            __m128i tmp3 = _mm_unpackhi_epi32(row2, row3); // (c2 d2 c3 d3)

            row0 = _mm_unpacklo_epi64(tmp0, tmp2); // (a0, b0, c0, d0) 
            row1 = _mm_unpackhi_epi64(tmp0, tmp2); // (a1, b1, c1, d1)
            row2 = _mm_unpacklo_epi64(tmp1, tmp3); // (a2, b2, c2, d2)
            row3 = _mm_unpackhi_epi64(tmp1, tmp3); // (a3, b3, c3, d3)

            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(j, i)), row0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(j + 1, i)), row1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(j + 2, i)), row2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(j + 3, i)), row3);
        }
    }

    return transposed;
}

template<typename T>
Matrix<T> Matrix<T>::avx88() {
    Matrix<T> transposed(cols_, rows_);

    for(size_t i = 0; i < rows_ / 8; i++) {
        for(size_t j = 0; j < cols_ / 8; j++) {
            __m256i row0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i    , j))));
            __m256i row1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 1, j))));
            __m256i row2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 2, j))));
            __m256i row3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 3, j))));
            __m256i row4 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 4, j))));
            __m256i row5 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 5, j))));
            __m256i row6 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 6, j))));
            __m256i row7 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator(i + 7, j))));

            __m256i first0 = _mm256_unpacklo_epi32(row0, row1); // a0 b0 a1 b1 a2 b2 a3 b3
            __m256i first1 = _mm256_unpackhi_epi32(row0, row1); // a4 b4 a5 b5 a6 b6 a7 b7 
            __m256i first2 = _mm256_unpacklo_epi32(row2, row3); // c0 d0 c1 d1 c2 d2 c3 d3
            __m256i first3 = _mm256_unpackhi_epi32(row2, row3); // c4 d4 c5 d5 c6 d6 c7 d7
            __m256i first4 = _mm256_unpacklo_epi32(row4, row5); // e0 f0 e1 f1 e2 f2 e3 f3
            __m256i first5 = _mm256_unpackhi_epi32(row4, row5); // e4 f4 e5 f5 e6 f6 e7 f7
            __m256i first6 = _mm256_unpacklo_epi32(row6, row7); // g0 h0 g1 h1 g2 h2 g3 h3
            __m256i first7 = _mm256_unpackhi_epi32(row6, row7); // g4 h4 g5 h5 g6 h6 g7 h7

            __m256i second0 = _mm256_unpacklo_epi64(first0, first2); // a0 b0 c0 d0 a1 b1 c1 d1
            __m256i second1 = _mm256_unpackhi_epi64(first0, first2); // a2 b2 c2 d2 a3 b3 c3 d3
            __m256i second2 = _mm256_unpacklo_epi64(first1, first3); // a4 b4 c4 d4 a5 b5 c5 d5
            __m256i second3 = _mm256_unpackhi_epi64(first1, first3); // a6 b6 c6 d6 a7 b7 c7 d7

        }
    }
}