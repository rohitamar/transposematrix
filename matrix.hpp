#include <memory>
#include <immintrin.h>
#include <vector>

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(size_t n) {
        return static_cast<T*>(_mm_malloc(n * sizeof(T), 64)); // 64-byte aligned
    }

    void deallocate(T* p, std::size_t) noexcept {
        _mm_free(p);
    }
};

template<typename T>
class Matrix {
    private:
    std::vector<T, AlignedAllocator<T>> data_;
    size_t rows_, cols_;

    public:
    Matrix(size_t R, size_t C) : rows_(R), cols_(C) {
        data_.resize(rows_ * cols_);

        // fill with an upper-triangular pattern  
        for(size_t i = 0; i < R; i++) {
            auto ptr = data_.begin() + i * cols_ + i; 
            std::iota(ptr, ptr + C - i, 1LL);
        }
    }
    
    T& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }

    bool operator==(const Matrix& o) const {
        if(rows_ != o.rows_ || cols_ != o.cols_) return false;
        return data_ == o.data_;
    }

    size_t rows() const { return rows_; } 
    size_t cols() const { return cols_; } 

    inline void print() const {
        for(size_t i = 0; i < rows_; i++) {
            for(size_t j = 0; j < cols_; j++) {
                std::cout << this->operator()(i, j) << " ";
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
            __m128i row0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(4 * i    , 4 * j)))); // (a0 a1 a2 a3)
            __m128i row1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(4 * i + 1, 4 * j)))); // (b0 b1 b2 b3)
            __m128i row2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(4 * i + 2, 4 * j)))); // (c0 c1 c2 c3)
            __m128i row3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&(this->operator()(4 * i + 3, 4 * j)))); // (d0 d1 d2 d3)

            __m128i tmp0 = _mm_unpacklo_epi32(row0, row1); // (a0 b0 a1 b1) 
            __m128i tmp1 = _mm_unpackhi_epi32(row0, row1); // (a2 b2 a3 b3)
            __m128i tmp2 = _mm_unpacklo_epi32(row2, row3); // (c0 d0 c1 d1)
            __m128i tmp3 = _mm_unpackhi_epi32(row2, row3); // (c2 d2 c3 d3)

            row0 = _mm_unpacklo_epi64(tmp0, tmp2); // (a0, b0, c0, d0) 
            row1 = _mm_unpackhi_epi64(tmp0, tmp2); // (a1, b1, c1, d1)
            row2 = _mm_unpacklo_epi64(tmp1, tmp3); // (a2, b2, c2, d2)
            row3 = _mm_unpackhi_epi64(tmp1, tmp3); // (a3, b3, c3, d3)

            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(4 * j    , 4 * i)), row0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(4 * j + 1, 4 * i)), row1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(4 * j + 2, 4 * i)), row2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&transposed(4 * j + 3, 4 * i)), row3);

        }
    }

    return transposed;
}

template<typename T>
Matrix<T> Matrix<T>::avx88() {
    Matrix<T> transposed(cols_, rows_);

    for(size_t i = 0; i < rows_ / 8; i++) {
        for(size_t j = 0; j < cols_ / 8; j++) {
            __m256i row0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i    , 8 * j))));
            __m256i row1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 1, 8 * j))));
            __m256i row2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 2, 8 * j))));
            __m256i row3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 3, 8 * j))));
            __m256i row4 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 4, 8 * j))));
            __m256i row5 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 5, 8 * j))));
            __m256i row6 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 6, 8 * j))));
            __m256i row7 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&(this->operator()(8 * i + 7, 8 * j))));

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
            __m256i second4 = _mm256_unpacklo_epi64(first4, first6); // e0 f0 g0 h0 e1 f1 g1 h1
            __m256i second5 = _mm256_unpackhi_epi64(first4, first6); // e2 f2 g2 h2 e3 f3 g3 h3
            __m256i second6 = _mm256_unpacklo_epi64(first5, first7); // e4 f4 g4 h4 e5 f5 g5 h5
            __m256i second7 = _mm256_unpackhi_epi64(first5, first7); // e6 f6 h6 g6 e7 f7 g7 h7

            __m256i col0 = _mm256_permute2x128_si256(second0, second4, 0x20);
            __m256i col1 = _mm256_permute2x128_si256(second1, second5, 0x20);
            __m256i col2 = _mm256_permute2x128_si256(second2, second6, 0x20);
            __m256i col3 = _mm256_permute2x128_si256(second3, second7, 0x20);
            __m256i col4 = _mm256_permute2x128_si256(second0, second4, 0x31);
            __m256i col5 = _mm256_permute2x128_si256(second1, second5, 0x31);
            __m256i col6 = _mm256_permute2x128_si256(second2, second6, 0x31);
            __m256i col7 = _mm256_permute2x128_si256(second3, second7, 0x31);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j    , 8 * i)), col0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 1, 8 * i)), col1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 2, 8 * i)), col2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 3, 8 * i)), col3);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 4, 8 * i)), col4);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 5, 8 * i)), col5);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 6, 8 * i)), col6);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&transposed(8 * j + 7, 8 * i)), col7);
        }
    }

    return transposed;
}