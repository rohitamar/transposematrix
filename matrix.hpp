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
    Matrix<T> sse();
    Matrix<T> avx();
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
Matrix<T> Matrix<T>::sse() {
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
Matrix<T> Matrix<T>::avx() {

}