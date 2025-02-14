#include <iostream> 
#include <memory>
#include <algorithm>
#include <numeric>

typedef long long int ll;

const ll R = 1LL << 20;
const ll C = 1LL << 20;

ll mat[R][C];
ll transposed[C][R];

void fill() {
    for(ll i = 0; i < C; i++) {
        std::fill_n(transposed[i], R, 0LL);
    }
    for(ll i = 0; i < R; i++) {
        std::fill_n(mat[i], C, 0LL);
        std::iota(mat[i] + i, mat[i] + C, 1LL);
    }
}

void bf() {
    for(ll i = 0; i < R; i++) {
        for(ll j = 0; j < C; j++) {
            transposed[j][i] = mat[i][j];
        }
    }
}

void sse() {

}

int main() {
    fill();
    bf();
    
    fill();
    sse();
    
    fill();
}