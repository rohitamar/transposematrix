# transposematrix
Transpose a matrix in C++ with SSE (4 by 4) and AVX2 (8 by 8)

## Results 

| ```N``` | ```brute_force``` | ```SSE, 4 by 4``` | ```AVX2, 8 by 8``` | 
|---|-------------------|-------------------|--------------------|
|10 | 10 | 6 | 5 |
|11 | 46 | 25 | 26 |
|12 | 337 | 149 | 106 | 
| 13 | 1342 | 553 | 412 |
| 14 | 5215 | 2222 | 1721 |
