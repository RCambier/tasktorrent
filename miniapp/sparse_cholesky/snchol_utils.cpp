#include "snchol_utils.hpp"

// Creates a random (-1, 1) vector
// @pre size >= 0
// @post x, st -1 <= x[i] <= 1 for all 0 <= i < size
Eigen::VectorXd random(int size, int seed)
{
    std::mt19937 rng;
    rng.seed(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXd x(size);
    for (int i = 0; i < size; i++)
    {
        x[i] = dist(rng);
    }
    return x;
}

Eigen::SparseMatrix<double> make_neglapl(const int n, const int d) {

    assert(d == 2 || d == 3);
    const int N = pow(n, d);
    std::vector<Eigen::Triplet<double>> triplets;

    for(int col = 0; col < N; col++) {
        if(d == 2) {
            
            const int offset_i[5] = {0,  0,  0,  -1,  1};
            const int offset_j[5] = {0, -1,  1,   0,  0};
            const double values[5] = {-4.0, 1.0, 1.0, 1.0, 1.0};
            
            const int i = (col        ) / n;
            const int j = (col - i * n) / 1;
            
            for(int p = 0; p < 5; p++) {
                const int ii = i + offset_i[p];
                const int jj = j + offset_j[p];
                const double vv = values[p];
                if(ii >= 0 && ii < n && jj >= 0 && jj < n) {
                    const int row = ii * n + jj;
                    triplets.push_back({row, col, vv});
                }
            }

        } else {

            const int offset_i[7] = {0, -1,  1,  0,  0,  0,  0};
            const int offset_j[7] = {0,  0,  0, -1,  1,  0,  0};
            const int offset_k[7] = {0,  0,  0,  0,  0, -1,  1};
            const double values[7] = {-6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

            const int i = (col                    ) / (n * n);
            const int j = (col - i * n * n        ) / n;
            const int k = (col - i * n * n - j * n) / 1;

            for(int p = 0; p < 7; p++) {
                const int ii = i + offset_i[p];
                const int jj = j + offset_j[p];
                const int kk = k + offset_k[p];
                const double vv = values[p];
                if(ii >= 0 && ii < n && jj >= 0 && jj < n && kk >= 0 && kk < n) {
                    const int row = ii * n * n + jj * n + kk;
                    triplets.push_back({row, col, vv});
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return - A;
}