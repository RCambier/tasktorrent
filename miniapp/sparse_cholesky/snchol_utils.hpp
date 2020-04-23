#ifndef __TASKTORRENT_MINIAPP_SPARSE_CHOLESKY_SNCHOL_UTILS_HPP__
#define __TASKTORRENT_MINIAPP_SPARSE_CHOLESKY_SNCHOL_UTILS_HPP__

#include <vector>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

Eigen::SparseMatrix<double> make_neglapl(const int n, const int d);
Eigen::VectorXd random(int size, int seed);

#endif
