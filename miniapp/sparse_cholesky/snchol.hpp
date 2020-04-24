#ifndef __TASKTORRENT_MINIAPP_SPARSE_CHOLESKY_SNCHOL_HPP__
#define __TASKTORRENT_MINIAPP_SPARSE_CHOLESKY_SNCHOL_HPP__

#include <fstream>
#include <array>
#include <random>
#include <fstream>
#include <iostream>
#include <set>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <exception>
#include <map>
#include <mutex>
#include <tuple>
#include <memory>
#include <utility>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <scotch.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "runtime.hpp"
#include "util.hpp"
#include "mmio.hpp"
#include "communications.hpp"
#include "runtime.hpp"

#include "snchol_utils.hpp"

typedef Eigen::SparseMatrix<double> SpMat;
typedef std::array<int, 2> int2;
typedef std::array<int, 3> int3;

struct MatBlock
{
  public:
    // row irow of the block
    int i;
    // col irow of the block
    int j;
    // the matrix block
    unique_ptr<Eigen::MatrixXd> matA;
    // A subset of nodes[i]->start ... nodes[i]->end
    std::vector<int> rows;
    // A subset of nodes[j]->start ... nodes[j]->end
    std::vector<int> cols;
    // Accumulation structures
    // number of gemm to accumulate on this block
    int n_accumulate; 
    // data to be accumulated
    std::mutex to_accumulate_mtx;
    std::map<int, std::unique_ptr<Eigen::MatrixXd>> to_accumulate; // The blocks to be accumulated on this
    // Debugging only
    std::atomic<bool> accumulating_busy;
    std::atomic<int> accumulated;
    MatBlock(int i_, int j_);
    Eigen::MatrixXd* A();
};

struct Node
{
  public:
    // irow of this node
    int irow;
    // start, end, size of this node
    int start;
    int end;
    int size;
    // nbrs, after (below the diagonal, in the column) and excluding self
    std::vector<int> nbrs;
    // used in the solve phase
    Eigen::VectorXd xsol;
    // children in the etree
    std::vector<int> children;
    // parent in the etree
    int parent;
    Node(int irow_, int s_, int l_);
};

struct DistMat
{
  private:
    const int my_rank;
    const int n_ranks;

    // All the nodes
    // nodes[i] = ith pivot (diagonal bloc)
    std::vector<std::unique_ptr<Node>> nodes;
    // blocks[i,j] = non zero part of the matrix
    std::map<int2, std::unique_ptr<MatBlock>> blocks;
    // permutation from natural ordering
    Eigen::VectorXi perm;
    // original A
    SpMat A;
    // App = P A P^T
    SpMat App;
    // Super of supernodes
    int nblk;
    // Map nodes to ranks
    std::vector<int> node2rank;
    // Depth (in the ND tree) of each node
    std::vector<int> depth;
    // Map col to rank
    int col2rank(int col);
    // Number of gemm to accumulate on block ij
    int n_to_accumulate(int2 ij);
    // Number of gemm accumulated on block ij
    int accumulated(int2 ij);
    
    // Some statistics/timings
    std::atomic<long long> gemm_us;
    std::atomic<long long> trsm_us;
    std::atomic<long long> potf_us;
    std::atomic<long long> scat_us;
    std::atomic<long long> allo_us;

    // Logging
    const int block_size;
    const int nlevels;
    const int verb;
    const bool do_log;
    const std::string folder;

    // Build all the supernodes based on rangtab
    // Returns i2irow, mapping row -> irow
    Eigen::VectorXi build_supernodes(Eigen::VectorXi& rangtab);

    // Build elimination tree, ie, compute 
    // - node->nbrs (irow)
    // - node->parent (irow)
    // - node->children (irow)
    // - block->rows (row)
    // - block->cols (row)
    // row = unknowns in App
    // irow = block #
    // Returns roots, the roots of the elimination tree (usually 1 - not always)
    std::vector<int> build_tree(Eigen::VectorXi &i2irow);

    // Fill node2rank[k] with a rank for supernode k
    void distribute_tree(std::vector<int> &roots);

    // Allocate blocks[i,k]->A() when column k resides on this rank
    // Fill blocks[i,k]->A() when column k resides on this rank
    void allocate_blocks(Eigen::VectorXi& i2irow);

    void print();
    void potf(int krow);
    void trsm(int2 kirow);
    void gemm(int3 kijrow);
    void accumulate(int3 kijrow);

  public:

    DistMat(const SpMat& A, int nlevels, int block_size, int verb, bool log, std::string folder);
    void factorize(int n_threads);
    Eigen::VectorXd solve(Eigen::VectorXd &b);
    SpMat get_A() const;
};

std::unique_ptr<DistMat> make_DistMat_from_file(std::string filename, int nlevels, int block_size, int verb, bool log, std::string folder);
std::unique_ptr<DistMat> make_DistMat_neglapl(int n, int d, int nlevels, int block_size, int verb, bool log, std::string folder);

#endif