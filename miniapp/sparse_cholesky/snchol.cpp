#include "snchol.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

struct range
{
    int lb;
    int ub;
    int k;
};

// Given (k, i, j), returns (k, max(i, j), min(i, j))
int3 lower(int3 kij)
{
    int k = kij[0];
    int i = kij[1];
    int j = kij[2];
    return {k, std::max(i,j), std::min(i,j)};
};

// Find the positions of c_rows into p_rows
//
// @in c_rows is a subset of p_rows
// @in c_rows, p_rows are sorted
// @post out[i] = j <=> c_rows[i] == p_rows[j]
vector<int> get_subids(const vector<int> &c_rows, const vector<int> &p_rows)
{
    int cn = c_rows.size();
    vector<int> subids(cn);
    int l = 0;
    for (int i = 0; i < cn; i++)
    {
        while (c_rows[i] != p_rows[l])
        {
            l++;
        }
        assert(l < p_rows.size());
        assert(p_rows[l] == c_rows[i]);
        subids[i] = l;
    }
    return subids;
};

// Given a sparse matrix A, creates rowval and colptr, the CSC representation of A
// without the self loops on the diagonal
// @pre A is a sparse symmetrix matrix
void get_csc_no_diag(const SpMat &A, VectorXi *rowval, VectorXi *colptr) {
    int N = A.rows();
    int nnz = A.nonZeros();
    *rowval = VectorXi(nnz);
    *colptr = VectorXi(N + 1);
    int k = 0;
    (*colptr)[0] = 0;
    for (int j = 0; j < N; j++)
    {
        for (SpMat::InnerIterator it(A, j); it; ++it)
        {
            int i = it.row();
            if (i != j)
            {
                (*rowval)[k] = i;
                k++;
            }
        }
        (*colptr)[j + 1] = k;
    }
    rowval->conservativeResize(k);
}

// Algebraic partitioning
// N is the matrix size
// rowval, colptr is the CSC of symmetric A, without the diagonal
// nlevels, block_size are ND and partitioning parameters
// Outputs are
// - permtab, the permuation 
// - rangtab, the colptr of the clusters (cluster i goes from rangtab[i] ... rangtab[i+1] in permtab ordering)
void algebraic_partitioning(int N, VectorXi &rowval, VectorXi &colptr, int nlevels, int block_size, VectorXi *permtab, VectorXi *rangtab, vector<int> *depth) {
    assert(N == colptr.size() - 1);
    SCOTCH_Graph *graph = SCOTCH_graphAlloc();
    int err = SCOTCH_graphInit(graph);
    assert(err == 0);
    err = SCOTCH_graphBuild(graph, 0, N, colptr.data(), nullptr, nullptr, nullptr, rowval.size(), rowval.data(), nullptr);
    assert(err == 0);
    err = SCOTCH_graphCheck(graph);
    assert(err == 0);
    // Create strat
    SCOTCH_Strat *strat = SCOTCH_stratAlloc();
    err = SCOTCH_stratInit(strat);
    assert(err == 0);
    assert(nlevels > 0);
    string orderingstr = "n{sep=(/levl<" + to_string(nlevels - 1) + "?g:z;),ose=b{cmin=" + to_string(block_size) + "}}";
    cout << "Using ordering " << orderingstr << endl;
    // string orderingstr = "n{sep=(/levl<" + to_string(nlevels-1) + "?g:z;)}";
    err = SCOTCH_stratGraphOrder(strat, orderingstr.c_str());
    assert(err == 0);
    // Order with SCOTCH
    *permtab = VectorXi::Zero(N);
    VectorXi peritab = VectorXi::Zero(N);
    *rangtab = VectorXi::Zero(N + 1);
    VectorXi treetab = VectorXi::Zero(N);
    int nblk = 0;
    err = SCOTCH_graphOrder(graph, strat, permtab->data(), peritab.data(), &nblk, rangtab->data(), treetab.data());
    assert(err == 0);
    assert(nblk >= 0);
    rangtab->conservativeResize(nblk + 1);
    // Compute depth
    *depth = vector<int>(nblk, 0);
    for (int i = 0; i < nblk; i++)
    {
        int p = treetab[i];
        while (p != -1)
        {
            p = treetab[p];
            (*depth)[i]++;
            assert(p == -1 || (p >= 0 && p < nblk));
        }
    }
    SCOTCH_graphExit(graph);
}

MatBlock::MatBlock(int i_, int j_) : i(i_), j(j_), matA(nullptr), n_accumulate(0), accumulating_busy(false), accumulated(0) {};

Eigen::MatrixXd* MatBlock::A() { 
    return matA.get(); 
}

Node::Node(int irow_, int s_, int l_) : irow(irow_), start(s_), size(l_), parent(-1) {
    end = start + size;
};

int DistMat::col2rank(int col)
{
    assert(node2rank.size() == nblk);
    assert(col >= 0 && col < nblk);
    return node2rank[col];
}
 
// Number of gemm to accumulate on block ij
int DistMat::n_to_accumulate(int2 ij)
{
    return blocks.at(ij)->n_accumulate;
}

// Number of gemm accumulated on block ij
int DistMat::accumulated(int2 ij)
{
    return blocks.at(ij)->accumulated.load();
}
    
 
// Build all the supernodes based on rangtab
// Returns i2irow, mapping row -> irow
VectorXi DistMat::build_supernodes(VectorXi& rangtab) {
    nodes = vector<unique_ptr<Node>>(nblk);
    int N = App.rows();
    assert(rangtab.size() > 0);
    assert(rangtab[0] == 0);
    assert(nblk == rangtab.size()-1);
    assert(rangtab(nblk) == N);
    VectorXi i2irow(N);
    double mean = 0.0;
    int mini = App.rows();
    int maxi = -1;
    for (int i = 0; i < nblk; i++)
    {
        nodes[i] = make_unique<Node>(i, rangtab[i], rangtab[i + 1] - rangtab[i]);
        for (int j = rangtab[i]; j < rangtab[i + 1]; j++)
        {
            i2irow[j] = i;
        }
        mean += nodes.at(i)->size;
        mini = min(mini, nodes.at(i)->size);
        maxi = max(maxi, nodes.at(i)->size);
    }
    printf("[%d] %d blocks, min size %d, mean size %f, max size %d\n", comm_rank(), nblk, mini, mean / nblk, maxi);
    return i2irow;
}

// Build elimination tree, ie, compute 
// - node->nbrs (irow)
// - node->parent (irow)
// - node->children (irow)
// - block->rows (row)
// - block->cols (row)
// row = unknowns in App
// irow = block #
// Returns roots, the roots of the elimination tree (usually 1 - not always)
vector<int> DistMat::build_tree(VectorXi &i2irow) {
    vector<set<int>> rows_tmp(nblk); // The mapping block -> rows under
    vector<int> roots(0);
    for (int k = 0; k < nblk; k++)
    {
        auto &n = nodes.at(k);
        // Add local rows
        for (int j = n->start; j < n->end; j++)
        {
            for (SpMat::InnerIterator it(App, j); it; ++it)
            {
                int i = it.row();
                if (i >= n->end)
                {
                    rows_tmp.at(k).insert(i);
                }
            }
        }
        // Get sorted set of neighbros
        vector<int> rows(rows_tmp.at(k).begin(), rows_tmp.at(k).end());
        sort(rows.begin(), rows.end());
        // Convert to neighbors
        set<int> nbrs_tmp;
        for (auto i : rows)
            nbrs_tmp.insert(i2irow(i));
        n->nbrs = vector<int>(nbrs_tmp.begin(), nbrs_tmp.end());
        sort(n->nbrs.begin(), n->nbrs.end());
        // Diagonal bloc
        blocks[{k, k}] = std::make_unique<MatBlock>(k, k);
        auto &b = blocks[{k, k}];
        b->rows = vector<int>(n->size);
        for (int i = 0; i < n->size; i++)
            b->rows[i] = n->start + i;
        b->cols = b->rows;
        // Below-diagonal bloc
        for (auto nirow : n->nbrs)
        {
            auto &nbr = nodes.at(nirow);
            blocks[{nirow, k}] = std::make_unique<MatBlock>(nirow, k);
            auto &b = blocks[{nirow, k}];
            // Find rows
            auto lower = lower_bound(rows.begin(), rows.end(), nbr->start);
            auto upper = upper_bound(rows.begin(), rows.end(), nbr->end - 1);
            b->rows = vector<int>(lower, upper);
            b->cols = blocks.at({k, k})->rows;
        }
        // Add to parent
        if (n->nbrs.size() > 0)
        {
            assert(n->nbrs[0] > k);
            int prow = n->nbrs[0]; // parent in etree = first non zero in column
            n->parent = prow;
            auto &p = nodes.at(prow);
            for (auto i : rows_tmp.at(k))
            {
                if (i >= p->end)
                {
                    rows_tmp.at(prow).insert(i);
                }
            }
            p->children.push_back(k);
            // if(comm_rank() == 0) printf("%d -> %d\n", k, prow);
        }
        else
        {
            n->parent = -1;
            roots.push_back(k);
        }
    }
    printf("%lu roots (should be 1 in general)\n", roots.size());
    return roots;
}

// Fill node2rank[k] with a rank for supernode k
void DistMat::distribute_tree(vector<int> &roots) {
    queue<int> toexplore;
    vector<range> node2range(nblk);
    for (auto k : roots)
    {
        node2range[k] = {0, comm_size(), 0};
        toexplore.push(k);
    }
    // Distribute tree
    while (!toexplore.empty())
    {
        int k = toexplore.front();
        // if(comm_rank() == 0) printf("exploring %d\n", k);
        auto r = node2range.at(k);
        toexplore.pop();
        auto &n = nodes.at(k);
        if (n->children.size() == 0)
        {
            // Done
        }
        else if (n->children.size() == 1)
        {
            // Same range, just cycle by 1
            auto c = n->children[0];
            assert(r.ub > r.lb);
            int newk = r.lb + (r.k - r.lb + 1) % (r.ub - r.lb);
            node2range[c] = {r.lb, r.ub, newk};
            // if(comm_rank() == 0) printf(" children %d\n", c);
            toexplore.push(c);
        }
        else
        {
            int nc = n->children.size();
            int step = (r.ub - r.lb) / nc;
            // if(comm_rank() == 0) printf("lb ub step nc %d %d %d %d\n", r.lb, r.ub, step, nc);
            if (step == 0)
            { // To many children. Cycle by steps of 1.
                for (int i = 0; i < nc; i++)
                {
                    auto c = n->children[i];
                    auto start = r.lb + i % (r.ub - r.lb);
                    auto end = start + 1;
                    assert(start < end);
                    node2range[c] = {start, end, start};
                    toexplore.push(c);
                    // if(comm_rank() == 0) printf(" children %d start %d end %d\n", c, start, end);
                }
            }
            else
            {
                for (int i = 0; i < nc; i++)
                {
                    auto c = n->children[i];
                    auto start = r.lb + i * step;
                    auto end = r.lb + (i + 1) * step;
                    assert(start < end);
                    node2range[c] = {start, end, start};
                    toexplore.push(c);
                    // if(comm_rank() == 0) printf(" children %d start %d end %d\n", c, start, end);
                }
            }
        }
        // if(comm_rank() == 0) printf(" childrnode2range size %d\n", node2range.size());
    }
    node2rank = vector<int>(nblk, 0);
    for (int i = 0; i < nblk; i++)
    {
        node2rank[i] = node2range.at(i).k;
        // if(comm_rank() == 0)
        // printf("[%d] Node %d - %d\n", comm_rank(), i, node2rank[i]);
    }
}

// Allocate blocks[i,k]->A() when column k resides on this rank
// Fill blocks[i,k]->A() when column k resides on this rank
void DistMat::allocate_blocks(VectorXi& i2irow) {
    size_t total_allocated = 0;
    for (int k = 0; k < nblk; k++)
    {
        if (col2rank(k) == comm_rank())
        {
            // Allocate
            auto &n = nodes.at(k);
            auto &b = blocks.at({k, k});
            b->matA = make_unique<MatrixXd>(n->size, n->size);
            total_allocated += b->matA->size() * sizeof(double);
            b->A()->setZero();
            for (auto nirow : n->nbrs)
            {
                auto &b = blocks.at({nirow, k});
                b->matA = make_unique<MatrixXd>(b->rows.size(), n->size);
                total_allocated += b->matA->size() * sizeof(double);
                b->A()->setZero();
            }
            // Fill
            for (int j = n->start; j < n->end; j++)
            {
                for (SpMat::InnerIterator it(App, j); it; ++it)
                {
                    int i = it.row();
                    if (i >= n->start)
                    {
                        // Find bloc
                        int irow = i2irow[i];
                        auto &b = blocks.at({irow, k});
                        // Find row
                        auto found = lower_bound(b->rows.begin(), b->rows.end(), i);
                        assert(found != b->rows.end());
                        int ii = distance(b->rows.begin(), found);
                        int jj = j - n->start;
                        (*b->A())(ii, jj) = it.value();
                    }
                }
            }

        }
    }
    printf("%zd B allocated for blocks\n", total_allocated);
}

DistMat::DistMat(const SpMat& Ain, int nlevels, int block_size, int verb, bool log, std::string folder) : 
    my_rank(comm_rank()), n_ranks(comm_size()), A(Ain), nblk(-1), nlevels(nlevels), block_size(block_size), gemm_us(0), trsm_us(0), potf_us(0), scat_us(0), allo_us(0), verb(verb), do_log(log), folder(folder)
{
    // Initialize & prepare
    int N = A.rows();
    VectorXi colptr, rowval;
    get_csc_no_diag(A, &rowval, &colptr);
    // Partition the matrix
    VectorXi rangtab;
    algebraic_partitioning(N, rowval, colptr, nlevels, block_size, &perm, &rangtab, &depth);
    assert(rangtab.size() > 0);
    assert(perm.size() == N);
    nblk = rangtab.size()-1;
    // Permute matrix
    App = perm.asPermutation() * A * perm.asPermutation().transpose();
    // Create all supernodes
    // i2irow maps row/col -> irow (row/col -> block)
    VectorXi i2irow = build_supernodes(rangtab);
    // Compute elimination tree & neighbors
    // roots are the etree roots
    vector<int> roots = build_tree(i2irow);
    // Distribute the tree
    distribute_tree(roots);
    // Allocate blocks
    allocate_blocks(i2irow);
}

void DistMat::print()
{
    MatrixXd Aff = MatrixXd::Zero(perm.size(), perm.size());
    for (int k = 0; k < nblk; k++)
    {
        auto &n = nodes.at(k);
        int start = n->start;
        int size = n->size;
        Aff.block(start, start, size, size) = blocks.at({k, k})->A()->triangularView<Lower>();
        for (auto i : n->nbrs)
        {
            MatrixXd *Aik = blocks.at({i, k})->A();
            for (int ii = 0; ii < Aik->rows(); ii++)
            {
                for (int jj = 0; jj < Aik->cols(); jj++)
                {
                    Aff(blocks.at({i, k})->rows[ii], blocks.at({i, k})->cols[jj]) = (*Aik)(ii, jj);
                }
            }
        }
    }
    cout << Aff << endl;
}

// Factor a diagonal pivot in-place
void DistMat::potf(int krow)
{
    MatrixXd *Ass = blocks.at({krow, krow})->A();
    timer t0 = wctime();
    int err = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', Ass->rows(), Ass->data(), Ass->rows());
    timer t1 = wctime();
    potf_us += (long long)(elapsed(t0, t1) * 1e6);
    assert(err == 0);
}

// Trsm a panel block in-place
void DistMat::trsm(int2 kirow)
{
    int krow = kirow[0];
    int irow = kirow[1];
    MatrixXd *Ass = blocks.at({krow, krow})->A();
    MatrixXd *Ais = blocks.at({irow, krow})->A();
    timer t0 = wctime();
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                Ais->rows(), Ais->cols(), 1.0, Ass->data(), Ass->rows(), Ais->data(), Ais->rows());
    timer t1 = wctime();
    trsm_us += (long long)(elapsed(t0, t1) * 1e6);
}

// Perform a gemm between (i,k) and (j,k) and store the result at (i,j) in to_accumulate
void DistMat::gemm(int3 kijrow)
{
    int krow = kijrow[0];
    int irow = kijrow[1];
    int jrow = kijrow[2];
    MatrixXd *Ais = blocks.at({irow, krow})->A();
    MatrixXd *Ajs = blocks.at({jrow, krow})->A();
    // Do the math
    timer t0, t1, t2;
    t0 = wctime();
    auto Aij_acc = make_unique<MatrixXd>(Ais->rows(), Ajs->rows());
    t1 = wctime();
    if (jrow == irow)
    { // Aii_ = -Ais Ais^T
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                    Ais->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
    }
    else
    { // Aij_ = -Ais Ajs^T
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    Ais->rows(), Ajs->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), Ajs->data(), Ajs->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
    }
    t2 = wctime();
    {
        auto &mtx = blocks.at({irow, jrow})->to_accumulate_mtx;
        auto &acc = blocks.at({irow, jrow})->to_accumulate;
        lock_guard<mutex> lock(mtx);
        acc[krow] = move(Aij_acc);
    }
    allo_us += (long long)(elapsed(t0, t1) * 1e6);
    gemm_us += (long long)(elapsed(t1, t2) * 1e6);
}

void DistMat::accumulate(int3 kijrow)
{
    int krow = kijrow[0];
    int irow = kijrow[1];
    int jrow = kijrow[2];
    auto &mtx = blocks.at({irow, jrow})->to_accumulate_mtx;
    auto &acc = blocks.at({irow, jrow})->to_accumulate;
    {
        assert(!blocks.at({irow, jrow})->accumulating_busy.load());
        blocks.at({irow, jrow})->accumulating_busy.store(true);
    }
    unique_ptr<MatrixXd> Aij_acc;
    MatrixXd *Aij = blocks.at({irow, jrow})->A();
    timer t0, t1;
    {
        lock_guard<mutex> lock(mtx);
        Aij_acc = move(acc.at(krow));
        acc.erase(acc.find(krow));
    }
    t0 = wctime();
    if (jrow == irow)
    { // Aii_ = -Ais Ais^T
        auto Iids = get_subids(blocks.at({irow, krow})->rows, blocks.at({irow, jrow})->rows);
        for (int j = 0; j < Aij_acc->cols(); j++)
        {
            for (int i = j; i < Aij_acc->rows(); i++)
            {
                (*Aij)(Iids[i], Iids[j]) += (*Aij_acc)(i, j);
            }
        }
    }
    else
    { // Aij_ = -Ais Ajs^T
        auto Iids = get_subids(blocks.at({irow, krow})->rows, blocks.at({irow, jrow})->rows);
        auto Jids = get_subids(blocks.at({jrow, krow})->rows, blocks.at({irow, jrow})->cols);
        for (int j = 0; j < Aij_acc->cols(); j++)
        {
            for (int i = 0; i < Aij_acc->rows(); i++)
            {
                (*Aij)(Iids[i], Jids[j]) += (*Aij_acc)(i, j);
            }
        }
    }
    t1 = wctime();
    scat_us += (long long)(elapsed(t0, t1) * 1e6);
    {
        assert(blocks.at({irow, jrow})->accumulating_busy.load());
        blocks.at({irow, jrow})->accumulating_busy.store(false);
    }
}

void DistMat::factorize(int n_threads)
{
    for (int k = 0; k < nblk; k++) {
        const auto &n = nodes.at(k);
        for (auto i : n->nbrs) {
            for (auto j : n->nbrs) {
                if (i >= j) {
                    auto &b = blocks.at({i, j});
                    b->n_accumulate++;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer t0 = wctime();
    printf("Rank %d starting w/ %d threads\n", my_rank, n_threads);
    Logger log(1000000);
    Communicator comm(verb);
    Threadpool tp(n_threads, &comm, verb, "[" + to_string(my_rank) + "]_");
    Taskflow<int> pf(&tp, verb);
    Taskflow<int2> tf(&tp, verb);
    Taskflow<int3> gf(&tp, verb);
    Taskflow<int3> rf(&tp, verb);

    auto am_send_panel = comm.make_active_msg(
        [&](int &i, int &k, int &isize, int &ksize, view<double> &Aik, view<int> &js) {
            auto *b = this->blocks.at({i, k}).get();
            b->matA = make_unique<MatrixXd>(isize, ksize);
            memcpy(b->A()->data(), Aik.data(), Aik.size() * sizeof(double));
            vector<int> jss(js.size());
            memcpy(jss.data(), js.data(), js.size() * sizeof(int));
            for (auto j : jss)
            {
                gf.fulfill_promise(lower({k, i, j}));
            }
        });

    if (do_log)
    {
        tp.set_logger(&log);
        comm.set_logger(&log);
    }

    pf
        .set_mapping([&](int k) {
            assert(col2rank(k) == my_rank);
            return (k % n_threads);
        })
        .set_indegree([&](int k) {
            assert(col2rank(k) == my_rank);
            int ngemms = n_to_accumulate({k, k});
            return ngemms == 0 ? 1 : ngemms; // # gemms before ?
        })
        .set_task([&](int k) {
            assert(accumulated({k, k}) == n_to_accumulate({k, k}));
            assert(col2rank(k) == my_rank);
            potf(k);
        })
        .set_fulfill([&](int k) {
            assert(accumulated({k, k}) == n_to_accumulate({k, k}));
            assert(col2rank(k) == my_rank);
            auto &n = nodes.at(k);
            for (auto i : n->nbrs)
            {
                tf.fulfill_promise({k, i});
            }
        })
        .set_name([&](int k) {
            return "[" + to_string(my_rank) + "]_potf_" + to_string(k) + "_lvl" + to_string(depth.at(k));
        })
        .set_priority([](int k) {
            return 3.0;
        });

    tf
        .set_mapping([&](int2 ki) {
            assert(col2rank(ki[0]) == my_rank);
            return (ki[0] % n_threads);
        })
        .set_indegree([&](int2 ki) {
            assert(col2rank(ki[0]) == my_rank);
            int k = ki[0];
            int i = ki[1];
            assert(i > k);
            return n_to_accumulate({i, k}) + 1; // # gemm before + potf
        })
        .set_task([&](int2 ki) {
            assert(col2rank(ki[0]) == my_rank);
            assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));
            trsm(ki);
        })
        .set_fulfill([&](int2 ki) {
            assert(col2rank(ki[0]) == my_rank);
            assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));
            int k = ki[0];
            int i = ki[1];
            Node *n = nodes.at(k).get();
            map<int, vector<int>> deps;
            for (auto j : n->nbrs)
            {
                int dest = col2rank(lower({k, i, j})[2]);
                if (dest != my_rank)
                {
                    deps[dest] = {};
                }
            }
            for (auto j : n->nbrs)
            {
                int dest = col2rank(lower({k, i, j})[2]);
                if (dest != my_rank)
                {
                    deps[dest].push_back(j);
                }
                else
                {
                    gf.fulfill_promise(lower({k, i, j}));
                }
            }
            for (auto dep : deps)
            {
                int dest = dep.first;
                auto js = dep.second;
                MatrixXd *Aik = blocks.at({i, k})->A();
                int isize = Aik->rows();
                int ksize = Aik->cols();
                assert(Aik->size() == isize * ksize);
                auto vAik = view<double>(Aik->data(), Aik->size());
                auto vJs = view<int>(js.data(), js.size());
                am_send_panel->named_send(dest, "trsm_" + to_string(k) + "_" + to_string(i),
                                            i, k, isize, ksize, vAik, vJs);
            }
        })
        .set_name([&](int2 ki) {
            return "[" + to_string(my_rank) + "]_trsm_" + to_string(ki[0]) + "_" + to_string(ki[1]) + "_lvl" + to_string(depth[ki[0]]);
        })
        .set_priority([](int2 k) {
            return 2.0;
        });

    gf
        .set_mapping([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            return (kij[0] % n_threads);
        })
        .set_indegree([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            int i = kij[1];
            int j = kij[2];
            assert(j <= i);
            return (i == j ? 1 : 2); // Trsms
        })
        .set_task([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            gemm(kij);
        })
        .set_fulfill([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            assert(k <= j);
            assert(j <= i);
            // printf("gf %d %d %d -> rf %d %d %d\n", my_rank, k, i, j, k, i, j);
            rf.fulfill_promise(kij);
        })
        .set_name([&](int3 kij) {
            return "[" + to_string(my_rank) + "]_gemm_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_lvl" + to_string(depth[kij[0]]);
        })
        .set_priority([](int3) {
            return 1.0;
        });

    rf
        .set_mapping([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            return (kij[1] + kij[2]) % n_threads; // any i & j -> same thread. So k cannot appear in this expression
        })
        .set_indegree([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            return 1; // The corresponding gemm
        })
        .set_task([&](int3 kij) {
            assert(col2rank(kij[2]) == my_rank);
            blocks.at({kij[1], kij[2]})->accumulated++;
            accumulate(kij);
        })
        .set_fulfill([&](int3 kij) {
            int i = kij[1];
            int j = kij[2];
            if (i == j)
            {
                // printf("rf %d %d %d -> pf %d\n", k, i, j, i);
                pf.fulfill_promise(i);
            }
            else
            {
                // printf("rf %d %d %d -> tf %d %d\n", k, i, j, j, i);
                tf.fulfill_promise({j, i});
            }
        })
        .set_name([&](int3 kij) {
            return "[" + to_string(my_rank) + "]_acc_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_lvl" + to_string(depth[kij[0]]);
        })
        .set_priority([](int3) {
            return 4.0;
        })
        .set_binding([](int3) {
            return true;
        });

    for (int k = 0; k < nblk; k++)
    {
        if (col2rank(k) == my_rank)
        {
            if (n_to_accumulate({k, k}) == 0)
            {
                pf.fulfill_promise(k);
            }
        }
    }

    tp.join();
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Tp & Comms done\n");
    timer t1 = wctime();
    const double total_time = elapsed(t0, t1);
    const int nrows = A.rows();
    printf("Factorization done, time %3.2e s.\n", total_time);
    printf("Potf %3.2e s., %3.2e s./thread\n", double(potf_us / 1e6), double(potf_us / 1e6) / n_threads);
    printf("Trsm %3.2e s., %3.2e s./thread\n", double(trsm_us / 1e6), double(trsm_us / 1e6) / n_threads);
    printf("Gemm %3.2e s., %3.2e s./thread\n", double(gemm_us / 1e6), double(gemm_us / 1e6) / n_threads);
    printf("Allo %3.2e s., %3.2e s./thread\n", double(allo_us / 1e6), double(allo_us / 1e6) / n_threads);
    printf("Scat %3.2e s., %3.2e s./thread\n", double(scat_us / 1e6), double(scat_us / 1e6) / n_threads);
    const int n_threads_total = n_threads * n_ranks;
    const size_t n_rows_2 = (size_t)nrows * (size_t)nrows;
    const size_t n_rows_2_per_threads_total = n_rows_2 / n_threads_total;
    printf("FORMAT [my_rank] my_rank n_ranks n_threads n_threads_total nblk block_size nlevels nrows n_rows_2 n_rows_2_per_threads_total total_time\n");
    printf("[%d]>>>>snchol %d %d %d %d %d %d %d %d %zd %zd %3.2e\n", 
            my_rank, my_rank, n_ranks, n_threads, n_threads_total,
            nblk, block_size, nlevels, nrows, 
            n_rows_2, n_rows_2_per_threads_total, total_time);

    if (do_log > 0)
    {
        ofstream logfile;
        string filename = folder + "/snchol_" + to_string(n_ranks) + "_" + to_string(n_threads) + "_" + to_string(App.rows()) + ".log." + to_string(my_rank);
        printf("[%d] Logger saved to %s\n", my_rank, filename.c_str());
        logfile.open(filename);
        logfile << log;
        logfile.close();
    }
}

VectorXd DistMat::solve(VectorXd &b)
{
    // Send to rank 0
    for (int k = 0; k < nblk; k++)
    {
        if (my_rank != 0 && col2rank(k) == my_rank)
        {
            // Send column to 0
            {
                MatrixXd *Akk = blocks.at({k, k})->A();
                const int ksize = Akk->rows();
                MPI_Send(Akk->data(), ksize * ksize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            auto &n = nodes.at(k);
            for (auto i : n->nbrs)
            {
                MatrixXd *Aik = blocks.at({i, k})->A();
                const int isize = Aik->rows();
                const int ksize = Aik->cols();
                MPI_Send(Aik->data(), ksize * isize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        } else if (my_rank == 0 && col2rank(k) != 0) {
            const int from = col2rank(k);
            // Receive column
            {
                MatBlock* b = blocks.at({k,k}).get();
                const int ksize = b->rows.size();
                assert(b->cols.size() == ksize);
                b->matA = make_unique<MatrixXd>(ksize, ksize);
                MatrixXd* Akk = b->A();
                MPI_Recv(Akk->data(), ksize * ksize, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            auto &n = nodes.at(k);
            for (auto i : n->nbrs)
            {
                MatBlock* b = blocks.at({i,k}).get();
                const int isize = b->rows.size();
                const int ksize = b->cols.size();
                b->matA = make_unique<MatrixXd>(isize, ksize);
                MatrixXd* Aik = b->A();
                MPI_Recv(Aik->data(), ksize * isize, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // Solve
    VectorXd xglob = perm.asPermutation() * b;
    if(my_rank == 0) {
        // Set solution on each node
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            k->xsol = xglob.segment(k->start, k->size);
        }
        // Forward
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            // Pivot xs <- Lss^-1 xs
            MatrixXd *Lss = blocks.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocks.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // xn = -Lns xs
                cblas_dgemv(CblasColMajor, CblasNoTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), k->xsol.data(), 1, 0.0, xn.data(), 1);
                // Reduce into xn
                auto Iids = get_subids(blocks.at({irow, krow})->rows, blocks.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    n->xsol(Iids[i]) += xn(i);
                }
            }
        }
        // Backward
        for (int krow = nblk - 1; krow >= 0; krow--)
        {
            auto &k = nodes.at(krow);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocks.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // Fetch from xn
                auto Iids = get_subids(blocks.at({irow, krow})->rows, blocks.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    xn(i) = n->xsol(Iids[i]);
                }
                // xs -= Lns^T xn
                cblas_dgemv(CblasColMajor, CblasTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), xn.data(), 1, 1.0, k->xsol.data(), 1);
            }
            // xs = Lss^-T xs
            MatrixXd *Lss = blocks.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
        }
        // Back to x
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            xglob.segment(k->start, k->size) = k->xsol;
        }
    }
    return perm.asPermutation().transpose() * xglob;
}

SpMat DistMat::get_A() const {
    return A;
}

std::unique_ptr<DistMat> make_DistMat_from_file(std::string filename, int nlevels, int block_size, int verb, bool log, std::string folder) {
    printf("Building matrix from %s\n", filename.c_str());
    SpMat A = mmio::sp_mmread<double, int>(filename);
    auto dm = std::make_unique<DistMat>(A, nlevels, block_size, verb, log, folder);
    return dm;
}

std::unique_ptr<DistMat> make_DistMat_neglapl(int n, int d, int nlevels, int block_size, int verb, bool log, std::string folder) {
    printf("Building matrix as laplacian n = %d, d = %d\n", n, d);
    SpMat A = make_neglapl(n, d);
    auto dm = std::make_unique<DistMat>(A, nlevels, block_size, verb, log, folder);
    return dm;
}