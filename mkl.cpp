#include <eigen3/Eigen/Core>

#include <chrono>
#include <iostream>

#include <mkl/mkl_lapacke.h>
#include <mkl/mkl_cblas.h>

template <typename F> auto time1(F f) -> double {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

template <typename F> auto timeit(F f) -> double {
  auto time_limit = 1e-0;
  unsigned n_iters = 1;
  while (true) {
    auto t = time1([&] {
      for (unsigned i = 0; i < n_iters; ++i) {
        f();
      }
    });

    if (t >= time_limit || n_iters > 1'000'000'000) {
      return t / n_iters;
    }

    n_iters = 2 * std::max<unsigned>(n_iters, time_limit / t);
  }
}

auto main() -> int {
  using Mat = Eigen::Matrix<double, -1, -1>;
  int ns[] = {4,   8,    16,   32,   64,   128,  256,  512,
              768, 1000, 2000, 3000, 4000, 5000, 7500, 10000};
  for (auto n : ns) {
    Mat c(n, n);
    Mat a(n, n);
    Mat b(n, n);
    c.setRandom();
    a.setRandom();
    b.setRandom();

    double alpha = 1.0;
    double beta = 1.0;
    auto time = timeit([&] {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
                  a.data(), n, b.data(), n, beta, c.data(), n);
    });
    std::cout << time << ',' << std::endl;
  }

  for (auto n : ns) {
    Mat c(n, n);
    Mat work(n, n);
    Mat qr(n, n);
    Mat tau(n, 1);
    c.setRandom();

    auto time = timeit([&] {
      qr = c;
      LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, n, n, qr.data(), n, tau.data(),
                          work.data(), (n * n));
    });
    std::cout << time << ',' << std::endl;
  }

  for (auto n : ns) {
    Mat c(n, n);
    Mat work(n, n);
    c.setRandom();

    Mat evd(n, n);
    Mat ul(n, n);
    Mat ur(n, n);
    Mat s_re(n, 1);
    Mat s_im(n, 1);

    auto time = timeit([&] {
      evd = c;
      LAPACKE_dgeev_work(LAPACK_COL_MAJOR, 'V', 'N', n, evd.data(), n,
                         s_re.data(), s_im.data(), ul.data(), n, ur.data(), n,
                         work.data(), (n * n));
    });
    std::cout << time << ',' << std::endl;
  }
}
