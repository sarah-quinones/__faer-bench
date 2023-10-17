#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <iostream>
#include <limits>
#include <fstream>

using f32 = float;
using f64 = double;
using c32 = std::complex<f32>;
using c64 = std::complex<f64>;

template <typename T> struct EigenSolver {
  using Type = Eigen::EigenSolver<T>;
};

template <typename T>
struct EigenSolver<Eigen::Matrix<std::complex<T>, -1, -1>> {
  using Type =
      Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<T>, -1, -1>>;
};

namespace double_f64 {
struct DoubleF64 {
  f64 x0;
  f64 x1;

  DoubleF64() = default;
  inline DoubleF64(double x) : x0{x}, x1{} {}
  inline DoubleF64(double x, double y) : x0{x}, x1{y} {}
};

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
inline double quick_two_sum(double a, double b, double &err) {
  double s = a + b;
  err = b - (s - a);
  return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
inline double quick_two_diff(double a, double b, double &err) {
  double s = a - b;
  err = (a - s) - b;
  return s;
}

/* Computes fl(a+b) and err(a+b).  */
inline double two_sum(double a, double b, double &err) {
  double s = a + b;
  double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

/* Computes fl(a-b) and err(a-b).  */
inline double two_diff(double a, double b, double &err) {
  double s = a - b;
  double bb = s - a;
  err = (a - (s - bb)) - (b + bb);
  return s;
}

/* Computes fl(a*b) and err(a*b). */
inline double two_prod(double a, double b, double &err) {
  double p = a * b;
  err = std::fma(a, b, -p);
  return p;
}

inline DoubleF64 operator+(const DoubleF64 &a, const DoubleF64 &b) {
  double s, e;

  s = two_sum(a.x0, b.x0, e);
  e += (a.x1 + b.x1);
  s = quick_two_sum(s, e, e);
  return DoubleF64{s, e};
}

inline DoubleF64 operator-(const DoubleF64 &a) {
  return DoubleF64{-a.x0, -a.x1};
}

inline DoubleF64 operator-(const DoubleF64 &a, const DoubleF64 &b) {
  double s1, s2, t1, t2;
  s1 = two_diff(a.x0, b.x0, s2);
  t1 = two_diff(a.x1, b.x1, t2);
  s2 += t1;
  s1 = quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = quick_two_sum(s1, s2, s2);
  return DoubleF64{s1, s2};
}

inline DoubleF64 operator*(const DoubleF64 &a, const DoubleF64 &b) {
  double p1, p2;

  p1 = two_prod(a.x0, b.x0, p2);
  p2 += (a.x0 * b.x1 + a.x1 * b.x0);
  p1 = quick_two_sum(p1, p2, p2);
  return DoubleF64(p1, p2);
}

inline DoubleF64 operator/(const DoubleF64 &a, const DoubleF64 &b) {
  double s1, s2;
  double q1, q2;
  DoubleF64 r;

  q1 = a.x0 / b.x0; /* approximate quotient */

  /* compute  this - q1 * dd */
  r = b * q1;
  s1 = two_diff(a.x0, r.x0, s2);
  s2 -= r.x1;
  s2 += a.x1;

  /* get next approximation */
  q2 = (s1 + s2) / b.x0;

  /* renormalize */
  r.x0 = quick_two_sum(q1, q2, r.x1);
  return r;
}

inline DoubleF64 &operator+=(DoubleF64 &a, const DoubleF64 &b) {
  a = a + b;
  return a;
}
inline DoubleF64 &operator-=(DoubleF64 &a, const DoubleF64 &b) {
  a = a - b;
  return a;
}
inline DoubleF64 &operator*=(DoubleF64 &a, const DoubleF64 &b) {
  a = a * b;
  return a;
}
inline DoubleF64 &operator/=(DoubleF64 &a, const DoubleF64 &b) {
  a = a / b;
  return a;
}

inline DoubleF64 sqrt(DoubleF64 const &a) {
  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto infty = std::numeric_limits<double>::infinity();
  if (a.x0 == 0.0) {
    return DoubleF64{};
  } else if (a.x0 < 0.0) {
    return DoubleF64{nan, nan};
  } else if (a.x0 == infty) {
    return DoubleF64{infty, infty};
  } else {
    auto x = 1.0 / std::sqrt(a.x0);
    auto ax = DoubleF64{a.x0 * x};
    return ax + (a - ax * ax) * DoubleF64{x * 0.5};
  }
}

inline DoubleF64 fabs(DoubleF64 const &a) { return a.x0 < 0.0 ? -a : a; }
inline DoubleF64 abs(DoubleF64 const &a) { return a.x0 < 0.0 ? -a : a; }

inline bool isfinite(DoubleF64 const &a) { return std::isfinite(a.x0); }
inline bool isinf(DoubleF64 const &a) { return std::isinf(a.x0); }
inline bool isnan(DoubleF64 const &a) {
  return std::isnan(a.x0) || std::isnan(a.x1);
}

inline bool operator==(const DoubleF64 &a, const DoubleF64 &b) {
  return a.x0 == b.x0 && a.x1 == b.x1;
}
inline bool operator!=(const DoubleF64 &a, const DoubleF64 &b) {
  return !(a == b);
}
inline bool operator<(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 < b.x1);
}
inline bool operator>(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 > b.x1);
}
inline bool operator<=(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 <= b.x1);
}
inline bool operator>=(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 >= b.x1);
}
} // namespace double_f64

using f128 = double_f64::DoubleF64;
using c128 = std::complex<double_f64::DoubleF64>;

namespace std {
template <> struct numeric_limits<f128> {
  static constexpr auto is_specialized = true;
  static constexpr auto is_signed = true;
  static constexpr auto is_integer = false;
  static constexpr auto is_exact = false;
  static constexpr auto has_infinity = true;
  static constexpr auto has_quiet_NaN = true;
  static constexpr auto has_signaling_NaN = true;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = std::round_to_nearest;
  static constexpr auto is_iec559 = false;
  static constexpr auto is_bounded = true;
  static constexpr auto is_modulo = false;
  static constexpr auto digits = 100;

  static f128 epsilon() { return f128{1e-30}; }
  static f128 min() { return f128{1e-200}; }
  static f128 max() { return 1.0 / min(); }
  static f128 quiet_NaN() {
    return f128{
        std::numeric_limits<f64>::quiet_NaN(),
        std::numeric_limits<f64>::quiet_NaN(),
    };
  }
  static f128 infinity() {
    return f128{
        std::numeric_limits<f64>::infinity(),
        std::numeric_limits<f64>::infinity(),
    };
  }
};
} // namespace std

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
  std::ofstream file("eigen");
  {
    using Mat = Eigen::Matrix<f64, -1, -1>;
    int ns[] = {4,   8,    16,   32,   64,   128,  256,  512,
                768, 1000, 2000, 3000, 4000, 5000, 7500, 10000};

    std::cout << "eigen_matmul = [" << '\n';
    file << "eigen_matmul = [" << '\n';
    for (auto n : ns) {
      Mat c(n, n);
      Mat a(n, n);
      Mat b(n, n);
      c.setRandom();
      a.setRandom();
      b.setRandom();

      double alpha = 1.0;
      double beta = 1.0;
      auto time = timeit([&] { c.noalias() += a * b; });
      std::cout << time << ',' << std::endl;
      file << time << ',' << std::endl;
    }
    std::cout << ']' << '\n';
    file << ']' << '\n';

    std::cout << "eigen_qr = [" << '\n';
    file << "eigen_qr = [" << '\n';
    for (auto n : ns) {
      Mat c(n, n);
      Mat work(n, n);
      Mat qr(n, n);
      Mat tau(n, 1);
      c.setRandom();

      Eigen::HouseholderQR<Mat> b(n, n);

      auto time = timeit([&] { b.compute(c); });
      std::cout << time << ',' << std::endl;
      file << time << ',' << std::endl;
    }
    std::cout << ']' << '\n';
    file << ']' << '\n';

    std::cout << "eigen_evd = [" << '\n';
    file << "eigen_evd = [" << '\n';
    for (auto n : ns) {
      if (n > 4000) continue;
      Mat c(n, n);
      Mat work(n, n);
      c.setRandom();

      Eigen::EigenSolver<Mat> evd(n);

      auto time = timeit([&] { evd.compute(c); });
      std::cout << time << ',' << std::endl;
      file << time << ',' << std::endl;
    }
    std::cout << ']' << '\n';
    file << ']' << '\n';
  }

  {
    using Mat = Eigen::Matrix<f128, -1, -1>;
    int ns[] = {4, 8, 16, 32, 64, 128, 256, 512, 768, 1000, 2000, 3000};

    std::cout << "eigen_matmul_f128 = [" << '\n';
    file << "eigen_matmul_f128 = [" << '\n';
    for (auto n : ns) {
      Mat c(n, n);
      Mat a(n, n);
      Mat b(n, n);
      c.setRandom();
      a.setRandom();
      b.setRandom();

      double alpha = 1.0;
      double beta = 1.0;
      auto time = timeit([&] { c.noalias() += a * b; });
      std::cout << time << ',' << std::endl;
      file << time << ',' << std::endl;
    }
    std::cout << ']' << '\n';
    file << ']' << '\n';

    std::cout << "eigen_qr_f128 = [" << '\n';
    file << "eigen_qr_f128 = [" << '\n';
    for (auto n : ns) {
      Mat c(n, n);
      Mat work(n, n);
      Mat qr(n, n);
      Mat tau(n, 1);
      c.setRandom();

      Eigen::HouseholderQR<Mat> b(n, n);

      auto time = timeit([&] { b.compute(c); });
      std::cout << time << ",\n";
      file << time << ",\n";
    }
    std::cout << ']' << '\n';
    file << ']' << '\n';

    std::cout << "eigen_evd_f128 = [" << '\n';
    file << "eigen_evd_f128 = [" << '\n';
    for (auto n : ns) {
      if (n > 2000) continue;
      Mat c(n, n);
      Mat work(n, n);
      c.setRandom();

      Eigen::EigenSolver<Mat> evd(n);

      auto time = timeit([&] { evd.compute(c); });
      std::cout << time << ',' << std::endl;
      file << time << ',' << std::endl;
    }
    std::cout << ']' << std::endl;
    file << ']' << std::endl;
  }
}
