import matplotlib.pyplot as plt
import numpy as np

sizes = np.array(
    [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1000,
        2000,
        3000,
        4000,
        5000,
        7500,
        10000,
    ]
)
sizes_f128 = np.array(
    [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1000,
        2000,
        3000,
    ]
)

# ----------------------------------------------------------------------
# data goes here


# ----------------------------------------------------------------------


plt.style.use("ggplot")
plt.rcParams["text.usetex"] = True

fig, axis = plt.subplots(1, 2)

axis[0].plot(sizes, sizes**3 / faer_matmul, label="faer")
axis[0].plot(sizes, sizes**3 / eigen_matmul, label="Eigen")
axis[0].plot(sizes, sizes**3 / openblas_matmul, label="OpenBLAS")
axis[0].plot(sizes, sizes**3 / mkl_matmul, label="MKL")
axis[0].plot(sizes, sizes**3 / blis_matmul, label="BLIS")
axis[0].legend()
axis[0].set_xlabel("f64 matrix dimension")
axis[0].set_ylabel("$n^3/$second")

axis[1].plot(sizes_f128, sizes_f128**3 / faer_matmul_f128, label="faer")
axis[1].plot(sizes_f128, sizes_f128**3 / eigen_matmul_f128, label="Eigen")
axis[1].legend()
axis[1].set_xlabel("f128 matrix dimension")
axis[1].set_ylabel("$n^3/$second")

axis[0].semilogx()
axis[1].semilogx()

fig.set_figwidth(4.8 * 2)
fig.tight_layout()
fig.savefig("matmul.png")

fig, axis = plt.subplots(1, 2)

axis[0].plot(sizes, sizes**3 / faer_qr, label="faer")
axis[0].plot(sizes, sizes**3 / eigen_qr, label="Eigen")
axis[0].plot(sizes, sizes**3 / openblas_qr, label="OpenBLAS")
axis[0].plot(sizes, sizes**3 / mkl_qr, label="MKL")
axis[0].plot(sizes, sizes**3 / blis_qr, label="BLIS")
axis[0].legend()
axis[0].set_xlabel("f64 matrix dimension")
axis[0].set_ylabel("$n^3/$second")

axis[1].plot(sizes_f128, sizes_f128**3 / faer_qr_f128, label="faer")
axis[1].plot(sizes_f128, sizes_f128**3 / eigen_qr_f128, label="Eigen")
axis[1].legend()
axis[1].set_xlabel("f128 matrix dimension")
axis[1].set_ylabel("$n^3/$second")

axis[0].semilogx()
axis[1].semilogx()


fig.set_figwidth(4.8 * 2)
fig.tight_layout()
fig.savefig("qr.png")

fig, axis = plt.subplots(1, 2)

axis[0].plot(sizes, sizes**3 / faer_evd, label="faer")
axis[0].plot(sizes[:-3], sizes[:-3]**3 / eigen_evd, label="Eigen")
axis[0].plot(sizes, sizes**3 / openblas_evd, label="OpenBLAS")
axis[0].plot(sizes, sizes**3 / mkl_evd, label="MKL")
axis[0].plot(sizes, sizes**3 / blis_evd, label="BLIS")
axis[0].legend()
axis[0].set_xlabel("f64 matrix dimension")
axis[0].set_ylabel("$n^3/$second")

axis[1].plot(sizes_f128, sizes_f128**3 / faer_evd_f128, label="faer")
axis[1].plot(sizes_f128[:-1], sizes_f128[:-1] ** 3 / eigen_evd_f128, label="Eigen")
axis[1].legend()
axis[1].set_xlabel("f128 matrix dimension")
axis[1].set_ylabel("$n^3/$second")

axis[0].semilogx()
axis[1].semilogx()


fig.set_figwidth(4.8 * 2)
fig.tight_layout()
fig.savefig("evd.png")
