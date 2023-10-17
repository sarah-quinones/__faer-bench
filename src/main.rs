use dyn_stack::PodStack;
use faer_core::{ComplexField, Mat};
use std::io::Write;
use std::time::Instant;

fn time(f: impl FnOnce()) -> f64 {
    let now = Instant::now();
    f();
    now.elapsed().as_secs_f64()
}

extern crate openmp_sys;

fn timeit(mut f: impl FnMut(), time_limit: f64) -> f64 {
    let mut n_iters: u32 = 1;
    loop {
        let t = time(|| {
            for _ in 0..n_iters {
                f();
            }
        });

        if t >= time_limit || n_iters > 1_000_000_000 {
            return t / n_iters as f64;
        }

        let new_n_iters = (time_limit / t) as u32;
        if new_n_iters > n_iters {
            n_iters = new_n_iters;
        }
        n_iters *= 2;
    }
}

fn main() {
    let time_limit = 1.0;
    let sizes = [
        4, 8, 16, 32, 64, 128, 256, 512, 768, 1000, 2000, 3000, 4000, 5000, 7500, 10000,
    ];

    let n_threads = "12";

    std::env::set_var("RAYON_NUM_THREADS", n_threads);
    std::env::set_var("BLIS_NUM_THREADS", n_threads);
    std::env::set_var("OPENBLAS_NUM_THREADS", n_threads);
    std::env::set_var("MKL_NUM_THREADS", n_threads);
    std::env::set_var("OMP_NUM_THREADS", n_threads);

    let filename = std::env::var("BACKEND").unwrap_or("OPENBLAS".to_string());
    let mut blas_file = std::fs::File::create(&filename).unwrap();
    let mut faer_file = std::fs::File::create("faer").unwrap();

    let parallelism = faer_core::Parallelism::Rayon(0);

    let f64_ = true;
    let f128_ = true;

    let faer = true;
    let blas = true;

    let matmul = true;
    let qr = true;
    let evd = true;

    if f64_ {
        if matmul {
            if faer {
                let faer_matmul = sizes
                    .into_iter()
                    .map(|n| {
                        let mut c = Mat::<f64>::zeros(n, n);
                        let a = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let b = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());

                        let t = timeit(
                            || {
                                faer_core::mul::matmul(
                                    c.as_mut(),
                                    a.as_ref(),
                                    b.as_ref(),
                                    Some(1.0),
                                    1.0,
                                    parallelism,
                                )
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_matmul = {faer_matmul:?}").unwrap();
            }

            if blas {
                let blas_matmul = sizes
                    .into_iter()
                    .map(|n| {
                        let mut c = Mat::<f64>::zeros(n, n);
                        let a = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let b = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());

                        let t = timeit(
                            || unsafe {
                                blas_sys::dgemm_(
                                    "N\0".as_ptr() as _,
                                    "N\0".as_ptr() as _,
                                    &(n as _),
                                    &(n as _),
                                    &(n as _),
                                    &1.0,
                                    a.as_ptr(),
                                    &(a.col_stride() as _),
                                    b.as_ptr(),
                                    &(b.col_stride() as _),
                                    &1.0,
                                    c.as_mut_ptr(),
                                    &(c.col_stride() as _),
                                );
                            },
                            time_limit,
                        );
                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(blas_file, "{filename}_matmul = {blas_matmul:?}").unwrap();
            }
        }

        if qr {
            if faer {
                let faer_qr = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let mut qr = Mat::<f64>::zeros(n, n);
                        let bs = faer_qr::no_pivoting::compute::recommended_blocksize::<f64>(n, n);
                        let mut tau = Mat::<f64>::zeros(bs, n);

                        let mut mem = dyn_stack::GlobalPodBuffer::new(
                            faer_qr::no_pivoting::compute::qr_in_place_req::<f64>(
                                n,
                                n,
                                bs,
                                parallelism,
                                Default::default(),
                            )
                            .unwrap(),
                        );

                        let t = timeit(
                            || {
                                qr.clone_from(&c);
                                faer_qr::no_pivoting::compute::qr_in_place(
                                    qr.as_mut(),
                                    tau.as_mut(),
                                    parallelism,
                                    PodStack::new(&mut mem),
                                    Default::default(),
                                )
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_qr = {faer_qr:?}").unwrap();
            }

            if blas {
                let blas_qr = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let mut qr = Mat::<f64>::zeros(n, n);
                        let mut work = Mat::<f64>::zeros(n, n);
                        let mut tau = Mat::<f64>::zeros(n, 1);

                        let t = timeit(
                            || unsafe {
                                qr.clone_from(&c);
                                lapacke_sys::LAPACKE_dgeqrf_work(
                                    lapacke_sys::LAPACK_COL_MAJOR,
                                    n as _,
                                    n as _,
                                    qr.as_mut_ptr(),
                                    qr.col_stride() as _,
                                    tau.as_mut_ptr(),
                                    work.as_mut_ptr(),
                                    (n * n) as _,
                                );
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(blas_file, "{filename}_qr = {blas_qr:?}").unwrap();
            }
        }

        if evd {
            if faer {
                let faer_evd = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let mut u = Mat::<f64>::zeros(n, n);
                        let mut s_re = Mat::<f64>::zeros(n, 1);
                        let mut s_im = Mat::<f64>::zeros(n, 1);

                        let params = Default::default();
                        let mut mem = dyn_stack::GlobalPodBuffer::new(
                            faer_evd::compute_evd_req::<f64>(
                                n,
                                faer_evd::ComputeVectors::Yes,
                                parallelism,
                                params,
                            )
                            .unwrap(),
                        );

                        let t = timeit(
                            || {
                                faer_evd::compute_evd_real(
                                    c.as_ref(),
                                    s_re.as_mut(),
                                    s_im.as_mut(),
                                    Some(u.as_mut()),
                                    parallelism,
                                    PodStack::new(&mut mem),
                                    params,
                                );
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_evd = {faer_evd:?}").unwrap();
            }

            if blas {
                let blas_evd = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());
                        let mut evd = Mat::<f64>::zeros(n, n);
                        let mut ul = Mat::<f64>::zeros(n, n);
                        let mut ur = Mat::<f64>::zeros(n, n);
                        let mut s_re = Mat::<f64>::zeros(n, 1);
                        let mut s_im = Mat::<f64>::zeros(n, 1);
                        let mut work = Mat::<f64>::zeros(n, n);

                        let t = timeit(
                            || unsafe {
                                evd.clone_from(&c);
                                lapacke_sys::LAPACKE_dgeev_work(
                                    lapacke_sys::LAPACK_COL_MAJOR,
                                    b'V' as _,
                                    b'N' as _,
                                    n as _,
                                    evd.as_mut_ptr(),
                                    evd.col_stride() as _,
                                    s_re.as_mut_ptr(),
                                    s_im.as_mut_ptr(),
                                    ul.as_mut_ptr(),
                                    ul.col_stride() as _,
                                    ur.as_mut_ptr(),
                                    ur.col_stride() as _,
                                    work.as_mut_ptr(),
                                    (n * n) as _,
                                );
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(blas_file, "{filename}_evd = {blas_evd:?}").unwrap();
            }
        }
    }

    if f128_ {
        let sizes = [4, 8, 16, 32, 64, 128, 256, 512, 768, 1000, 2000, 3000];
        #[allow(non_camel_case_types)]
        type f128 = qd::Double<f64>;

        if matmul {
            if faer {
                let faer_matmul = sizes
                    .into_iter()
                    .map(|n| {
                        let mut c = Mat::<f128>::zeros(n, n);
                        let a = Mat::<f128>::from_fn(n, n, |_, _| {
                            f128::faer_from_f64(rand::random::<f64>())
                        });
                        let b = Mat::<f128>::from_fn(n, n, |_, _| {
                            f128::faer_from_f64(rand::random::<f64>())
                        });

                        let t = timeit(
                            || {
                                faer_core::mul::matmul(
                                    c.as_mut(),
                                    a.as_ref(),
                                    b.as_ref(),
                                    Some(f128::faer_one()),
                                    f128::faer_one(),
                                    parallelism,
                                )
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_matmul_f128 = {faer_matmul:?}").unwrap();
            }
        }

        if qr {
            if faer {
                let faer_qr = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f128>::from_fn(n, n, |_, _| {
                            f128::faer_from_f64(rand::random::<f64>())
                        });
                        let mut qr = Mat::<f128>::zeros(n, n);
                        let bs = faer_qr::no_pivoting::compute::recommended_blocksize::<f128>(n, n);
                        let mut tau = Mat::<f128>::zeros(bs, n);

                        let mut mem = dyn_stack::GlobalPodBuffer::new(
                            faer_qr::no_pivoting::compute::qr_in_place_req::<f128>(
                                n,
                                n,
                                bs,
                                parallelism,
                                Default::default(),
                            )
                            .unwrap(),
                        );

                        let t = timeit(
                            || {
                                qr.clone_from(&c);
                                faer_qr::no_pivoting::compute::qr_in_place(
                                    qr.as_mut(),
                                    tau.as_mut(),
                                    parallelism,
                                    PodStack::new(&mut mem),
                                    Default::default(),
                                )
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_qr_f128 = {faer_qr:?}").unwrap();
            }
        }

        if evd {
            if faer {
                let faer_evd = sizes
                    .into_iter()
                    .map(|n| {
                        let c = Mat::<f128>::from_fn(n, n, |_, _| {
                            f128::faer_from_f64(rand::random::<f64>())
                        });
                        let mut u = Mat::<f128>::zeros(n, n);
                        let mut s_re = Mat::<f128>::zeros(n, 1);
                        let mut s_im = Mat::<f128>::zeros(n, 1);

                        let params = Default::default();
                        let mut mem = dyn_stack::GlobalPodBuffer::new(
                            faer_evd::compute_evd_req::<f128>(
                                n,
                                faer_evd::ComputeVectors::Yes,
                                parallelism,
                                params,
                            )
                            .unwrap(),
                        );

                        let t = timeit(
                            || {
                                faer_evd::compute_evd_real(
                                    c.as_ref(),
                                    s_re.as_mut(),
                                    s_im.as_mut(),
                                    Some(u.as_mut()),
                                    parallelism,
                                    PodStack::new(&mut mem),
                                    params,
                                );
                            },
                            time_limit,
                        );

                        dbg!(n, t);
                        t
                    })
                    .collect::<Vec<_>>();
                writeln!(faer_file, "faer_evd_f128 = {faer_evd:?}").unwrap();
            }
        }
    }
}
