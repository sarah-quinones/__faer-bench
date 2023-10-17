fn main() {
    let backend = std::env::var("BACKEND").unwrap_or("OPENBLAS".to_string());

    println!("cargo:rerun-if-env-changed=BACKEND");

    println!("cargo:rustc-link-search=./target/release");
    println!("cargo:rustc-link-lib=lapacke");
    match &*backend {
        "OPENBLAS" => println!("cargo:rustc-link-lib=openblas"),
        _ => {
            println!("cargo:rustc-link-lib=blis");
            println!("cargo:rustc-link-lib=lapack");
            println!("cargo:rustc-link-lib=gfortran");
        }
    }
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
}
