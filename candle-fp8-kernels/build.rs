// Build script compiling the block-wise FP8 fused dequant+GEMM CUDA kernel into a static lib.
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels/fp8_gemm.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let builder = KernelBuilder::new()
        .source_files(vec!["kernels/fp8_gemm.cu"])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC");

    let out_file = out_dir.join("libfp8kernels.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=fp8kernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    Ok(())
}
