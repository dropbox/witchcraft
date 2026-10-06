fn main() {
    build_kernels();
    println!("cargo:rerun-if-changed=src/capi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    std::fs::create_dir_all("include").expect("create include directory");
    cbindgen::Builder::new()
        .with_src("src/capi.rs")
        .with_config(cbindgen::Config::from_file("cbindgen.toml").expect("read cbindgen.toml"))
        .generate()
        .expect("generate C API header")
        .write_to_file("include/witchcraft.h");

    if std::env::var("CARGO_FEATURE_NAPI").is_ok() {
        napi_build::setup();
    }
}

fn build_kernels() {
    use std::{env, fs, io::Read, path::PathBuf, process::Command};

    let target = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let (platform, archive_name, loader) = match target.as_str() {
        "macos" if env::var_os("CARGO_FEATURE_NESO_METAL").is_some() =>
            ("metal_nosimd", "kernels_metal_nosimd.tar.zst", "neso_metal_gen.rs"),
        "windows" if env::var_os("CARGO_FEATURE_NESO_D3D12").is_some() =>
            ("hlsl", "kernels_dxil.tar.zst", "neso_d3d12_gen.rs"),
        _ => return,
    };
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let neso = env::var_os("NESO_DIR").map(PathBuf::from)
        .unwrap_or_else(|| root.parent().unwrap().join("neso"));
    let python = neso.join(if cfg!(windows) { "env/Scripts/python.exe" } else { "env/bin/python" });
    let cache = root.join("kernels/out").join(archive_name);
    println!("cargo:rerun-if-env-changed=NESO_DIR");
    println!("cargo:rerun-if-env-changed=DXC_PATH");
    println!("cargo:rerun-if-changed={}", python.display());
    println!("cargo:rerun-if-changed={}", neso.join("src").display());
    println!("cargo:rerun-if-changed={}", cache.display());
    for file in ["build.py", "compile_step.py", "gen_rust.py", "kernel_configs.py", "encoder_kernels.py"] {
        println!("cargo:rerun-if-changed=kernels/{file}");
    }
    if python.is_file() {
        let built = Command::new(&python).arg(root.join("kernels/build.py")).arg(platform)
            .env("NESO_DIR", &neso).status().map(|status| status.success()).unwrap_or(false);
        if !built {
            println!("cargo:warning=Neso compilation failed; using cached {archive_name}");
        }
    } else {
        println!("cargo:warning=Neso unavailable; using cached {archive_name}");
    }
    let compressed = fs::File::open(&cache).unwrap_or_else(|error|
        panic!("Cannot open {}: {error}. Install Neso and run kernels/build.py {platform} to generate the cache.", cache.display()));
    let decoder = zstd::stream::read::Decoder::new(compressed).expect("decode kernel cache");
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().expect("read kernel cache") {
        let mut entry = entry.expect("read kernel cache entry");
        if entry.path().expect("read kernel cache path").as_ref() == std::path::Path::new(loader) {
            let mut source = Vec::new();
            entry.read_to_end(&mut source).expect("read cached kernel loader");
            let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join(loader);
            if fs::read(&output).ok().as_deref() != Some(source.as_slice()) {
                fs::write(output, source).expect("write cached kernel loader");
            }
            return;
        }
    }
    panic!("Kernel cache {} is missing {loader}", cache.display());
}
