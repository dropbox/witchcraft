fn main() {
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
