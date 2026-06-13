fn main() {
    if std::env::var("CARGO_FEATURE_NAPI").is_ok() {
        napi_build::setup();
    }
}
