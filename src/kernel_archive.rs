use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

pub(crate) type KernelCache = OnceLock<Result<HashMap<String, Vec<u8>>, String>>;

pub(crate) fn load_kernel(
    name: &str,
    extension: &str,
    compressed: &'static [u8],
    cache: &'static KernelCache,
) -> anyhow::Result<&'static [u8]> {
    let kernels = cache.get_or_init(|| unpack(compressed).map_err(|error| error.to_string()));
    let kernels = kernels.as_ref().map_err(|error| anyhow::anyhow!("{error}"))?;
    let filename = format!("{name}.{extension}");
    kernels.get(&filename).map(Vec::as_slice)
        .ok_or_else(|| anyhow::anyhow!("No embedded kernel for {filename}"))
}

fn unpack(compressed: &[u8]) -> anyhow::Result<HashMap<String, Vec<u8>>> {
    let decoder = zstd::stream::read::Decoder::new(compressed)?;
    let mut archive = tar::Archive::new(decoder);
    let mut kernels = HashMap::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let name = entry.path()?.to_string_lossy().into_owned();
        if !name.ends_with(".metallib") && !name.ends_with(".dxil") {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        kernels.insert(name, bytes);
    }
    Ok(kernels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_kernel_cache_preserves_bytes_and_reports_missing_kernels() {
        static CACHE: KernelCache = OnceLock::new();
        let mut archive = tar::Builder::new(Vec::new());
        for (name, bytes) in [("first.metallib", &b"metal\0bytes"[..]), ("second.dxil", &b"dxil\0bytes"[..])] {
            let mut header = tar::Header::new_ustar();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, bytes).unwrap();
        }
        let compressed = zstd::stream::encode_all(&archive.into_inner().unwrap()[..], 0).unwrap();
        let compressed = Box::leak(compressed.into_boxed_slice());
        let first = load_kernel("first", "metallib", compressed, &CACHE).unwrap();
        assert_eq!(first, b"metal\0bytes");
        assert_eq!(load_kernel("second", "dxil", compressed, &CACHE).unwrap(), b"dxil\0bytes");
        assert_eq!(first.as_ptr(), load_kernel("first", "metallib", compressed, &CACHE).unwrap().as_ptr());
        assert!(load_kernel("missing", "metallib", compressed, &CACHE).is_err());
    }

    #[test]
    fn corrupt_kernel_cache_returns_an_error() {
        static CACHE: KernelCache = OnceLock::new();
        assert!(load_kernel("first", "metallib", b"invalid zstd", &CACHE).is_err());
    }
}
