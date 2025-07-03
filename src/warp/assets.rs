
use once_cell::sync::OnceCell;
use std::io::Cursor;

#[cfg(not(feature = "embed-assets"))]
use std::path::Path;

/// A Zstandard‑compressed asset that is decompressed at most once and
/// cached for the program’s lifetime.
pub struct Asset {
    /// When `embed-assets` is **on**, we hold the compile‑time embedded bytes.
    #[cfg(feature = "embed-assets")]
    compressed: &'static [u8],

    /// When `embed-assets` is **off**, we just remember the on‑disk path.
    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,

    /// Once‑initialised buffer containing the decompressed payload.
    decompressed: OnceCell<Vec<u8>>,
}

impl Asset {
    /// Constructor used when assets are **embedded**.
    #[cfg(feature = "embed-assets")]
    pub const fn new_embedded(compressed: &'static [u8]) -> Self {
        Self { compressed, decompressed: OnceCell::new() }
    }

    /// Constructor used when assets are kept **on disk**.
    #[cfg(not(feature = "embed-assets"))]
    pub const fn new_file(path: &'static str) -> Self {
        Self { path, decompressed: OnceCell::new() }
    }

    /// Return the decompressed bytes; performs work only on the first call.
    pub fn bytes(&'static self) -> &'static [u8] {
        self.decompressed.get_or_init(|| {
            // 1. Obtain the compressed data according to build mode.
            #[cfg(feature = "embed-assets")]
            let compressed: Vec<u8> = self.compressed.to_vec();

            #[cfg(not(feature = "embed-assets"))]
            let compressed: Vec<u8> = std::fs::read(Path::new(self.path))
                .expect("failed to read compressed asset file");

            // 2. Decompress the entire buffer.
            let mut cursor = Cursor::new(compressed);
            zstd::stream::decode_all(&mut cursor)
                .expect("zstd decompression failed")
        })
    }
    pub fn as_str(&'static self) -> &'static str {
        std::str::from_utf8(self.bytes())
            .expect("asset is not valid UTF-8")
    }
}

/// Embed a Zstd‑compressed file as a lazily decompressed [`Asset`].

#[macro_export]
macro_rules! embed_zst_asset {
    ($vis:vis $name:ident, $path:literal) => {
        // --- Embedded build (default) -----------------------------------
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::warp::assets::Asset =
            $crate::warp::assets::Asset::new_embedded(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/", $path)
            ));

        // --- Disk build (no default features) ---------------------------
        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::warp::assets::Asset =
            $crate::warp::assets::Asset::new_file(concat!(
                env!("CARGO_MANIFEST_DIR"), "/", $path
            ));
    };
}
