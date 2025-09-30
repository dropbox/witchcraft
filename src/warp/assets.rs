//! assets.rs — generic Zstd asset loader with optional compile-time embedding.

use once_cell::sync::OnceCell;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};

#[cfg(not(feature = "embed-assets"))]
use std::fs;

/// A Zstandard-compressed asset that is decompressed at most once and
/// cached for the program’s lifetime.
pub struct Asset {
    #[cfg(feature = "embed-assets")]
    compressed: &'static [u8],

    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,

    decompressed: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    #[cfg(feature = "embed-assets")]
    pub const fn new_embedded(compressed: &'static [u8]) -> Self {
        Self {
            compressed,
            decompressed: OnceCell::new(),
        }
    }

    #[cfg(not(feature = "embed-assets"))]
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            decompressed: OnceCell::new(),
        }
    }

    /// Return the decompressed bytes; performs work only on first call.
    pub fn bytes(&'static self, assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.decompressed
            .get_or_init(|| {
                #[cfg(feature = "embed-assets")]
                let compressed: Vec<u8> = self.compressed.to_vec();

                #[cfg(not(feature = "embed-assets"))]
                let compressed: Vec<u8> = match fs::read(Path::new(&assets.join(self.path))) {
                    Ok(c) => c,
                    Err(e) => {
                        println!("failed to read warp asset {}: {}", self.path, e);
                        return Err(());
                    }
                };

                let mut cursor = Cursor::new(compressed);
                match zstd::stream::decode_all(&mut cursor) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        println!("failed to decompress warp asset {}: {}", self.path, e);
                        Err(())
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

#[macro_export]
macro_rules! embed_zst_asset {
    ($vis:vis $name:ident, $path:literal) => {
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::warp::assets::Asset =
            $crate::warp::assets::Asset::new_embedded(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/", $path)
            ));

        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::warp::assets::Asset =
            $crate::warp::assets::Asset::new_file($path);
    };
}
