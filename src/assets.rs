//! assets.rs — Simple asset file loader with lazy caching.

#[cfg(not(feature = "embed-assets"))]
use once_cell::sync::OnceCell;
use std::path::Path;

/// A raw (uncompressed) asset that can be memory-mapped.
pub struct Asset {
    #[cfg(feature = "embed-assets")]
    data: &'static [u8],

    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,
    #[cfg(not(feature = "embed-assets"))]
    cached: OnceCell<Vec<u8>>,
}

impl Asset {
    #[cfg(feature = "embed-assets")]
    pub const fn new(data: &'static [u8]) -> Self {
        Self { data }
    }

    #[cfg(not(feature = "embed-assets"))]
    pub const fn new(path: &'static str) -> Self {
        Self {
            path,
            cached: OnceCell::new(),
        }
    }

    #[cfg(feature = "embed-assets")]
    pub fn bytes(&'static self, _assets: &Path) -> Result<&'static [u8], ()> {
        Ok(self.data)
    }

    #[cfg(not(feature = "embed-assets"))]
    pub fn bytes(&'static self, assets: &Path) -> Result<&'static [u8], ()> {
        self.cached
            .get_or_try_init(|| match std::fs::read(assets.join(self.path)) {
                Ok(bytes) => Ok(bytes),
                Err(e) => {
                    log::warn!("failed to read warp asset {}: {}", self.path, e);
                    Err(())
                }
            })
            .map(|bytes| bytes.as_slice())
    }
}

#[macro_export]
macro_rules! embed_asset {
    ($vis:vis $name:ident, $path:literal) => {
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $path)
            ));

        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new($path);
    };
}
