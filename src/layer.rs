use std::fmt;

/// Errors specific to layer operations.
#[derive(Debug)]
pub enum LayerError {
    NotFound(i64),
    HasChildren(i64),
    NotActive(i64),
    /// stop_at is not an ancestor of the compaction source layer
    NotInChain { stop_at: i64, layer_id: i64 },
    Sqlite(rusqlite::Error),
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerError::NotFound(id) => write!(f, "layer {id} not found"),
            LayerError::HasChildren(id) => write!(f, "layer {id} has children"),
            LayerError::NotActive(id) => write!(f, "layer {id} is not active"),
            LayerError::NotInChain { stop_at, layer_id } => {
                write!(f, "layer {stop_at} is not an ancestor of {layer_id}")
            }
            LayerError::Sqlite(e) => write!(f, "sqlite error: {e}"),
        }
    }
}

impl std::error::Error for LayerError {}

impl From<rusqlite::Error> for LayerError {
    fn from(e: rusqlite::Error) -> Self {
        LayerError::Sqlite(e)
    }
}

/// Layer status in the persistent tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerStatus {
    /// Accepting writes.
    Active,
    /// Immutable (advisory — application enforced, not IVF-driven).
    Sealed,
    /// Replaced by compaction output, pending GC.
    Compacted,
}

impl LayerStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            LayerStatus::Active => "active",
            LayerStatus::Sealed => "sealed",
            LayerStatus::Compacted => "compacted",
        }
    }
}

impl std::str::FromStr for LayerStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(LayerStatus::Active),
            "sealed" => Ok(LayerStatus::Sealed),
            "compacted" => Ok(LayerStatus::Compacted),
            other => Err(format!("unknown layer status: {other}")),
        }
    }
}

impl fmt::Display for LayerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A layer in the persistent tree.
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub name: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub status: LayerStatus,
}

/// Identifies a position in the layer tree for query resolution.
/// For the common flat case (overlay + baseline), this is just two IDs.
#[derive(Debug, Clone)]
pub struct LayerChain {
    /// Layer IDs from leaf to root, with depth indices.
    /// For flat topology: [(overlay_id, 0), (baseline_id, 1)]
    /// For baseline-only: [(baseline_id, 0)]
    pub chain: Vec<(i64, u32)>,
}
