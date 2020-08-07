//! Implements the base structure (i.e. [WasiNnCtx]) that will provide the implementation of the
//! wasi-nn API.
use crate::witx::types::Graph;
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

/// Possible errors for interacting with [WasiNnCtx].
#[derive(Debug, Error)]
pub enum WasiNnError {
    #[error("TODO")]
    Unknown,
}

type WasiNnResult<T> = std::result::Result<T, WasiNnError>;

/// This structure provides the Rust-side context necessary for implementing the wasi-nn API. At the
/// moment, it is specialized for a single inference implementation (i.e. OpenVINO) but conceivably
/// this could support more than one backing implementation.
pub struct WasiNnCtx {
    pub(crate) core: openvino::Core,
    pub(crate) graphs: RefCell<HashMap<Graph, openvino::CNNNetwork>>,
}

impl WasiNnCtx {
    /// Make a new `WasiNnCtx` with the default settings.
    pub fn new() -> WasiNnResult<Self> {
        Ok(Self {
            core: openvino::Core::new(None),
            graphs: RefCell::from(HashMap::new()),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn instantiate() {
        WasiNnCtx::new();
    }
}
