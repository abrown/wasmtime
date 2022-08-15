//! Contains the macro-generated implementation of wasi-nn from the its WITX
//! definition file.
use crate::{WasiParallel, WasiParallelError};

// Generate the traits and types of wasi-parallel to several Rust modules (e.g.
// `types`).
wiggle::from_witx!({
    witx: ["$WASI_ROOT/phases/ephemeral/witx/wasi_ephemeral_parallel.witx"],
    errors: { par_errno => WasiParallelError },
});

use types::ParErrno;

// Additionally, we must let Wiggle know which of our error codes represents a
// successful operation.
impl wiggle::GuestErrorType for ParErrno {
    fn success() -> Self {
        Self::Success
    }
}

// Provide a way to map errors from the `WasiEphemeralTrait` (see `impl.rs`) to
// the WITX-defined error type.
impl wasi_ephemeral_parallel::UserErrorConversion for WasiParallel {
    fn par_errno_from_wasi_parallel_error(
        &mut self,
        _e: anyhow::Error,
    ) -> Result<crate::witx::types::ParErrno, wiggle::Trap> {
        todo!()
    }
}
