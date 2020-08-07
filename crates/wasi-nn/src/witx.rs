//! Contains the macro-generated implementation of wasi-nn from the its witx definition file.
use crate::ctx::WasiNnCtx;

// Generate the traits and types of wasi-nn in several Rust modules (e.g. `types`).
wiggle::from_witx!({
    witx: ["$WASI_ROOT/spec/phases/ephemeral/witx/wasi_ephemeral_nn.witx"],
    ctx: WasiNnCtx,
});

pub use types::Errno;
/// Exposes a helpful `Result` type for Rust-side users.
pub type Result<T> = std::result::Result<T, Errno>;

/// Wiggle generates code that performs some input validation on the arguments passed in by users of
/// wasi-nn. Here we convert the validation error into one (or more, eventually) of the error
/// variants defined in the witx.
impl types::GuestErrorConversion for WasiNnCtx {
    fn into_errno(&self, e: wiggle::GuestError) -> Errno {
        eprintln!("Guest error: {:?}", e);
        Errno::InvalidArgument
    }
}

/// Additionally, we must let Wiggle know which of our error codes represents a successful operation.
impl wiggle::GuestErrorType for Errno {
    fn success() -> Self {
        Self::Success
    }
}
