pub mod backend;
mod ctx;
mod registry;

pub use ctx::{preload, WasiNnCtx};
pub use registry::{GraphRegistry, InMemoryRegistry};
pub mod wit;
pub mod witx;

use std::sync::Arc;

/// A backend-defined graph (i.e., ML model).
#[derive(Clone)]
pub struct Graph(Arc<dyn backend::BackendGraph>);
impl From<Box<dyn backend::BackendGraph>> for Graph {
    fn from(value: Box<dyn backend::BackendGraph>) -> Self {
        Self(value.into())
    }
}
impl std::ops::Deref for Graph {
    type Target = dyn backend::BackendGraph;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// A backend-defined execution context.
pub struct ExecutionContext(Box<dyn backend::BackendExecutionContext>);
impl From<Box<dyn backend::BackendExecutionContext>> for ExecutionContext {
    fn from(value: Box<dyn backend::BackendExecutionContext>) -> Self {
        Self(value)
    }
}
impl std::ops::Deref for ExecutionContext {
    type Target = dyn backend::BackendExecutionContext;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for ExecutionContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

/// A Backend
pub struct Backend(Box<dyn backend::BackendInner>);
impl std::ops::Deref for Backend {
    type Target = dyn backend::BackendInner;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for Backend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}
impl<T: backend::BackendInner + 'static> From<T> for Backend {
    fn from(value: T) -> Self {
        Self(Box::new(value))
    }
}

pub struct Registry(Box<dyn GraphRegistry>);
impl std::ops::Deref for Registry {
    type Target = dyn GraphRegistry;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for Registry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}
impl<T> From<T> for Registry
where
    T: GraphRegistry + 'static,
{
    fn from(value: T) -> Self {
        Self(Box::new(value))
    }
}

#[cfg(feature = "test-check")]
pub mod test_check {
    use anyhow::{anyhow, Context, Result};
    use std::{env, fs, fs::File, path::Path, path::PathBuf};

    #[macro_export]
    macro_rules! test_check {
        () => {
            if let Err(e) = $crate::test_check::check() {
                if std::env::var_os("CI").is_some() || std::env::var_os("").is_some() {
                    return Err(e);
                } else {
                    println!("> ignoring test: {}", e);
                    return Ok(());
                }
            }
        };
    }

    /// Return `Ok` if all checks pass.
    pub fn check() -> Result<()> {
        check_openvino_is_installed()?;
        check_openvino_artifacts_are_available()?;
        Ok(())
    }

    /// Return `Ok` if we find a working OpenVINO installation.
    pub fn check_openvino_is_installed() -> Result<()> {
        match std::panic::catch_unwind(|| {
            println!("> found openvino version: {}", openvino::version())
        }) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("unable to find an OpenVINO installation: {:?}", e)),
        }
    }

    pub fn artifacts_dir() -> PathBuf {
        PathBuf::from(env!("OUT_DIR")).join("mobilenet")
    }

    /// Return `Ok` if we find the cached MobileNet test artifacts; this will
    /// download the artifacts if necessary.
    fn check_openvino_artifacts_are_available() -> Result<()> {
        const BASE_URL: &str = "https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet";
        let artifacts_dir = artifacts_dir();
        fs::create_dir(&artifacts_dir)?;
        for (from, to) in [
            ("mobilenet.bin", "model.bin"),
            ("mobilenet.xml", "model.xml"),
            ("tensor-1x224x224x3-f32.bgr", "tensor.bgr"),
        ] {
            let remote_url = [BASE_URL, from].join("/");
            let local_path = artifacts_dir.join(to);
            if !local_path.is_file() {
                download(&remote_url, &local_path)
                    .with_context(|| "unable to retrieve test artifact")?;
            } else {
                println!("> using cached artifact: {}", local_path.display())
            }
        }
        Ok(())
    }

    /// Retrieve the bytes at the `from` URL and place them in the `to` file.
    fn download(from: &str, to: &Path) -> anyhow::Result<()> {
        println!("> downloading:\n  {} ->\n  {}", from, to.display());
        let mut file = File::create(to)?;
        let _ = reqwest::blocking::get(from)?.copy_to(&mut file)?;
        Ok(())
    }
}
