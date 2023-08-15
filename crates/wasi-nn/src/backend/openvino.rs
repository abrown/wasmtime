//! Implements a `wasi-nn` [`Backend`] using OpenVINO.

use super::{Backend, BackendError, BackendExecutionContext, BackendGraph};
use crate::types::{ExecutionTarget, Tensor, TensorType};
use openvino::{InferenceError, Layout, Precision, SetupError, TensorDesc};
use std::{fs::File, io::Read, path::Path, sync::Arc};

#[derive(Default)]
pub(crate) struct OpenvinoBackend(Option<openvino::Core>);
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl Backend for OpenvinoBackend {
    fn name(&self) -> &str {
        "openvino"
    }

    fn load(
        &mut self,
        builders: &[&[u8]],
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        if builders.len() != 2 {
            return Err(BackendError::InvalidNumberOfBuilders(2, builders.len()).into());
        }

        // Construct the context if none is present; this is done lazily (i.e.
        // upon actually loading a model) because it may fail to find and load
        // the OpenVINO libraries. The laziness limits the extent of the error
        // only to wasi-nn users, not all WASI users.
        if self.0.is_none() {
            self.0.replace(openvino::Core::new(None)?);
        }

        // Read the guest array.
        let xml = &builders[0];
        let weights = &builders[1];

        // Construct OpenVINO graph structures: `cnn_network` contains the graph
        // structure, `exec_network` can perform inference.
        let core = self
            .0
            .as_mut()
            .expect("openvino::Core was previously constructed");
        let mut cnn_network = core.read_network_from_buffer(&xml, &weights)?;

        // TODO: this is a temporary workaround. We need a more elegant way to
        // specify the layout in the long run. However, without this newer
        // versions of OpenVINO will fail due to parameter mismatch.
        for i in 0..cnn_network.get_inputs_len()? {
            let name = cnn_network.get_input_name(i)?;
            cnn_network.set_input_layout(&name, Layout::NHWC)?;
        }

        let exec_network =
            core.load_network(&cnn_network, map_execution_target_to_string(target))?;

        Ok(Box::new(OpenvinoGraph(Arc::new(cnn_network), exec_network)))
    }

    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        let model = read(&path.join("model.xml"))?;
        let weights = read(&path.join("model.bin"))?;
        let graph = self.load(&[&model, &weights], target)?;
        Ok(graph)
    }
}

struct OpenvinoGraph(Arc<openvino::CNNNetwork>, openvino::ExecutableNetwork);

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        let infer_request = self.1.create_infer_request()?;
        Ok(Box::new(OpenvinoExecutionContext(
            self.0.clone(),
            infer_request,
        )))
    }
}

struct OpenvinoExecutionContext(Arc<openvino::CNNNetwork>, openvino::InferRequest);

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input<'a>(&mut self, index: u32, tensor: &Tensor<'a>) -> Result<(), BackendError> {
        let input_name = self.0.get_input_name(index as usize)?;

        // Construct the blob structure. TODO: there must be some good way to
        // discover the layout here; `desc` should not have to default to NHWC.
        let precision = map_tensor_type_to_precision(tensor.ty);
        let desc = TensorDesc::new(Layout::NHWC, tensor.dims, precision);
        let blob = openvino::Blob::new(&desc, tensor.data)?;

        // Actually assign the blob to the request.
        self.1.set_blob(&input_name, &blob)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.1.infer()?;
        Ok(())
    }

    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        let output_name = self.0.get_output_name(index as usize)?;
        let blob = self.1.get_blob(&output_name)?;
        let blob_size = blob.byte_len()?;
        if blob_size > destination.len() {
            return Err(BackendError::NotEnoughMemory(blob_size));
        }

        // Copy the tensor data into the destination buffer.
        destination[..blob_size].copy_from_slice(blob.buffer()?);
        Ok(blob_size as u32)
    }
}

impl From<InferenceError> for BackendError {
    fn from(e: InferenceError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<SetupError> for BackendError {
    fn from(e: SetupError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> &'static str {
    match target {
        ExecutionTarget::CPU => "CPU",
        ExecutionTarget::GPU => "GPU",
        ExecutionTarget::TPU => unimplemented!("OpenVINO does not support TPU execution targets"),
    }
}

/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
fn map_tensor_type_to_precision(tensor_type: TensorType) -> openvino::Precision {
    match tensor_type {
        TensorType::F16 => Precision::FP16,
        TensorType::F32 => Precision::FP32,
        TensorType::U8 => Precision::U8,
        TensorType::I32 => Precision::I32,
        TensorType::BF16 => todo!(),
    }
}

/// Read a file into a byte vector.
fn read(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![];
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use std::{env, mem, panic, path::Path, slice};

    #[test]
    fn image_classification_directly_on_backend() -> anyhow::Result<()> {
        if !check_openvino() || !check_openvino_artifacts() {
            println!("> unable to run `image_classification_directly_on_backend` test");
            return Ok(());
        }

        // Compute a MobileNet classification using the test artifacts.
        let mut backend = OpenvinoBackend::default();
        let mut graph = backend.load_from_dir(Path::new(env!("OUT_DIR")), ExecutionTarget::CPU)?;
        let mut context = graph.init_execution_context()?;
        let data = read(&Path::new(env!("OUT_DIR")).join("tensor.bgr"))?;
        let tensor = Tensor {
            dims: &[1, 3, 224, 224],
            ty: TensorType::F32,
            data: &data,
        };
        context.set_input(0, &tensor)?;
        context.compute()?;
        let mut destination = vec![0f32; 1001];
        let destination_ = unsafe {
            slice::from_raw_parts_mut(
                destination.as_mut_ptr().cast(),
                destination.len() * mem::size_of::<f32>(),
            )
        };
        context.get_output(0, destination_)?;

        // Find the top score which should be the entry for "pizza" (see
        // https://github.com/leferrad/tensorflow-mobilenet/blob/master/imagenet/labels.txt,
        // e.g.)
        let (id, score) = destination
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        println!("> top match: label #{} = {}", id, score);
        assert_eq!(id, 964);

        Ok(())
    }

    /// Return `true` if we find a working OpenVINO installation.
    fn check_openvino() -> bool {
        panic::catch_unwind(|| println!("> found openvino version: {}", openvino::version()))
            .is_ok()
    }

    /// Return `true` if we find the cached MobileNet test artifacts; this will
    /// download the artifacts if necessary.
    fn check_openvino_artifacts() -> bool {
        const BASE_URL: &str = "https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet";
        let artifacts_dir = Path::new(env!("OUT_DIR"));
        for (from, to) in [
            ("mobilenet.bin", "model.bin"),
            ("mobilenet.xml", "model.xml"),
            ("tensor-1x224x224x3-f32.bgr", "tensor.bgr"),
        ] {
            let remote_url = [BASE_URL, from].join("/");
            let local_path = artifacts_dir.join(to);
            if !local_path.is_file() {
                download(&remote_url, &local_path)
                    .with_context(|| "unable to retrieve test artifact")
                    .unwrap();
            } else {
                println!("> using cached artifact: {}", local_path.display())
            }
        }
        true
    }

    /// Retrieve the bytes at the `from` URL and place them in the `to` file.
    fn download(from: &str, to: &Path) -> anyhow::Result<()> {
        println!("> downloading:\n  {} ->\n  {}", from, to.display());
        let mut file = File::create(to)?;
        let _ = reqwest::blocking::get(from)?.copy_to(&mut file)?;
        Ok(())
    }
}
