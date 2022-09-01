//! Implements the wasi-nn API using TensorFlow.

use crate::api::{Backend, BackendError, BackendExecutionContext, BackendGraph};
use crate::witx::types::{ExecutionTarget, GraphBuilderArray, Tensor, TensorType};
use anyhow::anyhow;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use tensorflow::{
    Graph, SavedModelBundle, Session, SessionOptions, SessionRunArgs, SignatureDef, Status,
    Tensor as TFTensor, DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

#[derive(Default)]
pub(crate) struct TensorflowBackend;

impl Backend for TensorflowBackend {
    fn name(&self) -> &str {
        "tensorflow"
    }

    fn load(
        &mut self,
        builders: &GraphBuilderArray<'_>,
        _target: ExecutionTarget,
        map_dirs: &Vec<(String, String)>,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        if !map_dirs.is_empty() {
            if builders.len() < 1 {
                // TODO incorrect.
                return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
            }

            // Initialize the Tensorflow backend.
            tensorflow::library::load().or_else(|e| {
                println!("Error loading the TensorFlow backend: \n {}", e);
                Err(BackendError::BackendAccess(anyhow!("e")))
            })?;

            // Tensorflow wants to read models from a directory. This path here
            // is the guest-side (WebAssembly) version of that path which we map
            // (from `--mapdir` CLI option) to the host-side path with the
            // actual files. If in the future Tensorflow allows loading models
            // from bytes, that would be a better solution (TODO).
            let builders_len = builders.len();
            let builders = builders.as_ptr();
            let guest_map = builders.read()?.as_slice()?;
            let mapped_directory =
                build_path(&guest_map, map_dirs).ok_or(BackendError::MissingMapDir())?;
            let mut tags: Vec<String> = vec![];
            let mut itr: u32 = 1;

            // Get all the user provided options.
            let mut signature = DEFAULT_SERVING_SIGNATURE_DEF_KEY.to_string();
            while itr < builders_len {
                let opt = builders.add(itr)?.read()?.as_slice()?.to_owned();
                let mut opt_str = str::from_utf8(&opt).ok().unwrap().split(',');

                match opt_str.next().unwrap() {
                    "signature" => {
                        signature = opt_str.next().unwrap().to_owned();
                    }
                    "tag" => tags.push(opt_str.next().unwrap().to_owned()),
                    o => {
                        println!("** Unknown Tensorflow option {}, ignoring... **", o);
                    }
                }
                itr += 1;
            }

            // Load the model.
            let mut graph = Graph::new();
            let bundle = SavedModelBundle::load(
                &SessionOptions::new(),
                &tags,
                &mut graph,
                mapped_directory,
            )?;

            // Extract the model signature.
            let signature = bundle.meta_graph_def().get_signature(&signature)?.clone();

            return Ok(Box::new(TensorflowGraph {
                graph: Arc::new(graph),
                session: Arc::new(bundle.session),
                signature: Arc::new(signature),
            }));
        }

        Err(BackendError::MissingMapDir())
    }
}

/// Map the `guest_path` to its equivalent host path *if* there is a mapping for
/// it in the `map_dirs`.
fn build_path(guest_path: &[u8], map_dirs: &Vec<(String, String)>) -> Option<PathBuf> {
    let guest_path = Path::new(str::from_utf8(guest_path).ok()?);
    for (guest_base, host_base) in map_dirs {
        let host_base = Path::new(host_base);
        // If this is the mapped directory we are looking for...
        if guest_path.starts_with(guest_base) {
            // ...then map the guest path to its host equivalent.
            let guest_suffix = guest_path.strip_prefix(guest_base).ok()?;
            let host_path: PathBuf = [host_base, guest_suffix].iter().collect();
            // Now canonicalize the host path to check that the guest path
            // has not escaped the host's base path.
            let canon_path = host_path.canonicalize().ok()?;
            return canon_path.starts_with(&host_base).then_some(host_path);
        }
    }
    None
}

struct TensorflowGraph {
    graph: Arc<Graph>,
    session: Arc<Session>,
    signature: Arc<SignatureDef>,
}

impl<'a> BackendGraph for TensorflowGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        Ok(Box::new(TensorflowExecutionContext {
            graph: self.graph.clone(),
            session: self.session.clone(),
            signature: self.signature.clone(),
            tensors: Vec::new(),
            args: SessionRunArgs::new(),
        }))
    }
}

struct TensorflowExecutionContext<'a> {
    graph: Arc<Graph>,
    session: Arc<Session>,
    signature: Arc<SignatureDef>,
    tensors: Vec<TFTensor<u8>>,
    args: SessionRunArgs<'a>,
}

impl<'a> BackendExecutionContext for TensorflowExecutionContext<'a> {
    fn set_input(&mut self, index: u32, tensor: &Tensor<'_>) -> Result<(), BackendError> {
        // Return an error if the index doesn't exist in the signature.
        if index as usize > self.signature.inputs().len() - 1 {
            return Err(BackendError::InvalidTensorIndex(index as usize));
        }

        // Sort the input keys alphabetically so that we know we always index
        // into the same key. (Note that TF's `HashMap::keys()` is returned in
        // arbitrary order).
        let mut input_keys: Vec<String> = self.signature.inputs().keys().cloned().collect();
        input_keys.sort();
        let input_key = &input_keys[index as usize];

        // Check that the tensor data type provided matches the one in the
        // model.
        let tensor_info = self.signature.get_input(input_key)?;
        match_tensor_type(index as usize, tensor_info.dtype(), tensor.type_)?;

        // Now, figure out what TF operation to bind this tensor to.
        let operation = self
            .graph
            .operation_by_name_required(&tensor_info.name().name)?;

        // Convert the dimensions to `u64`s.
        let dims = tensor
            .dimensions
            .as_slice()?
            .iter()
            .map(|d| *d as u64)
            .collect::<Vec<_>>();

        // Copy the tensor bytes to the Tensorflow container. We pretend the
        // tensor has byte elements (though it may contain elements of any
        // `TensorType`) because we expect the user to provide the tensor in the
        // exact, compatible byte format for Tensorflow. Ideally we would avoid
        // the copy here and just point to the original bytes (TODO: investigate
        // unsafely using `as_mut_ptr`).
        let mut tf_tensor = TFTensor::new(&dims);
        tf_tensor.copy_from_slice(&tensor.data.as_slice()?);
        self.tensors.push(tf_tensor);

        // Assign the tensor to the session arguments. The `add_feed`
        // documentation says that because most operations have only one output
        // (and presumably one input), so the input index is likely 0. Note that
        // we need to do some awkward hoop-jumping here:
        // - in order to maintain the lifetime of `SessionRunArgs`, which
        //   borrows the tensor data, we copy the tensor data into our `Self`
        //   above (otherwise we cannot guarantee that the borrowed data will be
        //   there when we actually need it, in `compute`).
        // - but we also must fit within the Wiggle-generated
        //   `BackendExecutionContext` trait, which says this function must take
        //   `&mut self`. So we pretend that `&mut self` lives as long as, well,
        //   itself (`'a`) using `transmute` and use our new `self_` to borrow
        //   the tensor data we copied to `Self`.
        assert_eq!(operation.num_inputs(), 1);
        let self_ = unsafe { std::mem::transmute::<&mut Self, &'a mut Self>(self) };
        let tensor_ref = &self_.tensors[self_.tensors.len()];
        self_.args.add_feed(&operation, 0, tensor_ref);

        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.session.run(&mut self.args)?;
        Ok(())
    }

    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        // Sort the output keys alphabetically so that we know we always index
        // into the same key. (Note that TF's `HashMap::keys()` is returned in
        // arbitrary order).
        let mut output_keys: Vec<String> = self.signature.inputs().keys().cloned().collect();
        output_keys.sort();
        let output_key = &output_keys[index as usize];

        // Now, figure out what TF operation to bind this tensor to.
        let tensor_info = self.signature.get_output(output_key)?;
        let operation = self
            .graph
            .operation_by_name_required(&tensor_info.name().name)?;

        // Retrieve the output tensor.
        let token = self.args.request_fetch(&operation, 0);
        let tensor = self.args.fetch::<u8>(token)?;

        // Copy the tensor as bytes (though it may contain elements of any
        // `TensorType`). We expect the user to handle the tensor data
        // appropriately.
        if tensor.len() > destination.len() {
            Err(BackendError::NotEnoughMemory(tensor.len()))
        } else {
            let tensor = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), tensor.len()) };
            destination.copy_from_slice(tensor);
            Ok(tensor.len() as u32)
        }
    }
}

/// Check that the data type of the user-provided tensor matches the one
/// expected by Tensorflow.
fn match_tensor_type(
    index: usize,
    expected: tensorflow::DataType,
    provided: TensorType,
) -> Result<(), BackendError> {
    if let Some(expected) = convert_tensor_type(expected) {
        if expected != provided {
            let expected = format!("{:?}", expected);
            let provided = format!("{:?}", provided);
            return Err(BackendError::InvalidTensorType(index, expected, provided));
        }
    } else {
        let expected = expected.to_string();
        let provided = format!("{:?}", provided);
        return Err(BackendError::InvalidTensorType(index, expected, provided));
    }
    Ok(())
}

/// Convert the Tensorflow data type to its wasi-nn type, if possible.
fn convert_tensor_type(tensor_type: tensorflow::DataType) -> Option<TensorType> {
    match tensor_type {
        tensorflow::DataType::UInt8 => Some(TensorType::U8),
        tensorflow::DataType::Half => Some(TensorType::F16),
        tensorflow::DataType::Int32 => Some(TensorType::I32),
        tensorflow::DataType::Float => Some(TensorType::F32),
        _ => None,
    }
}

impl From<Status> for BackendError {
    fn from(e: Status) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{Builder, TempDir};

    fn create_temp_dir<P: AsRef<Path>>(p: P) -> (TempDir, PathBuf) {
        let parent_dir = Builder::new().prefix("wasi-nn-tests").tempdir().unwrap();
        let child_dir = parent_dir.path().join(p);
        std::fs::create_dir_all(&child_dir).unwrap();
        (parent_dir, child_dir)
    }

    #[test]
    fn valid_path() {
        let (_tmp_dir, foo_bar_dir) = create_temp_dir("foo/bar");
        let foo_dir = foo_bar_dir.parent().unwrap();
        let map_dirs = vec![("/baz".to_string(), foo_dir.to_string_lossy().to_string())];

        // Map `/baz/bar` to `<host path>/foo/bar`.
        let result = build_path(b"/baz/bar", &map_dirs);
        assert!(result.is_some());
    }

    #[test]
    fn valid_path_with_parent_dots() {
        let (_tmp_dir, foo_bar_dir) = create_temp_dir("foo/bar");
        let foo_dir = foo_bar_dir.parent().unwrap();
        let map_dirs = vec![("/baz".to_string(), foo_dir.to_string_lossy().to_string())];

        // Map `/baz/bar/..` to `<host path>/foo`.
        let result = build_path(b"/baz/bar", &map_dirs);
        assert!(result.is_some());
    }

    #[test]
    fn invalid_path_escape_attempt() {
        let (_tmp_dir, foo_bar_dir) = create_temp_dir("foo/bar");
        let foo_dir = foo_bar_dir.parent().unwrap();
        let map_dirs = vec![("/baz".to_string(), foo_dir.to_string_lossy().to_string())];

        // It is invalid to map `/baz/..` because it would escape the mapping.
        let result = build_path(b"/baz/..", &map_dirs);
        assert!(result.is_none());
    }
}
