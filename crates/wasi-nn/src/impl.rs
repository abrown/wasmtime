//! Implements the wasi-nn API.
use crate::ctx::{ExecutionContext, WasiNnResult as Result};
use crate::witx::types::{
    ExecutionTarget, Graph, GraphBuilderArray, GraphEncoding, GraphExecutionContext, Tensor,
    TensorType,
};
use crate::witx::wasi_ephemeral_nn::WasiEphemeralNn;
use crate::WasiNnCtx;
use openvino::{Layout, Precision, TensorDesc};
use thiserror::Error;
use wiggle::GuestPtr;

#[derive(Debug, Error)]
pub enum UsageError {
    #[error("Only OpenVINO's IR is currently supported, passed encoding: {0}")]
    InvalidEncoding(GraphEncoding),
    #[error("OpenVINO expects only two buffers (i.e. [ir, weights]), passed: {0}")]
    InvalidNumberOfBuilders(u32),
    #[error("Invalid graph handle; has it been loaded?")]
    InvalidGraphHandle,
    #[error("Invalid execution context handle; has it been initialized?")]
    InvalidExecutionContextHandle,
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(u32),
}

impl<'a> WasiEphemeralNn for WasiNnCtx {
    fn load<'b>(
        &self,
        builders: &GraphBuilderArray<'_>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        if encoding != GraphEncoding::Openvino {
            return Err(UsageError::InvalidEncoding(encoding).into());
        }
        if builders.len() != 2 {
            return Err(UsageError::InvalidNumberOfBuilders(builders.len()).into());
        }
        let builders = builders.as_ptr();
        let xml = builders.read()?.as_slice()?;
        let weights = builders.add(1)?.read()?.as_slice()?;
        let graph = self
            .ctx
            .borrow_mut()
            .core
            .read_network_from_buffer(&xml, &weights)?;
        let executable_graph = self
            .ctx
            .borrow_mut()
            .core
            .load_network(&graph, &target.to_string())?;
        let id = self
            .ctx
            .borrow_mut()
            .graphs
            .insert((graph, executable_graph));
        Ok(id)
    }

    fn init_execution_context(&self, graph: Graph) -> Result<GraphExecutionContext> {
        if let Some((_, executable_graph)) = self.ctx.borrow_mut().graphs.get_mut(graph) {
            let request = executable_graph.create_infer_request()?;
            let execution_context = ExecutionContext::new(graph, request);
            let handle = self.ctx.borrow_mut().executions.insert(execution_context);
            Ok(handle)
        } else {
            Err(UsageError::InvalidExecutionContextHandle.into())
        }
    }

    fn set_input<'b>(
        &self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor<'b>,
    ) -> Result<()> {
        if let Some(execution) = self.ctx.borrow_mut().executions.get_mut(context) {
            // Retrieve name for this input index.
            let name = if let Some((graph, _)) = self.ctx.borrow().graphs.get(execution.graph) {
                graph.get_input_name(index as usize)?
            } else {
                unreachable!("It should be impossible to attempt to access an execution's graph and for that graph not to exist--this is a bug.")
            };

            // Construct the blob structure.
            let dimensions = tensor
                .dimensions
                .as_slice()?
                .iter()
                .map(|d| *d as u64)
                .collect::<Vec<_>>();
            let precision = match tensor.type_ {
                TensorType::F16 => Precision::FP16,
                TensorType::F32 => Precision::FP32,
                TensorType::U8 => Precision::U8,
                TensorType::I32 => Precision::I32,
            };
            // TODO how to discover layout?
            let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
            let data = tensor.data.as_slice()?;
            let blob = openvino::Blob::new(desc, &data)?;

            // Actually assign the blob to the request.
            execution.request.set_blob(&name, blob)?;
            Ok(())
        } else {
            return Err(UsageError::InvalidExecutionContextHandle.into());
        }
    }

    fn compute(&self, context: GraphExecutionContext) -> Result<()> {
        if let Some(execution) = self.ctx.borrow_mut().executions.get_mut(context) {
            Ok(execution.request.infer()?)
        } else {
            return Err(UsageError::InvalidExecutionContextHandle.into());
        }
    }

    fn get_output<'b>(
        &self,
        context: GraphExecutionContext,
        index: u32,
        out_buffer: &GuestPtr<'_, u8>,
        out_buffer_max_size: u32,
    ) -> Result<u32> {
        if let Some(execution) = self.ctx.borrow_mut().executions.get_mut(context) {
            // Retrieve name for this output index.
            let name = if let Some((graph, _)) = self.ctx.borrow().graphs.get(execution.graph) {
                graph.get_output_name(index as usize)?
            } else {
                unreachable!("It should be impossible to attempt to access an execution's graph and for that graph not to exist--this is a bug.")
            };

            // Retrieve the tensor data.
            let mut blob = execution.request.get_blob(&name)?; // TODO shouldn't need to be mut
            let blob_size = blob.byte_len()? as u32;
            if blob_size > out_buffer_max_size {
                return Err(UsageError::NotEnoughMemory(blob_size).into());
            }

            // Copy the tensor data over to the `out_buffer`.
            let mut out_slice = out_buffer.as_array(out_buffer_max_size).as_slice()?;
            (&mut out_slice[..blob_size as usize]).copy_from_slice(blob.buffer()?);

            Ok(blob_size)
        } else {
            return Err(UsageError::InvalidExecutionContextHandle.into());
        }
    }
}
