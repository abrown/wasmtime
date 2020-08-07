//! Implements the wasi-nn API.
use crate::witx::types::{
    ExecutionTarget, Graph, GraphEncoding, GraphExecutionContext, Size, Tensor,
};
use crate::witx::wasi_ephemeral_nn::WasiEphemeralNn;
use crate::witx::Result;
use crate::WasiNnCtx;

impl<'a> WasiEphemeralNn for WasiNnCtx {
    fn load<'b>(
        &self,
        graph_buf: &wiggle::GuestPtr<'b, u8>,
        graph_buf_len: Size,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        unimplemented!()
    }

    fn init_execution_context(&self, graph: Graph) -> Result<GraphExecutionContext> {
        unimplemented!()
    }

    fn set_input<'b>(
        &self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor<'b>,
    ) -> Result<()> {
        unimplemented!()
    }

    fn get_output<'b>(&self, context: GraphExecutionContext, index: u32) -> Result<Tensor<'b>> {
        unimplemented!()
    }

    fn compute(&self, context: GraphExecutionContext) -> Result<()> {
        unimplemented!()
    }
}
