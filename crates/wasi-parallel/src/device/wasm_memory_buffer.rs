//! Contains a reference to a slice of Wasm memory.
use std::any::Any;

use super::Buffer;
use crate::witx::types::BufferAccessKind;
use anyhow::{anyhow, Result};
use wiggle::GuestPtr;

/// This kind of buffer is designed to live exclusively in Wasm memory--it
/// contains the information necessary for reading and writing to Wasm memory.
/// Its lifecycle involves:
/// - The buffer is created by `create_buffer`; this associates a buffer ID with
///   a device ID, but the buffer has no knowledge of what data it may contain,
///   only its length.
/// - The buffer may then be written to by `write_buffer`; at this point the
///   buffer will record the its offset within the Wasm memory.
/// - When used by `for`, this buffer will simply pass its offset and length to
///   the parallel kernel, where it will be mutated by a Wasm function.
/// - The buffer contents may "read" from one section of the Wasm memory to
///   another.
pub struct WasmMemoryBuffer {
    pub(crate) offset: Option<u32>,
    length: u32,
    access: BufferAccessKind,
}

impl WasmMemoryBuffer {
    pub fn new(size: u32, access: BufferAccessKind) -> Self {
        Self {
            offset: None,
            length: size,
            access,
        }
    }
}

impl Buffer for WasmMemoryBuffer {
    fn len(&self) -> u32 {
        self.length
    }

    fn access(&self) -> BufferAccessKind {
        self.access
    }

    /// Does not copy data: simply checks that the lengths of the buffer and
    /// guest slice match and then records the starting location of the guest
    /// pointer. This will require some re-thinking once multiple memories are
    /// possible (TODO).
    fn write(&mut self, slice: GuestPtr<[u8]>) -> Result<()> {
        if slice.len() == self.len() {
            self.offset = Some(slice.offset_base());
            Ok(())
        } else {
            Err(anyhow!(
                "The slice to write did not match the buffer size: {} != {}",
                slice.len(),
                self.len(),
            ))
        }
    }

    /// This implementation of `read` will attempt to copy the device data, held
    /// in Wasm memory, to another location in Wasm memory. Currently it will
    /// fail if the slices are overlapping (TODO). At some point, this should
    /// also see if the `read` is from and to the same slice and avoid the copy
    /// entirely (TODO).
    fn read(&self, slice: GuestPtr<[u8]>) -> Result<()> {
        debug_assert_eq!(slice.len(), self.len());
        let mem = slice.mem().base();
        let mem = unsafe { std::slice::from_raw_parts_mut(mem.0, mem.1 as usize) };
        copy_within_a_slice(
            mem,
            self.offset.unwrap() as usize,
            slice.offset_base() as usize,
            slice.len() as usize,
        );
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

/// This helper copies one sublice to another within a mutable slice. It will
/// panic if the slices are overlapping. See
/// https://stackoverflow.com/a/45082624.
fn copy_within_a_slice<T: Clone>(v: &mut [T], from: usize, to: usize, len: usize) {
    if from > to {
        let (dst, src) = v.split_at_mut(from);
        dst[to..to + len].clone_from_slice(&src[..len]);
    } else {
        let (src, dst) = v.split_at_mut(to);
        dst[..len].clone_from_slice(&src[from..from + len]);
    }
}
