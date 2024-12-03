//! Contains the traits that a user of this assembler must implement.

use arbitrary::Arbitrary;
use std::{num::NonZeroU8, ops::Index, vec::Vec};

/// The encoding formats in this module all require a way of placing bytes into
/// a buffer.
pub trait CodeSink {
    /// Add 1 byte to the code section.
    fn put1(&mut self, _: u8);

    /// Add 2 bytes to the code section.
    fn put2(&mut self, _: u16);

    /// Add 4 bytes to the code section.
    fn put4(&mut self, _: u32);

    /// Add 8 bytes to the code section.
    fn put8(&mut self, _: u64);

    /// TODO: I would prefer to keep this away from this trait (but see how
    /// `Amode::RipRelative` is lowered).
    fn cur_offset(&self) -> u32;

    /// TODO: I would prefer to keep this away from this trait (but see how
    /// `Amode::RipRelative` is lowered).
    fn use_label_at_offset(&mut self, offset: u32, label: Label);

    /// TODO: I would prefer to keep this away from this trait (but see
    /// how we lower `Amode`).
    fn add_trap(&mut self, code: TrapCode);
}

#[derive(Debug, Clone, Arbitrary)]
pub struct Label(pub u32);

#[derive(Debug, Clone, Copy, Arbitrary)]
pub struct TrapCode(pub NonZeroU8);

/// Provide a convenient implementation for testing.
impl CodeSink for Vec<u8> {
    fn put1(&mut self, v: u8) {
        self.extend_from_slice(&[v]);
    }

    fn put2(&mut self, v: u16) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn put4(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn put8(&mut self, v: u64) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn cur_offset(&self) -> u32 {
        self.len().try_into().unwrap()
    }

    fn use_label_at_offset(&mut self, _: u32, _: Label) {}

    fn add_trap(&mut self, _: TrapCode) {}
}

/// A table mapping `KnownOffset` identifiers to their `i32` offset values.
///
/// When encoding instructions, Cranelift may not know all of the information
/// needed to construct an immediate. Specifically, addressing modes that
/// require knowing the size of the tail arguments or outgoing arguments (see
/// `SyntheticAmode::finalize`) will not know these sizes until emission.
///
/// This table allows up to do a "late" look up of these values by their
/// `KnownOffset`.
pub trait KnownOffsetTable: Index<KnownOffset, Output = i32> {}
impl KnownOffsetTable for Vec<i32> {}
impl KnownOffsetTable for [i32; 0] {}

/// A `KnownOffset` is a unique identifier for a specific offset known only at
/// emission time.
pub type KnownOffset = usize;
