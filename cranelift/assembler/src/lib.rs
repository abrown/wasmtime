#![allow(non_camel_case_types)]

mod alloc;
pub mod fuzz;
mod imm;
mod mem;
mod reg;

use alloc::RegallocVisitor;
use cranelift_codegen::isa::x64;
use cranelift_codegen::isa::x64::encoding::rex::{emit_simm, encode_modrm, RexFlags};
use cranelift_codegen::isa::x64::encoding::ByteSink;
use cranelift_codegen::MachBuffer;
use imm::{Imm16, Imm32, Imm8};
use mem::{emit_modrm_sib_disp, GprMem};
use reg::{Gpr, Size};

// Include code generated by the `meta` crate; this
include!(concat!(env!("OUT_DIR"), "/assembler.rs"));

/// Helper function to make code generation simpler.
fn emit_modrm<BS: ByteSink + ?Sized>(buffer: &mut BS, enc_reg_g: u8, rm_e: u8) {
    let modrm = encode_modrm(0b11, enc_reg_g & 7, rm_e & 7);
    buffer.put1(modrm);
}
