//! A Cranelift-specific x64 assembler.
//!
//! All instructions known to this assembler are listed in the [`inst`] module.
//! The [`Inst`] enumeration contains a variant for each, allowing matching over
//! all these instructions. All of this is parameterized by a [`Registers`]
//! trait, allowing users of this assembler to plug in their own register types.
//!
//! ```
//! # use cranelift_assembler_x64::{Feature, Fixed, Imm8, inst, Inst, Registers};
//! // Tell the assembler the type of registers we're using; we can always
//! // encode a HW register as a `u8` (e.g., `eax = 0`).
//! pub struct Regs;
//! impl Registers for Regs {
//!     type ReadGpr = u8;
//!     type ReadWriteGpr = u8;
//!     type ReadXmm = u8;
//!     type ReadWriteXmm = u8;
//! }
//!
//! // Then, build one of the `AND` instructions; this one operates on an
//! // implicit `AL` register with an immediate. We can collect a sequence of
//! // instructions by converting to the `Inst` type.
//! let rax: u8 = 0;
//! let and = inst::andb_i::new(Fixed(rax), Imm8::new(0b10101010));
//! let seq: Vec<Inst<Regs>> = vec![and.into()];
//!
//! // Now we can encode this sequence into a code buffer, checking that each
//! // instruction is valid in 64-bit mode.
//! let mut buffer = vec![];
//! let offsets = vec![];
//! for inst in seq {
//!     if inst.features().contains(&Feature::_64b) {
//!         inst.encode(&mut buffer, &offsets);
//!     }
//! }
//! assert_eq!(buffer, vec![0x24, 0b10101010]);
//! ```
//!
//! With an [`Inst`], we can encode the instruction into a code buffer; see the
//! [example](Inst).

#![allow(
    non_camel_case_types,
    reason = "all of the generated struct names use snake case"
)]

mod api;
pub mod gpr;
mod imm;
pub mod inst;
mod mem;
mod op;
mod rex;
pub mod xmm;

#[cfg(any(test, feature = "fuzz"))]
pub mod fuzz;

/// An assembly instruction; contains all instructions known to the assembler.
///
/// This wraps all [`inst`] structures into a single enumeration for collecting
/// instructions.
#[doc(inline)]
// This re-exports, and documents, a module that is more convenient to use at
// the library top-level.
pub use inst::Inst;

/// A CPU feature.
///
/// This is generated from the `dsl::Feature` enumeration defined in the `meta`
/// crate (i.e., an exact replica). It describes the CPUID features required by
/// an instruction; see [`Inst::features`].
#[doc(inline)]
// Like `Inst` above, a convenient re-export.
pub use inst::Feature;

pub use api::{
    AsReg, CodeSink, Constant, KnownOffset, KnownOffsetTable, Label, RegisterVisitor, Registers,
    TrapCode,
};
pub use gpr::{Gpr, NonRspGpr, Size};
pub use imm::{Extension, Imm16, Imm32, Imm8, Simm16, Simm32, Simm8};
pub use mem::{
    Amode, AmodeOffset, AmodeOffsetPlusKnownOffset, DeferredTarget, GprMem, Scale, XmmMem,
};
pub use op::Operand;
pub use rex::RexFlags;
pub use xmm::Xmm;

/// List the files generated to create this assembler.
pub fn generated_files() -> Vec<std::path::PathBuf> {
    include!(concat!(env!("OUT_DIR"), "/generated-files.rs"))
}

// enum Op<R: AsReg, M: AsReg> {
//     Imm,
//     Gpr(R),
//     Fixed(u8),
//     Amode(M),
// }
// enum Inst2 {
//     Add,
//     Sub,
//     Mul,
// }
// impl Inst2 {
//     fn operands<AR: AsReg + 'static>(&self) -> Box<dyn ExactSizeIterator<Item = Op<AR>>> {
//         match self {
//             Inst2::Add => Box::new(add_operands::<AR>().into_iter()),
//             Inst2::Sub => Box::new(sub_operands::<AR>().into_iter()),
//             Inst2::Mul => Box::new(mul_operands::<AR>().into_iter()),
//         }
//     }
//     fn operands_vec<AR: AsReg>(&self) -> Vec<Op<AR>> {
//         match self {
//             Inst2::Add => Box::new(add_operands::<AR>().into_iter()),
//             Inst2::Sub => Box::new(sub_operands::<AR>().into_iter()),
//             Inst2::Mul => Box::new(mul_operands::<AR>().into_iter()),
//         }
//     }
// }
// // type Iter = impl ExactSizeIterator<Item = Op<AR>>;
// // type Iter<AR> = std::array::IntoIter<Op<AR>, 2>;
// type Iter<AR, const N: usize> = [Op<AR>; N];
// fn add_operands<AR: AsReg>() -> Iter<AR, 2> {
//     [Op::Fixed(8), Op::Imm]
// }
// fn sub_operands<AR: AsReg>() -> Iter<AR, 0> {
//     // let x = AR::new(8);
//     // let y = AR::new(9);
//     // [Op::Amode(x), Op::Gpr(y)]
//     []
// }
// fn mul_operands<AR: AsReg>() -> Iter<AR, 0> {
//     []
// }

// #[test]
// fn us() {
//     assert_eq!(Inst2::Add.operands::<u8>().len(), 2);
//     // assert_eq!(Inst2::Sub.operands::<u8>().len(), 2);
//     assert_eq!(Inst2::Mul.operands::<u8>().len(), 0);
// }

// struct addb<R: AsReg>(Fixed<R, { gpr::enc::RAX }>, Imm8);

#[derive(Clone, Debug)]
pub struct Fixed<R, const E: u8>(pub R);
impl<R: AsReg, const E: u8> AsReg for Fixed<R, E> {
    #[cfg(any(test, feature = "fuzz"))]
    fn new(reg: u8) -> Self {
        assert!(reg == E);
        Self(R::new(reg))
    }
    fn enc(&self) -> u8 {
        assert!(self.0.enc() == E);
        self.0.enc()
    }
}
impl<R, const E: u8> AsRef<R> for Fixed<R, E> {
    fn as_ref(&self) -> &R {
        &self.0
    }
}
