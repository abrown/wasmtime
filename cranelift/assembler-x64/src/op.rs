use crate::{Amode, AsReg, Fixed, Gpr, GprMem, Imm16, Imm32, Imm8, Simm16, Simm32, Simm8, Xmm};

/// An instruction operand.
///
/// This is useful for iterating over the operands of an [`Inst`][crate::Inst].
///
/// ```
/// # use cranelift_assembler_x64::{Fixed, Imm8, inst, Inst, Registers};
/// pub struct Regs;
/// impl Registers for Regs {
///     type ReadGpr = u8;
///     type ReadWriteGpr = u8;
///     type ReadXmm = u8;
///     type ReadWriteXmm = u8;
/// }
///
/// let rax = 0;
/// let inst: Inst<Regs> = inst::addb_i::new(Fixed(rax), Imm8::new(0x42)).into();
/// // let operands = inst.operands();
/// ```
pub enum Operand<R: AsReg> {
    Read(OperandKind<R>),
    ReadWrite(OperandKind<R>),
}

///
// #[expect(missing_docs, reason = "self-describing variants")]
pub enum OperandKind<R: AsReg> {
    // Memory operands.
    Amode(Amode<R>),

    // Register operands.
    Gpr(Gpr<R>),
    FixedGpr(Gpr<R>),
    Xmm(Xmm<R>),
    FixedXmm(Xmm<R>),

    // Immediate operands.
    Imm8(Imm8),
    Imm16(Imm16),
    Imm32(Imm32),
    Simm8(Simm8),
    Simm16(Simm16),
    Simm32(Simm32),
}

impl<R: AsReg> From<GprMem<R, R>> for OperandKind<R> {
    fn from(gpr_mem: GprMem<R, R>) -> Self {
        match gpr_mem {
            GprMem::Gpr(gpr) => Gpr::new(gpr).into(),
            GprMem::Mem(amode) => amode.into(),
        }
    }
}

impl<R: AsReg> From<Amode<R>> for OperandKind<R> {
    fn from(amode: Amode<R>) -> Self {
        OperandKind::Amode(amode)
    }
}

impl<R: AsReg> From<Gpr<R>> for OperandKind<R> {
    fn from(gpr: Gpr<R>) -> Self {
        OperandKind::Gpr(gpr)
    }
}

impl<R: AsReg, const E: u8> From<Fixed<Gpr<R>, E>> for OperandKind<R> {
    fn from(fixed: Fixed<Gpr<R>, E>) -> Self {
        OperandKind::FixedGpr(fixed.0)
    }
}

impl<R: AsReg> From<Xmm<R>> for OperandKind<R> {
    fn from(xmm: Xmm<R>) -> Self {
        OperandKind::Xmm(xmm)
    }
}

impl<R: AsReg, const E: u8> From<Fixed<Xmm<R>, E>> for OperandKind<R> {
    fn from(fixed: Fixed<Xmm<R>, E>) -> Self {
        OperandKind::FixedXmm(fixed.0)
    }
}

impl<R: AsReg> From<Imm8> for OperandKind<R> {
    fn from(imm: Imm8) -> Self {
        OperandKind::Imm8(imm)
    }
}

impl<R: AsReg> From<Imm16> for OperandKind<R> {
    fn from(imm: Imm16) -> Self {
        OperandKind::Imm16(imm)
    }
}

impl<R: AsReg> From<Imm32> for OperandKind<R> {
    fn from(imm: Imm32) -> Self {
        OperandKind::Imm32(imm)
    }
}

impl<R: AsReg> From<Simm8> for OperandKind<R> {
    fn from(imm: Simm8) -> Self {
        OperandKind::Simm8(imm)
    }
}

impl<R: AsReg> From<Simm16> for OperandKind<R> {
    fn from(imm: Simm16) -> Self {
        OperandKind::Simm16(imm)
    }
}

impl<R: AsReg> From<Simm32> for OperandKind<R> {
    fn from(imm: Simm32) -> Self {
        OperandKind::Simm32(imm)
    }
}
