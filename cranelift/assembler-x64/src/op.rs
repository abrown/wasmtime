use crate::{Amode, AsReg, Gpr, GprMem, Imm16, Imm32, Imm8, Simm16, Simm32, Simm8, Xmm};

pub enum OperandKind<R: AsReg> {
    Readable(Operand<R>),
    Writable(Operand<R>),
}

///
#[expect(missing_docs, reason = "self-describing variants")]
pub enum Operand<R: AsReg> {
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

impl<R: AsReg> From<GprMem<R, R>> for Operand<R> {
    fn from(gpr_mem: GprMem<R, R>) -> Self {
        match gpr_mem {
            GprMem::Gpr(gpr) => Gpr::new(gpr).into(),
            GprMem::Mem(amode) => amode.into(),
        }
    }
}

impl<R: AsReg> From<Amode<R>> for Operand<R> {
    fn from(amode: Amode<R>) -> Self {
        Operand::Amode(amode)
    }
}

impl<R: AsReg> From<Gpr<R>> for Operand<R> {
    fn from(gpr: Gpr<R>) -> Self {
        Operand::Gpr(gpr)
    }
}

// impl<R: AsReg> From<FixedGpr<R>> for Operand<R> {
//     fn from(gpr: FixedGpr<R>) -> Self {
//         Operand::FixedGpr(gpr)
//     }
// }

impl<R: AsReg> From<Xmm<R>> for Operand<R> {
    fn from(xmm: Xmm<R>) -> Self {
        Operand::Xmm(xmm)
    }
}

// impl<R: AsReg> From<FixedXmm<R>> for Operand<R> {
//     fn from(xmm: FixedXmm<R>) -> Self {
//         Operand::FixedXmm(xmm)
//     }
// }

impl<R: AsReg> From<Imm8> for Operand<R> {
    fn from(imm: Imm8) -> Self {
        Operand::Imm8(imm)
    }
}

impl<R: AsReg> From<Imm16> for Operand<R> {
    fn from(imm: Imm16) -> Self {
        Operand::Imm16(imm)
    }
}

impl<R: AsReg> From<Imm32> for Operand<R> {
    fn from(imm: Imm32) -> Self {
        Operand::Imm32(imm)
    }
}

impl<R: AsReg> From<Simm8> for Operand<R> {
    fn from(imm: Simm8) -> Self {
        Operand::Simm8(imm)
    }
}

impl<R: AsReg> From<Simm16> for Operand<R> {
    fn from(imm: Simm16) -> Self {
        Operand::Simm16(imm)
    }
}

impl<R: AsReg> From<Simm32> for Operand<R> {
    fn from(imm: Simm32) -> Self {
        Operand::Simm32(imm)
    }
}
