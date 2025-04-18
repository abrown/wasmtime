use crate::{
    Amode, Gpr, GprMem, Imm16, Imm32, Imm8, Registers, Simm16, Simm32, Simm8, Xmm, XmmMem,
};

/// An instruction operand.
///
/// This is useful for iterating over the operands of an [`Inst`][crate::Inst].
///
/// ```
/// # use cranelift_assembler_x64::{Fixed, Imm8, inst, Inst, Operand, Registers};
/// pub struct Regs;
/// impl Registers for Regs {
///     type ReadGpr = u8;
///     type ReadWriteGpr = u8;
///     type ReadXmm = u8;
///     type ReadWriteXmm = u8;
/// }
///
/// let rax = 0;
/// let mut inst: Inst<Regs> = inst::addb_i::new(Fixed(rax), Imm8::new(0x42)).into();
/// let operands = inst.operands();
/// assert_eq!(operands.len(), 2);
/// assert!(matches!(operands[0], Operand::ReadWriteGpr { fixed: true, .. }));
/// assert!(matches!(operands[1], Operand::Imm8(_)));
/// ```

/// TODO
#[derive(Debug)]
pub enum Operand<'a, R: Registers> {
    // Memory operands.
    Amode(&'a mut Amode<R::ReadGpr>),

    // Register operands.
    ReadGpr {
        gpr: &'a mut R::ReadGpr,
        fixed: bool,
    },
    ReadWriteGpr {
        gpr: &'a mut R::ReadWriteGpr,
        fixed: bool,
    },
    ReadXmm {
        xmm: &'a mut R::ReadXmm,
        fixed: bool,
    },
    ReadWriteXmm {
        xmm: &'a mut R::ReadWriteXmm,
        fixed: bool,
    },

    // Immediate operands.
    Imm8(&'a mut Imm8),
    Imm16(&'a mut Imm16),
    Imm32(&'a mut Imm32),
    Simm8(&'a mut Simm8),
    Simm16(&'a mut Simm16),
    Simm32(&'a mut Simm32),
}

impl<'a, R: Registers> Operand<'a, R> {
    pub fn from_read_gpr(gpr: &'a mut Gpr<R::ReadGpr>) -> Self {
        let gpr = &mut gpr.0;
        Operand::ReadGpr { gpr, fixed: false }
    }
    pub fn from_read_write_gpr(gpr: &'a mut Gpr<R::ReadWriteGpr>) -> Self {
        let gpr = &mut gpr.0;
        Operand::ReadWriteGpr { gpr, fixed: false }
    }
    pub fn from_read_fixed_gpr(gpr: &'a mut R::ReadGpr) -> Self {
        Operand::ReadGpr { gpr, fixed: true }
    }
    pub fn from_read_write_fixed_gpr(gpr: &'a mut R::ReadWriteGpr) -> Self {
        Operand::ReadWriteGpr { gpr, fixed: true }
    }
    pub fn from_read_xmm(xmm: &'a mut Xmm<R::ReadXmm>) -> Self {
        let xmm = &mut xmm.0;
        Operand::ReadXmm { xmm, fixed: false }
    }
    pub fn from_read_write_xmm(xmm: &'a mut Xmm<R::ReadWriteXmm>) -> Self {
        let xmm = &mut xmm.0;
        Operand::ReadWriteXmm { xmm, fixed: false }
    }
    pub fn from_read_gpr_mem(gpr_mem: &'a mut GprMem<R::ReadGpr, R::ReadGpr>) -> Self {
        match gpr_mem {
            GprMem::Gpr(gpr) => Operand::ReadGpr { gpr, fixed: false },
            GprMem::Mem(amode) => Operand::Amode(amode),
        }
    }
    pub fn from_read_write_gpr_mem(gpr_mem: &'a mut GprMem<R::ReadWriteGpr, R::ReadGpr>) -> Self {
        match gpr_mem {
            GprMem::Gpr(gpr) => Operand::ReadWriteGpr { gpr, fixed: false },
            GprMem::Mem(amode) => Operand::Amode(amode),
        }
    }
    pub fn from_read_xmm_mem(xmm_mem: &'a mut XmmMem<R::ReadXmm, R::ReadGpr>) -> Self {
        match xmm_mem {
            XmmMem::Xmm(xmm) => Operand::ReadXmm { xmm, fixed: false },
            XmmMem::Mem(amode) => Operand::Amode(amode),
        }
    }
    pub fn from_read_write_xmm_mem(xmm_mem: &'a mut XmmMem<R::ReadWriteXmm, R::ReadGpr>) -> Self {
        match xmm_mem {
            XmmMem::Xmm(xmm) => Operand::ReadWriteXmm { xmm, fixed: false },
            XmmMem::Mem(amode) => Operand::Amode(amode),
        }
    }
    pub fn from_amode(amode: &'a mut Amode<R::ReadGpr>) -> Self {
        Operand::Amode(amode)
    }
}

// impl<R: Registers + AsReg> From<GprMem<R, R>> for Operand<'_, R> {
//     fn from(gpr_mem: GprMem<R, R>) -> Self {
//         match gpr_mem {
//             GprMem::Gpr(gpr) => Gpr::new(gpr).into(),
//             GprMem::Mem(amode) => amode.into(),
//         }
//     }
// }

// impl<'a, R: Registers> From<&mut Amode<R::ReadGpr>> for Operand<'a, R> {
//     fn from(amode: &'a mut Amode<R::ReadGpr>) -> Self {
//         Operand::Amode(amode)
//     }
// }

// impl<R: Registers, X: R::Registers::ReadGpr> From<Gpr<X>> for Operand<'a, R> {
//     fn from(gpr: Gpr<X>) -> Self {
//         Operand::ReadGpr(&mut gpr.0)
//     }
// }

// impl<'a, R: Registers> From<Gpr<R::ReadGpr>> for Operand<'a, R> {
//     fn from(gpr: &'a mut Gpr<R::ReadGpr>) -> Self {
//         Operand::ReadGpr(&mut gpr.0)
//     }
// }

// impl<'a, R: Registers> From<Gpr<R::ReadWriteGpr>> for Operand<'a, R> {
//     fn from(gpr: &'a mut Gpr<R::ReadWriteGpr>) -> Self {
//         Operand::ReadWriteGpr(&mut gpr.0)
//     }
// }

// impl<R: Registers, const E: u8> From<Fixed<R::ReadGpr, E>> for Operand<R> {
//     fn from(fixed: Fixed<R::ReadGpr, E>) -> Self {
//         Operand::FixedReadGpr(fixed.0)
//     }
// }

// impl<R: Registers> From<Xmm<R>> for Operand<R> {
//     fn from(xmm: Xmm<R>) -> Self {
//         Operand::ReadXmm(xmm)
//     }
// }

// impl<R: Registers, const E: u8> From<Fixed<R::ReadXmm, E>> for Operand<R> {
//     fn from(fixed: Fixed<R, E>) -> Self {
//         todo!()
//     }
// }

// impl<R: Registers> From<Imm8> for Operand<'_, R> {
//     fn from(imm: Imm8) -> Self {
//         Operand::Imm8(imm)
//     }
// }

// impl<R: Registers> From<Imm16> for Operand<'_, R> {
//     fn from(imm: Imm16) -> Self {
//         Operand::Imm16(imm)
//     }
// }

// impl<R: Registers> From<Imm32> for Operand<'_, R> {
//     fn from(imm: Imm32) -> Self {
//         Operand::Imm32(imm)
//     }
// }

// impl<R: Registers> From<Simm8> for Operand<'_, R> {
//     fn from(imm: Simm8) -> Self {
//         Operand::Simm8(imm)
//     }
// }

// impl<R: Registers> From<Simm16> for Operand<'_, R> {
//     fn from(imm: Simm16) -> Self {
//         Operand::Simm16(imm)
//     }
// }

// impl<R: Registers> From<Simm32> for Operand<'_, R> {
//     fn from(imm: Simm32) -> Self {
//         Operand::Simm32(imm)
//     }
// }
