//! Pure register operands.

use crate::{alloc::RegallocVisitor, rex::RexFlags};
use arbitrary::Arbitrary;

pub const ENC_XMM0: u8 = 0;
pub const ENC_XMM1: u8 = 1;
pub const ENC_XMM2: u8 = 2;
pub const ENC_XMM3: u8 = 3;
pub const ENC_XMM4: u8 = 4;
pub const ENC_XMM5: u8 = 5;
pub const ENC_XMM6: u8 = 6;
pub const ENC_XMM7: u8 = 7;
pub const ENC_XMM8: u8 = 8;
pub const ENC_XMM9: u8 = 9;
pub const ENC_XMM10: u8 = 10;
pub const ENC_XMM11: u8 = 11;
pub const ENC_XMM12: u8 = 12;
pub const ENC_XMM13: u8 = 13;
pub const ENC_XMM14: u8 = 14;
pub const ENC_XMM15: u8 = 15;

pub const ENC_RAX: u8 = 0;
pub const ENC_RCX: u8 = 1;
pub const ENC_RDX: u8 = 2;
pub const ENC_RBX: u8 = 3;
pub const ENC_RSP: u8 = 4;
pub const ENC_RBP: u8 = 5;
pub const ENC_RSI: u8 = 6;
pub const ENC_RDI: u8 = 7;
pub const ENC_R8: u8 = 8;
pub const ENC_R9: u8 = 9;
pub const ENC_R10: u8 = 10;
pub const ENC_R11: u8 = 11;
pub const ENC_R12: u8 = 12;
pub const ENC_R13: u8 = 13;
pub const ENC_R14: u8 = 14;
pub const ENC_R15: u8 = 15;

/// A general purpose x64 register (e.g., `%rax`).
///
/// This holds a larger value than needed to accommodate register allocation.
/// Cranelift's register allocator expects to modify an instruction's operands
/// in place (see [`Gpr::as_mut`]); Cranelift assigns a virtual register to each
/// operand and only later replaces these with true HW registers. A consequence:
/// register allocation _must happen_ before encoding the register (see
/// [`Gpr::enc`]).
#[derive(Clone, Copy, Debug)]
pub struct Gpr(pub(crate) u32);

impl Gpr {
    /// Create a [`Gpr`] that may be real (emit-able in machine code) or virtual
    /// (waiting for register allocation).
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    pub fn enc(&self) -> u8 {
        assert!(self.0 < 16, "invalid register: {}", self.0);
        self.0.try_into().expect("invalid register")
    }

    pub fn always_emit_if_8bit_needed(&self, rex: &mut RexFlags) {
        let enc_reg = self.enc();
        if (4..=7).contains(&enc_reg) {
            rex.always_emit();
        }
    }

    pub fn to_string(&self, size: Size) -> &str {
        use Size::{Byte, Doubleword, Quadword, Word, DoubleQuadword};
        match self.enc() {
            ENC_RAX => match size {
                Byte => "%al",
                Word => "%ax",
                Doubleword => "%eax",
                Quadword => "%rax",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RBX => match size {
                Byte => "%bl",
                Word => "%bx",
                Doubleword => "%ebx",
                Quadword => "%rbx",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RCX => match size {
                Byte => "%cl",
                Word => "%cx",
                Doubleword => "%ecx",
                Quadword => "%rcx",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RDX => match size {
                Byte => "%dl",
                Word => "%dx",
                Doubleword => "%edx",
                Quadword => "%rdx",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RSI => match size {
                Byte => "%sil",
                Word => "%si",
                Doubleword => "%esi",
                Quadword => "%rsi",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RDI => match size {
                Byte => "%dil",
                Word => "%di",
                Doubleword => "%edi",
                Quadword => "%rdi",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RBP => match size {
                Byte => "%bpl",
                Word => "%bp",
                Doubleword => "%ebp",
                Quadword => "%rbp",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_RSP => match size {
                Byte => "%spl",
                Word => "%sp",
                Doubleword => "%esp",
                Quadword => "%rsp",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R8 => match size {
                Byte => "%r8b",
                Word => "%r8w",
                Doubleword => "%r8d",
                Quadword => "%r8",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R9 => match size {
                Byte => "%r9b",
                Word => "%r9w",
                Doubleword => "%r9d",
                Quadword => "%r9",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R10 => match size {
                Byte => "%r10b",
                Word => "%r10w",
                Doubleword => "%r10d",
                Quadword => "%r10",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R11 => match size {
                Byte => "%r11b",
                Word => "%r11w",
                Doubleword => "%r11d",
                Quadword => "%r11",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R12 => match size {
                Byte => "%r12b",
                Word => "%r12w",
                Doubleword => "%r12d",
                Quadword => "%r12",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R13 => match size {
                Byte => "%r13b",
                Word => "%r13w",
                Doubleword => "%r13d",
                Quadword => "%r13",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R14 => match size {
                Byte => "%r14b",
                Word => "%r14w",
                Doubleword => "%r14d",
                Quadword => "%r14",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            ENC_R15 => match size {
                Byte => "%r15b",
                Word => "%r15w",
                Doubleword => "%r15d",
                Quadword => "%r15",
                DoubleQuadword => unreachable!("Gpr should never be DoubleQuadword"),
            },
            _ => panic!("%invalid{}", self.0), // TODO: print instead?
        }
    }

    pub fn read(&mut self, visitor: &mut impl RegallocVisitor) {
        visitor.read(self.as_mut());
    }

    pub fn read_write(&mut self, visitor: &mut impl RegallocVisitor) {
        visitor.read_write(self.as_mut());
    }

    /// Allow the register allocator to modify this register in place.
    pub fn as_mut(&mut self) -> &mut u32 {
        &mut self.0
    }

    /// Allow external users to inspect this register.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Xmm(pub(crate) u32);

impl Xmm {
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    pub fn enc(&self) -> u8 {
        assert!(self.0 < 16, "invalid register: {}", self.0);
        self.0.try_into().expect("invalid register")
    }

    pub fn to_string(&self) -> &str {
        match self.enc() {
            ENC_XMM0 => "%xmm0",
            ENC_XMM1 => "%xmm1",
            ENC_XMM2 => "%xmm2",
            ENC_XMM3 => "%xmm3",
            ENC_XMM4 => "%xmm4",
            ENC_XMM5 => "%xmm5",
            ENC_XMM6 => "%xmm6",
            ENC_XMM7 => "%xmm7",
            ENC_XMM8 => "%xmm8",
            ENC_XMM9 => "%xmm9",
            ENC_XMM10 => "%xmm10",
            ENC_XMM11 => "%xmm11",
            ENC_XMM12 => "%xmm12",
            ENC_XMM13 => "%xmm13",
            ENC_XMM14 => "%xmm14",
            ENC_XMM15 => "%xmm15",
            _ => panic!("invalid register encoding: {}", self.0),
        }
    }

    pub fn read(&mut self, visitor: &mut impl RegallocVisitor) {
        visitor.read(self.as_mut());
    }

    pub fn read_write(&mut self, visitor: &mut impl RegallocVisitor) {
        visitor.read_write(self.as_mut());
    }

    /// Allow the register allocator to modify this register in place.
    pub fn as_mut(&mut self) -> &mut u32 {
        &mut self.0
    }

    /// Allow external users to inspect this register.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl<'a> Arbitrary<'a> for Xmm {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self(u.int_in_range(0..=15)?))
    }
}

impl<'a> Arbitrary<'a> for Gpr {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self(u.int_in_range(0..=15)?))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Size {
    Byte,
    Word,
    Doubleword,
    Quadword,
    DoubleQuadword,
}

#[derive(Clone, Debug)]
pub struct Gpr2MinusRsp(Gpr);

impl Gpr2MinusRsp {
    pub fn as_mut(&mut self) -> &mut u32 {
        self.0.as_mut()
    }
    pub fn enc(&self) -> u8 {
        self.0.enc()
    }
    pub fn to_string(&self, size: Size) -> &str {
        self.0.to_string(size)
    }
}

impl<'a> Arbitrary<'a> for Gpr2MinusRsp {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let gpr = u.choose(&[
            ENC_RAX, ENC_RCX, ENC_RDX, ENC_RBX, ENC_RBP, ENC_RSI, ENC_RDI, ENC_R8, ENC_R9, ENC_R10,
            ENC_R11, ENC_R12, ENC_R13, ENC_R14, ENC_R15,
        ])?;
        Ok(Self(Gpr(u32::from(*gpr))))
    }
}
