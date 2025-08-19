//! Pure register operands; see [`Gpr`].

use crate::AsReg;

/// A general purpose x64 register (e.g., `%rax`).
///
/// This container wraps true register type `R` to allow users to specify their
/// own; by default this will use `u8`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gpr<R: AsReg = u8>(pub(crate) R);

impl<R: AsReg> Gpr<R> {
    /// Create a [`Gpr`] that may be real (immediately emit-able in machine
    /// code) or virtual (waiting for register allocation).
    pub fn new(reg: R) -> Self {
        Self(reg)
    }

    /// Return the register's hardware encoding; the underlying type `R` _must_
    /// be a real register at this point.
    ///
    /// # Panics
    ///
    /// Panics if the register is not a valid x64 register.
    pub fn enc(&self) -> u8 {
        let enc = self.0.enc();
        assert!(enc < 16, "invalid register: {enc}");
        enc
    }

    /// Return the register name at the given `size`.
    pub fn to_string(&self, size: Size) -> String {
        self.0.to_string(Some(size))
    }
}

impl<R: AsReg> AsRef<R> for Gpr<R> {
    fn as_ref(&self) -> &R {
        &self.0
    }
}

impl<R: AsReg> AsMut<R> for Gpr<R> {
    fn as_mut(&mut self) -> &mut R {
        &mut self.0
    }
}

impl<R: AsReg> From<R> for Gpr<R> {
    fn from(reg: R) -> Gpr<R> {
        Gpr(reg)
    }
}

/// A single x64 register encoding can access a different number of bits.
#[derive(Copy, Clone, Debug)]
pub enum Size {
    /// An 8-bit access.
    Byte,
    /// A 16-bit access.
    Word,
    /// A 32-bit access.
    Doubleword,
    /// A 64-bit access.
    Quadword,
}

/// Like [`Gpr`], but with `%rsp` disallowed.
///
/// This is due to avoid special cases of REX encodings, see Intel SDM Vol. 2A,
/// table 2-5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonRspGpr<R: AsReg>(R);

impl<R: AsReg> NonRspGpr<R> {
    /// See [`Gpr::new`].
    pub fn new(reg: R) -> Self {
        Self(reg)
    }

    /// See [`Gpr::to_string`].
    pub fn to_string(&self, size: Size) -> String {
        self.0.to_string(Some(size))
    }

    /// See [`Gpr::enc`].
    ///
    /// # Panics
    ///
    /// Panics if the register is invalid or `%rsp`.
    pub fn enc(&self) -> u8 {
        let enc = self.0.enc();
        assert!(enc < 16, "invalid register: {enc}");
        assert_ne!(enc, enc::RSP, "invalid register: %rsp");
        enc
    }
}

impl<R: AsReg> AsRef<R> for NonRspGpr<R> {
    fn as_ref(&self) -> &R {
        &self.0
    }
}

impl<R: AsReg> AsMut<R> for NonRspGpr<R> {
    fn as_mut(&mut self) -> &mut R {
        &mut self.0
    }
}

impl<R: AsReg> From<R> for NonRspGpr<R> {
    fn from(reg: R) -> NonRspGpr<R> {
        NonRspGpr(reg)
    }
}

/// Encode x64 registers.
pub mod enc {
    use super::Size;

    pub const RAX: u8 = 0;
    pub const RCX: u8 = 1;
    pub const RDX: u8 = 2;
    pub const RBX: u8 = 3;
    pub const RSP: u8 = 4;
    pub const RBP: u8 = 5;
    pub const RSI: u8 = 6;
    pub const RDI: u8 = 7;
    pub const R8: u8 = 8;
    pub const R9: u8 = 9;
    pub const R10: u8 = 10;
    pub const R11: u8 = 11;
    pub const R12: u8 = 12;
    pub const R13: u8 = 13;
    pub const R14: u8 = 14;
    pub const R15: u8 = 15;
    pub const R16: u8 = 16;
    pub const R17: u8 = 17;
    pub const R18: u8 = 18;
    pub const R19: u8 = 19;
    pub const R20: u8 = 20;
    pub const R21: u8 = 21;
    pub const R22: u8 = 22;
    pub const R23: u8 = 23;
    pub const R24: u8 = 24;
    pub const R25: u8 = 25;
    pub const R26: u8 = 26;
    pub const R27: u8 = 27;
    pub const R28: u8 = 28;
    pub const R29: u8 = 29;
    pub const R30: u8 = 30;
    pub const R31: u8 = 31;

    /// Return the name of a GPR encoding (`enc`) at the given `size`.
    ///
    /// # Panics
    ///
    /// This function will panic if the encoding is not a valid x64 register.
    pub fn to_string(enc: u8, size: Size) -> &'static str {
        use Size::{Byte, Doubleword, Quadword, Word};
        match enc {
            RAX => match size {
                Byte => "%al",
                Word => "%ax",
                Doubleword => "%eax",
                Quadword => "%rax",
            },
            RBX => match size {
                Byte => "%bl",
                Word => "%bx",
                Doubleword => "%ebx",
                Quadword => "%rbx",
            },
            RCX => match size {
                Byte => "%cl",
                Word => "%cx",
                Doubleword => "%ecx",
                Quadword => "%rcx",
            },
            RDX => match size {
                Byte => "%dl",
                Word => "%dx",
                Doubleword => "%edx",
                Quadword => "%rdx",
            },
            RSI => match size {
                Byte => "%sil",
                Word => "%si",
                Doubleword => "%esi",
                Quadword => "%rsi",
            },
            RDI => match size {
                Byte => "%dil",
                Word => "%di",
                Doubleword => "%edi",
                Quadword => "%rdi",
            },
            RBP => match size {
                Byte => "%bpl",
                Word => "%bp",
                Doubleword => "%ebp",
                Quadword => "%rbp",
            },
            RSP => match size {
                Byte => "%spl",
                Word => "%sp",
                Doubleword => "%esp",
                Quadword => "%rsp",
            },
            R8 => match size {
                Byte => "%r8b",
                Word => "%r8w",
                Doubleword => "%r8d",
                Quadword => "%r8",
            },
            R9 => match size {
                Byte => "%r9b",
                Word => "%r9w",
                Doubleword => "%r9d",
                Quadword => "%r9",
            },
            R10 => match size {
                Byte => "%r10b",
                Word => "%r10w",
                Doubleword => "%r10d",
                Quadword => "%r10",
            },
            R11 => match size {
                Byte => "%r11b",
                Word => "%r11w",
                Doubleword => "%r11d",
                Quadword => "%r11",
            },
            R12 => match size {
                Byte => "%r12b",
                Word => "%r12w",
                Doubleword => "%r12d",
                Quadword => "%r12",
            },
            R13 => match size {
                Byte => "%r13b",
                Word => "%r13w",
                Doubleword => "%r13d",
                Quadword => "%r13",
            },
            R14 => match size {
                Byte => "%r14b",
                Word => "%r14w",
                Doubleword => "%r14d",
                Quadword => "%r14",
            },
            R15 => match size {
                Byte => "%r15b",
                Word => "%r15w",
                Doubleword => "%r15d",
                Quadword => "%r15",
            },
            R16 => match size {
                Byte => "%r16b",
                Word => "%r16w",
                Doubleword => "%r16d",
                Quadword => "%r16",
            },
            R17 => match size {
                Byte => "%r17b",
                Word => "%r17w",
                Doubleword => "%r17d",
                Quadword => "%r17",
            },
            R18 => match size {
                Byte => "%r18b",
                Word => "%r18w",
                Doubleword => "%r18d",
                Quadword => "%r18",
            },
            R19 => match size {
                Byte => "%r19b",
                Word => "%r19w",
                Doubleword => "%r19d",
                Quadword => "%r19",
            },
            R20 => match size {
                Byte => "%r20b",
                Word => "%r20w",
                Doubleword => "%r20d",
                Quadword => "%r20",
            },
            R21 => match size {
                Byte => "%r21b",
                Word => "%r21w",
                Doubleword => "%r21d",
                Quadword => "%r21",
            },
            R22 => match size {
                Byte => "%r22b",
                Word => "%r22w",
                Doubleword => "%r22d",
                Quadword => "%r22",
            },
            R23 => match size {
                Byte => "%r23b",
                Word => "%r23w",
                Doubleword => "%r23d",
                Quadword => "%r23",
            },
            R24 => match size {
                Byte => "%r24b",
                Word => "%r24w",
                Doubleword => "%r24d",
                Quadword => "%r24",
            },
            R25 => match size {
                Byte => "%r25b",
                Word => "%r25w",
                Doubleword => "%r25d",
                Quadword => "%r25",
            },
            R26 => match size {
                Byte => "%r26b",
                Word => "%r26w",
                Doubleword => "%r26d",
                Quadword => "%r26",
            },
            R27 => match size {
                Byte => "%r27b",
                Word => "%r27w",
                Doubleword => "%r27d",
                Quadword => "%r27",
            },
            R28 => match size {
                Byte => "%r28b",
                Word => "%r28w",
                Doubleword => "%r28d",
                Quadword => "%r28",
            },
            R29 => match size {
                Byte => "%r29b",
                Word => "%r29w",
                Doubleword => "%r29d",
                Quadword => "%r29",
            },
            R30 => match size {
                Byte => "%r30b",
                Word => "%r30w",
                Doubleword => "%r30d",
                Quadword => "%r30",
            },
            R31 => match size {
                Byte => "%r31b",
                Word => "%r31w",
                Doubleword => "%r31d",
                Quadword => "%r31",
            },
            _ => panic!("%invalid{enc}"),
        }
    }
}
