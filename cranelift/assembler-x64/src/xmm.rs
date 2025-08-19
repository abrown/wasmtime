//! Xmm register operands; see [`Xmm`].

use crate::{AsReg, CodeSink, rex::encode_modrm};

/// An x64 SSE register (e.g., `%xmm0`).
#[derive(Clone, Copy, Debug)]
pub struct Xmm<R: AsReg = u8>(pub(crate) R);

impl<R: AsReg> Xmm<R> {
    /// Create a new [`Xmm`] register.
    pub fn new(reg: R) -> Self {
        Self(reg)
    }

    /// Return the register's hardware encoding; the underlying type `R` _must_
    /// be a real register at this point.
    ///
    /// # Panics
    ///
    /// Panics if the register is not a valid Xmm register.
    pub fn enc(&self) -> u8 {
        let enc = self.0.enc();
        assert!(enc < 16, "invalid register: {enc}");
        enc
    }

    /// Return the register name.
    pub fn to_string(&self) -> String {
        self.0.to_string(None)
    }

    /// Emit this register as the `r/m` field of a ModR/M byte.
    pub(crate) fn encode_modrm(&self, sink: &mut impl CodeSink, enc_reg: u8) {
        sink.put1(encode_modrm(0b11, enc_reg & 0b111, self.enc() & 0b111));
    }

    /// Return the registers for encoding the `b` and `x` bits (e.g., in a VEX
    /// prefix).
    ///
    /// This is primarily used for `*Mem` variants (see
    /// `XmmMem::encode_bx_regs`), but when used on a single `Xmm` register,
    /// only the `b` bit is set by the topmost bit (the fourth bit) of this
    /// register. We expect this register to be in the `rm` slot.
    pub(crate) fn encode_bx_regs(&self) -> (Option<u8>, Option<u8>) {
        (Some(self.enc()), None)
    }
}

impl<R: AsReg> AsRef<R> for Xmm<R> {
    fn as_ref(&self) -> &R {
        &self.0
    }
}

impl<R: AsReg> AsMut<R> for Xmm<R> {
    fn as_mut(&mut self) -> &mut R {
        &mut self.0
    }
}

impl<R: AsReg> From<R> for Xmm<R> {
    fn from(reg: R) -> Xmm<R> {
        Xmm(reg)
    }
}

/// Encode xmm registers.
pub mod enc {
    pub const XMM0: u8 = 0;
    pub const XMM1: u8 = 1;
    pub const XMM2: u8 = 2;
    pub const XMM3: u8 = 3;
    pub const XMM4: u8 = 4;
    pub const XMM5: u8 = 5;
    pub const XMM6: u8 = 6;
    pub const XMM7: u8 = 7;
    pub const XMM8: u8 = 8;
    pub const XMM9: u8 = 9;
    pub const XMM10: u8 = 10;
    pub const XMM11: u8 = 11;
    pub const XMM12: u8 = 12;
    pub const XMM13: u8 = 13;
    pub const XMM14: u8 = 14;
    pub const XMM15: u8 = 15;
    pub const XMM16: u8 = 16;
    pub const XMM17: u8 = 17;
    pub const XMM18: u8 = 18;
    pub const XMM19: u8 = 19;
    pub const XMM20: u8 = 20;
    pub const XMM21: u8 = 21;
    pub const XMM22: u8 = 22;
    pub const XMM23: u8 = 23;
    pub const XMM24: u8 = 24;
    pub const XMM25: u8 = 25;
    pub const XMM26: u8 = 26;
    pub const XMM27: u8 = 27;
    pub const XMM28: u8 = 28;
    pub const XMM29: u8 = 29;
    pub const XMM30: u8 = 30;
    pub const XMM31: u8 = 31;

    /// Return the name of a XMM encoding (`enc`).
    ///
    /// # Panics
    ///
    /// This function will panic if the encoding is not a valid x64 register.
    pub fn to_string(enc: u8) -> &'static str {
        match enc {
            XMM0 => "%xmm0",
            XMM1 => "%xmm1",
            XMM2 => "%xmm2",
            XMM3 => "%xmm3",
            XMM4 => "%xmm4",
            XMM5 => "%xmm5",
            XMM6 => "%xmm6",
            XMM7 => "%xmm7",
            XMM8 => "%xmm8",
            XMM9 => "%xmm9",
            XMM10 => "%xmm10",
            XMM11 => "%xmm11",
            XMM12 => "%xmm12",
            XMM13 => "%xmm13",
            XMM14 => "%xmm14",
            XMM15 => "%xmm15",
            XMM16 => "%xmm16",
            XMM17 => "%xmm17",
            XMM18 => "%xmm18",
            XMM19 => "%xmm19",
            XMM20 => "%xmm20",
            XMM21 => "%xmm21",
            XMM22 => "%xmm22",
            XMM23 => "%xmm23",
            XMM24 => "%xmm24",
            XMM25 => "%xmm25",
            XMM26 => "%xmm26",
            XMM27 => "%xmm27",
            XMM28 => "%xmm28",
            XMM29 => "%xmm29",
            XMM30 => "%xmm30",
            XMM31 => "%xmm31",
            _ => panic!("%invalid{enc}"),
        }
    }
}
