//! Encoding logic for REX instructions.

use crate::api::CodeSink;

/// Encode the ModR/M byte.
#[inline]
pub(crate) fn encode_modrm(m0d: u8, enc_reg_g: u8, rm_e: u8) -> u8 {
    debug_assert!(m0d < 4);
    debug_assert!(enc_reg_g < 8);
    debug_assert!(rm_e < 8);
    ((m0d & 3) << 6) | ((enc_reg_g & 7) << 3) | (rm_e & 7)
}

/// Encode the SIB byte (scale-index-base).
#[inline]
pub(crate) fn encode_sib(scale: u8, enc_index: u8, enc_base: u8) -> u8 {
    debug_assert!(scale < 4);
    debug_assert!(enc_index < 8);
    debug_assert!(enc_base < 8);
    ((scale & 3) << 6) | ((enc_index & 7) << 3) | (enc_base & 7)
}

/// Tests whether `enc` is `rsp`, `rbp`, `rsi`, or `rdi`. If 8-bit register
/// sizes are used then it means a REX prefix is required.
///
/// This function is used below in combination with `uses_8bit` booleans to
/// determine the [`RexPrefix::must_emit`] flag. Table 3-2 in volume 1 of the
/// Intel manual details how referencing `dil`, the low 8-bits of `rdi`,
/// requires the use of the REX prefix as without it it would otherwise
/// reference the `AH` register.
///
/// This is used whenever a register is encoded with a `RexPrefix` and is also
/// only used if the register is referenced in its 8-bit form. That means for
/// example that when encoding addressing modes this function is not used.
/// Addressing modes use 64-bit versions of registers meaning that the 8-bit
/// special case does not apply.
const fn is_special_if_8bit(enc: u8) -> bool {
    enc >= 4 && enc <= 7
}

/// Construct and emit the REX prefix byte.
///
/// For more details, see section 2.2.1, "REX Prefixes" in Intel's reference
/// manual.
#[derive(Clone, Copy)]
pub struct RexPrefix {
    byte: u8,
    must_emit: bool,
}

impl RexPrefix {
    /// Construct the [`RexPrefix`] for a unary instruction.
    ///
    /// Used with a single register operand:
    /// - `x` and `r` are unused.
    /// - `b` extends the `enc` register, allowing access to r8-r15, or the top
    ///   bit of the opcode digit.
    #[inline]
    #[must_use]
    pub const fn one_op(enc: u8, w_bit: bool, uses_8bit: bool) -> Self {
        let must_emit = uses_8bit && is_special_if_8bit(enc);
        let w = if w_bit { 1 } else { 0 };
        let r = 0;
        let x = 0;
        let b = (enc >> 3) & 1;
        let byte = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
        Self { byte, must_emit }
    }

    /// Construct the [`RexPrefix`] for a binary instruction.
    ///
    /// Used without a SIB byte or for register-to-register addressing:
    /// - `r` extends the `reg` operand, allowing access to r8-r15.
    /// - `x` is unused.
    /// - `b` extends the `r/m` operand, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn two_op(enc_reg: u8, enc_rm: u8, w_bit: bool, uses_8bit: bool) -> Self {
        let mut ret = RexPrefix::mem_op(enc_reg, enc_rm, w_bit, uses_8bit);
        if uses_8bit && is_special_if_8bit(enc_rm) {
            ret.must_emit = true;
        }
        ret
    }

    /// Construct the [`RexPrefix`] for a binary instruction where one operand
    /// is a memory address.
    ///
    /// This is the same as [`RexPrefix::two_op`] except that `enc_rm` is
    /// guaranteed to address a 64-bit register. This has a slightly different
    /// meaning when `uses_8bit` is `true` to omit the REX prefix in more cases
    /// than `two_op` would emit.
    #[inline]
    #[must_use]
    pub const fn mem_op(enc_reg: u8, enc_rm: u8, w_bit: bool, uses_8bit: bool) -> Self {
        let must_emit = uses_8bit && is_special_if_8bit(enc_reg);
        let w = if w_bit { 1 } else { 0 };
        let r = (enc_reg >> 3) & 1;
        let x = 0;
        let b = (enc_rm >> 3) & 1;
        let byte = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
        Self { byte, must_emit }
    }

    /// Construct the [`RexPrefix`] for an instruction using an opcode digit.
    ///
    /// Similar to [`RexPrefix::two_op`] except that:
    /// - `r` extends the opcode digit.
    /// - `x` is unused.
    /// - `b` extends the `reg` operand, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn with_digit(digit: u8, enc_reg: u8, w_bit: bool, uses_8bit: bool) -> Self {
        Self::two_op(digit, enc_reg, w_bit, uses_8bit)
    }

    /// Construct the [`RexPrefix`] for a ternary instruction, typically using a
    /// memory address.
    ///
    /// Used with a SIB byte:
    /// - `r` extends the `reg` operand, allowing access to r8-r15.
    /// - `x` extends the index register, allowing access to r8-r15.
    /// - `b` extends the base register, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn three_op(
        enc_reg: u8,
        enc_index: u8,
        enc_base: u8,
        w_bit: bool,
        uses_8bit: bool,
    ) -> Self {
        let must_emit = uses_8bit && is_special_if_8bit(enc_reg);
        let w = if w_bit { 1 } else { 0 };
        let r = (enc_reg >> 3) & 1;
        let x = (enc_index >> 3) & 1;
        let b = (enc_base >> 3) & 1;
        let byte = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
        Self { byte, must_emit }
    }

    /// Possibly emit the REX prefix byte.
    ///
    /// This will only be emitted if the REX prefix is not `0x40` (the default)
    /// or if the instruction uses 8-bit operands.
    #[inline]
    pub fn encode(&self, sink: &mut impl CodeSink) {
        if self.byte != 0x40 || self.must_emit {
            sink.put1(self.byte);
        }
    }
}

/// Indicate the legacy opcode map used by an instruction.
pub enum LegacyMap {
    /// Legacy opcode map 0, which does not use an escape prefix.
    SingleByte,
    /// Legacy opcode map 1, which uses the escape prefix `0x0F`.
    Escaped,
}

impl LegacyMap {
    const fn bit(&self) -> u8 {
        match self {
            LegacyMap::SingleByte => 0,
            LegacyMap::Escaped => 1,
        }
    }
}

/// Construct and emit the REX2 prefix bytes.
///
/// Intel APX adds the ability to use 16 extended general-purpose registers
/// (EGPRs). The REX2 prefix, a two-byte prefix, adds two high bits to encode
/// registers `r16`-`r31` for the `R`, `X`, and `B` fields::
///
/// ```text
/// +------+  +--------------------------------------+
/// | 0xD5 |  | M0 | R4 | X4 | B4 | W | R3 | X3 | B3 |
/// +------+  +--------------------------------------+
/// ```
///
/// [`Rex2Prefix`] can replace [`RexPrefix`] for instructions with a single-byte
/// opcode (legacy opcode map 0, no escape prefix) and with a two-byte, escaped
/// opcode (legacy opcode map 1, escape prefix `0x0F`). It cannot be used with
/// three-byte opcodes (legacy opcode map 2--`0x0F38`--and 3--`0x0F3A`). The
/// `M0` bit is set to `0` for [`LegacyMap::SingleByte`] and `1` for
/// [`LegacyMap::Escaped`].
///
/// The `W` operates the same as in [`RexPrefix`].
///
/// Like [`RexPrefix`], when the operand size is 8 bits, the presence of the
/// [`Rex2Prefix`] makes GPR encodings `4-7` address byte registers
/// `[SPL,BPL,SIL,DIL]` instead of `[AH,CH,DH,BH]`.
///
/// For more details, see the Intel APX Architecture Specification, section
/// 3.1.2.1.
#[derive(Clone, Copy)]
pub struct Rex2Prefix {
    byte: u8,
    must_emit: bool,
}

impl Rex2Prefix {
    /// Construct the [`Rex2Prefix`] for a unary instruction.
    ///
    /// Used with a single register operand:
    /// - `x` and `r` are unused.
    /// - `b` extends the `enc` register, allowing access to r8-r31.
    #[inline]
    #[must_use]
    pub const fn one_op(enc: u8, w_bit: bool, uses_8bit: bool, map: LegacyMap) -> Self {
        let must_emit = uses_8bit && is_special_if_8bit(enc);
        let w = if w_bit { 1 } else { 0 };
        let b3 = (enc >> 3) & 1;
        let b4 = (enc >> 4) & 1;
        let byte = (map.bit() << 7) | (b4 << 4) | (w << 3) | b3;
        Self { byte, must_emit }
    }
}
