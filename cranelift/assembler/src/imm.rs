//! Immediate operands to instructions.

#![allow(clippy::module_name_repetitions)]
#![allow(unused_comparisons)] // Necessary to use maybe_print_hex! with `u*` values.
#![allow(clippy::cast_possible_wrap)] // Necessary to cast to `i*` for sign extension.

use crate::sink::{KnownOffset, KnownOffsetTable};
use arbitrary::Arbitrary;

/// This helper function prints the hexadecimal representation of the immediate
/// value, but only if the value is greater than or equal to 10. This is
/// necessary for how Capstone pretty-prints immediate values.
macro_rules! maybe_print_hex {
    ($n:expr) => {
        if $n >= 0 && $n < 10 {
            format!("${:x}", $n)
        } else {
            format!("$0x{:x}", $n)
        }
    };
}

#[derive(Arbitrary, Clone, Copy, Debug)]
pub struct Imm8(u8);

impl Imm8 {
    pub fn new(value: u8) -> Self {
        Self(value)
    }

    #[must_use]
    pub fn value(&self) -> u8 {
        self.0
    }

    pub fn to_string(&self, extend: Extension) -> String {
        use Extension::{None, SignExtendLong, SignExtendQuad, SignExtendWord, ZeroExtend};
        match extend {
            None => maybe_print_hex!(self.0),
            SignExtendWord => maybe_print_hex!(i16::from(self.0 as i8)),
            SignExtendLong => maybe_print_hex!(i32::from(self.0 as i8)),
            SignExtendQuad => maybe_print_hex!(i64::from(self.0 as i8)),
            ZeroExtend => maybe_print_hex!(u64::from(self.0)),
        }
    }
}

#[derive(Arbitrary, Clone, Debug)]
pub struct Imm16(u16);

impl Imm16 {
    #[must_use]
    pub fn value(&self) -> u16 {
        self.0
    }

    pub fn to_string(&self, extend: Extension) -> String {
        use Extension::{None, SignExtendLong, SignExtendQuad, SignExtendWord, ZeroExtend};
        match extend {
            None => maybe_print_hex!(self.0),
            SignExtendWord => maybe_print_hex!(self.0 as i16),
            SignExtendLong => maybe_print_hex!(i32::from(self.0 as i16)),
            SignExtendQuad => maybe_print_hex!(i64::from(self.0 as i16)),
            ZeroExtend => maybe_print_hex!(u64::from(self.0)),
        }
    }
}

#[derive(Arbitrary, Clone, Debug)]
pub struct Imm32(u32);

impl Imm32 {
    #[must_use]
    pub fn value(&self) -> u32 {
        self.0
    }
    pub fn to_string(&self, extend: Extension) -> String {
        use Extension::{None, SignExtendLong, SignExtendQuad, SignExtendWord, ZeroExtend};
        match extend {
            None => maybe_print_hex!(self.0),
            SignExtendWord => unreachable!("cannot sign extend a 32-bit value"),
            SignExtendLong => maybe_print_hex!(self.0 as i32),
            SignExtendQuad => maybe_print_hex!(i64::from(self.0 as i32)),
            ZeroExtend => maybe_print_hex!(u64::from(self.0)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Extension {
    None,
    SignExtendQuad,
    SignExtendLong,
    SignExtendWord,
    ZeroExtend,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub struct Simm32(i32);

impl Simm32 {
    #[must_use]
    pub fn value(self) -> i32 {
        self.0
    }
}

impl std::fmt::LowerHex for Simm32 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0 == 0 {
            return Ok(());
        }
        if self.0 < 0 {
            write!(f, "-")?;
        }
        if self.0 > 9 || self.0 < -9 {
            write!(f, "0x")?;
        }
        let abs = match self.0.checked_abs() {
            Some(i) => i,
            None => -2_147_483_648,
        };
        std::fmt::LowerHex::fmt(&abs, f)
    }
}

#[derive(Clone, Debug)]
pub struct Simm32PlusKnownOffset {
    pub simm32: Simm32,
    pub offset: Option<KnownOffset>,
}

impl Simm32PlusKnownOffset {
    #[must_use]
    pub fn value(&self, offsets: &impl KnownOffsetTable) -> i32 {
        let offset = match self.offset {
            Some(offset) => offsets[offset],
            None => 0,
        };
        self.simm32
            .value()
            .checked_add(offset)
            .expect("no wrapping")
    }
}

impl Arbitrary<'_> for Simm32PlusKnownOffset {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        // For now, we don't generate offsets (TODO).
        Ok(Self { simm32: Simm32::arbitrary(u)?, offset: None })
    }
}

impl std::fmt::LowerHex for Simm32PlusKnownOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(offset) = self.offset {
            write!(f, "<offset:{offset}>+")?;
        }
        std::fmt::LowerHex::fmt(&self.simm32, f)
    }
}
