/// Abbreviated constructor for an instruction "format."
///
/// These model what the specification calls "instruction operand encodings,"
/// usually defined in a table after an instruction's opcodes.
pub fn fmt(
    name: impl Into<String>,
    operands: impl IntoIterator<Item = impl Into<Operand>>,
) -> Format {
    Format {
        name: name.into(),
        operands: operands.into_iter().map(Into::into).collect(),
    }
}

#[must_use]
pub fn rw(location: Location) -> Operand {
    Operand {
        location,
        mutability: Mutability::ReadWrite,
        extension: Extension::default(),
    }
}

#[must_use]
pub fn r(location: Location) -> Operand {
    Operand {
        location,
        mutability: Mutability::Read,
        extension: Extension::None,
    }
}

#[must_use]
pub fn sxq(location: Location) -> Operand {
    Operand {
        location,
        mutability: Mutability::Read,
        extension: Extension::SignExtendQuad,
    }
}

#[must_use]
pub fn sxl(location: Location) -> Operand {
    Operand {
        location,
        mutability: Mutability::Read,
        extension: Extension::SignExtendLong,
    }
}

#[must_use]
pub fn sxw(location: Location) -> Operand {
    Operand {
        location,
        mutability: Mutability::Read,
        extension: Extension::SignExtendWord,
    }
}

pub struct Format {
    pub name: String,
    pub operands: Vec<Operand>,
}

impl Format {
    pub fn uses_memory(&self) -> Option<Location> {
        debug_assert!(
            self.locations()
                .copied()
                .filter(Location::uses_memory)
                .count()
                <= 1
        );
        self.locations().copied().find(Location::uses_memory)
    }

    pub fn locations(&self) -> impl Iterator<Item = &Location> + '_ {
        self.operands.iter().map(|o| &o.location)
    }

    pub fn operands_by_bits(&self) -> Vec<u8> {
        self.locations().map(Location::bits).collect()
    }

    pub fn operands_by_kind(&self) -> Vec<OperandKind> {
        self.locations().map(Location::kind).collect()
    }
}

impl core::fmt::Display for Format {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let Format { name, operands } = self;
        let operands = operands
            .iter()
            .map(|operand| format!("{operand}"))
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{name}({operands})")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Operand {
    pub location: Location,
    pub mutability: Mutability,
    pub extension: Extension,
}

impl core::fmt::Display for Operand {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let Self { location, mutability, extension } = self;
        write!(f, "{location}")?;
        let has_default_mutability = matches!(mutability, Mutability::Read);
        let has_default_extension = matches!(extension, Extension::None);
        match (has_default_mutability, has_default_extension) {
            (true, true) => {}
            (true, false) => write!(f, "[{extension}]")?,
            (false, true) => write!(f, "[{mutability}]")?,
            (false, false) => write!(f, "[{mutability},{extension}]")?,
        }
        Ok(())
    }
}

impl From<Location> for Operand {
    fn from(location: Location) -> Self {
        let mutability = Mutability::default();
        let extension = Extension::default();
        Self { location, mutability, extension }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum Location {
    al,
    ax,
    eax,
    rax,

    imm8,
    imm16,
    imm32,

    r8,
    r16,
    r32,
    r64,
    
    xmm,

    rm8,
    rm16,
    rm32,
    rm64,
    rm128,
}

impl Location {
    #[must_use]
    pub fn bits(&self) -> u8 {
        use Location::*;
        match self {
            al | imm8 | r8 | rm8 => 8,
            ax | imm16 | r16 | rm16 => 16,
            eax | imm32 | r32 | rm32 => 32,
            rax | r64 | rm64 => 64,
            xmm | rm128 => 128,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> u8 {
        self.bits() / 8
    }

    #[must_use]
    pub fn uses_memory(&self) -> bool {
        use Location::*;
        match self {
            al | ax | eax | rax | imm8 | imm16 | imm32 | r8 | r16 | r32 | r64 | xmm => false,
            rm8 | rm16 | rm32 | rm64 | rm128 => true,
        }
    }

    #[must_use]
    pub fn kind(&self) -> OperandKind {
        use Location::*;
        match self {
            al | ax | eax | rax => OperandKind::FixedReg(*self),
            imm8 | imm16 | imm32 => OperandKind::Imm(*self),
            r8 | r16 | r32 | r64 => OperandKind::Reg(*self),
            xmm => OperandKind::XmmReg(*self),
            rm8 | rm16 | rm32 | rm64 => OperandKind::RegMem(*self),
            rm128 => OperandKind::XmmRegMem(*self),
        }
    }
}

impl core::fmt::Display for Location {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use Location::*;
        match self {
            al => write!(f, "al"),
            ax => write!(f, "ax"),
            eax => write!(f, "eax"),
            rax => write!(f, "rax"),

            imm8 => write!(f, "imm8"),
            imm16 => write!(f, "imm16"),
            imm32 => write!(f, "imm32"),

            r8 => write!(f, "r8"),
            r16 => write!(f, "r16"),
            r32 => write!(f, "r32"),
            r64 => write!(f, "r64"),

            xmm => write!(f, "xmm"),

            rm8 => write!(f, "rm8"),
            rm16 => write!(f, "rm16"),
            rm32 => write!(f, "rm32"),
            rm64 => write!(f, "rm64"),
            rm128 => write!(f, "rm128"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OperandKind {
    FixedReg(Location),
    Imm(Location),
    Reg(Location),
    RegMem(Location),
    XmmReg(Location),
    XmmRegMem(Location),
}

#[derive(Clone, Copy, Debug)]
pub enum Mutability {
    Read,
    ReadWrite,
}

impl Default for Mutability {
    fn default() -> Self {
        Self::Read
    }
}

impl core::fmt::Display for Mutability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "r"),
            Self::ReadWrite => write!(f, "rw"),
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

impl Default for Extension {
    fn default() -> Self {
        Self::None
    }
}

impl core::fmt::Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Extension::None => write!(f, ""),
            Extension::SignExtendQuad => write!(f, "sxq"),
            Extension::SignExtendLong => write!(f, "sxl"),
            Extension::SignExtendWord => write!(f, "sxw"),
            Extension::ZeroExtend => write!(f, "zx"),
        }
    }
}
