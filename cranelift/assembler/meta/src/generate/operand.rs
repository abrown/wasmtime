use crate::dsl;

impl dsl::Location {
    /// `<operand type>`
    #[must_use]
    pub fn generate_type(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax => None,
            imm8 => Some("Imm8"),
            imm16 => Some("Imm16"),
            imm32 => Some("Imm32"),
            r8 | r16 | r32 | r64 => Some("Gpr"),
            rm8 | rm16 | rm32 | rm64 => Some("GprMem"),
        }
    }

    /// `self.<operand>.to_string(...)`
    #[must_use]
    pub fn generate_to_string(&self, extension: dsl::Extension) -> String {
        use dsl::Location::*;
        match self {
            al => "\"%al\"".into(),
            ax => "\"%ax\"".into(),
            eax => "\"%eax\"".into(),
            rax => "\"%rax\"".into(),
            imm8 | imm16 | imm32 => {
                let variant = extension.generate_variant();
                format!("self.{self}.to_string({variant})")
            }
            r8 | r16 | r32 | r64 | rm8 | rm16 | rm32 | rm64 => match self.generate_size() {
                Some(size) => format!("self.{self}.to_string({size})"),
                None => unreachable!(),
            },
        }
    }

    /// `Size::<operand size>`
    #[must_use]
    pub fn generate_size(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax | imm8 | imm16 | imm32 => None,
            r8 | rm8 => Some("Size::Byte"),
            r16 | rm16 => Some("Size::Word"),
            r32 | rm32 => Some("Size::Doubleword"),
            r64 | rm64 => Some("Size::Quadword"),
        }
    }

    /// `Gpr(regs::...)`
    #[must_use]
    pub fn generate_fixed_reg(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax => Some("Gpr::new(reg::ENC_RAX.into())"),
            imm8 | imm16 | imm32 | r8 | r16 | r32 | r64 | rm8 | rm16 | rm32 | rm64 => None,
        }
    }
}

impl dsl::Mutability {
    #[must_use]
    pub fn generate_regalloc_call(&self) -> &str {
        match self {
            dsl::Mutability::Read => "read",
            dsl::Mutability::ReadWrite => "read_write",
        }
    }
}

impl dsl::Extension {
    /// `Extension::...`
    #[must_use]
    pub fn generate_variant(&self) -> &str {
        use dsl::Extension::*;
        match self {
            None => "Extension::None",
            SignExtendWord => "Extension::SignExtendWord",
            SignExtendLong => "Extension::SignExtendLong",
            SignExtendQuad => "Extension::SignExtendQuad",
            ZeroExtend => "Extension::ZeroExtend",
        }
    }
}
