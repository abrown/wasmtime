use super::{fmtln, generate_derive, Formatter};
use crate::dsl;

impl dsl::Inst {
    /// `struct <inst> { <op>: Reg, <op>: Reg, ... }`
    pub fn generate_struct(&self, f: &mut Formatter) {
        let struct_name = self.struct_name();
        let struct_fields = self.struct_fields();

        f.line(format!("/// `{self}`"), None);
        generate_derive(f);
        fmtln!(f, "pub struct {struct_name} {{ {struct_fields} }}");
    }

    /// `pub <op>: <type>, *`
    #[must_use]
    fn struct_fields(&self) -> String {
        self.format
            .locations()
            .filter_map(|l| {
                let ty = l.generate_type()?;
                Some(format!("pub {l}: {ty}"))
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// `<class name>_<format name>`
    #[must_use]
    fn struct_name(&self) -> String {
        format!("{}_{}", self.name.to_lowercase(), self.format.name.to_lowercase())
    }

    /// `<inst>(<inst>),`
    pub fn generate_enum_variant(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        fmtln!(f, "{variant_name}({variant_name}),");
    }

    // `Self::<inst>(i) => write!(f, "{}", i),`
    pub fn generate_variant_display(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        fmtln!(f, "Self::{variant_name}(i) => write!(f, \"{{i}}\"),");
    }

    // `Self::<inst>(i) => i.encode(b, o),`
    pub fn generate_variant_encode(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        fmtln!(f, "Self::{variant_name}(i) => i.encode(b, o),");
    }

    // `Self::<inst>(i) => i.regalloc(v),`
    pub fn generate_variant_regalloc(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        fmtln!(f, "Self::{variant_name}(i) => i.regalloc(v),");
    }

    /// `impl <inst> { ... }`
    pub fn generate_struct_impl(&self, f: &mut Formatter) {
        let struct_name = self.struct_name();
        fmtln!(f, "impl {struct_name} {{");
        f.indent_push();
        self.generate_encode_function(f);
        f.empty_line();
        self.generate_regalloc_function(f);
        f.indent_pop();
        fmtln!(f, "}}");
    }

    /// `fn encode(&self, buf: &mut impl CodeSink, off: &impl KnownOffsetTable) { ... }`
    pub fn generate_encode_function(&self, f: &mut Formatter) {
        let off = if self.format.uses_memory().is_some() {
            "off"
        } else {
            "_"
        };
        fmtln!(
            f,
            "pub fn encode(&self, buf: &mut impl CodeSink, {off}: &impl KnownOffsetTable) {{"
        );
        f.indent_push();

        // Emit trap.
        if let Some(op) = self.format.uses_memory() {
            f.empty_line();
            f.comment("Emit trap.");
            fmtln!(f, "if let GprMem::Mem({op}) = &self.{op} {{");
            f.indent(|f| {
                fmtln!(f, "if let Some(trap_code) = {op}.trap_code() {{");
                f.indent(|f| {
                    fmtln!(f, "buf.add_trap(trap_code);");
                });
                fmtln!(f, "}}");
            });
            fmtln!(f, "}}");
        }

        match &self.encoding {
            dsl::Encoding::Rex(rex) => self.format.generate_rex_encoding(f, rex),
            dsl::Encoding::Vex(_) => todo!(),
        }

        f.indent_pop();
        fmtln!(f, "}}");
    }

    /// `fn regalloc(&self) -> String { ... }`
    pub fn generate_regalloc_function(&self, f: &mut Formatter) {
        use dsl::OperandKind::*;
        fmtln!(f, "pub fn regalloc(&mut self, visitor: &mut impl RegallocVisitor) {{");
        f.indent(|f| {
            for o in &self.format.operands {
                match o.location.kind() {
                    Imm(_) => {
                        // Immediates do not need register allocation.
                    }
                    FixedReg(_) => {
                        let call = o.mutability.generate_regalloc_call();
                        let Some(fixed) = o.location.generate_fixed_reg() else {
                            unreachable!()
                        };
                        fmtln!(f, "visitor.fixed_{call}({fixed}.as_mut());");
                    }
                    Reg(reg) => {
                        let call = o.mutability.generate_regalloc_call();
                        fmtln!(f, "self.{reg}.{call}(visitor);");
                    }
                    RegMem(rm) => {
                        let call = o.mutability.generate_regalloc_call();
                        fmtln!(f, "self.{rm}.{call}(visitor);");
                    }
                }
            }
        });
        fmtln!(f, "}}");
    }

    /// `impl Debug for <inst> { ... }`
    pub fn generate_display_impl(&self, f: &mut Formatter) {
        let struct_name = self.struct_name();
        fmtln!(f, "impl std::fmt::Display for {struct_name} {{");
        f.indent_push();
        fmtln!(f, "fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{");

        f.indent_push();
        for op in &self.format.operands {
            let location = op.location;
            let to_string = location.generate_to_string(op.extension);
            fmtln!(f, "let {location} = {to_string};");
        }

        let inst_name = &self.name;
        let ordered_ops = self.format.generate_att_style_operands();
        fmtln!(f, "write!(f, \"{inst_name} {ordered_ops}\")");
        f.indent_pop();
        fmtln!(f, "}}");

        f.indent_pop();
        fmtln!(f, "}}");
    }
}
