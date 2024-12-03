mod format;
mod formatter;
mod inst;
mod operand;

use crate::dsl;
use formatter::fmtln;
pub use formatter::Formatter;

/// Top-level function for generating Rust code.
pub fn generate(f: &mut Formatter, insts: &[dsl::Inst]) {
    // Generate "all instructions" enum.
    generate_inst_enum(f, insts);
    generate_inst_display_impl(f, insts);
    generate_inst_impl(f, insts);

    // Generate per-instruction structs.
    f.empty_line();
    for inst in insts {
        inst.generate_struct(f);
        inst.generate_struct_impl(f);
        inst.generate_display_impl(f);
        f.empty_line();
    }
}

/// `enum Inst { ... }`
fn generate_inst_enum(f: &mut Formatter, insts: &[dsl::Inst]) {
    generate_derive(f);
    fmtln!(f, "pub enum Inst {{");
    f.indent_push();
    for inst in insts {
        inst.generate_enum_variant(f);
    }
    f.indent_pop();
    fmtln!(f, "}}");
}

/// `#[derive(...)]`
fn generate_derive(f: &mut Formatter) {
    f.line("#[derive(arbitrary::Arbitrary, Clone, Debug)]", None);
}

/// `impl std::fmt::Display for Inst { ... }`
fn generate_inst_display_impl(f: &mut Formatter, insts: &[dsl::Inst]) {
    fmtln!(f, "impl std::fmt::Display for Inst {{");
    f.indent(|f| {
        fmtln!(f, "fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{");
        f.indent(|f| {
            fmtln!(f, "match self {{");
            f.indent_push();
            for inst in insts {
                inst.generate_variant_display(f);
            }
            f.indent_pop();
            fmtln!(f, "}}");
        });
        fmtln!(f, "}}");
    });
    fmtln!(f, "}}");
}

/// `impl Inst { ... }`
fn generate_inst_impl(f: &mut Formatter, insts: &[dsl::Inst]) {
    fmtln!(f, "impl Inst {{");
    f.indent(|f| {
        fmtln!(f, "pub fn encode(&self, b: &mut impl CodeSink, o: &impl KnownOffsetTable) {{");
        f.indent(|f| {
            fmtln!(f, "match self {{");
            f.indent_push();
            for inst in insts {
                inst.generate_variant_encode(f);
            }
            f.indent_pop();
            fmtln!(f, "}}");
        });
        fmtln!(f, "}}");
    });
    f.indent(|f| {
        fmtln!(f, "pub fn regalloc(&mut self, v: &mut impl RegallocVisitor) {{");
        f.indent(|f| {
            fmtln!(f, "match self {{");
            f.indent_push();
            for inst in insts {
                inst.generate_variant_regalloc(f);
            }
            f.indent_pop();
            fmtln!(f, "}}");
        });
        fmtln!(f, "}}");
    });
    fmtln!(f, "}}");
}

pub fn maybe_file_loc(fmtstr: &str, file: &'static str, line: u32) -> Option<FileLocation> {
    if fmtstr.ends_with(['{', '}']) {
        None
    } else {
        Some(FileLocation { file, line })
    }
}

pub struct FileLocation {
    file: &'static str,
    line: u32,
}

impl core::fmt::Display for FileLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}
