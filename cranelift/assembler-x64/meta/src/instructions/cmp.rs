use crate::dsl::{EflagsMutability::*, Feature::*, Inst, Location::*};
use crate::dsl::{fmt, inst, r, rex};

#[rustfmt::skip] // Keeps instructions on a single line.
pub fn list() -> Vec<Inst> {
    vec![
        inst("ucomisd", fmt("A", [r(xmm), r(xmm_m64)]).flags(W), rex([0x66, 0x0F, 0x2E]).r(), _64b | compat | sse2),
        inst("ucomiss", fmt("A", [r(xmm), r(xmm_m32)]).flags(W), rex([0x0F, 0x2E]).r(), _64b | compat | sse),
    ]
}
