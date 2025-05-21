use crate::dsl::{Feature::*, Inst, Location::*};
use crate::dsl::{fmt, inst, r, rex, rw};
#[rustfmt::skip] // Keeps instructions on a single line.
pub fn list() -> Vec<Inst> {
    vec![
        inst("cmppd", fmt("A", [rw(xmm), r(xmm_m128), r(imm8)]), rex([0x66, 0x0F, 0xC2]).r().ib(), _64b | compat | sse2),
        inst("cmpps", fmt("A", [rw(xmm), r(xmm_m128), r(imm8)]), rex([0x0F, 0xC2]).r().ib(), _64b | compat | sse),
    ]
}
