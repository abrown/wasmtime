use crate::dsl::{align, fmt, inst, r, rex, rw, w};
use crate::dsl::{Feature::*, Inst, Location::*};

#[rustfmt::skip] // Keeps instructions on a single line.
pub fn list() -> Vec<Inst> {
    // When a destination register is marked as read-write (`rw`) in these
    // instructions, it means that the moved value is merged into the
    // destination register--some bits remain unchanged.
    vec![
        // Move GPR integers to and from XMM locations. From the reference
        // manual: "when the destination operand is an XMM register, the source
        // operand is written to the low doubleword of the register, and the
        // register is zero-extended to 128 bits."
        inst("movd", fmt("A", [w(xmm), r(rm32)]), rex([0x66, 0x0F, 0x6E]).r(), _64b | compat | sse2),
        inst("movq", fmt("A", [w(xmm), r(rm64)]), rex([0x66, 0x0F, 0x6E]).w().r(), _64b | sse2),
        inst("movd", fmt("B" [w(rm32), r(xmm)]), rex([0x66, 0x0F, 0x7E]).r(), _64b | compat | sse2),
        inst("movq", fmt("B" [w(rm64), r(xmm)]), rex([0x66, 0x0F, 0x7E]).w().r(), _64b | sse2),
        // Move floating-point values to and from XMM locations. Note that some
        // memory-loading versions of `movs*` clear the upper bits of the XMM
        // destination, hence the added `X` suffix to the format name.
        inst("movss", fmt("A", [rw(xmm), r(xmm)]), rex([0xF3, 0x0F, 0x10]).r(), _64b | compat | sse),
        inst("movss", fmt("AX", [w(xmm), r(m32)]), rex([0xF3, 0x0F, 0x10]).r(), _64b | compat | sse),
        inst("movss", fmt("C", [rw(xmm_m32), r(xmm)]), rex([0xF3, 0x0F, 0x11]).r(), _64b | compat | sse),
        inst("movsd", fmt("A", [rw(xmm), r(xmm)]), rex([0xF2, 0x0F, 0x10]).r(), _64b | compat | sse2),
        inst("movsd", fmt("AX", [rw(xmm), r(m64)]), rex([0xF2, 0x0F, 0x10]).r(), _64b | compat | sse2),
        inst("movsd", fmt("C", [rw(xmm_m64), r(xmm)]), rex([0xF2, 0x0F, 0x11]).r(), _64b | compat | sse2),
        // Move aligned 128-bit values to and from XMM locations.
        inst("movaps", fmt("A", [w(xmm), r(align(xmm_m128))]), rex([0x0F, 0x28]).r(), _64b | compat | sse),
        inst("movaps", fmt("B", [w(align(xmm_m128)), r(xmm)]), rex([0x0F, 0x29]).r(), _64b | compat | sse),
        inst("movapd", fmt("A", [w(xmm), r(align(xmm_m128))]), rex([0x66, 0x0F, 0x28]).r(), _64b | compat | sse2),
        inst("movapd", fmt("B", [w(align(xmm_m128)), r(xmm)]), rex([0x66, 0x0F, 0x29]).r(), _64b | compat | sse2),
        inst("movdqa", fmt("A", [w(xmm), r(align(xmm_m128))]), rex([0x66, 0x0F, 0x6f]).r(), _64b | compat | sse2),
        inst("movdqa", fmt("B", [w(align(xmm_m128)), r(xmm)]), rex([0x66, 0x0F, 0x7f]).r(), _64b | compat | sse2),
        // Move unaligned 128-bit values to and from XMM locations.
        inst("movups", fmt("A", [w(xmm), r(xmm_m128)]), rex([0x0F, 0x10]).r(), _64b | compat | sse),
        inst("movups", fmt("B", [w(xmm_m128), r(xmm)]), rex([0x0F, 0x11]).r(), _64b | compat | sse),
        inst("movupd", fmt("A", [w(xmm), r(xmm_m128)]), rex([0x66, 0x0F, 0x10]).r(), _64b | compat | sse2),
        inst("movupd", fmt("B", [w(xmm_m128), r(xmm)]), rex([0x66, 0x0F, 0x11]).r(), _64b | compat | sse2),
        inst("movdqu", fmt("A", [w(xmm), r(xmm_m128)]), rex([0xF3, 0x0F, 0x6f]).r(), _64b | compat | sse2),
        inst("movdqu", fmt("B", [w(xmm_m128), r(xmm)]), rex([0xF3, 0x0F, 0x7f]).r(), _64b | compat | sse2),
        // Move two lower 32-bit floats to the high two lanes.
        inst("movlhps", fmt("RM", [rw(xmm), r(xmm)]), rex([0x0F, 0x29]).r(), _64b | compat | sse),
        // Extract sign masks from the floating-point lanes.
        inst("movmskps", fmt("RM", [w(r64), r(xmm)]), rex([0x0F, 0x50]).r(), _64b | compat | sse),
        inst("movmskpd", fmt("RM", [w(r64), r(xmm)]), rex([0x66, 0x0F, 0x50]).r(), _64b | compat | sse2),
        inst("pmovmskb", fmt("RM", [w(r64), r(xmm)]), rex([0x66, 0x0F, 0xD7]).r(), _64b | compat | sse2),
        // Move and extend packed integers to and from XMM locations with sign extension.
        inst("pmovsxbw", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x20]).r(), _64b | compat | sse41),
        inst("pmovsxbd", fmt("A", [w(xmm), r(xmm_m32)]), rex([0x66, 0x0F, 0x38, 0x21]).r(), _64b | compat | sse41),
        inst("pmovsxbq", fmt("A", [w(xmm), r(xmm_m16)]), rex([0x66, 0x0F, 0x38, 0x22]).r(), _64b | compat | sse41),
        inst("pmovsxwd", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x23]).r(), _64b | compat | sse41),
        inst("pmovsxwq", fmt("A", [w(xmm), r(xmm_m32)]), rex([0x66, 0x0F, 0x38, 0x24]).r(), _64b | compat | sse41),
        inst("pmovsxdq", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x25]).r(), _64b | compat | sse41),
        // Move and extend packed integers to and from XMM locations with zero extension.
        inst("pmovzxbw", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x30]).r(), _64b | compat | sse41),
        inst("pmovzxbd", fmt("A", [w(xmm), r(xmm_m32)]), rex([0x66, 0x0F, 0x38, 0x31]).r(), _64b | compat | sse41),
        inst("pmovzxbq", fmt("A", [w(xmm), r(xmm_m16)]), rex([0x66, 0x0F, 0x38, 0x32]).r(), _64b | compat | sse41),
        inst("pmovzxwd", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x33]).r(), _64b | compat | sse41),
        inst("pmovzxwq", fmt("A", [w(xmm), r(xmm_m32)]), rex([0x66, 0x0F, 0x38, 0x34]).r(), _64b | compat | sse41),
        inst("pmovzxdq", fmt("A", [w(xmm), r(xmm_m64)]), rex([0x66, 0x0F, 0x38, 0x35]).r(), _64b | compat | sse41),

    ]
}
