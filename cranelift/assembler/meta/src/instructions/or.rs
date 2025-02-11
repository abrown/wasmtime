use crate::dsl::{
    fmt, inst, r, rex, rw, sxl, sxq, Features::*, Inst, LegacyPrefixes::*, Location::*
};

pub fn list() -> Vec<Inst> {
    vec![
        inst("orb", fmt("I", [rw(al), r(imm8)]), rex(0x0C).ib(), None),
        inst("orw", fmt("I", [rw(ax), r(imm16)]), rex(0x0D).prefix(_66).iw(), None),
        inst("orl", fmt("I", [rw(eax), r(imm32)]), rex(0x0D).id(), None),
        inst("orq", fmt("I_SXLQ", [rw(rax), sxq(imm32)]), rex(0x0D).w().id(), None),
        inst("orb", fmt("MI", [rw(rm8), r(imm8)]), rex(0x80).digit(1).ib(), None),
        inst("orw", fmt("MI", [rw(rm16), r(imm16)]), rex(0x81).prefix(_66).digit(1).iw(), None),
        inst("orl", fmt("MI", [rw(rm32), r(imm32)]), rex(0x81).digit(1).id(), None),
        inst("orq", fmt("MI_SXLQ", [rw(rm64), sxq(imm32)]), rex(0x81).w().digit(1).id(), None),
        //inst("orw", fmt("MI_SXWL", [rw(rm16), sxl(imm8)]), rex(0x83).digit(1).ib(), None),
        inst("orl", fmt("MI_SXBL", [rw(rm32), sxl(imm8)]), rex(0x83).digit(1).ib(), None),
        inst("orq", fmt("MI_SXBQ", [rw(rm64), sxq(imm8)]), rex(0x83).w().digit(1).ib(), None),
        inst("orb", fmt("MR", [rw(rm8), r(r8)]), rex(0x08).r(), None),
        inst("orw", fmt("MR", [rw(rm16), r(r16)]), rex(0x09).prefix(_66).r(), None),
        inst("orl", fmt("MR", [rw(rm32), r(r32)]), rex(0x09).r(), None),
        inst("orq", fmt("MR", [rw(rm64), r(r64)]), rex(0x09).w().r(), None),
        inst("orb", fmt("RM", [rw(r8), r(rm8)]), rex(0x0A).r(), None),
        inst("orw", fmt("RM", [rw(r16), r(rm16)]), rex(0x0B).prefix(_66).r(), None),
        inst("orl", fmt("RM", [rw(r32), r(rm32)]), rex(0x0B).r(), None),
        inst("orq", fmt("RM", [rw(r64), r(rm64)]), rex(0x0B).w().r(), None),
        inst("orpd", fmt("A", [rw(xmm), r(rm128)]), rex(0x560F).prefix(_66).r(), None),
    ]
}
