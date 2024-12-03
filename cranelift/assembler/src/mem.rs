//! Memory operands to instructions.

use crate::alloc::RegallocVisitor;
use crate::imm::{Simm32, Simm32PlusKnownOffset};
use crate::reg::{self, Gpr, Gpr2MinusRsp, Size};
use crate::rex::{encode_modrm, encode_sib, Imm, RexFlags};
use crate::sink::{CodeSink, KnownOffsetTable, Label, TrapCode};
use arbitrary::Arbitrary;

#[derive(Arbitrary, Clone, Debug)]
pub enum Amode {
    ImmReg {
        base: Gpr,
        simm32: Simm32PlusKnownOffset,
        trap: Option<TrapCode>,
    },
    ImmRegRegShift {
        base: Gpr,
        index: Gpr2MinusRsp,
        scale: Scale,
        simm32: Simm32,
        trap: Option<TrapCode>,
    },
    RipRelative {
        target: Label,
    },
}
impl Amode {
    pub fn trap_code(&self) -> Option<TrapCode> {
        match self {
            Amode::ImmReg { trap, .. } | Amode::ImmRegRegShift { trap, .. } => *trap,
            Amode::RipRelative { .. } => None,
        }
    }

    pub fn emit_rex_prefix(&self, rex: RexFlags, enc_g: u8, sink: &mut impl CodeSink) {
        match self {
            Amode::ImmReg { base, .. } => {
                let enc_e = base.enc();
                rex.emit_two_op(sink, enc_g, enc_e);
            }
            Amode::ImmRegRegShift { base: reg_base, index: reg_index, .. } => {
                let enc_base = reg_base.enc();
                let enc_index = reg_index.enc();
                rex.emit_three_op(sink, enc_g, enc_index, enc_base);
            }
            Amode::RipRelative { .. } => {
                // note REX.B = 0.
                rex.emit_two_op(sink, enc_g, 0);
            }
        }
    }

    pub fn read(&mut self, visitor: &mut impl RegallocVisitor) {
        match self {
            Amode::ImmReg { base, .. } => {
                // TODO: allow regalloc to replace a virtual register with a real one.
                // TODO: should this be a debug_assert instead, like below?
                if base.enc() != reg::ENC_RBP && base.enc() != reg::ENC_RSP {
                    visitor.read(base.as_mut());
                }
            }
            Amode::ImmRegRegShift { base, index, .. } => {
                // TODO: allow regalloc to replace a virtual register with a real one.
                debug_assert_ne!(base.enc(), reg::ENC_RBP);
                debug_assert_ne!(base.enc(), reg::ENC_RSP);
                visitor.read(base.as_mut());
                debug_assert_ne!(index.enc(), reg::ENC_RBP);
                debug_assert_ne!(index.enc(), reg::ENC_RSP);
                visitor.read(index.as_mut());
            }
            Amode::RipRelative { .. } => todo!(),
        }
    }
}

impl std::fmt::Display for Amode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Amode::ImmReg { simm32, base, .. } => {
                // Note: size is always 8; the address is 64 bits,
                // even if the addressed operand is smaller.
                write!(f, "{:x}({})", simm32, base.to_string(Size::Quadword))
            }
            Amode::ImmRegRegShift { simm32, base, index, scale, .. } => {
                if scale.shift() > 1 {
                    write!(
                        f,
                        "{:x}({}, {}, {})",
                        simm32,
                        base.to_string(Size::Quadword),
                        index.to_string(Size::Quadword),
                        scale.shift()
                    )
                } else {
                    write!(
                        f,
                        "{:x}({}, {})",
                        simm32,
                        base.to_string(Size::Quadword),
                        index.to_string(Size::Quadword)
                    )
                }
            }
            Amode::RipRelative { .. } => write!(f, "(%rip)"),
        }
    }
}

#[derive(Arbitrary, Clone, Debug)]
pub enum Scale {
    One,
    Two,
    Four,
    Eight,
}
impl Scale {
    fn enc(&self) -> u8 {
        match self {
            Scale::One => 0b00,
            Scale::Two => 0b01,
            Scale::Four => 0b10,
            Scale::Eight => 0b11,
        }
    }
    fn shift(&self) -> u8 {
        1 << self.enc()
    }
}

#[derive(Arbitrary, Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub enum GprMem {
    Gpr(Gpr),
    Mem(Amode),
}
impl GprMem {
    pub fn always_emit_if_8bit_needed(&self, rex: &mut RexFlags) {
        match self {
            GprMem::Gpr(gpr) => gpr.always_emit_if_8bit_needed(rex),
            GprMem::Mem(_) => {}
        }
    }

    pub fn to_string(&self, size: Size) -> String {
        match self {
            GprMem::Gpr(gpr2) => gpr2.to_string(size).to_owned(),
            GprMem::Mem(amode) => amode.to_string(),
        }
    }

    pub fn read(&mut self, visitor: &mut impl RegallocVisitor) {
        match self {
            GprMem::Gpr(gpr) => gpr.read(visitor),
            GprMem::Mem(amode) => amode.read(visitor),
        }
    }

    pub fn read_write(&mut self, visitor: &mut impl RegallocVisitor) {
        match self {
            GprMem::Gpr(gpr) => gpr.read_write(visitor),
            GprMem::Mem(amode) => amode.read(visitor),
        }
    }
}

pub fn emit_modrm_sib_disp(
    sink: &mut impl CodeSink,
    offsets: &impl KnownOffsetTable,
    enc_g: u8,
    mem_e: &Amode,
    bytes_at_end: u8,
    evex_scaling: Option<i8>,
) {
    match mem_e.clone() {
        Amode::ImmReg { simm32, base, .. } => {
            let enc_e = base.enc();
            let mut imm = Imm::new(simm32.value(offsets), evex_scaling);

            // Most base registers allow for a single ModRM byte plus an
            // optional immediate. If rsp is the base register, however, then a
            // SIB byte must be used.
            let enc_e_low3 = enc_e & 7;
            if enc_e_low3 == reg::ENC_RSP {
                // Displacement from RSP is encoded with a SIB byte where
                // the index and base are both encoded as RSP's encoding of
                // 0b100. This special encoding means that the index register
                // isn't used and the base is 0b100 with or without a
                // REX-encoded 4th bit (e.g. rsp or r12)
                sink.put1(encode_modrm(imm.m0d(), enc_g & 7, 0b100));
                sink.put1(0b00_100_100);
                imm.emit(sink);
            } else {
                // If the base register is rbp and there's no offset then force
                // a 1-byte zero offset since otherwise the encoding would be
                // invalid.
                if enc_e_low3 == reg::ENC_RBP {
                    imm.force_immediate();
                }
                sink.put1(encode_modrm(imm.m0d(), enc_g & 7, enc_e & 7));
                imm.emit(sink);
            }
        }

        Amode::ImmRegRegShift {
            simm32, base: reg_base, index: reg_index, scale, ..
        } => {
            let enc_base = reg_base.enc();
            let enc_index = reg_index.enc();

            // Encoding of ModRM/SIB bytes don't allow the index register to
            // ever be rsp. Note, though, that the encoding of r12, whose three
            // lower bits match the encoding of rsp, is explicitly allowed with
            // REX bytes so only rsp is disallowed.
            assert!(enc_index != reg::ENC_RSP);

            // If the offset is zero then there is no immediate. Note, though,
            // that if the base register's lower three bits are `101` then an
            // offset must be present. This is a special case in the encoding of
            // the SIB byte and requires an explicit displacement with rbp/r13.
            let mut imm = Imm::new(simm32.value(), evex_scaling);
            if enc_base & 7 == reg::ENC_RBP {
                imm.force_immediate();
            }

            // With the above determined encode the ModRM byte, then the SIB
            // byte, then any immediate as necessary.
            sink.put1(encode_modrm(imm.m0d(), enc_g & 7, 0b100));
            sink.put1(encode_sib(scale.enc(), enc_index & 7, enc_base & 7));
            imm.emit(sink);
        }

        Amode::RipRelative { target } => {
            // RIP-relative is mod=00, rm=101.
            sink.put1(encode_modrm(0b00, enc_g & 7, 0b101));

            let offset = sink.cur_offset();
            sink.use_label_at_offset(offset, target.clone());

            // N.B.: some instructions (XmmRmRImm format for example)
            // have bytes *after* the RIP-relative offset. The
            // addressed location is relative to the end of the
            // instruction, but the relocation is nominally relative
            // to the end of the u32 field. So, to compensate for
            // this, we emit a negative extra offset in the u32 field
            // initially, and the relocation will add to it.
            #[allow(clippy::cast_sign_loss)]
            sink.put4(-(i32::from(bytes_at_end)) as u32);
        }
    }
}
