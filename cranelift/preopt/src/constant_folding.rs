//! Fold operations on constants at compile time.
//!
//! This algorithm is lifted Cooper and Torczon's "Engineering a Compiler."
#![allow(clippy::float_arithmetic)]

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{self, dfg::ValueDef, types, DataFlowGraph, InstBuilder, Opcode, Type};
use cranelift_entity::PrimaryMap;
use cranelift_interpreter::instruction::DfgInstructionContext;
use cranelift_interpreter::state::ImmutableRegisterState;
use cranelift_interpreter::step::{step, ControlFlow};
use cranelift_interpreter::value::{Value, ValueConversionKind, ValueError, ValueResult};
use log::trace;
use std::collections::HashMap;
use std::iter::FromIterator;

/// Fold operations on constants.
///
/// It's important to note that this will not remove unused constants. It's
/// assumed that the DCE pass will take care of them.
pub fn fold_constants(func: &mut ir::Function) {
    // Initialize the worklist and initial value mapping.
    let mut worklist: Vec<ir::Value> = Vec::with_capacity(func.dfg.num_values());
    let mut values = Values::setup(&func.dfg, &mut worklist);

    // Find the uses of each SSA names.
    let uses = SsaNameUses::new(&mut FuncCursor::new(func));

    // Find the inputs for each block parameter.
    let inputs = PhiValues::new(&mut FuncCursor::new(func));

    // Work through the worklist.
    while let Some(v) = worklist.pop() {
        trace!("Evaluating {:?} with worklist: {:?}", v, worklist);
        for &inst in uses.get(v) {
            if func.dfg[inst].opcode().is_branch() {
                // We do not need to update values at the branch site as long as no branch
                // instructions modifies values; instead, we attempt to join (using
                // LatticeValue::meet) the various input values of each destination branch
                // parameter in hopes that we will see a change and continue propagating
                // constants into the destination branch. TODO this should happen regardless of
                // whether we can replace a branch or not:
                if let Some(block_name) =
                    find_block_parameter_for_branch_argument(&func.dfg, inst, v)
                {
                    trace!("Attempting to meet: {}", block_name);
                    if let Some(changed) = meet(&inputs, &values, block_name) {
                        trace!("Joined at meet point: {} = {:?}", block_name, changed);
                        values.set(block_name, changed);
                        worklist.push(block_name);
                    }
                }
            }

            match interpret(&func.dfg, inst, &values) {
                // If interpretation returns an assignment and these values do not match the
                // previously mapped values, we:
                //  - if possible, replace the instruction with a constant instruction
                //  - update our mapping of SSA names to values
                //  - add the changed SSA name to the worklist
                Some(ControlFlow::Assign(result_values)) => {
                    if result_values.len() == 1 {
                        if let LatticeValue::Constant(c) = result_values[0].clone() {
                            replace_inst(&mut func.dfg, inst, c);
                        }
                    }
                    let results = func.dfg.inst_results(inst);
                    for (&result_name, result_value) in results.iter().zip(result_values) {
                        if values.get(result_name) != &result_value {
                            values.set(result_name, result_value);
                            worklist.push(result_name)
                        }
                    }
                }
                // If interpretation can tell us which direction a branch led to, we replace the
                // branch with an unconditional jump and update our mapping of SSA names to values.
                Some(ControlFlow::ContinueAt(block, _)) => {
                    possibly_replace_branch_with_jump(func, inst, block);
                }
                Some(ControlFlow::Continue) if func.dfg[inst].opcode().is_branch() => {
                    func.dfg.replace(inst).nop();
                }
                _ => {}
            }
        }
    }
}

/// Coalesce the values at a block parameter using [LatticeValue::meet] and return `Some` if this
/// changes the value.
fn meet(inputs: &PhiValues, values: &Values, param_name: ir::Value) -> Option<LatticeValue> {
    let previous = values.get(param_name);
    let folded = inputs
        .inputs_of(param_name)
        .iter()
        .fold(LatticeValue::Top, |acc, &name| {
            acc.meet(values.get(name).clone())
        });
    if &folded != previous {
        Some(folded)
    } else {
        None
    }
}

/// Return `Some` arguments for the jump if the branch was replaced (these values should propagate)
fn possibly_replace_branch_with_jump(
    func: &mut ir::Function,
    inst: ir::Inst,
    block: ir::Block,
) -> Option<Vec<ir::Value>> {
    let opcode = func.dfg[inst].opcode();
    debug_assert!(opcode.is_branch());

    match opcode {
        Opcode::Jump | Opcode::Fallthrough => {
            trace!(
                "Skipping unconditional jump: {}",
                func.dfg.display_inst(inst, None),
            );
            None
        }
        Opcode::IndirectJumpTableBr | Opcode::BrTable => {
            trace!(
                "TODO teach SSCP how to replace tables, skipping: {}",
                func.dfg.display_inst(inst, None),
            );
            None
        }
        _ => {
            let params = branch_arguments(&func.dfg, inst).to_vec();
            trace!(
                "Replacing branch with jump: {} -> jump {}{:?}",
                func.dfg.display_inst(inst, None),
                block,
                params
            );

            func.dfg.replace(inst).jump(block, &params);
            // Remove the rest of the block to avoid verifier errors.
            while let Some(next_inst) = func.layout.next_inst(inst) {
                func.layout.remove_inst(next_inst);
            }
            Some(params)
        }
    }
}

/// Interpret an instruction using a mapping of SSA names to [LatticeValue]s.
fn interpret<'a>(
    dfg: &ir::DataFlowGraph,
    inst: ir::Inst,
    values: &'a Values,
) -> Option<ControlFlow<'a, LatticeValue>> {
    let mut state = ImmutableRegisterState::new(&values.to_map());
    let context = DfgInstructionContext::new(inst, dfg);
    match step(&mut state, context) {
        Ok(r) => {
            trace!("Step result: {:?}", r);
            Some(r)
        }
        Err(e) => {
            trace!("Step error: {}", e);
            None
        }
    }
}

/// Check if the value of [ir::Value] will be statically knowable; e.g. `load` operations are
/// considered unknowable.
fn is_value_unknowable(dfg: &ir::DataFlowGraph, value: ir::Value) -> bool {
    let original = dfg.resolve_aliases(value);
    if let ValueDef::Result(inst, _) = dfg.value_def(original) {
        if dfg[inst].opcode().can_load() {
            return true;
        }
    }
    false
}

/// Convert an SSA name (i.e. [ir::Value]) to its immediate [DataValue], if the SSA name was
/// generated by a `const` instruction.
fn resolve_value_to_imm(dfg: &ir::DataFlowGraph, value: ir::Value) -> Option<DataValue> {
    let original = dfg.resolve_aliases(value);

    let inst = match dfg.value_def(original) {
        ValueDef::Result(inst, _) => inst,
        ValueDef::Param(_, _) => return None,
    };

    match dfg[inst].opcode() {
        // These constants declare their immediate with a size that may be larger than the size
        // specified by the controlling type variable; this means we have to perform an exact
        // conversion to get the correct variant of `DataValue`.
        Opcode::Iconst | Opcode::Bconst => Some(
            dfg[inst]
                .imm_value()
                .expect("iconst/bconst must have an immediate value")
                .convert(ValueConversionKind::Exact(dfg.ctrl_typevar(inst)))
                .expect("iconst/bconst must have a valid controlling type variable"),
        ),
        // FP constants declare their immediate with the exact size so that `imm_value` can return
        // the correct `DataValue`.
        Opcode::F32const | Opcode::F64const => Some(dfg[inst].imm_value().unwrap()),
        // No other instructions generate constants. TODO add `vconst`; loading from constant pool.
        _ => None,
    }
}

/// Map each SSA name to its current [LatticeValue].
struct Values(PrimaryMap<ir::Value, LatticeValue>);
impl Values {
    fn setup(dfg: &DataFlowGraph, worklist: &mut Vec<ir::Value>) -> Self {
        let mut values = PrimaryMap::from_iter(vec![LatticeValue::Top; dfg.num_values()]);
        for name in dfg.values() {
            let lattice_value = if is_value_unknowable(dfg, name) {
                LatticeValue::Bottom
            } else if let Some(c) = resolve_value_to_imm(dfg, name) {
                LatticeValue::Constant(c)
            } else {
                LatticeValue::Top
            };
            if lattice_value != LatticeValue::Top {
                worklist.push(name);
            }
            values[name] = lattice_value;
        }
        Self(values)
    }

    fn get(&self, name: ir::Value) -> &LatticeValue {
        match self.0.get(name) {
            None => panic!("{} should have a value set during setup", name),
            Some(v) => v,
        }
    }

    pub fn set(&mut self, name: ir::Value, value: LatticeValue) {
        self.0[name] = value;
    }

    fn to_map(&self) -> &PrimaryMap<ir::Value, LatticeValue> {
        &self.0
    }
}

/// Gather all of the possible [ir::Value] inputs to a block parameter.
struct PhiValues(HashMap<ir::Value, Vec<ir::Value>>);
impl PhiValues {
    fn new(pos: &mut FuncCursor) -> Self {
        let mut mapping = HashMap::with_capacity(pos.func.dfg.num_values());

        // Pre-populate all block parameters with empty vectors.
        while let Some(block) = pos.next_block() {
            pos.func.dfg.block_params(block).iter().for_each(|&p| {
                mapping.insert(p, Vec::new());
            });
        }

        // For each branching instruction, map any input arguments to their respective block
        // parameters.
        while let Some(_) = pos.next_block() {
            while let Some(inst) = pos.next_inst() {
                if pos.func.dfg[inst].opcode().is_branch() {
                    trace!(
                        "Gathering branch arguments: {}",
                        pos.func.dfg.display_inst(inst, None)
                    );
                    let to_block = pos.func.dfg[inst]
                        .branch_destination()
                        .expect("branch must have a destination");
                    for (&from_name, to_name) in branch_arguments(&pos.func.dfg, inst)
                        .iter()
                        .zip(pos.func.dfg.block_params(to_block))
                    {
                        trace!(
                            "Mapped branch argument (source -> destination): {} -> {}",
                            from_name,
                            to_name
                        );
                        mapping
                            .get_mut(to_name)
                            .expect("to be filled in setup")
                            .push(from_name)
                    }
                }
            }
        }
        Self(mapping)
    }

    fn inputs_of(&self, block_parameter: ir::Value) -> &[ir::Value] {
        match self.0.get(&block_parameter) {
            Some(inputs) => &inputs,
            None => panic!(
                "cannot request the inputs of a non-block parameter: {}",
                block_parameter
            ),
        }
    }
}

/// Capture all uses of an SSA [ir::Value] in a function.
struct SsaNameUses(HashMap<ir::Value, Vec<ir::Inst>>);
impl SsaNameUses {
    fn new(pos: &mut FuncCursor) -> Self {
        let mut uses: HashMap<ir::Value, Vec<ir::Inst>> =
            HashMap::with_capacity(pos.func.dfg.num_values());
        while let Some(_block) = pos.next_block() {
            while let Some(inst) = pos.next_inst() {
                for value in pos.func.dfg[inst].arguments(&pos.func.dfg.value_lists) {
                    let instructions = uses.entry(*value).or_insert_with(Vec::new);
                    instructions.push(inst);
                }
            }
        }
        Self(uses)
    }

    fn get(&self, value: ir::Value) -> &[ir::Inst] {
        match self.0.get(&value) {
            Some(insns) => &insns,
            None => <&[ir::Inst]>::default(),
        }
    }
}

fn find_block_parameter_for_branch_argument(
    dfg: &ir::DataFlowGraph,
    inst: ir::Inst,
    arg: ir::Value,
) -> Option<ir::Value> {
    assert!(dfg[inst].opcode().is_branch());
    let block = dfg[inst]
        .branch_destination()
        .expect("a branch to have a block destination");
    let block_params = dfg.block_params(block);
    for (&branch_name, &block_name) in branch_arguments(dfg, inst).iter().zip(block_params) {
        if branch_name == arg {
            return Some(block_name);
        }
    }
    None
}

fn branch_arguments(dfg: &ir::DataFlowGraph, inst: ir::Inst) -> &[ir::Value] {
    match dfg[inst].opcode() {
        Opcode::Jump | Opcode::Fallthrough => dfg.inst_args(inst),
        i if i.is_branch() => &dfg.inst_args(inst)[1..],
        _ => panic!("not a branch instruction: {}", dfg.display_inst(inst, None)),
    }
}

/// Replace the given [ir::Inst] with a constant instruction holding `const_imm`.
fn replace_inst(dfg: &mut ir::DataFlowGraph, inst: ir::Inst, const_imm: DataValue) {
    use self::DataValue::*;
    trace!(
        "Replacing instruction with constant: {} -> {}",
        dfg.display_inst(inst, None),
        const_imm
    );
    match const_imm {
        i if i.ty().is_int() => {
            let typevar = dfg.ctrl_typevar(inst);
            dfg.replace(inst).iconst(typevar, i.into_int().unwrap());
        }
        F32(imm) => {
            dfg.replace(inst).f32const(imm);
        }
        F64(imm) => {
            dfg.replace(inst).f64const(imm);
        }
        B(imm) => {
            let typevar = dfg.ctrl_typevar(inst);
            dfg.replace(inst).bconst(typevar, imm);
        }
        _ => unimplemented!(),
    }
}

/// Use a semilattice to describe the set of all possible values during this optimization.
#[derive(Clone, PartialEq, Debug)]
enum LatticeValue {
    /// Optimistically specifies that the algorithm does not know the value of the SSA name--yet.
    Top,
    /// Specifies that the value of the SSA name is known--a constant [DataValue].
    Constant(DataValue),
    /// The algorithm can never know the value of the SSA name--e.g. a `load` or a changing input
    /// to a block.
    Bottom,
}

impl LatticeValue {
    pub fn meet(self, other: Self) -> Self {
        match (self, other) {
            (LatticeValue::Top, x) | (x, LatticeValue::Top) => x,
            (LatticeValue::Bottom, _) | (_, LatticeValue::Bottom) => LatticeValue::Bottom,
            (LatticeValue::Constant(a), LatticeValue::Constant(b)) => {
                if a == b {
                    LatticeValue::Constant(a)
                } else {
                    LatticeValue::Bottom
                }
            }
        }
    }
}

impl Value for LatticeValue {
    fn ty(&self) -> Type {
        match self {
            LatticeValue::Top | LatticeValue::Bottom => types::INVALID,
            LatticeValue::Constant(v) => v.ty(),
        }
    }

    fn int(n: i64, ty: Type) -> ValueResult<Self> {
        DataValue::int(n, ty).map(LatticeValue::Constant)
    }

    fn into_int(self) -> ValueResult<i64> {
        convert(DataValue::into_int, self)
    }

    fn float(f: u64, ty: Type) -> ValueResult<Self> {
        DataValue::float(f, ty).map(LatticeValue::Constant)
    }

    fn into_float(self) -> ValueResult<f64> {
        convert(DataValue::into_float, self)
    }

    fn is_nan(&self) -> ValueResult<bool> {
        convert_ref(DataValue::is_nan, self)
    }

    fn bool(b: bool, ty: Type) -> ValueResult<Self> {
        DataValue::bool(b, ty).map(LatticeValue::Constant)
    }

    fn into_bool(self) -> ValueResult<bool> {
        convert(DataValue::into_bool, self)
    }

    fn vector(v: [u8; 16], ty: Type) -> ValueResult<Self> {
        DataValue::vector(v, ty).map(LatticeValue::Constant)
    }

    fn convert(self, kind: ValueConversionKind) -> ValueResult<Self> {
        use LatticeValue::*;
        match self {
            Constant(a) => a.convert(kind).map(LatticeValue::Constant),
            _ => unimplemented!(),
        }
    }

    fn eq(&self, other: &Self) -> ValueResult<bool> {
        match (self, other) {
            (LatticeValue::Constant(a), LatticeValue::Constant(b)) => Value::eq(a, b),
            _ => unimplemented!(),
        }
    }

    fn gt(&self, other: &Self) -> ValueResult<bool> {
        match (self, other) {
            (LatticeValue::Constant(a), LatticeValue::Constant(b)) => Value::gt(a, b),
            _ => unimplemented!(),
        }
    }

    fn uno(&self, other: &Self) -> ValueResult<bool> {
        match (self, other) {
            (LatticeValue::Constant(a), LatticeValue::Constant(b)) => Value::uno(a, b),
            _ => unimplemented!(),
        }
    }

    fn add(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::add, self, other)
    }

    fn sub(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::sub, self, other)
    }

    fn mul(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::mul, self, other)
    }

    fn div(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::div, self, other)
    }

    fn rem(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::rem, self, other)
    }

    fn shl(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::shl, self, other)
    }

    fn ushr(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::ushr, self, other)
    }

    fn ishr(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::ishr, self, other)
    }

    fn rotl(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::rotl, self, other)
    }

    fn rotr(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::rotr, self, other)
    }

    fn and(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::and, self, other)
    }

    fn or(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::or, self, other)
    }

    fn xor(self, other: Self) -> ValueResult<Self> {
        binary(DataValue::xor, self, other)
    }

    fn not(self) -> ValueResult<Self> {
        unary(DataValue::not, self)
    }
}

#[inline]
fn binary(
    op: fn(DataValue, DataValue) -> ValueResult<DataValue>,
    a: LatticeValue,
    b: LatticeValue,
) -> ValueResult<LatticeValue> {
    use LatticeValue::*;
    match (a, b) {
        (Bottom, _) | (_, Bottom) => Ok(Bottom),
        (Constant(a), Constant(b)) => Ok(Constant(op(a, b)?)),
        (Top, _) | (_, Top) => Ok(Top),
    }
}

#[inline]
fn unary(
    op: fn(DataValue) -> ValueResult<DataValue>,
    a: LatticeValue,
) -> ValueResult<LatticeValue> {
    use LatticeValue::*;
    match a {
        Bottom => Ok(Bottom),
        Constant(a) => Ok(Constant(op(a)?)),
        Top => Ok(Top),
    }
}

#[inline]
fn convert<V>(op: fn(DataValue) -> ValueResult<V>, a: LatticeValue) -> ValueResult<V> {
    use LatticeValue::*;
    match a {
        Constant(a) => op(a),
        _ => Err(ValueError::InvalidValue(a.ty())),
    }
}

#[inline]
fn convert_ref<V>(op: fn(&DataValue) -> ValueResult<V>, a: &LatticeValue) -> ValueResult<V> {
    use LatticeValue::*;
    match a {
        Constant(a) => op(a),
        _ => unimplemented!(),
    }
}

impl From<DataValue> for LatticeValue {
    fn from(dv: DataValue) -> Self {
        LatticeValue::Constant(dv)
    }
}
