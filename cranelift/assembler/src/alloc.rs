//! Register allocation.

pub trait RegallocVisitor {
    fn read(&mut self, reg: &mut u32);
    fn read_write(&mut self, reg: &mut u32);
    fn fixed_read(&mut self, reg: &mut u32);
    fn fixed_read_write(&mut self, reg: &mut u32);
}
