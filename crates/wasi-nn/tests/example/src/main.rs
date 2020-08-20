use std::fs;

#[link(wasm_import_module = "wasi_nn")]
extern "C" {
    pub fn load(s: u32) -> u32;
}

pub fn main() {
    let xml = fs::read_to_string("fixture/frozen_inference_graph.xml").unwrap();
    println!("First 50 characters of graph: {}", &xml[..50]);

    let weights = fs::read("fixture/frozen_inference_graph.bin").unwrap();
    println!("Size of weights: {}", weights.len());

    println!("foo: {}", unsafe { load(3) });
    // println!("bar2: {}", unsafe { bar(4) })
}
