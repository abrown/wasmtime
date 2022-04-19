use std::sync::{Arc, RwLock};

use anyhow::Result;
use wasmtime::*;

#[test]
fn test_instantiate_shared_memory() -> Result<()> {
    let wat = r#"
    (module (memory 1 1 shared))
    "#;

    let mut config = Config::new();
    config.wasm_threads(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wat)?;
    let mut store = Store::new(&engine, ());
    let _instance = Instance::new(&mut store, &module, &[])?;
    Ok(())
}

#[test]
fn test_import_shared_memory() -> Result<()> {
    let wat = r#"
    (module (import "env" "memory" (memory 1 5 shared)))
    "#;

    let mut config = Config::new();
    config.wasm_threads(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wat)?;
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::shared(1, 5))?;
    let _instance = Instance::new(&mut store, &module, &[memory.into()])?;
    Ok(())
}

#[test]
fn test_share_external_memory() -> Result<()> {
    let wat = r#"
    (module (import "env" "memory" (memory 1 5 shared)))
    "#;

    let mut config = Config::new();
    config.wasm_threads(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wat)?;
    let mut store = Store::new(&engine, ());
    let shared_memory = SharedMemory::new(1, 5)?;
    let memory = Memory::from_shared_memory(&mut store, shared_memory)?;
    let _instance = Instance::new(&mut store, &module, &[memory.into()])?;
    Ok(())
}

#[test]
fn test_probe_shared_memory() -> Result<()> {
    let wat = r#"
    (module
        (memory 1 1 shared)
        (func (export "size") (result i32) (memory.size))
    )
    "#;

    let mut config = Config::new();
    config.wasm_threads(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wat)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let size = instance.get_typed_func::<(), i32, _>(&mut store, "size")?;
    let current_size = size.call(&mut store, ())?;
    assert_eq!(current_size, 1);
    Ok(())
}

#[test]
fn test_grow_memory_in_multiple_threads() -> Result<()> {
    let wat = r#"
    (module
        (import "env" "memory" (memory 1 5 shared))
        (func (export "grow") (param $delta i32) (result i32) (memory.grow (local.get $delta)))
    )
    "#;

    let mut config = Config::new();
    config.wasm_threads(true);
    let engine = Arc::new(Engine::new(&config)?);
    let module = Arc::new(Module::new(&engine, wat)?);
    let shared_memory = SharedMemory::new(1, 5)?;
    let mut threads = vec![];
    let sizes = Arc::new(RwLock::new(vec![]));

    // Spawn several threads using a single shared memory and grow the memory
    // concurrently on all threads.
    for _ in 0..4 {
        let engine = engine.clone();
        let module = module.clone();
        let sizes = sizes.clone();
        let shared_memory = shared_memory.clone();
        let thread = std::thread::spawn(move || {
            let mut store = Store::new(&engine, ());
            let memory = Memory::from_shared_memory(&mut store, shared_memory).unwrap();
            let instance = Instance::new(&mut store, &module, &[memory.into()]).unwrap();
            let grow = instance
                .get_typed_func::<i32, i32, _>(&mut store, "grow")
                .unwrap();
            for _ in 0..4 {
                let old_size = grow.call(&mut store, 1).unwrap();
                sizes.write().unwrap().push(old_size as u32);
            }
        });
        threads.push(thread);
    }

    // Wait for all threads to finish.
    for t in threads {
        t.join().unwrap()
    }

    // Ensure the returned "old memory sizes" were pushed in increasing order,
    // indicating that the lock worked.
    println!("Returned memory sizes: {:?}", sizes);
    assert!(is_sorted(sizes.read().unwrap().as_slice()));

    Ok(())
}

fn is_sorted(data: &[u32]) -> bool {
    data.windows(2).all(|d| d[0] <= d[1])
}
