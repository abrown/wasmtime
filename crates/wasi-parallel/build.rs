//! This build script:
//!  - has the configuration necessary for the wiggle and witx macros
//!  - generates Wasm from the files in `tests/rust` to `tests/wasm`
use std::{
    env,
    fs::DirBuilder,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    // This is necessary for wiggle/witx macros.
    let cwd = std::env::current_dir().unwrap();
    let wasi_root = cwd.join("spec");
    println!("cargo:rustc-env=WASI_ROOT={}", wasi_root.display());

    // Also automatically rebuild if the Witx files change.
    for entry in walkdir::WalkDir::new(wasi_root) {
        println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
    }

    println!("cargo:rerun-if-changed=tests/cpp/wasi_parallel.h");

    //build_wasm("tests");
    //build_wasm("benches");
}

fn build_wasm<P: AsRef<Path>>(root: P) {
    let root_dir = Path::new(root.as_ref().as_os_str());
    let wasm_dir = root_dir.join("wasm");

    DirBuilder::new().recursive(true).create(&wasm_dir).unwrap();

    #[cfg(feature = "opencl")]
    if root_dir.join("cl").exists() {
        DirBuilder::new()
            .recursive(true)
            .create(root_dir.join("spv"))
            .unwrap();
        for entry in walkdir::WalkDir::new(root_dir.join("cl")) {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
            if entry.path().is_file() {
                compile_cl(entry.path(), root_dir.join("spv"))
            }
        }
    }

    // Automatically rebuild any Rust tests.
    if root_dir.join("rust").exists() {
        for entry in walkdir::WalkDir::new(root_dir.join("rust")) {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
            if entry.path().is_file() && entry.file_name() != "wasi_parallel.rs" {
                compile_rust(entry.path(), &wasm_dir)
            }
        }
    }

    if root_dir.join("cpp").exists() {
        #[cfg(feature = "opencl")]
        let temp_dir = Path::new(&env::var("OUT_DIR").unwrap()).join(&root_dir);
        #[cfg(feature = "opencl")]
        DirBuilder::new().recursive(true).create(&temp_dir).unwrap();
        for entry in walkdir::WalkDir::new(root_dir.join("cpp")) {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
            if entry.path().is_file() && entry.file_name() != "wasi_parallel.h" {
                #[cfg(feature = "opencl")]
                let spirv_file = spirv_file(entry.path(), root_dir.join("spv"));
                #[cfg(feature = "opencl")]
                if spirv_file.exists() {
                    let temp_file = compile_cpp(entry.path(), &temp_dir);
                    attach_spirv(temp_file, spirv_file, &wasm_dir)
                } else {
                    compile_cpp(entry.path(), &wasm_dir);
                }
                #[cfg(not(feature = "opencl"))]
                compile_cpp(entry.path(), &wasm_dir);
            }
        }
    }
}

/// Use rustc to compile a Rust file to a Wasm file that uses the wasi-parallel
/// API.
fn compile_rust<P1: AsRef<Path>, P2: AsRef<Path>>(source_file: P1, destination_dir: P2) {
    let stem = source_file.as_ref().file_stem().unwrap();
    let mut destination_file: PathBuf = [destination_dir.as_ref().as_os_str(), stem]
        .iter()
        .collect();
    destination_file.set_extension("wasm");

    let mut command = Command::new("rustc");
    command
        .arg("--target")
        .arg("wasm32-wasi")
        .arg(source_file.as_ref().to_str().unwrap())
        .arg("-o")
        .arg(destination_file.to_str().unwrap());

    let status = command
        .status()
        .expect("Failed to execute 'rustc' command to generate Wasm file.");

    assert!(
        status.success(),
        "Failed to compile test program: {:?}",
        command
    )
}

fn compile_cpp<P1: AsRef<Path>, P2: AsRef<Path>>(source_file: P1, destination_dir: P2) -> PathBuf {
    let stem = source_file.as_ref().file_stem().unwrap();
    let mut destination_file: PathBuf = [destination_dir.as_ref().as_os_str(), stem]
        .iter()
        .collect();
    destination_file.set_extension("wasm");

    let mut command = Command::new("clang");
    command
        .arg("-O3")
        .arg("--target=wasm32-wasi")
        .arg("-Xlinker")
        .arg("--export-table")
        .arg("-mthread-model")
        .arg("single")
        .arg("--sysroot")
        .arg(
            Path::new(&env::var("WASI_SDK_PATH").unwrap())
                .join("share")
                .join("wasi-sysroot"),
        )
        .arg(source_file.as_ref().to_str().unwrap())
        .arg("-o")
        .arg(destination_file.to_str().unwrap());

    let status = command
        .status()
        .expect("Failed to execute 'clang' command to generate Wasm file.");

    assert!(
        status.success(),
        "Failed to compile test program: {:?}",
        command
    );
    destination_file
}

#[cfg(feature = "opencl")]
fn compile_cl<P1: AsRef<Path>, P2: AsRef<Path>>(source_file: P1, destination_dir: P2) {
    let stem = source_file.as_ref().file_stem().unwrap();
    let destination_file_base: PathBuf = [destination_dir.as_ref().as_os_str(), stem]
        .iter()
        .collect();

    let mut bc_file = destination_file_base.clone();
    bc_file.set_extension("bc");

    let mut command = Command::new("clang");
    command
        .arg("--std=cl2.0")
        .arg("--target=spir-unknown-unknown")
        .arg("-c")
        .arg("-emit-llvm")
        .arg("-Xclang")
        .arg("-finclude-default-header")
        .arg(source_file.as_ref().to_str().unwrap())
        .arg("-o")
        .arg(bc_file.to_str().unwrap());

    let status = command
        .status()
        .expect("Failed to execute 'clang' command to generate bc file for cl file.");

    assert!(
        status.success(),
        "Failed to compile cl program: {:?}",
        command
    );

    let mut spirv_file = destination_file_base.clone();
    spirv_file.set_extension("spv");

    command = Command::new("llvm-spirv");
    command
        .arg(bc_file.to_str().unwrap())
        .arg("-o")
        .arg(spirv_file.to_str().unwrap());

    let status = command
        .status()
        .expect("Failed to execute 'llvm-spirv' command to generate SPIR-V file.");

    assert!(
        status.success(),
        "Failed to compile generate SPIR-V file: {:?}",
        command
    );
}

#[cfg(feature = "opencl")]
fn spirv_file<P1: AsRef<Path>, P2: AsRef<Path>>(source_file: P1, spirv_dir: P2) -> PathBuf {
    let stem = source_file.as_ref().file_stem().unwrap();
    let mut spirv_file: PathBuf = [spirv_dir.as_ref().as_os_str(), stem].iter().collect();
    spirv_file.set_extension("spv");
    spirv_file
}

#[cfg(feature = "opencl")]
fn attach_spirv<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    wasm_file: P1,
    spirv_file: P2,
    destination_dir: P3,
) {
    // Code copied from "attach-spirv" to remove build depedency loop

    use std::{
        fs::{self, File},
        io::{Read, Write},
    };

    use wasm_encoder::{CustomSection, Section};

    let mut wasm_bytes = wat::parse_bytes(&fs::read(&wasm_file).unwrap())
        .unwrap()
        .to_vec();

    // TODO: Id 1 only work for clang, we need to find a better way to get the id.
    let mut new_section_bytes = 1_i32.to_le_bytes().to_vec();
    File::open(&spirv_file)
        .unwrap()
        .read_to_end(&mut new_section_bytes)
        .unwrap();

    let custom_section = CustomSection {
        name: "wasi-parallel",
        data: &new_section_bytes,
    };

    // Append the new custom section at the end. Note that the encoded Wasm module accepts a custom
    // section pretty much anywhere, unlike other sections:
    // https://webassembly.github.io/spec/core/binary/modules.html#binary-module.
    wasm_bytes.push(custom_section.id());
    custom_section.encode(&mut wasm_bytes);

    let stem = wasm_file.as_ref().file_stem().unwrap();
    let mut destination_file: PathBuf = [destination_dir.as_ref().as_os_str(), stem]
        .iter()
        .collect();
    destination_file.set_extension("wasm");

    File::create(&destination_file)
        .unwrap()
        .write_all(&wasm_bytes)
        .unwrap();
}
