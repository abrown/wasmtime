//! This build script:
//!  - has configuration necessary for the wiggle and witx macros
//!  - compiles a Wasm file from `tests/example` and generates a `#[test]` to execute it.

use std::env;
use std::fs::{read_to_string, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use toml;

fn main() {
    // This is necessary for Wiggle/Witx macros.
    let wasi_root = PathBuf::from(".").canonicalize().unwrap();
    println!("cargo:rustc-env=WASI_ROOT={}", wasi_root.display());

    let project_dir = PathBuf::from(".").canonicalize().unwrap();
    let example_project_dir = project_dir.join("tests/example");
    let out_dir =
        PathBuf::from(env::var("OUT_DIR").expect("The OUT_DIR environment variable must be set"));

    // Also rebuild if the example project changes
    let cb = |p: PathBuf| println!("cargo:rerun-if-changed={}", p.display());
    visit_dirs(&example_project_dir, &cb).expect("to visit source files");

    // Compile the Wasm file from the sources in `tests/example`.
    let wasm_file = compile_to_wasm(&example_project_dir, &out_dir).expect("compile to Wasm");

    // Generate tests running the compiled Wasm file.
    let preopen_dir = project_dir.join("tests");
    let mut test_file =
        File::create(out_dir.join("tests.rs")).expect("error generating test source file");
    generate_test(
        "example",
        preopen_dir.to_str().unwrap(),
        wasm_file.to_str().unwrap(),
        &mut test_file,
    )
    .expect("generating tests");
}

/// Generate a test that executes a Wasm file; relies on the definition of `execute_wasi_nn_program`
/// in `tests/main.rs`.
fn generate_test(
    module_name: &str,
    preopen_dir: &str,
    wasm_file: &str,
    out: &mut File,
) -> io::Result<()> {
    writeln!(out, "mod {} {{", module_name)?;
    writeln!(out, "    use super::*;")?;
    writeln!(out, "    #[test]")?;
    writeln!(out, "    fn execute() -> anyhow::Result<()> {{")?;
    writeln!(
        out,
        "        execute_wasi_nn_program(\"{}\", \"{}\")",
        preopen_dir, wasm_file
    )?;
    writeln!(out, "    }}")?;
    writeln!(out, "}}",)
}

/// Compile the Cargo `project_dir` to a Wasm file in `out_dir`, returning the location of the
/// compiled file.
fn compile_to_wasm(project_dir: &Path, out_dir: &Path) -> io::Result<PathBuf> {
    // `cargo build` the project to a Wasm file.
    let mut cmd = Command::new("cargo");
    cmd.args(&[
        "build",
        "--release",
        "--target=wasm32-wasi",
        "--target-dir",
        out_dir.to_str().unwrap(),
    ])
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit())
    .current_dir(project_dir);
    let output = cmd.output()?;

    // Check that the build succeeded.
    let status = output.status;
    if !status.success() {
        panic!(
            "Building the example failed with exit code: {}",
            status.code().unwrap()
        );
    }

    // Determine the path to the compiled Wasm file using both pre-defined information and by
    // inspecting the Cargo.toml file.
    let cargo_toml = project_dir.join("Cargo.toml");
    let compiled_wasm_file = out_dir
        .join("wasm32-wasi/release")
        .join(get_project_name(&cargo_toml))
        .with_extension("wasm");

    Ok(compiled_wasm_file)
}

/// Retrieve the project name from a Cargo file.
fn get_project_name(cargo_toml: &Path) -> String {
    let table: toml::Value = read_to_string(cargo_toml)
        .expect("readable contents")
        .parse()
        .expect("a parsable toml file");
    let name_value = table
        .get("package")
        .expect("a [package] declaration")
        .get("name")
        .expect("a project name");
    name_value
        .as_str()
        .expect("the package named to be a string")
        .to_string()
}

/// Helper for recursively visiting the files in this directory; see https://doc.rust-lang.org/std/fs/fn.read_dir.html.
fn visit_dirs(dir: &Path, cb: &dyn Fn(PathBuf)) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(entry.path());
            }
        }
    }
    Ok(())
}
