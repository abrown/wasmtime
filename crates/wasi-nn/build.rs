use std::path::PathBuf;

fn main() {
    // This is necessary for Wiggle/Witx macros.
    let wasi_root = PathBuf::from(".").canonicalize().unwrap();
    println!("cargo:rustc-env=WASI_ROOT={}", wasi_root.display());

    // TODO Remove this and use pkg-config instead.
    let openvino_libs = PathBuf::from("../../../openvino/bin/intel64/Debug/lib")
        .canonicalize()
        .unwrap();
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}",
        openvino_libs.display()
    );
}
