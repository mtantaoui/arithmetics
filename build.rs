fn main() {
    // Compile C code
    cc::Build::new()
        .file("src/add.c")
        // .flag("-mavx512f")
        .flag("-march=native")
        .flag("-fPIC")
        .flag("-shared")
        .compile("add");

    println!("cargo:rerun-if-changed=src/add.c");
    println!("cargo:rerun-if-changed=src/add.h");
}
