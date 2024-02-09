use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use anyhow::bail;

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let mut cmd = Command::new("verilator");
    cmd.arg("--cc")
        .arg("--Mdir")
        .arg(&out_dir)
        .arg("--relative-includes")
        .arg("--trace-fst")
        .arg("--assert")
        .arg("-Wno-fatal");
    if env::var_os("CARGO_CFG_FUZZING").is_some() {
        // Mimic the flags `cargo fuzz` passes to `rustc`:
        // * -Cpasses=sancov-module: I think this is implied by `-fsanitize-coverage`?
        // * -Cllvm-args=-sanitizer-coverage-level=4: idk
        // * -Cllvm-args=-sanitizer-coverage-inline-8bit-counters,
        //   -Cllvm-args=-sanitizer-coverage-pc-table,
        //   -Cllvm-args=-sanitizer-coverage-trace-compares:
        //   -fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp
        // * --cfg fuzzing: Irrelevant.
        // * -Clink-dead-code: We aren't doing any linking, so irrelevant.
        // * -Zsanitizer=address: -fsanitize=address
        // * -Cdebug-assertions, -C codegen-units=1: maybe we should do these but they
        //   aren't needed to make things work.
        cmd.arg("-CFLAGS")
            .arg("-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp -fsanitize=address");
    }
    cmd.arg(manifest_dir.join("../../fpga/net-finder.srcs/sources_1/new/generated.sv"))
        .arg(manifest_dir.join("../../fpga/net-finder.srcs/sources_1/new/instruction_neighbour.sv"))
        .arg(manifest_dir.join("../../fpga/net-finder.srcs/sources_1/new/valid_checker.sv"))
        .arg(manifest_dir.join("../../fpga/net-finder.srcs/sources_1/new/core.sv"))
        .arg(manifest_dir.join("wrapper.cpp"));
    let status = cmd.status()?;
    if !status.success() {
        bail!("verilator failed");
    }

    let mut cmd = Command::new("make");
    cmd.arg("-C").arg(&out_dir).arg("-f").arg("Vcore.mk");
    // Verilator's makefiles don't respect `CXX` when passed as an environment
    // variable, but they do respect it when passed as an argument to `make`. So if
    // `CXX` is set, pass it to `make`.
    if let Ok(cxx) = env::var("CXX") {
        cmd.arg(format!("CXX={cxx}"));
    }
    cmd.arg("Vcore__ALL.a")
        .arg("verilated.o")
        .arg("verilated_fst_c.o")
        .arg("verilated_threads.o")
        // Verilator doesn't include any wrapper files you provide unless you pass `--exe`, but it
        // still generates make targets for them. So we invoke it manually.
        .arg("wrapper.o");
    let status = cmd.status()?;
    if !status.success() {
        bail!("make failed");
    }

    // Add a `lib` prefix.
    fs::copy(out_dir.join("Vcore__ALL.a"), out_dir.join("libVcore.a"))?;

    // Add the last few loose object files that we need into `libVcore.a`.
    //
    // This is probably overkill to just run `ar`, but eh.
    let status = cc::Build::new()
        .cpp(true)
        .get_archiver()
        .arg("-q")
        .arg(out_dir.join("libVcore.a"))
        .arg(out_dir.join("verilated.o"))
        .arg(out_dir.join("verilated_fst_c.o"))
        .arg(out_dir.join("verilated_threads.o"))
        .arg(out_dir.join("wrapper.o"))
        .status()?;
    if !status.success() {
        bail!("ar failed");
    }

    println!("cargo:rerun-if-changed=../../fpga/net-finder.srcs/sources_1/new");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=Vcore");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=z");

    Ok(())
}
