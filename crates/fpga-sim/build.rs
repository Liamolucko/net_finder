use std::fs::File;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use anyhow::bail;

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let verilog_path = out_dir.join("core.v");
    let status = Command::new("python3")
        .current_dir(manifest_dir.join("../.."))
        .arg("-m")
        .arg("net_finder.core.fuzz_target")
        .stdout(File::create(&verilog_path)?)
        .status()?;
    if !status.success() {
        bail!("failed to generate verilog")
    }

    let mut verilator = Command::new("verilator");
    verilator
        .arg("--cc")
        .arg("--Mdir")
        .arg(&out_dir)
        .arg("--trace")
        .arg("--assert")
        .arg("-Wno-fatal");
    if env::var_os("CARGO_CFG_FUZZING").is_some() {
        verilator.arg("-CFLAGS").arg("-fsanitize=fuzzer");
    }
    verilator
        .arg(verilog_path)
        .arg(manifest_dir.join("wrapper.cpp"));
    let status = verilator.status()?;
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
        .arg("verilated_vcd_c.o")
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
        .arg(out_dir.join("verilated_vcd_c.o"))
        .arg(out_dir.join("verilated_threads.o"))
        .arg(out_dir.join("wrapper.o"))
        .status()?;
    if !status.success() {
        bail!("ar failed");
    }

    println!("cargo:rerun-if-changed=../../net_finder/core");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=Vcore");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=z");

    Ok(())
}
