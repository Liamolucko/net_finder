use anyhow::bail;
use clap::Parser;
use net_finder::Cuboid;
use net_finder_cpu::CpuRuntime;

#[derive(Parser)]
struct Options {
    #[arg(long)]
    resume: bool,
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    match *options.cuboids.as_slice() {
        [] => bail!("must specify at least 1 cuboid"),
        [a] => net_finder::drive(CpuRuntime::new([a]), options.resume),
        [a, b] => net_finder::drive(CpuRuntime::new([a, b]), options.resume),
        [a, b, c] => net_finder::drive(CpuRuntime::new([a, b, c]), options.resume),
        _ => bail!("only up to 3 cuboids are currently supported"),
    }
}
