use clap::Parser;
use net_finder::{Cuboid, Zdd};

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    for cuboid in options.cuboids {
        let zdd = Zdd::construct(cuboid);
        let size = zdd.size();
        println!("{size} nets found");
        for (i, net) in zdd.nets().enumerate() {
            print!("net {i} / {size}");
            if let Some(colored) = net.color(cuboid) {
                println!(":\n{colored}");
            } else {
                println!(" (broken):\n{net}");
            }
            println!();
        }
    }
    Ok(())
}
