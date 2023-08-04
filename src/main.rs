use std::time::Instant;

use clap::Parser;
use net_finder::{find_nets, resume, Cuboid, Net, MAX_CUBOIDS};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    resume: bool,
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    assert!(options.cuboids.len() <= MAX_CUBOIDS.into());

    let mut count = 0;
    let start = Instant::now();
    let callback = |net: Net| {
        count += 1;
        println!("#{count} after {:?}:", start.elapsed());
        let mut nets = vec![net.to_string()];
        for &cuboid in options.cuboids.iter() {
            nets.push(net.color(cuboid).unwrap().to_string());
        }
        println!("{}\n", join_horizontal(nets));
    };
    if options.resume {
        resume(&options.cuboids)?.for_each(callback);
    } else {
        find_nets(&options.cuboids)?.for_each(callback);
    }
    println!("Number of nets: {count} (took {:?})", start.elapsed());
    Ok(())
}

fn join_horizontal(strings: Vec<String>) -> String {
    let mut lines: Vec<_> = strings.iter().map(|s| s.lines()).collect();
    let mut out = String::new();
    loop {
        for (i, iter) in lines.iter_mut().enumerate() {
            if i != 0 {
                out += " ";
            }
            if let Some(line) = iter.next() {
                out += line;
            } else {
                return out;
            }
        }
        out += "\n";
    }
}
