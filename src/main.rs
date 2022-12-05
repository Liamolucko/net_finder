use std::env;
use std::time::Instant;

use anyhow::bail;
use net_finder::{find_nets, Cuboid};

fn main() -> anyhow::Result<()> {
    let args = env::args()
        .into_iter()
        .skip(1)
        .map(|arg| arg.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if args.len() % 3 != 0 {
        bail!(
            "expected sets of 3 arguments representing the width, height and depth of the cuboids."
        );
    }
    let cuboids: Vec<_> = args
        .chunks(3)
        .map(|chunk| Cuboid::new(chunk[0], chunk[1], chunk[2]))
        .collect();
    let mut count = 0;
    let start = Instant::now();
    for net in find_nets(cuboids.clone())? {
        count += 1;
        println!("#{count} after {:?}:", start.elapsed());
        let mut nets = vec![net.to_string()];
        for &cuboid in cuboids.iter() {
            nets.push(net.color(cuboid).unwrap().to_string());
        }
        println!("{}\n", join_horizontal(nets));
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
