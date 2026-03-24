//! For all areas less than or equal to the area specified which are produced by
//! more than one cuboid with integer side lengths, prints out all of those
//! cuboids.

use std::collections::BTreeMap;
use std::env;

use anyhow::{bail, Context};
use itertools::Itertools;
use net_finder::Cuboid;

fn main() -> anyhow::Result<()> {
    let Some([max_area]) = env::args().skip(1).collect_array() else {
        bail!("usage: valid_dims <max_area>");
    };
    let max_area: usize = max_area.parse().context("area should be an integer")?;

    let mut area_to_cuboids: BTreeMap<usize, Vec<Cuboid>> = BTreeMap::new();
    // For all integers a, b, c >= 1, 2(ab + bc + ca) >= 4a + 1, 4b + 1, 4c + 1.
    // Hence if 2(ab + bc + ca) <= max_area, a, b, c <= (max_area - 1) / 4.
    // (And for all integers x and reals y, if x <= y, x <= floor(y).)
    let max_dim: u8 = ((max_area - 1) / 4)
        .try_into()
        .context("only dimensions < 256 are supported right now")?;
    for width in 1..=max_dim {
        // always write dimensions in sorted order (which also prevents duplicates).
        for depth in width..=max_dim {
            for height in depth..=max_dim {
                let cuboid = Cuboid::new(width, depth, height);
                area_to_cuboids
                    .entry(cuboid.surface_area())
                    .or_default()
                    .push(cuboid);
            }
        }
    }

    for (area, cuboids) in area_to_cuboids {
        if area <= max_area && cuboids.len() > 1 {
            print!("{area}: ");
            for (i, cuboid) in cuboids.iter().enumerate() {
                if i != 0 {
                    print!(", ");
                }
                print!("{cuboid}");
            }
            println!();
        }
    }

    Ok(())
}
