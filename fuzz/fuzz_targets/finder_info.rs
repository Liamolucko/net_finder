#![no_main]
use std::any::Any;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::time::Duration;

use libfuzzer_sys::arbitrary::{self, Unstructured};
use libfuzzer_sys::{fuzz_target, Corpus};
use net_finder::{Cuboid, Finder, FinderCtx, FinderInfo};
use pretty_assertions::assert_eq;

fuzz_target!(|input: &[u8]| -> Corpus {
    let input = Unstructured::new(input);
    match main(input) {
        Ok(()) => Corpus::Keep,
        Err(_) => Corpus::Reject,
    }
});

fn main(mut input: Unstructured) -> arbitrary::Result<()> {
    let cuboids: Vec<Cuboid> = input.arbitrary()?;

    match cuboids.as_slice() {
        &[a] => main_inner(input, [a]),
        &[a, b] => main_inner(input, [a, b]),
        &[a, b, c] => main_inner(input, [a, b, c]),
        _ => Err(arbitrary::Error::IncorrectFormat),
    }
}

fn main_inner<const CUBOIDS: usize>(
    mut input: Unstructured,
    cuboids: [Cuboid; CUBOIDS],
) -> arbitrary::Result<()> {
    let info: FinderInfo<CUBOIDS> = input.arbitrary()?;

    // Cache the `FinderCtx`s we've used before (i.e., their equivalence classes).
    //
    // TODO: cache to disk as well? For 3 cuboids creating `FinderCtx`s can take
    // multiple seconds, disk access is significantly faster than that.
    thread_local! {
        static CTXS: RefCell<HashMap<Vec<Cuboid>, &'static dyn Any>> = RefCell::new(HashMap::new());
    }

    let ctx: &'static FinderCtx<CUBOIDS> = CTXS.with_borrow_mut(|ctxs| {
        let entry = match ctxs.entry(cuboids.into()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => *entry.insert(Box::leak(Box::new({
                // This will catch it if `cuboids` is invalid.
                FinderCtx::new(cuboids, Duration::ZERO)
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?
            }))),
        };
        Ok(entry.downcast_ref::<FinderCtx<CUBOIDS>>().unwrap())
    })?;

    // This will catch it if `info` is invalid.
    let finder = Finder::new(&ctx, &info).map_err(|_| arbitrary::Error::IncorrectFormat)?;

    // Make sure we get the same `FinderInfo` back from `Finder::info` that we
    // started with.
    assert_eq!(finder.clone().into_info(&ctx), info);

    // Make sure that after running the `Finder` for a bit, it can still be
    // round-tripped through a `FinderInfo` without changing anything.
    //
    // Specifically, we run until the decisions change, which is signalled by a
    // change in area. Up until that point, `stepped_finder` will keep giving the
    // same `FinderInfo` as `Finder`, since instructions run after a decision aren't
    // taken into account.
    let mut stepped_finder = finder.clone();
    while stepped_finder.area() == finder.area() {
        if !stepped_finder.step(&ctx) {
            // Stop early if the finder finishes.
            break;
        }

        assert_eq!(
            &Finder::new(&ctx, &stepped_finder.clone().into_info(&ctx)).unwrap(),
            if stepped_finder.area() != finder.area() {
                // If the area's changed the decisions have changed.
                &stepped_finder
            } else {
                // Otherwise they haven't and so this step won't be included.
                &finder
            }
        );
    }

    // And now by induction, any `Finder` created from a `FinderInfo` and then run
    // until just after a decision happens can be represented losslessly by a
    // `FinderInfo`.
    Ok(())
}
