#![no_main]
use std::collections::HashSet;
use std::iter::zip;

use libfuzzer_sys::{fuzz_target, Corpus};
use net_finder::{Combinations, Cuboid, Mapping, MappingData, SkipSet, SquareCache};

const CUBOIDS: usize = 3;

fuzz_target!(|input: (
    [Cuboid; CUBOIDS],
    Vec<Mapping<CUBOIDS>>,
    Vec<Mapping<CUBOIDS>>
)|
 -> Corpus {
    let (cuboids, included, other) = input;
    if cuboids.into_iter().any(|cuboid| {
        cuboid.surface_area() != cuboids[0].surface_area()
            || cuboid.width == 0
            || cuboid.depth == 0
            || cuboid.height == 0
    }) || cuboids[0].surface_area() > 64
        || included.iter().chain(other.iter()).any(|mapping| {
            mapping
                .cursors()
                .iter()
                .any(|cursor| usize::from(cursor.0) >= cuboids[0].surface_area())
        })
    {
        return Corpus::Reject;
    }

    let square_caches = cuboids.map(SquareCache::new);
    let mut set = SkipSet::new(cuboids);
    for mapping in included.iter().copied() {
        set.insert(&square_caches, mapping);
    }
    let expected: HashSet<_> = included
        .iter()
        .flat_map(|mapping| {
            Combinations::new(
                &zip(&square_caches, mapping.cursors())
                    .map(|(cache, cursor)| cursor.to_data(cache).equivalents())
                    .collect::<Vec<_>>(),
            )
            .map(|cursors| Mapping::from_data(&square_caches, &MappingData::new(cursors)))
            .collect::<Vec<_>>()
        })
        .collect();
    let actual: HashSet<_> = set.into_iter().collect();
    assert_eq!(expected, actual);

    for &mapping in expected.iter() {
        assert!(set.contains(mapping));
    }

    for mapping in other {
        if expected.contains(&mapping) {
            return Corpus::Reject;
        }
        assert!(!set.contains(mapping));
    }

    Corpus::Keep
});
