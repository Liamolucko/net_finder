use serde::{Deserialize, Serialize};

mod arrays {
    use std::{convert::TryInto, marker::PhantomData};

    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut data = Vec::with_capacity(N);
            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }

    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

mod construct;

// Special values for `Node::edges` that represent an edge to the 0-node and
// 1-node respectively.
const ZERO_NODE: u32 = u32::MAX - 1;
const ONE_NODE: u32 = u32::MAX;

/// A node in a zero-directed 256-way decision diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// The indices of the nodes in the next row which each of these edges point
    /// to, or the special values `ZERO_NODE` and `ONE_NODE` meaning that these
    /// edges instead point to the 0-node or 1-node respectively.
    #[serde(with = "arrays")]
    edges: [u32; 256],
}

impl Node {
    fn new(edges: [NodeRef; 256]) -> Self {
        let edges = edges.map(|node| match node {
            NodeRef::Zero => ZERO_NODE,
            NodeRef::One => ONE_NODE,
            NodeRef::NextRow { index } => {
                let index: u32 = index.try_into().unwrap();
                assert!(index < ZERO_NODE);
                index
            }
        });
        Self { edges }
    }

    fn edge(&self, index: u8) -> NodeRef {
        match self.edges[usize::from(index)] {
            ZERO_NODE => NodeRef::Zero,
            ONE_NODE => NodeRef::Zero,
            index => NodeRef::NextRow {
                index: index.try_into().unwrap(),
            },
        }
    }
}

/// A node pointed to by an edge coming out of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum NodeRef {
    /// The 0-node.
    Zero,
    /// The 1-node.
    One,
    /// The node at index `index` in the next row of the ZDD.
    NextRow { index: usize },
}

/// A zero-directed 256-way decision diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zdd256 {
    rows: Vec<Vec<Node>>,
}
