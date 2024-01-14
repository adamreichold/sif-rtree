#![forbid(unsafe_code)]
#![deny(missing_docs, missing_debug_implementations)]

//! A simple library implementing an immutable, flat representation of an [R-tree](https://en.wikipedia.org/wiki/R-tree)
//!
//! The library uses the same overlap-minimizing top-down bulk loading algorithm as the [rstar](https://github.com/georust/rstar) crate.
//! Its supports several kinds of spatial queries, nearest neighbour search and has a simple implementation as the objects in the index are fixed after construction.
//! This also enables a flat and thereby cache-friendly memory layout which can be backed by memory maps.
//!
//! The library provides optional integration with [serde] for (de-)serialization of the trees.
//!
//! # Example
//!
//! ```
//! use std::ops::ControlFlow;
//!
//! use sif_rtree::{DEF_NODE_LEN, Distance, RTree, Object};
//!
//! struct Something(usize, [f64; 2]);
//!
//! impl Object for Something {
//!     type Point = [f64; 2];
//!
//!     fn aabb(&self) -> (Self::Point, Self::Point) {
//!         (self.1, self.1)
//!     }
//! }
//!
//! impl Distance<[f64; 2]> for Something {
//!     fn distance_2(&self, point: &[f64; 2]) -> f64 {
//!         self.1.distance_2(point)
//!     }
//! }
//!
//! let index = RTree::new(
//!     DEF_NODE_LEN,
//!     vec![
//!         Something(0, [-0.4, -3.3]),
//!         Something(1, [-4.5, -1.8]),
//!         Something(2, [0.7, 2.0]),
//!         Something(3, [1.7, 1.5]),
//!         Something(4, [-1.3, 2.3]),
//!         Something(5, [2.2, 1.0]),
//!         Something(6, [-3.7, 3.8]),
//!         Something(7, [-3.2, -0.1]),
//!         Something(8, [1.4, 2.7]),
//!         Something(9, [3.1, -0.0]),
//!         Something(10, [4.3, 0.8]),
//!         Something(11, [3.9, -3.3]),
//!         Something(12, [0.4, -3.2]),
//!     ],
//! );
//!
//! let mut close_by = Vec::new();
//!
//! index.look_up_within_distance_of_point(&[0., 0.], 3., |thing| {
//!     close_by.push(thing.0);
//!
//!     ControlFlow::Continue(())
//! });
//!
//! assert_eq!(close_by, [3, 5, 4, 2]);
//! ```
//!
//! The [`RTree`] data structure is generic over its backing storage as long as it can be converted into a slice via the [`AsRef`] trait. This can for instance be used to memory map R-trees from persistent storage.
//!
//! ```no_run
//! # fn main() -> std::io::Result<()> {
//! use std::fs::File;
//! use std::mem::size_of;
//! use std::slice::from_raw_parts;
//!
//! use memmap2::Mmap;
//!
//! use sif_rtree::{Node, Object, Point, RTree};
//!
//! #[derive(Clone, Copy)]
//! struct Triangle([[f64; 3]; 3]);
//!
//! impl Object for Triangle {
//!     type Point = [f64; 3];
//!
//!     fn aabb(&self) -> (Self::Point, Self::Point) {
//!         let min = self.0[0].min(&self.0[1]).min(&self.0[2]);
//!         let max = self.0[0].max(&self.0[1]).max(&self.0[2]);
//!         (min, max)
//!     }
//! }
//!
//! let file = File::open("index.bin")?;
//! let map = unsafe { Mmap::map(&file)? };
//!
//! struct TriangleSoup(Mmap);
//!
//! impl AsRef<[Node<Triangle>]> for TriangleSoup {
//!     fn as_ref(&self) -> &[Node<Triangle>] {
//!         let ptr = self.0.as_ptr().cast();
//!         let len = self.0.len() / size_of::<Node<Triangle>>();
//!
//!         unsafe { from_raw_parts(ptr, len) }
//!     }
//! }
//!
//! let index = RTree::new_unchecked(TriangleSoup(map));
//! # Ok(()) }
//! ```

mod build;
mod iter;
mod look_up;
mod nearest;

pub use build::DEF_NODE_LEN;

use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Deref;

use num_traits::{Num, Zero};
#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Defines a [finite-dimensional][Self::DIM] space in terms of [coordinate values][Self::coord] along a chosen set of axes
pub trait Point: Clone {
    /// The dimension of the underlying space
    const DIM: usize;

    /// The type of the coordinate values
    type Coord: Num + Copy + PartialOrd;

    /// Access the coordinate value of the point along the given `axis`
    fn coord(&self, axis: usize) -> Self::Coord;

    /// Builds a new point by specifying its coordinate values along each axis
    ///
    /// # Example
    ///
    /// ```rust
    /// use sif_rtree::Point;
    ///
    /// fn scale<P>(point: P, factor: P::Coord) -> P
    /// where
    ///     P: Point,
    /// {
    ///     P::build(|axis| point.coord(axis) * factor)
    /// }
    /// ```
    fn build<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Coord;

    /// Computes the point which has the minimum coordinate values of `self` and `other` along each axis
    ///
    /// The default implementation is based on [`build`][Self::build] and [`PartialOrd`] and assumes that coordinate values are finite.
    fn min(&self, other: &Self) -> Self {
        Self::build(|axis| {
            let this = self.coord(axis);
            let other = other.coord(axis);

            if this < other {
                this
            } else {
                other
            }
        })
    }

    /// Computes the point which has the maximum coordinate values of `self` and `other` along each axis
    ///
    /// The default implementation is based on [`build`][Self::build] and [`PartialOrd`] and assumes that coordinate values are finite.
    fn max(&self, other: &Self) -> Self {
        Self::build(|axis| {
            let this = self.coord(axis);
            let other = other.coord(axis);

            if this > other {
                this
            } else {
                other
            }
        })
    }
}

/// `N`-dimensional space using [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
impl<T, const N: usize> Point for [T; N]
where
    T: Num + Copy + PartialOrd,
{
    const DIM: usize = N;

    type Coord = T;

    #[inline]
    fn coord(&self, axis: usize) -> Self::Coord {
        self[axis]
    }

    fn build<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Coord,
    {
        let mut res = [T::zero(); N];

        (0..N).for_each(|axis| res[axis] = f(axis));

        res
    }
}

impl<T, const N: usize> Distance<[T; N]> for [T; N]
where
    T: Num + Copy + PartialOrd,
{
    fn distance_2(&self, point: &[T; N]) -> T {
        (0..N).fold(T::zero(), |res, axis| {
            let diff = self[axis] - point[axis];

            res + diff * diff
        })
    }
}

/// Defines the objects which can be organized in an [`RTree`] by specifying their extent in the vector space defined via the [`Point`] trait
pub trait Object {
    /// The [`Point`] implementation used to represent the [axis-aligned bounding boxes (AABB)][`Self::aabb`] of these objects.
    type Point: Point;

    /// Return the axis-aligned bounding box (AABB) associated with this object.
    ///
    /// Note that this method is called repeatedly during construction and querying. Hence it might be necessary to cache the value internally to avoid the cost of computing it repeatedly.
    fn aabb(&self) -> (Self::Point, Self::Point);
}

/// Defines a distance metric between a type (objects, points, AABB, etc.) and [points][`Point`]
pub trait Distance<P>
where
    P: Point,
{
    /// Return the squared distance between `self` and `point`
    ///
    /// Generally, only the relation between two distance values is required so that computing square roots can be avoided.
    fn distance_2(&self, point: &P) -> P::Coord;

    /// Checks whether `self` contains `point`
    ///
    /// The default implementation just checks whether the squared distance is zero.
    fn contains(&self, point: &P) -> bool {
        self.distance_2(point) == P::Coord::zero()
    }
}

// Should be a power of two and as large as possible without `Twig` becoming the largest variant of `Node`.
const TWIG_LEN: usize = 4;

/// A node in the tree
///
/// The tree starts at a root node which is always a [`Branch`][Self::Branch] stored at index zero.
/// It ends with the [`Leaf`][Self::Leaf] nodes which contains the objects stored in the tree.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Node<O>
where
    O: Object,
{
    /// A branch in the tree
    ///
    /// The indices of the nodes belonging to this branch are stored separately in [`Twig`][Self::Twig] nodes immediately following this node.
    /// The first of these `Twig` nodes starts with padding if `len` is not a multiple of `TWIG_LEN`.
    Branch {
        /// The number of nodes belonging to this branch
        len: NonZeroUsize,
        /// The merged axis-aligned bounding box (AABB) of all the nodes belonging to this branch
        #[cfg_attr(
            feature = "serde",
            serde(bound = "O::Point: Serialize + DeserializeOwned")
        )]
        aabb: (O::Point, O::Point),
    },
    /// Contains the indices of nodes belonging to a branch
    Twig([usize; TWIG_LEN]),
    /// Contains an object stored in the tree
    Leaf(O),
}

const ROOT_IDX: usize = 0;

/// An immutable, flat representation of an [R-tree](https://en.wikipedia.org/wiki/R-tree)
///
/// Accelerates spatial queries by grouping objects based on their axis-aligned bounding boxes (AABB).
///
/// Note that this tree dereferences to and deserializes as a slice of nodes. Modifying node geometry through interior mutability or deserializing a modified sequence is safe but will lead to incorrect results.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct RTree<O, S = Box<[Node<O>]>>
where
    O: Object,
    S: AsRef<[Node<O>]>,
{
    nodes: S,
    _marker: PhantomData<O>,
}

impl<O, S> RTree<O, S>
where
    O: Object,
    S: AsRef<[Node<O>]>,
{
    /// Interprets the given `nodes` as a tree
    ///
    /// Supplying `nodes` which are not actually organized as an R-tree is safe but will lead to incorrect results.
    pub fn new_unchecked(nodes: S) -> Self {
        assert!(!nodes.as_ref().is_empty());

        Self {
            nodes,
            _marker: PhantomData,
        }
    }

    /// Iterators over the objects stored in the leaf nodes of the tree
    pub fn objects(&self) -> impl Iterator<Item = &O> {
        self.nodes.as_ref().iter().filter_map(|node| match node {
            Node::Branch { .. } | Node::Twig(_) => None,
            Node::Leaf(obj) => Some(obj),
        })
    }
}

impl<O, S> Deref for RTree<O, S>
where
    O: Object,
    S: AsRef<[Node<O>]>,
{
    type Target = [Node<O>];

    fn deref(&self) -> &Self::Target {
        self.nodes.as_ref()
    }
}

impl<O, S> AsRef<[Node<O>]> for RTree<O, S>
where
    O: Object,
    S: AsRef<[Node<O>]>,
{
    fn as_ref(&self) -> &[Node<O>] {
        self.nodes.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering;

    use proptest::{collection::vec, strategy::Strategy};

    pub fn random_points(len: usize) -> impl Strategy<Value = Vec<[f32; 3]>> {
        (
            vec(0.0_f32..=1.0, len),
            vec(0.0_f32..=1.0, len),
            vec(0.0_f32..=1.0, len),
        )
            .prop_map(|(x, y, z)| {
                x.into_iter()
                    .zip(y)
                    .zip(z)
                    .map(|((x, y), z)| [x, y, z])
                    .collect()
            })
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct RandomObject(pub [f32; 3], pub [f32; 3]);

    impl Eq for RandomObject {}

    impl PartialOrd for RandomObject {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for RandomObject {
        fn cmp(&self, other: &Self) -> Ordering {
            (self.0, self.1).partial_cmp(&(other.0, other.1)).unwrap()
        }
    }

    impl Object for RandomObject {
        type Point = [f32; 3];

        fn aabb(&self) -> (Self::Point, Self::Point) {
            (self.0, self.1)
        }
    }

    impl Distance<[f32; 3]> for RandomObject {
        fn distance_2(&self, point: &[f32; 3]) -> f32 {
            self.aabb().distance_2(point)
        }

        fn contains(&self, point: &[f32; 3]) -> bool {
            self.aabb().contains(point)
        }
    }

    pub fn random_objects(len: usize) -> impl Strategy<Value = Vec<RandomObject>> {
        (random_points(len), random_points(len)).prop_map(|(left, right)| {
            left.into_iter()
                .zip(right)
                .map(|(left, right)| {
                    let lower = left.min(&right);
                    let upper = left.max(&right);
                    RandomObject(lower, upper)
                })
                .collect()
        })
    }
}
