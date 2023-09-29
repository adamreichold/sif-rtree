use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::swap;

use num_traits::{Float, Zero};

use crate::{iter::BranchIter, Distance, Node, Object, Point, RTree};

impl<O, S> RTree<O, S>
where
    O: Object,
    O: Distance<O::Point>,
    O::Point: Distance<O::Point>,
    <O::Point as Point>::Coord: Float,
    S: AsRef<[Node<O>]>,
{
    /// Find the object nearest to the given `target`
    ///
    /// Returns `None` if no object has a finite distance to the `target`.
    pub fn nearest(&self, target: &O::Point) -> Option<&O> {
        let nodes = self.nodes.as_ref();

        let mut min_minmax_distance_2 = <O::Point as Point>::Coord::infinity();

        let mut items = BinaryHeap::new();

        items.push(NearestItem {
            idx: 0,
            distance_2: <O::Point as Point>::Coord::nan(),
        });

        while let Some(item) = items.pop() {
            match &nodes[item.idx] {
                Node::Branch { .. } => {
                    for idx in BranchIter::new(nodes, item.idx) {
                        let (aabb, distance_2) = match &nodes[idx] {
                            Node::Branch { aabb, .. } => (aabb.clone(), aabb.distance_2(target)),
                            Node::Twig(_) => unreachable!(),
                            Node::Leaf(obj) => (obj.aabb(), obj.distance_2(target)),
                        };

                        if min_minmax_distance_2 >= distance_2 {
                            let minmax_distance_2 = minmax_distance_2(&aabb, target);

                            if min_minmax_distance_2 > minmax_distance_2 {
                                min_minmax_distance_2 = minmax_distance_2;
                            }

                            items.push(NearestItem { idx, distance_2 });
                        }
                    }
                }
                Node::Twig(_) => unreachable!(),
                Node::Leaf(obj) => return Some(obj),
            }
        }

        None
    }
}

fn minmax_distance_2<P>(aabb: &(P, P), target: &P) -> P::Coord
where
    P: Point,
    P::Coord: Float,
{
    let mut max_diff = P::Coord::zero();
    let mut max_diff_axis = 0;
    let mut max_diff_min_2 = P::Coord::zero();

    let max_2 = P::build(|axis| {
        let lower = aabb.0.coord(axis);
        let upper = aabb.1.coord(axis);
        let target = target.coord(axis);

        let mut min_2 = (lower - target).powi(2);
        let mut max_2 = (upper - target).powi(2);

        if min_2 > max_2 {
            swap(&mut min_2, &mut max_2);
        }

        let diff = max_2 - min_2;

        if max_diff <= diff {
            max_diff = diff;
            max_diff_axis = axis;
            max_diff_min_2 = min_2;
        }

        max_2
    });

    (0..P::DIM).fold(P::Coord::zero(), |res, axis| {
        let minmax_2 = if axis == max_diff_axis {
            max_diff_min_2
        } else {
            max_2.coord(axis)
        };

        res + minmax_2
    })
}

struct NearestItem<F> {
    idx: usize,
    distance_2: F,
}

impl<F> PartialEq for NearestItem<F>
where
    F: Float,
{
    fn eq(&self, other: &Self) -> bool {
        other.distance_2 == self.distance_2
    }
}

impl<F> Eq for NearestItem<F> where F: Float {}

impl<F> PartialOrd for NearestItem<F>
where
    F: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F> Ord for NearestItem<F>
where
    F: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance_2.partial_cmp(&self.distance_2).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::test_runner::TestRunner;

    use crate::{
        tests::{random_objects, random_points},
        DEF_NODE_LEN,
    };

    #[test]
    fn random_nearest() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_points(10)),
                |(objects, targets)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for target in targets {
                        let result1 = index
                            .objects()
                            .min_by(|lhs, rhs| {
                                let lhs = lhs.distance_2(&target);
                                let rhs = rhs.distance_2(&target);

                                lhs.partial_cmp(&rhs).unwrap()
                            })
                            .unwrap();

                        let result2 = index.nearest(&target).unwrap();

                        assert_eq!(result1.distance_2(&target), result2.distance_2(&target));
                    }

                    Ok(())
                },
            )
            .unwrap();
    }
}
