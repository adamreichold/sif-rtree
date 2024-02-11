use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::swap;
use std::ops::ControlFlow;

use num_traits::{Float, Zero};

use crate::{iter::branch_for_each, Distance, Node, Object, Point, RTree, ROOT_IDX};

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
    /// Returns a reference to the object and its squared distance to the `target`.
    ///
    /// Returns `None` if no object has a finite distance to the `target`.
    pub fn nearest(&self, target: &O::Point) -> Option<(&O, <O::Point as Point>::Coord)> {
        let mut nearest = None;

        let mut min_minmax_distance_2 = <O::Point as Point>::Coord::infinity();

        from_near_to_far(
            self.nodes.as_ref(),
            target,
            |aabb, distance_2| {
                if min_minmax_distance_2 >= distance_2 {
                    let minmax_distance_2 = minmax_distance_2(aabb, target);

                    if min_minmax_distance_2 > minmax_distance_2 {
                        min_minmax_distance_2 = minmax_distance_2;
                    }

                    true
                } else {
                    false
                }
            },
            |object, distance_2| {
                nearest = Some((object, distance_2));
                ControlFlow::Break(())
            },
        );

        nearest
    }

    /// Visit all objects in ascending order of their distance to the given `target`
    ///
    /// Yields references to the objects and their squared distances to the `target`.
    pub fn from_near_to_far<'a, V>(&'a self, target: &O::Point, visitor: V) -> ControlFlow<()>
    where
        V: FnMut(&'a O, <O::Point as Point>::Coord) -> ControlFlow<()>,
    {
        from_near_to_far(
            self.nodes.as_ref(),
            target,
            |_aabb, _distance_2| true,
            visitor,
        )
    }
}

fn from_near_to_far<'a, O, F, V>(
    nodes: &'a [Node<O>],
    target: &O::Point,
    mut filter: F,
    mut visitor: V,
) -> ControlFlow<()>
where
    O: Object,
    O: Distance<O::Point>,
    O::Point: Distance<O::Point>,
    <O::Point as Point>::Coord: Float,
    F: FnMut(&(O::Point, O::Point), <O::Point as Point>::Coord) -> bool,
    V: FnMut(&'a O, <O::Point as Point>::Coord) -> ControlFlow<()>,
{
    let mut items = BinaryHeap::new();

    items.push(NearestItem {
        idx: ROOT_IDX,
        distance_2: <O::Point as Point>::Coord::nan(),
    });

    while let Some(item) = items.pop() {
        let [node, rest @ ..] = &nodes[item.idx..] else {
            unreachable!()
        };

        match node {
            Node::Branch { len, .. } => branch_for_each(len, rest, |idx| {
                let obj_aabb;

                let (aabb, distance_2) = match &nodes[idx] {
                    Node::Branch { aabb, .. } => (aabb, aabb.distance_2(target)),
                    Node::Twig(_) => unreachable!(),
                    Node::Leaf(obj) => {
                        obj_aabb = obj.aabb();

                        (&obj_aabb, obj.distance_2(target))
                    }
                };

                if filter(aabb, distance_2) {
                    items.push(NearestItem { idx, distance_2 });
                }

                ControlFlow::Continue(())
            })?,
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => visitor(obj, item.distance_2)?,
        }
    }

    ControlFlow::Continue(())
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
                            .map(|obj| obj.distance_2(&target))
                            .min_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap())
                            .unwrap();

                        let (obj, result2) = index.nearest(&target).unwrap();
                        assert_eq!(obj.distance_2(&target), result2);

                        assert_eq!(result1, result2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn random_from_near_to_far() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_points(10)),
                |(objects, targets)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for target in targets {
                        let mut result1 = index
                            .objects()
                            .map(|obj| obj.distance_2(&target))
                            .collect::<Vec<_>>();

                        result1.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());

                        let mut result2 = Vec::new();

                        index.from_near_to_far(&target, |obj, distance_2| {
                            assert_eq!(obj.distance_2(&target), distance_2);

                            result2.push(distance_2);
                            ControlFlow::Continue(())
                        });

                        assert_eq!(result1, result2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }
}
