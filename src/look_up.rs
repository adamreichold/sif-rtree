use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use num_traits::Zero;

use crate::{iter::branch_for_each, Distance, Node, Object, Point, RTree};

impl<O, S> RTree<O, S>
where
    O: Object,
    S: AsRef<[Node<O>]>,
{
    /// Locates all objects whose axis-aligned bounding box (AABB) is contained in the queried AABB
    pub fn look_up_aabb_contains<'a, V>(
        &'a self,
        query: &(O::Point, O::Point),
        visitor: V,
    ) -> ControlFlow<()>
    where
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        let query = |node: &Node<O>| match node {
            Node::Branch { aabb, .. } => intersects(aabb, query),
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => contains(&obj.aabb(), query),
        };

        self.look_up(query, visitor)
    }

    /// Locates all objects whose axis-aligned bounding box (AABB) intersects the queried AABB
    pub fn look_up_aabb_intersects<'a, V>(
        &'a self,
        query: &(O::Point, O::Point),
        visitor: V,
    ) -> ControlFlow<()>
    where
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        let query = |node: &Node<O>| match node {
            Node::Branch { aabb, .. } => intersects(aabb, query),
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => intersects(&obj.aabb(), query),
        };

        self.look_up(query, visitor)
    }

    /// Locates all objects which contain the queried point
    pub fn look_up_at_point<'a, V>(&'a self, query: &O::Point, visitor: V) -> ControlFlow<()>
    where
        O: Distance<O::Point>,
        O::Point: Distance<O::Point>,
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        let query = |node: &Node<O>| match node {
            Node::Branch { aabb, .. } => aabb.contains(query),
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => obj.contains(query),
        };

        self.look_up(query, visitor)
    }

    /// Locates all objects which are within the given `distance` of the given `center`
    pub fn look_up_within_distance_of_point<'a, V>(
        &'a self,
        center: &O::Point,
        distance: <O::Point as Point>::Coord,
        visitor: V,
    ) -> ControlFlow<()>
    where
        O: Distance<O::Point>,
        O::Point: Distance<O::Point>,
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        let distance_2 = distance * distance;

        let query = |node: &Node<O>| match node {
            Node::Branch { aabb, .. } => aabb.distance_2(center) <= distance_2,
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => obj.distance_2(center) <= distance_2,
        };

        self.look_up(query, visitor)
    }

    fn look_up<'a, Q, V>(&'a self, query: Q, visitor: V) -> ControlFlow<()>
    where
        Q: FnMut(&'a Node<O>) -> bool,
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        let mut args = LookUpArgs {
            nodes: self.nodes.as_ref(),
            query,
            visitor,
        };

        let (node, rest) = args.nodes.split_first().unwrap();

        if (args.query)(node) {
            match node {
                Node::Branch { len, .. } => look_up(&mut args, len, rest)?,
                Node::Twig(_) | Node::Leaf(_) => unreachable!(),
            }
        }

        ControlFlow::Continue(())
    }
}

struct LookUpArgs<'a, O, Q, V>
where
    O: Object,
{
    nodes: &'a [Node<O>],
    query: Q,
    visitor: V,
}

fn look_up<'a, O, Q, V>(
    args: &mut LookUpArgs<'a, O, Q, V>,
    mut len: &'a NonZeroUsize,
    mut twigs: &'a [Node<O>],
) -> ControlFlow<()>
where
    O: Object,
    Q: FnMut(&'a Node<O>) -> bool,
    V: FnMut(&'a O) -> ControlFlow<()>,
{
    loop {
        let mut branch = None;

        branch_for_each(len, twigs, |idx| {
            let (node, rest) = args.nodes[idx..].split_first().unwrap();

            if (args.query)(node) {
                match node {
                    Node::Branch { len, .. } => {
                        if let Some((len1, twigs1)) = branch.replace((len, rest)) {
                            look_up(args, len1, twigs1)?;
                        }
                    }
                    Node::Twig(_) => unreachable!(),
                    Node::Leaf(obj) => (args.visitor)(obj)?,
                }
            }

            ControlFlow::Continue(())
        })?;

        if let Some((len1, twigs1)) = branch {
            len = len1;
            twigs = twigs1;
        } else {
            return ControlFlow::Continue(());
        }
    }
}

fn intersects<P>(lhs: &(P, P), rhs: &(P, P)) -> bool
where
    P: Point,
{
    (0..P::DIM).all(|axis| {
        lhs.0.coord(axis) <= rhs.1.coord(axis) && lhs.1.coord(axis) >= rhs.0.coord(axis)
    })
}

fn contains<P>(lhs: &(P, P), rhs: &(P, P)) -> bool
where
    P: Point,
{
    (0..P::DIM).all(|axis| {
        lhs.0.coord(axis) <= rhs.0.coord(axis) && lhs.1.coord(axis) >= rhs.1.coord(axis)
    })
}

impl<P> Distance<P> for (P, P)
where
    P: Point + Distance<P>,
{
    fn distance_2(&self, point: &P) -> P::Coord {
        if !self.contains(point) {
            let min = self.1.min(&self.0.max(point));

            min.distance_2(point)
        } else {
            P::Coord::zero()
        }
    }

    fn contains(&self, point: &P) -> bool {
        (0..P::DIM).all(|axis| {
            self.0.coord(axis) <= point.coord(axis) && point.coord(axis) <= self.1.coord(axis)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::{collection::vec, test_runner::TestRunner};

    use crate::{
        tests::{random_objects, random_points},
        DEF_NODE_LEN,
    };

    #[test]
    fn random_look_up_aabb_contains() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_objects(10)),
                |(objects, queries)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for query in queries {
                        let mut results1 = index
                            .objects()
                            .filter(|obj| contains(&obj.aabb(), &query.aabb()))
                            .collect::<Vec<_>>();

                        let mut results2 = Vec::new();
                        index.look_up_aabb_contains(&query.aabb(), |obj| {
                            results2.push(obj);
                            ControlFlow::Continue(())
                        });

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn random_look_up_aabb_intersects() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_objects(10)),
                |(objects, queries)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for query in queries {
                        let mut results1 = index
                            .objects()
                            .filter(|obj| intersects(&obj.aabb(), &query.aabb()))
                            .collect::<Vec<_>>();

                        let mut results2 = Vec::new();
                        index.look_up_aabb_intersects(&query.aabb(), |obj| {
                            results2.push(obj);
                            ControlFlow::Continue(())
                        });

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn random_look_up_at_point() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_points(10)),
                |(objects, queries)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for query in queries {
                        let mut results1 = index
                            .objects()
                            .filter(|obj| obj.contains(&query))
                            .collect::<Vec<_>>();

                        let mut results2 = Vec::new();
                        index.look_up_at_point(&query, |obj| {
                            results2.push(obj);
                            ControlFlow::Continue(())
                        });

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn random_look_up_within_distance_of_point() {
        TestRunner::default()
            .run(
                &(
                    random_objects(100),
                    random_points(10),
                    vec(0.0_f32..=1.0, 10),
                ),
                |(objects, centers, distances)| {
                    let index = RTree::new(DEF_NODE_LEN, objects);

                    for (center, distance) in centers.iter().zip(distances) {
                        let mut results1 = index
                            .objects()
                            .filter(|obj| obj.distance_2(center) <= distance * distance)
                            .collect::<Vec<_>>();

                        let mut results2 = Vec::new();
                        index.look_up_within_distance_of_point(center, distance, |obj| {
                            results2.push(obj);
                            ControlFlow::Continue(())
                        });

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }
}
