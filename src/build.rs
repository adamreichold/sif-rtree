use std::marker::PhantomData;
use std::num::NonZeroUsize;

use crate::{iter::twig_len_pad, Node, Object, Point, RTree, TWIG_LEN};

/// A sensible default value for the node length, balancing query efficency against memory overhead
pub const DEF_NODE_LEN: usize = 6;

impl<O> RTree<O>
where
    O: Object,
{
    /// Builds a new [R-tree](https://en.wikipedia.org/wiki/R-tree) from a given set of `objects`
    ///
    /// The `node_len` parameter determines the length of branch nodes and thereby the three depth. It must be larger than one. [`DEF_NODE_LEN`] provides a sensible default.
    ///
    /// The `objects` parameter must not be empty.
    pub fn new(node_len: usize, objects: Vec<O>) -> Self {
        assert!(node_len > 1);
        assert!(!objects.is_empty());

        let mut nodes = Vec::new();
        let mut next_nodes = Vec::new();

        let root_idx = build(node_len, objects, &mut nodes, &mut next_nodes);
        debug_assert_eq!(root_idx, nodes.len() - 1);

        // The whole tree is reversed, so that iteration visits increasing memory addresses which measurably improves performance.
        nodes.reverse();

        for node in &mut nodes {
            if let Node::Twig(twig) = node {
                for idx in twig {
                    *idx = root_idx - *idx;
                }
            }
        }

        Self {
            nodes: nodes.into_boxed_slice(),
            _marker: PhantomData,
        }
    }
}

/// A reimplementation of the overlap-minimizing top-down bulk loading algorithm used by the [`rstar`] crate
///
/// For a given value of `node_len` (which is equivalent to [`rstar::RTreeParams::MAX_SIZE`]) and a given list of `objects`, it should produce the same tree structure.
fn build<O>(
    node_len: usize,
    objects: Vec<O>,
    nodes: &mut Vec<Node<O>>,
    next_nodes: &mut Vec<usize>,
) -> usize
where
    O: Object,
{
    let next_nodes_len = next_nodes.len();

    if objects.len() > node_len {
        let num_clusters = num_clusters(node_len, O::Point::DIM, objects.len());

        struct State<O> {
            objects: Vec<O>,
            axis: usize,
        }

        let mut state = vec![State {
            objects,
            axis: O::Point::DIM,
        }];

        while let Some(State {
            mut objects,
            mut axis,
        }) = state.pop()
        {
            if axis != 0 {
                axis -= 1;

                let cluster_len = (objects.len() + num_clusters - 1) / num_clusters;

                while objects.len() > cluster_len {
                    objects.select_nth_unstable_by(cluster_len, |lhs, rhs| {
                        let lhs = lhs.aabb().0.coord(axis);
                        let rhs = rhs.aabb().0.coord(axis);
                        lhs.partial_cmp(&rhs).unwrap()
                    });

                    let next_objects = objects.split_off(cluster_len);
                    state.push(State { objects, axis });
                    objects = next_objects;
                }

                if !objects.is_empty() {
                    state.push(State { objects, axis });
                }
            } else {
                let node = build(node_len, objects, nodes, next_nodes);
                next_nodes.push(node);
            }
        }
    } else {
        next_nodes.extend(nodes.len()..nodes.len() + objects.len());
        nodes.extend(objects.into_iter().map(Node::Leaf));
    }

    let node = add_branch(nodes, &next_nodes[next_nodes_len..]);
    next_nodes.truncate(next_nodes_len);
    node
}

fn num_clusters(node_len: usize, point_dim: usize, num_objects: usize) -> usize {
    let node_len = node_len as f64;
    let point_dim = point_dim as f64;
    let num_objects = num_objects as f64;

    let depth = num_objects.log(node_len).ceil() as usize;

    let subtree_len = node_len.powi(depth as i32 - 1);
    let num_subtree = (num_objects / subtree_len).ceil();

    num_subtree.powf(point_dim.recip()).ceil() as usize
}

fn add_branch<O>(nodes: &mut Vec<Node<O>>, next_nodes: &[usize]) -> usize
where
    O: Object,
{
    let nodes_len = nodes.len();

    let len = NonZeroUsize::new(next_nodes.len()).unwrap();

    let aabb = merge_aabb(nodes, next_nodes);

    {
        // Padding is inserted into the first twig, so that iteration is uniform over the following twigs.
        let (len, pad) = twig_len_pad(&len);

        nodes.reserve(len + 1);

        let mut twig = [0; TWIG_LEN];
        let mut pos = pad;

        for next_node in next_nodes {
            if pos == TWIG_LEN {
                nodes.push(Node::Twig(twig));
                pos = 0;
            }

            twig[pos] = *next_node;
            pos += 1;
        }

        if pos != 0 {
            nodes.push(Node::Twig(twig));
        }
    }

    let node = nodes.len();

    // The twigs in the branch are reversed, so that after reversing the whole tree, they will follow the branch in ascending order.
    nodes[nodes_len..node].reverse();

    nodes.push(Node::Branch { len, aabb });

    node
}

fn merge_aabb<O>(nodes: &[Node<O>], next_nodes: &[usize]) -> (O::Point, O::Point)
where
    O: Object,
{
    next_nodes
        .iter()
        .map(|idx| match &nodes[*idx] {
            Node::Branch { aabb, .. } => aabb.clone(),
            Node::Twig(_) => unreachable!(),
            Node::Leaf(obj) => obj.aabb(),
        })
        .reduce(|mut res, aabb| {
            res.0 = res.0.min(&aabb.0);
            res.1 = res.1.max(&aabb.1);

            res
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::ControlFlow;

    use proptest::test_runner::TestRunner;

    use crate::{
        iter::branch_for_each,
        tests::{random_objects, RandomObject},
    };

    impl rstar::RTreeObject for RandomObject {
        type Envelope = rstar::AABB<[f32; 3]>;

        fn envelope(&self) -> Self::Envelope {
            rstar::AABB::from_corners(self.0, self.1)
        }
    }

    fn collect_index<'a>(
        nodes: &'a [Node<RandomObject>],
        idx: usize,
        branches: &mut Vec<usize>,
        leaves: &mut Vec<&'a RandomObject>,
    ) {
        let (node, rest) = nodes[idx..].split_first().unwrap();
        let len = match node {
            Node::Branch { len, .. } => len,
            Node::Twig(_) | Node::Leaf(_) => unreachable!(),
        };
        branches.push(len.get());
        branch_for_each(len, rest, |idx| {
            match &nodes[idx] {
                Node::Branch { .. } => collect_index(nodes, idx, branches, leaves),
                Node::Twig(_) => unreachable!(),
                Node::Leaf(obj) => {
                    branches.push(0);
                    leaves.push(obj);
                }
            }
            ControlFlow::Continue(())
        });
    }

    fn collect_rstar_index<'a>(
        node: &'a rstar::ParentNode<RandomObject>,
        branches: &mut Vec<usize>,
        leaves: &mut Vec<&'a RandomObject>,
    ) {
        let children = node.children();
        branches.push(children.len());
        for child in children {
            match child {
                rstar::RTreeNode::Parent(node) => collect_rstar_index(node, branches, leaves),
                rstar::RTreeNode::Leaf(obj) => {
                    branches.push(0);
                    leaves.push(obj);
                }
            }
        }
    }

    #[test]
    fn random_trees() {
        TestRunner::default()
            .run(&random_objects(100), |objects| {
                let index = RTree::new(DEF_NODE_LEN, objects.clone());

                let mut branches = Vec::new();
                let mut leaves = Vec::new();

                collect_index(&index, 0, &mut branches, &mut leaves);

                let rstar_index = rstar::RTree::bulk_load(objects);

                let mut rstar_branches = Vec::new();
                let mut rstar_leaves = Vec::new();

                collect_rstar_index(rstar_index.root(), &mut rstar_branches, &mut rstar_leaves);

                assert_eq!(branches, rstar_branches);
                assert_eq!(leaves, rstar_leaves);

                Ok(())
            })
            .unwrap();
    }
}
