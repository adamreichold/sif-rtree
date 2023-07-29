use std::iter::{Copied, FusedIterator};
use std::slice::Iter;

use crate::{Node, Object, TWIG_LEN};

pub struct BranchIter<'a, O>
where
    O: Object,
{
    twigs: Iter<'a, Node<O>>,
    idx: Copied<Iter<'a, usize>>,
}

impl<'a, O> BranchIter<'a, O>
where
    O: Object,
{
    pub fn new(nodes: &'a [Node<O>], idx: usize) -> Self {
        let (branch, twigs) = nodes[idx..].split_first().unwrap();

        let len = match branch {
            Node::Branch { len, .. } => len,
            Node::Twig(_) | Node::Leaf(_) => unreachable!(),
        };

        let (len, pad) = twig_len_pad(len.get());

        let mut twigs = twigs[..len].iter();

        let idx = match twigs.next().unwrap() {
            Node::Twig(twig) => twig[pad..].iter().copied(),
            Node::Branch { .. } | Node::Leaf(_) => unreachable!(),
        };

        Self { twigs, idx }
    }
}

impl<O> Iterator for BranchIter<'_, O>
where
    O: Object,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.idx.next() {
            Some(idx) => Some(idx),
            None => match self.twigs.next() {
                Some(Node::Twig(twig)) => {
                    self.idx = twig.iter().copied();

                    self.idx.next()
                }
                Some(Node::Branch { .. }) | Some(Node::Leaf(_)) => unreachable!(),
                None => None,
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<O> ExactSizeIterator for BranchIter<'_, O>
where
    O: Object,
{
    fn len(&self) -> usize {
        self.twigs.len() * TWIG_LEN + self.idx.len()
    }
}

impl<O> FusedIterator for BranchIter<'_, O> where O: Object {}

pub fn twig_len_pad(len: usize) -> (usize, usize) {
    let quot = len / TWIG_LEN;
    let rem = len % TWIG_LEN;

    let len = if rem == 0 { quot } else { quot + 1 };
    let pad = if rem == 0 { 0 } else { TWIG_LEN - rem };

    (len, pad)
}
