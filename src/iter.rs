use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use crate::{Node, Object, TWIG_LEN};

pub fn branch_for_each<O, V>(
    len: &NonZeroUsize,
    twigs: &[Node<O>],
    mut visitor: V,
) -> ControlFlow<()>
where
    O: Object,
    V: FnMut(usize) -> ControlFlow<()>,
{
    let (len, pad) = twig_len_pad(len);

    let mut twigs = twigs[..len].iter();

    let mut twig = match twigs.next().unwrap() {
        Node::Twig(twig) => &twig[pad..],
        Node::Branch { .. } | Node::Leaf(_) => unreachable!(),
    };

    loop {
        for idx in twig {
            visitor(*idx)?;
        }

        twig = match twigs.next() {
            Some(Node::Twig(twig)) => twig,
            Some(Node::Branch { .. } | Node::Leaf(_)) => unreachable!(),
            None => break,
        };
    }

    ControlFlow::Continue(())
}

pub fn twig_len_pad(len: &NonZeroUsize) -> (usize, usize) {
    let len = len.get();
    let quot = len / TWIG_LEN;
    let rem = len % TWIG_LEN;

    let len = if rem == 0 { quot } else { quot + 1 };
    let pad = if rem == 0 { 0 } else { TWIG_LEN - rem };

    (len, pad)
}
