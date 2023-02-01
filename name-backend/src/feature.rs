use std::sync::Arc;
use num_traits::real::Real;
use crate::{ArrayConstraint, WithShape};
use crate::{Backend, Element};

macro_rules! binaryop {
    ($id:ident, $id_scalar:ident, $id_:ident, $id_scalar_:ident) => {
        fn $id(
            lhs: WithShape<&Self::Array, D>,
            rhs: WithShape<&Self::Array, D>,
        ) -> Self::Array;
        fn $id_scalar(
            lhs: WithShape<&Self::Array, D>,
            rhs: Elem
        ) -> Self::Array where ConstUsize<{D}>: Equals<1>;
        fn $id_(
            WithShape { array, shape }: WithShape<&mut Self::Array, D>,
            rhs: WithShape<&Self::Array, D>,
        ) {
             *array = Self::$id(WithShape { array, shape }, rhs);
        }
        fn $id_scalar_(
            WithShape { array, shape }: WithShape<&mut Self::Array, D>,
            rhs: Elem
        ) where ConstUsize<{D}>: Equals<1> {
            *array = Self::$id_scalar(WithShape { array, shape }, rhs);
        }
    };
}

macro_rules! unary_inplace_default {
    ($id:ident, $id_:ident) => {
        fn $id_(WithShape { array, shape }: WithShape<&mut Self::Array, D>) where ConstUsize<{D}>: Equals<1> {
            *array = Self::$id(WithShape{ array, shape });
        }
    };
}

pub trait FArray<Elem>: Backend {
    type Array: ArrayConstraint<Self::Device>;
    fn from_iterator<I: IntoIterator<Item=Elem>>(device: &Arc<Self::Device>, iter: I) -> Self::Array;
    fn new_uniform(device: &Arc<Self::Device>, count: usize) -> Self::Array;
}

pub struct ConstUsize<const N: usize>;

pub trait Equals<const N: usize> {}

impl<const N: usize> Equals<N> for ConstUsize<N> {}

pub trait FTensor<Elem, const D: usize>: FArray<Elem> {}

pub trait FTensorBool<const D: usize>: FTensor<bool, D> {
    fn neg(array: WithShape<&Self::Array, D>) -> Self::Array where ConstUsize<D>: Equals<1>;
    unary_inplace_default!(neg, neg_);
}

/// Contributor's note: add function here when you want to
/// these also implements arithmetics other than FArray
/// in addition, with broadcasting support
pub trait FTensorNum<Elem: Element, const D: usize>: FTensor<Elem, D> {
    fn powu_scalar(array: WithShape<&Self::Array, D>, rhs: u32) -> Self::Array where ConstUsize<D>: Equals<1>;
    fn powu_scalar_(array: WithShape<&Self::Array, D>, rhs: u32) where ConstUsize<D>: Equals<1>;
    fn powf_scalar(array: WithShape<&Self::Array, D>, rhs: f32) -> Self::Array where Elem: Real, ConstUsize<D>: Equals<1>;
    fn powf_scalar_(array: WithShape<&Self::Array, D>, rhs: f32) where Elem: Real, ConstUsize<D>: Equals<1>;
    fn erf(array: WithShape<&Self::Array, D>) -> Self::Array where ConstUsize<D>: Equals<1>;
    unary_inplace_default!(erf, erf_);
    fn gelu(WithShape { array, shape }: WithShape<&Self::Array, D>) -> Self::Array where ConstUsize<D>: Equals<1> {
        let mut x_inner = Self::div_scalar(WithShape { array, shape }, 2f64.into());
        let ref mut x = x_inner;
        Self::erf_(WithShape { array: x, shape });
        Self::add_scalar(WithShape { array: x, shape }, 1f64.into());
        Self::mul_(WithShape { array: x, shape }, WithShape { array, shape });
        Self::div_scalar(WithShape { array: x, shape }, 2f64.into());
        x_inner
    }
    unary_inplace_default!(gelu, gelu_);
    binaryop!(add, add_scalar, add_, add_scalar_);
    binaryop!(sub, sub_scalar, sub_, sub_scalar_);
    binaryop!(mul, mul_scalar, mul_, mul_scalar_);
    binaryop!(div, div_scalar, div_, div_scalar_);
    binaryop!(min, min_scalar, min_, min_scalar_);
    binaryop!(max, max_scalar, max_, max_scalar_);

    fn matmul(lhs: &WithShape<&Self::Array, D>, rhs: &WithShape<&Self::Array, D>) -> Self::Array where ConstUsize<{D}>: Equals<2>;
}