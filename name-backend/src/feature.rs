use std::sync::Arc;
use num_traits::real::Real;
use crate::{Shape, TensorPrimitiveConstraint};
use crate::{Backend, Element};

macro_rules! binaryop {
    ($id:ident, $id_scalar:ident, $id_:ident, $id_scalar_:ident) => {
        fn $id(
            lhs: &Self::TensorPrimitive,
            rhs: &Self::TensorPrimitive,
        ) -> Self::TensorPrimitive;
        fn $id_scalar(
            lhs: &Self::TensorPrimitive,
            rhs: Elem
        ) -> Self::TensorPrimitive;
        fn $id_(
            lhs: &mut Self::TensorPrimitive,
            rhs: &Self::TensorPrimitive,
        ) {
            *lhs = Self::$id(lhs, rhs);
        }
        fn $id_scalar_(
            lhs: &mut Self::TensorPrimitive,
            rhs: Elem
        ) {
            *lhs = Self::$id_scalar(lhs, rhs);
        }
    };
}

macro_rules! unary_inplace_default {
    ($id:ident, $id_:ident) => {
        fn $id_(tensor: &mut Self::TensorPrimitive) {
            *tensor = Self::$id(tensor);
        }
    };
}

pub trait Feature<Elem, const D: usize>: Backend {
    type TensorPrimitive: TensorPrimitiveConstraint<Self::Device, D>;
    fn from_iterator<I: IntoIterator<Item=Elem>>(device: &Arc<Self::Device>, iter: I) -> Self::TensorPrimitive where ConstUsize<{D}>: Equals<1>;
    fn reshape(tensor: Self::TensorPrimitive, shape: Shape<D>) -> Self::TensorPrimitive;
    fn shape(tensor: &Self::TensorPrimitive) -> Shape<D>;
}

pub trait FeatureBool<const D: usize>: Feature<bool, D> {
    fn false_(device: &Arc<Self::Device>, shape: Shape<D>) -> Self::TensorPrimitive;
    fn true_(device: &Arc<Self::Device>, shape: Shape<D>) -> Self::TensorPrimitive;
    fn neg(tensor: &Self::TensorPrimitive) -> Self::TensorPrimitive;
    unary_inplace_default!(neg, neg_);
}



pub struct ConstUsize<const N: usize>;

pub trait Equals<const N: usize> {}

impl<const N: usize> Equals<N> for ConstUsize<N> {}

pub trait FeatureNum<Elem: Element, const D: usize>: Feature<Elem, D> {

    fn zeros(device: &Arc<Self::Device>, shape: Shape<D>) -> Self::TensorPrimitive;
    fn ones(device: &Arc<Self::Device>, shape: Shape<D>) -> Self::TensorPrimitive;

    binaryop!(add, add_scalar, add_, add_scalar_);
    binaryop!(sub, sub_scalar, sub_, sub_scalar_);
    binaryop!(mul, mul_scalar, mul_, mul_scalar_);
    binaryop!(div, div_scalar, div_, div_scalar_);
    binaryop!(min, min_scalar, min_, min_scalar_);
    binaryop!(max, max_scalar, max_, max_scalar_);
    fn pow(&self, rhs: u32) -> Self::TensorPrimitive;
    fn powf(&self, rhs: f32) -> Self::TensorPrimitive where Elem: Real;

    fn relu(tensor: &Self::TensorPrimitive) -> Self::TensorPrimitive {
        Self::max_scalar(tensor, Elem::zero())
    }

    fn relu_(tensor: &mut Self::TensorPrimitive) {
        Self::max_scalar_(tensor, Elem::zero())
    }


    fn erf(tensor: &Self::TensorPrimitive) -> Self::TensorPrimitive;
    unary_inplace_default!(erf, erf_);

    fn gelu(tensor: &Self::TensorPrimitive) -> Self::TensorPrimitive {
        let mut x_inner = Self::div_scalar(tensor, 2f64.into());
        let ref mut x = x_inner;
        Self::erf_(x);
        Self::add_scalar(x, 1f64.into());
        Self::mul_(x, tensor);
        Self::div_scalar(x, 2f64.into());
        x_inner
    }
    unary_inplace_default!(gelu, gelu_);

    fn matmul(lhs: &Self::TensorPrimitive, rhs: &Self::TensorPrimitive) -> Self::TensorPrimitive where ConstUsize<{D}>: Equals<2>;
}