use std::ops::*;
use name_backend::{FeatureBool, Element, Feature, FeatureNum};

#[derive(Debug, Clone)]
pub struct Tensor<T, const D: usize, Backend: Feature<T, D>>(Backend::TensorPrimitive);

impl<T: Element, Backend: FeatureNum<T, 2>> Tensor<T, 2, Backend> {
    pub fn matmul(&self, rhs: &Self) -> Self {
        Self(Backend::matmul(&self.0, &rhs.0))
    }
}

impl<const D: usize, Backend: FeatureBool<D>> Neg for Tensor<bool, D, Backend> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        Backend::neg_(&mut self.0);
        self
    }
}

impl<const D: usize, Backend: FeatureBool<D>> Neg for &Tensor<bool, D, Backend> {
    type Output = Tensor<bool, D, Backend>;

    fn neg(self) -> Self::Output {
        Tensor(Backend::neg(&self.0))
    }
}

macro_rules! impl_binary_op {
    ($op:ident, $op_assign:ident, $op_fn:ident, $op_assign_fn:ident, $bop:ident, $bop_scalar:ident, $bop_:ident, $bop_scalar_:ident) => {
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op for Tensor<T, D, Backend> {
            type Output = Self;
        
            fn $op_fn(mut self, rhs: Self) -> Self::Output {
                Backend::$bop_(&mut self.0, &rhs.0);
                self
            }
        }
        
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op<T> for Tensor<T, D, Backend> {
            type Output = Self;
        
            fn $op_fn(mut self, rhs: T) -> Self::Output {
                Backend::$bop_scalar_(&mut self.0, rhs);
                self
            }
        }
        
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op for &Tensor<T, D, Backend> {
            type Output = Tensor<T, D, Backend>;
        
            fn $op_fn(self, rhs: Self) -> Self::Output {
                Tensor(Backend::$bop(&self.0, &rhs.0))
            }
        }
        
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op<T> for &Tensor<T, D, Backend> {
            type Output = Tensor<T, D, Backend>;
        
            fn $op_fn(self, rhs: T) -> Self::Output {
                Tensor(Backend::$bop_scalar(&self.0, rhs))
            }
        }
        
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op_assign for Tensor<T, D, Backend> {
            fn $op_assign_fn(&mut self, rhs: Self) {
                Backend::$bop_(&mut self.0, &rhs.0);
            }
        }
        
        impl<T: Element, const D: usize, Backend: FeatureNum<T, D>> $op_assign<T> for Tensor<T, D, Backend> {
            fn $op_assign_fn(&mut self, rhs: T) {
                Backend::$bop_scalar_(&mut self.0, rhs);
            }
        }
    };
}

impl_binary_op!(Add, AddAssign, add, add_assign, add, add_scalar, add_, add_scalar_);
impl_binary_op!(Sub, SubAssign, sub, sub_assign, sub, sub_scalar, sub_, sub_scalar_);
impl_binary_op!(Mul, MulAssign, mul, mul_assign, mul, mul_scalar, mul_, mul_scalar_);
impl_binary_op!(Div, DivAssign, div, div_assign, div, div_scalar, div_, div_scalar_);
