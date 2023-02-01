use backend::{FTensor, FTensorNum, Shape, WithShape, Element, Backend, ConstUsize, Equals, FArray};
use std::ops::*;

trait TensorAsRef: Sized {
    type T;
    const D: usize;
    type Backend: FArray<Self::T>;
    type TyReshape<const D0: usize>;
    fn reshape2<const D1: usize>(self, shape: Shape<D1>) -> Self::TyReshape<D1>;
    fn array_ref(&self) -> &<Self::Backend as FArray<Self::T>>::Array;
    fn shape_ref(&self) -> &Shape<{Self::D}>;
    fn shape_mut(&mut self) -> &mut Shape<{Self::D}>;

    fn as_with_shape(&self) -> WithShape<&<Self::Backend as FArray<Self::T>>::Array, {Self::D}> {
        WithShape {
            array: self.array_ref(),
            shape: *self.shape_ref()
        }
    }

    fn as_ref(&self) -> TensorRef<Self::T, {Self::D}, Self::Backend> {
        TensorRef {
            array: self.array_ref(),
            shape: *self.shape_ref(),
        }
    }
}


// marker traits; marks it doesn't own Tensor(it's not Tensor<T, D, B>)
// contributes optimization
trait TensorBorrowed {}

trait ArrayAsMut: TensorAsRef {
    fn array_mut(&mut self) -> &mut <Self::Backend as FArray<Self::T>>::Array;
    fn as_with_shape_mut(&mut self) -> WithShape<&<Self::Backend as FArray<Self::T>>::Array, {Self::D}> {
        WithShape {
            array: &mut self.array_mut(),
            shape: *self.shape_ref()
        }
    }

    fn as_ref(&self) -> TensorRef<Self::T, {Self::D}, Self::Backend> {
        TensorRef {
            array: self.array_ref(),
            shape: *self.shape_ref(),
        }
    }

    fn as_mut(&mut self) -> TensorMut<Self::T, {Self::D}, Self::Backend> {
        TensorMut {
            array: self.array_mut(),
            shape: *self.shape_ref(),
        }
    }
}

pub trait TensorExt: TensorAsRef {
    fn shape(&self) -> Shape<{Self::D}> {
        *self.shape_ref()
    }
    fn reshape(&mut self, shape: Shape<{Self::D}>) {
        shape.assert_reshape(shape);
        *self.shape_mut() = shape;
    }

    fn into_shape<const D: usize>(self, shape: Shape<D>) -> Self::TyReshape<D> {
        self.shape_ref().assert_reshape(shape);
        <Self as TensorAsRef>::reshape2(self, shape)
    }

    fn with_shape<const D: usize>(&self, shape: Shape<D>) -> TensorRef<Self::T, D, Self::Backend> {
        self.as_ref().into_shape(shape)
    }
}

impl<Tensor: TensorAsRef> TensorExt for Tensor {}

#[derive(Debug, Copy, Clone)]
pub struct TensorRef<'a, T, const D: usize, Backend: FArray<T>> {
    array: &'a Backend::Array,
    shape: Shape<D>,
}

impl<'a, T, const D: usize, Backend: FArray<T>> TensorAsRef for TensorRef<'a, T, D, Backend>{
    type T = T;
    const D: usize = D;
    type Backend = Backend;
    type TyReshape<const D0: usize> = TensorRef<'a, T, D0, Backend>;

    fn reshape2<const D1: usize>(self, shape: Shape<D1>) -> Self::TyReshape<D1> {
        TensorRef {
            array: self.array,
            shape,
        }
    }

    fn array_ref(&self) -> &Backend::Array {
        self.array
    }

    fn shape_ref(&self) -> &Shape<{Self::D}> {
        &self.shape
    }

    fn shape_mut(&mut self) -> &mut Shape<{Self::D}> {
        &mut self.shape
    }
}

#[derive(Debug)]
pub struct TensorMut<'a, T, const D: usize, Backend: FArray<T>> {
    array: &'a mut Backend::Array,
    shape: Shape<D>,
}

impl<'a, T, const D: usize, Backend: FArray<T>> TensorAsRef for TensorMut<'a, T, D, Backend>{
    type T = T;
    const D: usize = D;
    type Backend = Backend;
    type TyReshape<const D0: usize> = TensorRef<'a, T, D0, Backend>;

    fn reshape2<const D1: usize>(self, shape: Shape<D1>) -> Self::TyReshape<D1> {
        TensorRef {
            array: self.array,
            shape,
        }
    }

    fn array_ref(&self) -> &Backend::Array {
        self.array
    }

    fn shape_ref(&self) -> &Shape<{Self::D}> {
        &self.shape
    }

    fn shape_mut(&mut self) -> &mut Shape<{Self::D}> {
        &mut self.shape
    }
}

impl<'a, T, const D: usize, Backend: FArray<T>> ArrayAsMut for TensorMut<'a, T, D, Backend> {
    fn array_mut(&mut self) -> &mut Backend::Array {
        self.array
    }
}

/// simple owned tensor
#[derive(Debug, Clone)]
pub struct Tensor<T, const D: usize, Backend: FArray<T>>{
    array: Backend::Array,
    shape: Shape<D>
}

macro_rules! impl_binary_op {
    ($op:ident, $op_assign:ident, $op_fn:ident, $op_assign_fn:ident, $bop:ident, $bop_scalar:ident, $bop_:ident, $bop_scalar_:ident) => {
        // lhs is owned so can be reused
        impl<
            T: Element,
            const D: usize,
            Backend: FTensorNum<T, D>,
            Rhs: TensorAsRef<Backend=Backend, T=T>
        > $op<Rhs> for Tensor<T, D, Backend> where ConstUsize<D>: Equals<{Rhs::D}> {
            type Output = Tensor<T, D, Backend>;

            fn $op_fn(mut self, rhs: Rhs) -> Self::Output {
                Backend::$bop_(WithShape {
                    array: &mut self.array,
                    shape: self.shape
                }, rhs.as_with_shape());
                self
            }
        }

        // rhs is owned so may be reused
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T>
        // > $op<Tensor<T, D, Backend>> for Lhs {
        //     type Output = Tensor<T, D, Backend>;
        //
        //     fn $op_fn(mut self, rhs: Tensor<T, D, Backend>) -> Self::Output {
        //         self.shape_ref().assert_broadcast(rhs.shape);
        //         if *self.shape_ref() == rhs.shape {
        //             // rhs has same size reusing
        //             Backend::$bop_(WithShape {
        //                 array: &mut rhs.array,
        //                 shape: rhs.shape
        //             }, self.as_with_shape());
        //             rhs
        //         } else {
        //             Backend::$bop(
        //                 self.as_with_shape(),
        //                 rhs.as_with_shape()
        //             )
        //         }
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed,
        //     Rhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed
        // > $op for Lhs {
        //     type Output = Tensor<T, D, Backend>;
        //
        //     fn $op_fn(self, rhs: Rhs) -> Self::Output {
        //         Tensor {
        //             array: Backend::$bop(self.as_with_shape(), rhs.as_with_shape()),
        //             shape: *self.shape_ref()
        //         }
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed,
        //     Rhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed
        // > $op<Rhs> for &Lhs {
        //     type Output = Tensor<T, D, Backend>;
        //
        //     fn $op_fn(self, rhs: Rhs) -> Self::Output {
        //         Tensor {
        //             array: Backend::$bop(self.as_with_shape(), rhs.as_with_shape()),
        //             shape: *self.shape_ref()
        //         }
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed,
        //     Rhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed
        // > $op<&Rhs> for Lhs {
        //     type Output = Tensor<T, D, Backend>;
        //
        //     fn $op_fn(self, rhs: &Rhs) -> Self::Output {
        //         Tensor {
        //             array: Backend::$bop(self.as_with_shape(), rhs.as_with_shape()),
        //             shape: *self.shape_ref()
        //         }
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed,
        //     Rhs: TensorAsRef<Backend=Backend, T=T> + TensorBorrowed
        // > $op<&Rhs> for &Lhs {
        //     type Output = Tensor<T, D, Backend>;
        //
        //     fn $op_fn(self, rhs: &Rhs) -> Self::Output {
        //         Tensor {
        //             array: Backend::$bop(self.as_with_shape(), rhs.as_with_shape()),
        //             shape: *self.shape_ref()
        //         }
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + ArrayAsMut,
        //     Rhs: TensorAsRef<Backend=Backend, T=T>
        // > $op_assign<Rhs> for Lhs {
        //     fn $op_assign_fn(&mut self, rhs: Rhs) {
        //         Backend::$bop_(self.as_with_shape_mut(), rhs.as_with_shape());
        //     }
        // }
        //
        // impl<
        //     T: Element,
        //     const D: usize,
        //     Backend: FTensorNum<T, D>,
        //     Lhs: TensorAsRef<Backend=Backend, T=T> + ArrayAsMut,
        //     Rhs: TensorAsRef<Backend=Backend, T=T>
        // > $op_assign<&Rhs> for Lhs {
        //     fn $op_assign_fn(&mut self, rhs: &Rhs) {
        //         Backend::$bop_(self.as_with_shape_mut(), rhs.as_with_shape());
        //     }
        // }
    };
}

impl_binary_op!(Add, AddAssign, add, add_assign, add, add_scalar, add_, add_scalar_);
//impl_binary_op!(Sub, SubAssign, sub, sub_assign, sub, sub_scalar, sub_, sub_scalar_);
//impl_binary_op!(Mul, MulAssign, mul, mul_assign, mul, mul_scalar, mul_, mul_scalar_);
//impl_binary_op!(Div, DivAssign, div, div_assign, div, div_scalar, div_, div_scalar_);
