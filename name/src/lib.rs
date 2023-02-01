#![feature(generic_const_exprs)]
extern crate name_backend as backend;

macro_rules! include_mod {
    ($id:ident) => {
        mod $id;
        pub use $id::*;
    };
}

include_mod!(tensor);
include_mod!(array);