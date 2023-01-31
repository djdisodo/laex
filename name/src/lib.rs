macro_rules! include_mod {
    ($id:ident) => {
        mod $id;
        pub use $id::*;
    };
}

include_mod!(tensor);
