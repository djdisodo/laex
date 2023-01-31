#![feature(trait_alias)]

use std::fmt::{Debug, Display};
use std::sync::Arc;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use type_map::concurrent::TypeMap;
macro_rules! include_mod {
    ($id:ident) => {
        mod $id;
        pub use $id::*;
    };
}

include_mod!(element);

pub trait TensorPrimitiveConstraint<Device, const D: usize> =
    Clone
    + Display
    + Debug
    + Send
    + Sync
    + AsRef<Arc<Device>>
;

pub trait Backend: 'static {
    type Device: Send + Sync;
    fn name() -> String;

    fn default_device() -> Self::Device;
}

static ATOMIC_DEVICE_MAP: Lazy<RwLock<TypeMap>> = Lazy::new(Default::default);

pub trait GetDevice: Backend {
    fn device() -> Arc<Self::Device>;
    fn set_device(device: Arc<Self::Device>);
}

impl<T: Backend> GetDevice for T {
    fn device() -> Arc<Self::Device> {
        if let Some(device) = ATOMIC_DEVICE_MAP.read().get::<Arc<T::Device>>() {
            device.clone()
        } else {
            let ret = Arc::new(T::default_device());
            Self::set_device(ret.clone());
            ret
        }
    }

    fn set_device(device: Arc<Self::Device>) {
        ATOMIC_DEVICE_MAP.write().insert(device);
    }
}

include_mod!(feature);
include_mod!(shape);