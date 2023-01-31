use num_traits::Num;

pub trait Element: Num + From<u64> + From<f64> {}