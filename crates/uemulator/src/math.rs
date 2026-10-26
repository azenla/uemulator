//! The float methods `core` lacks, under their `std` names.

pub(crate) trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    fn round_ties_even(self) -> Self;
}

pub(crate) trait Float64Ext {
    fn powi(self, n: i32) -> f64;
    fn exp_m1(self) -> f64;
    fn log2(self) -> f64;
    fn tan(self) -> f64;
    fn sin(self) -> f64;
    fn cos(self) -> f64;
    fn atan2(self, x: f64) -> f64;
    fn powf(self, y: f64) -> f64;
}

impl FloatExt for f64 {
    fn sqrt(self) -> f64 {
        libm::sqrt(self)
    }
    fn floor(self) -> f64 {
        libm::floor(self)
    }
    fn ceil(self) -> f64 {
        libm::ceil(self)
    }
    fn trunc(self) -> f64 {
        libm::trunc(self)
    }
    fn round_ties_even(self) -> f64 {
        libm::rint(self)
    }
}

impl FloatExt for f32 {
    fn sqrt(self) -> f32 {
        libm::sqrtf(self)
    }
    fn floor(self) -> f32 {
        libm::floorf(self)
    }
    fn ceil(self) -> f32 {
        libm::ceilf(self)
    }
    fn trunc(self) -> f32 {
        libm::truncf(self)
    }
    fn round_ties_even(self) -> f32 {
        libm::rintf(self)
    }
}

impl Float64Ext for f64 {
    fn powi(self, n: i32) -> f64 {
        libm::pow(self, n as f64)
    }
    fn exp_m1(self) -> f64 {
        libm::expm1(self)
    }
    fn log2(self) -> f64 {
        libm::log2(self)
    }
    fn tan(self) -> f64 {
        libm::tan(self)
    }
    fn sin(self) -> f64 {
        libm::sin(self)
    }
    fn cos(self) -> f64 {
        libm::cos(self)
    }
    fn atan2(self, x: f64) -> f64 {
        libm::atan2(self, x)
    }
    fn powf(self, y: f64) -> f64 {
        libm::pow(self, y)
    }
}
