//! The `alloc` names `std` puts in scope implicitly, for the `no_std` UEFI build. Modules that
//! need them `use crate::prelude::*` under `cfg(freestanding)`.
pub(crate) use alloc::boxed::Box;
pub(crate) use alloc::string::{String, ToString};
pub(crate) use alloc::vec::Vec;
pub(crate) use alloc::{format, vec};
