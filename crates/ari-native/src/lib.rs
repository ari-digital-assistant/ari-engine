//! Binding C libraries that Ari opens at run time rather than links: LiteRT-LM,
//! the voice engines, and sherpa-onnx's C API. Each is declared once with
//! [`c_api!`], which loads every function it names together, so a library
//! missing one fails as a whole when it is opened rather than at the first call.

pub use libloading;

/// Declares a library's C functions and loads them all at once.
///
/// ```ignore
/// ari_native::c_api! {
///     struct Api in "liblitert-lm.so", prefix "litert_lm_" {
///         engine_delete: fn(*mut c_void);
///     }
/// }
/// ```
///
/// Each field is the C name without its prefix. `Api::bind` takes the opened
/// library and returns the table, or which function was missing.
#[macro_export]
macro_rules! c_api {
    (
        $(#[$meta:meta])*
        $vis:vis struct $api:ident in $library:literal, prefix $prefix:literal {
            $($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*
        }
    ) => {
        $(#[$meta])*
        $vis struct $api {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
            _library: $crate::libloading::Library,
        }

        impl $api {
            /// SAFETY: `library` must be the library this table describes,
            /// whose functions have the signatures above.
            unsafe fn bind(library: $crate::libloading::Library) -> Result<Self, String> {
                $(
                    // SAFETY: the caller vouches for the signature.
                    let $name = *unsafe {
                        library.get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                            concat!($prefix, stringify!($name)),
                        )
                    }
                    .map_err(|e| format!(concat!($library, " has no ", $prefix, "{}: {}"), stringify!($name), e))?;
                )*
                Ok($api { $($name,)* _library: library })
            }
        }
    };
}
