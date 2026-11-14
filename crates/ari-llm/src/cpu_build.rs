//! Which build of llama.cpp's CPU code runs the on-device assistant.
//!
//! On Android arm64, llama.cpp's maths kernels ship as several shared
//! libraries, each compiled for a different ARM generation, and the best
//! one the phone can run is loaded at startup. A build pinned to plain
//! ARMv8.0 left every newer instruction unused: on a Pixel 10 Pro Fold,
//! Medium answered in 8.1 s with it, 3.7 s with dot product (ARMv8.2) and
//! 3.1 s with int8 matrix multiply as well (ARMv8.6), same answers
//! (ari-android#32). Each library reports whether the CPU supports it, so
//! trying them best first and keeping the first that loads cannot pick one
//! that crashes.
//!
//! Everywhere else (desktop, tests, the x86_64 emulator) the CPU code is
//! compiled into the binary for the build machine's own target.

use std::sync::OnceLock;

/// The CPU build in use, for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuBuild {
    /// Short name: "ARMv8.6", "ARMv8.2", "ARMv8.0", or "built in".
    pub name: String,
    /// What the build uses, in words.
    pub description: String,
}

/// The builds Ari ships, best first: (library suffix, name, description).
/// llama.cpp builds four more (ARMv8.2 without half precision, ARMv9 with
/// SVE2 and SME); the app leaves them out of the APK until they are shown
/// to beat these on a phone.
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const VARIANTS: &[(&str, &str, &str)] = &[
    (
        "android_armv8.6_1",
        "ARMv8.6",
        "dot product, half-precision maths and int8 matrix multiply",
    ),
    ("android_armv8.2_2", "ARMv8.2", "dot product and half-precision maths"),
    ("android_armv8.0_1", "ARMv8.0", "baseline, no dot product"),
];

static CHOSEN: OnceLock<Option<CpuBuild>> = OnceLock::new();

/// Load the best CPU build this device supports, once. `None` when none
/// would load, which leaves llama.cpp with no CPU backend: model loads
/// then fail, the same as a missing model, rather than crashing.
pub(crate) fn ensure_loaded() -> Option<&'static CpuBuild> {
    CHOSEN.get_or_init(load).as_ref()
}

/// The CPU build in use, loading it first if nothing has yet.
pub fn cpu_build() -> Option<CpuBuild> {
    ensure_loaded().cloned()
}

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
fn load() -> Option<CpuBuild> {
    for (suffix, name, description) in VARIANTS {
        // A bare file name: Android's linker finds it among the app's own
        // native libraries, inside the APK, with no extraction needed.
        let library = std::ffi::CString::new(format!("libggml-cpu-{suffix}.so")).ok()?;
        // SAFETY: ggml_backend_load takes a NUL-terminated path, returns
        // null for a library that is missing or unsupported here, and
        // registers the backend for the life of the process otherwise.
        let reg = unsafe { llama_cpp_sys_2::ggml_backend_load(library.as_ptr()) };
        if !reg.is_null() {
            return Some(CpuBuild { name: name.to_string(), description: description.to_string() });
        }
    }
    None
}

#[cfg(not(all(target_os = "android", target_arch = "aarch64")))]
fn load() -> Option<CpuBuild> {
    Some(CpuBuild {
        name: "built in".to_string(),
        description: format!("compiled into Ari for {}", std::env::consts::ARCH),
    })
}

/// The CPU features this device reports that matter to llama.cpp, for
/// diagnostics: what it could use, next to [`cpu_build`]'s what it does.
pub fn cpu_features() -> Vec<String> {
    #[cfg(target_arch = "aarch64")]
    {
        // The kernel's own flag names, from /proc/cpuinfo's "Features" line.
        const LABELS: &[(&str, &str)] = &[
            ("asimddp", "dot product"),
            ("asimdhp", "half precision"),
            ("i8mm", "int8 matrix multiply"),
            ("bf16", "bfloat16"),
            ("sve", "SVE"),
            ("sve2", "SVE2"),
            ("sme", "SME"),
        ];
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let flags: Vec<&str> = cpuinfo
            .lines()
            .find(|line| line.starts_with("Features"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, flags)| flags.split_whitespace().collect())
            .unwrap_or_default();
        LABELS
            .iter()
            .filter(|(flag, _)| flags.contains(flag))
            .map(|(_, label)| label.to_string())
            .collect()
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        macro_rules! check {
            ($($feature:tt),* $(,)?) => {
                $(if std::arch::is_x86_feature_detected!($feature) {
                    features.push($feature.to_uppercase());
                })*
            };
        }
        check!("avx", "avx2", "fma", "f16c", "avx512f");
        features
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_desktop_build_reports_its_cpu_code_as_built_in() {
        // Tests run on the build machine, never Android arm64.
        let build = cpu_build().expect("a built-in build always loads");
        assert_eq!(build.name, "built in");
        assert!(build.description.contains(std::env::consts::ARCH));
    }
}
