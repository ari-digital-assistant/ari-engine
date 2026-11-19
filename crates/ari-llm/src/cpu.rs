//! The CPU the on-device assistant runs on, for diagnostics.

/// The CPU features this device reports that LiteRT-LM's CPU kernels
/// (XNNPACK) can use, for diagnostics.
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
