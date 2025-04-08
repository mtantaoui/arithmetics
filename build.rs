use std::cmp::Ordering;
use std::env;
use std::process::Command;

// CPU features we want to detect
#[derive(PartialEq, Eq, Debug)]
struct CpuFeature {
    name: &'static str,
    rustc_flag: &'static str,
    cfg_flag: &'static str,
    detected: bool,
}

impl CpuFeature {
    // define priority order between CPU Features
    fn priority(&self) -> usize {
        match self.name {
            "avx512f" => 0,
            "avx2" => 1,
            "sse41" => 2,
            _ => usize::MAX, // lowest priority by default
        }
    }

    // Groups all supported CPU features that uses optimizations in this crate
    // TODO: Do I have to define another method for ARM architectures ???
    fn features() -> Vec<CpuFeature> {
        let features = vec![
            CpuFeature {
                name: "sse41",
                rustc_flag: "+sse4.1",
                cfg_flag: "sse",
                detected: false,
            },
            CpuFeature {
                name: "avx512f",
                rustc_flag: "+avx512f",
                cfg_flag: "avx512",
                detected: false,
            },
            CpuFeature {
                name: "avx2",
                rustc_flag: "+avx2",
                cfg_flag: "avx2",
                detected: false,
            },
        ];

        features
    }
}

impl Ord for CpuFeature {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl PartialOrd for CpuFeature {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() {
    // Define the CPU features we're interested in (in order of preference)
    let mut features = CpuFeature::features();

    // Determine if we're cross-compiling
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let is_native_build = host == target;

    // Only run CPU detection for native builds
    // CPU features of build machine may be different from target machine
    if is_native_build {
        detect_cpu_features(&mut features);
    }

    // Set cargo flags for conditional compilation
    emit_cargo_config(&mut features);
}

fn detect_cpu_features(features: &mut [CpuFeature]) {
    // Detect features based on the current OS
    if cfg!(target_os = "linux") {
        detect_features_linux(features);
    } else if cfg!(target_os = "windows") {
        todo!();
    } else if cfg!(target_os = "macos") {
        detect_features_macos(features);
    }
}

fn detect_features_linux(features: &mut [CpuFeature]) {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for feature in features.iter_mut() {
            feature.detected = cpuinfo.contains(feature.name);
        }
    }
}

fn detect_features_macos(features: &mut [CpuFeature]) {
    let output = Command::new("sysctl").args(["-a"]).output();

    if let Ok(output) = output {
        let sysctl_output = String::from_utf8_lossy(&output.stdout).to_lowercase();

        for feature in features.iter_mut() {
            match feature.name {
                "avx512f" => feature.detected = sysctl_output.contains("avx512f"),
                "avx2" => feature.detected = sysctl_output.contains("avx2"),
                "sse41" => feature.detected = sysctl_output.contains("sse4.1"),
                _ => {}
            }
        }
    }
}

fn emit_cargo_config(features: &mut [CpuFeature]) {
    // sorting features by priority
    features.sort();

    // Find and use the highest detected feature (if any)
    // if no feature is detected, use baseline implementation
    let cfg_flag = features
        .iter()
        .find(|cpu_feature| cpu_feature.detected)
        .map(|cpu_feature| {
            println!("cargo:rustc-flag=-C");
            println!("cargo:rustc-flag=target-feature={}", cpu_feature.rustc_flag);
            cpu_feature.cfg_flag
        })
        .unwrap_or_else(|| "baseline");

    features.iter().for_each(|cpu_feature| {
        // Avoid `#[cfg(...)]` warning by registering it as a known config
        println!("cargo::rustc-check-cfg=cfg({})", cpu_feature.cfg_flag);
    });

    // Avoid `#[cfg(...)]` warning by registering it as a known config
    println!("cargo::rustc-check-cfg=cfg(baseline)");

    println!("cargo:rustc-cfg={}", cfg_flag);
}
