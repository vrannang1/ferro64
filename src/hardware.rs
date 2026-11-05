//! What this machine can run. On a Mac the GPU shares system memory, but
//! macOS only lets it use part of it (about two thirds up to 36 GB, three
//! quarters above, unless `iogpu.wired_limit_mb` raises it). On a PC the GPU
//! has its own memory (VRAM); llama.cpp can spill a model into system memory,
//! which works but is much slower.
use serde::Serialize;
use std::process::Command;

/// Memory is quoted the way Apple and GPU makers do: 1 GB = 1024³ bytes.
const GB: f64 = 1_073_741_824.0;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Hardware {
    /// "mac" (unified memory), "nvidia", "amd" or "cpu".
    pub kind: String,
    pub ram_gb: f64,
    /// Memory the GPU can hold a model in.
    pub gpu_gb: f64,
    pub gpu_name: Option<String>,
}

/// How a model fits this machine.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// Fits in GPU memory with room for its context.
    Good,
    /// Runs, but slowly (spills into system memory) or with little room.
    Tight,
    /// Will not load, or would make the machine swap.
    No,
}

/// Detected once per run.
pub fn this_machine() -> &'static Hardware {
    static HW: std::sync::OnceLock<Hardware> = std::sync::OnceLock::new();
    HW.get_or_init(detect)
}

fn sysctl(name: &str) -> Option<u64> {
    let out = Command::new("/usr/sbin/sysctl").args(["-n", name]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn first_line(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").trim().to_string())
}

/// The GPU's share of a Mac's memory, as macOS sets it.
pub fn mac_gpu_budget(ram_gb: f64, wired_limit_mb: Option<u64>) -> f64 {
    match wired_limit_mb.filter(|mb| *mb > 0) {
        Some(mb) => (mb as f64 / 1024.0).min(ram_gb),
        None if ram_gb <= 36.0 => ram_gb * 2.0 / 3.0,
        None => ram_gb * 0.75,
    }
}

/// FERRO64_HARDWARE pretends to be another machine, to see what it would
/// get: "mac:16" (memory in GB) or "nvidia:4:16" (GPU memory, then system).
/// A name can follow: "mac:16:Apple M2".
fn pretend() -> Option<Hardware> {
    let spec = std::env::var("FERRO64_HARDWARE").ok()?;
    let parts: Vec<&str> = spec.split(':').collect();
    let num = |i: usize| parts.get(i).and_then(|v| v.parse::<f64>().ok());
    match parts.first().copied()? {
        "mac" => {
            let ram = num(1)?;
            let name = parts.get(2).map(|n| n.to_string()).unwrap_or_else(|| format!("a {ram:.0} GB Mac (pretend)"));
            Some(Hardware { kind: "mac".into(), ram_gb: ram, gpu_gb: mac_gpu_budget(ram, None), gpu_name: Some(name) })
        }
        kind @ ("nvidia" | "amd") => {
            let name = parts.get(3).map(|n| n.to_string()).unwrap_or_else(|| format!("a {:.0} GB GPU (pretend)", num(1).unwrap_or(0.0)));
            Some(Hardware { kind: kind.into(), ram_gb: num(2)?, gpu_gb: num(1)?, gpu_name: Some(name) })
        }
        "cpu" => Some(Hardware { kind: "cpu".into(), ram_gb: num(1)?, gpu_gb: 0.0, gpu_name: None }),
        _ => None,
    }
}

pub fn detect() -> Hardware {
    if let Some(hw) = pretend() {
        return hw;
    }
    if cfg!(target_os = "macos") {
        let ram_gb = sysctl("hw.memsize").map(|b| b as f64 / GB).unwrap_or(8.0);
        let chip = first_line("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]);
        return Hardware { kind: "mac".into(), ram_gb, gpu_gb: mac_gpu_budget(ram_gb, sysctl("iogpu.wired_limit_mb")), gpu_name: chip };
    }
    #[cfg(windows)]
    let ram_gb = first_line("powershell.exe", &["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        .and_then(|bytes| bytes.parse::<f64>().ok())
        .map(|bytes| bytes / GB)
        .unwrap_or(8.0);
    #[cfg(not(windows))]
    let ram_gb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|kb| kb.parse::<f64>().ok()))
        .map(|kb| kb * 1024.0 / GB)
        .unwrap_or(8.0);
    // NVIDIA: total VRAM of the first GPU, in MiB.
    if let Some(line) = first_line("nvidia-smi", &["--query-gpu=memory.total,name", "--format=csv,noheader,nounits"]) {
        let mut parts = line.splitn(2, ',');
        if let Some(mib) = parts.next().and_then(|m| m.trim().parse::<f64>().ok()) {
            return Hardware { kind: "nvidia".into(), ram_gb, gpu_gb: mib / 1024.0, gpu_name: parts.next().map(|n| n.trim().to_string()) };
        }
    }
    // AMD: the kernel reports VRAM in bytes per card.
    let amd = std::fs::read_dir("/sys/class/drm").ok().into_iter().flatten().flatten().find_map(|card| {
        std::fs::read_to_string(card.path().join("device/mem_info_vram_total")).ok()?.trim().parse::<f64>().ok()
    });
    if let Some(bytes) = amd.filter(|b| *b > 0.0) {
        return Hardware { kind: "amd".into(), ram_gb, gpu_gb: bytes / GB, gpu_name: None };
    }
    Hardware { kind: "cpu".into(), ram_gb, gpu_gb: 0.0, gpu_name: None }
}

/// Memory a model needs while running: its files plus the context cache and
/// working buffers. An estimate, at the context this machine defaults to
/// (see `default_context`).
pub fn needed_gb(file_bytes: u64) -> f64 {
    let size = file_bytes as f64 / GB;
    size * 1.12 + 0.5
}

/// Context window when LLAMACPP_CTX_SIZE is not set: 32k from an ordinary 16 GB
/// machine up, 64k with a large GPU, and 8k below 16 GB.
pub fn default_context(hw: &Hardware) -> u32 {
    // At 8K a coding run has almost no room to work in.
    if hw.gpu_gb < 12.0 && hw.ram_gb < 16.0 {
        8_192
    } else if hw.gpu_gb < 40.0 {
        32_768
    } else {
        65_536
    }
}

/// How a model of `need` GB fits.
pub fn fit(hw: &Hardware, need: f64) -> Fit {
    if need <= hw.gpu_gb {
        return Fit::Good;
    }
    match hw.kind.as_str() {
        // Unified memory: past the GPU share, raising the limit can work if
        // most of the memory is still free for macOS.
        "mac" if need <= hw.ram_gb * 0.85 - 4.0 => Fit::Tight,
        // A GPU card: llama.cpp can keep part of the model in system memory.
        // A quarter of memory (at least 3 GB) stays for the system.
        "nvidia" | "amd" if need <= hw.gpu_gb + (hw.ram_gb - (hw.ram_gb * 0.25).max(3.0)).max(0.0) * 0.8 => Fit::Tight,
        // No GPU: all of it is already in system memory, with nothing to split.
        "cpu" if need <= hw.ram_gb - (hw.ram_gb * 0.25).max(3.0) => Fit::Tight,
        _ => Fit::No,
    }
}

/// One line explaining a fit, shown under the model.
pub fn explain(hw: &Hardware, need: f64, fit: Fit) -> String {
    let gpu = if hw.kind == "mac" { "this Mac can give the GPU" } else { "this GPU has" };
    match fit {
        Fit::Good => format!("needs about {need:.0} GB; {gpu} {:.0} GB", hw.gpu_gb),
        Fit::Tight if hw.kind == "mac" => format!(
            "needs about {need:.0} GB; {gpu} {:.0} GB. It can run if you raise the GPU limit (see the README), with little memory left for other apps",
            hw.gpu_gb
        ),
        Fit::Tight => format!("needs about {need:.0} GB; {gpu} {:.0} GB, so part runs in system memory and it will be slow", hw.gpu_gb),
        Fit::No => format!("needs about {need:.0} GB; {gpu} {:.0} GB and the machine has {:.0} GB in all. It won't run here", hw.gpu_gb, hw.ram_gb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mac(ram: f64) -> Hardware {
        Hardware { kind: "mac".into(), ram_gb: ram, gpu_gb: mac_gpu_budget(ram, None), gpu_name: None }
    }

    #[test]
    fn mac_gpu_share_follows_macos() {
        assert!((mac_gpu_budget(16.0, None) - 10.67).abs() < 0.01);
        assert!((mac_gpu_budget(64.0, None) - 48.0).abs() < 0.01);
        // A raised limit wins, but never past the memory there is.
        assert!((mac_gpu_budget(64.0, Some(57_344)) - 56.0).abs() < 0.01);
    }

    #[test]
    fn models_fit_the_machines_we_aim_for() {
        let qwen35_35b = needed_gb(22_016_023_168 + 899_283_648);
        let qwen35_9b = needed_gb(5_680_522_464 + 918_166_080);
        let qwen3_4b = needed_gb(2_497_280_256);
        // 64 GB Mac: everything we use.
        assert_eq!(fit(&mac(64.0), qwen35_35b), Fit::Good);
        // 16 GB Mac: the 9B and 4B, not the 35B.
        assert_eq!(fit(&mac(16.0), qwen35_9b), Fit::Good);
        assert_eq!(fit(&mac(16.0), qwen3_4b), Fit::Good);
        assert_eq!(fit(&mac(16.0), qwen35_35b), Fit::No);
        // A 4 GB card with 16 GB of memory: the 4B fits, the 9B spills over.
        let laptop = Hardware { kind: "nvidia".into(), ram_gb: 16.0, gpu_gb: 4.0, gpu_name: None };
        assert_eq!(fit(&laptop, qwen3_4b), Fit::Good);
        assert_eq!(fit(&laptop, qwen35_9b), Fit::Tight);
        assert_eq!(fit(&laptop, qwen35_35b), Fit::No);
        assert!(explain(&mac(16.0), qwen35_35b, Fit::No).contains("won't run"));
        // No GPU at all: the 4B still runs on the CPU, slowly.
        let cpu = Hardware { kind: "cpu".into(), ram_gb: 8.0, gpu_gb: 0.0, gpu_name: None };
        assert_eq!(fit(&cpu, qwen3_4b), Fit::Tight);
        assert_eq!(fit(&cpu, qwen35_9b), Fit::No);
        assert_eq!(default_context(&mac(16.0)), 32_768);
        assert_eq!(default_context(&mac(8.0)), 8_192);
        assert_eq!(default_context(&mac(36.0)), 32_768);
        assert_eq!(default_context(&mac(64.0)), 65_536);
    }

    #[test]
    fn the_default_model_runs_on_a_16_gb_laptop() {
        let laptop = Hardware { kind: "cpu".into(), ram_gb: 16.0, gpu_gb: 0.0, gpu_name: None };
        let need = crate::model_store::memory_needed(crate::model_store::catalog_entry("qwen3.5:9b").unwrap());
        assert!(fit(&laptop, need) == Fit::Tight, "needs {need:.1} GB");
        assert!(fit(&mac(16.0), need) == Fit::Good, "needs {need:.1} GB");
    }
}
