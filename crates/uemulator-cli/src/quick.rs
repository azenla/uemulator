use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::ValueEnum;

use crate::run::DEFAULT_MEMORY_MIB;

/// The library's upper memory bound: RAM must stay below the 0xC000_0000 PCI/MMIO window.
const MAX_MEMORY_MIB: usize = 3072;
const GIB_SHIFT: u32 = 30;
const DOWNLOAD_CHUNK_LEN: usize = 1 << 16;

#[derive(Clone, Copy, ValueEnum)]
pub enum Distro {
    Debian,
    Alpine,
    Fedora,
}

impl Distro {
    /// Fedora's stage2 image (~900 MB) is downloaded into RAM and must fit beside Anaconda, so it
    /// gets the maximum the library allows.
    pub fn default_memory_mib(self) -> usize {
        match self {
            Distro::Fedora => MAX_MEMORY_MIB,
            _ => DEFAULT_MEMORY_MIB,
        }
    }

    /// Kernel arguments that make the netboot image reach its installer over the user-mode network.
    /// `ip=dhcp` configures the NIC in the initramfs; Alpine's `modloop` must come from the same
    /// release as the cached kernel, so `--update` refreshes both together.
    pub fn cmdline(self) -> &'static str {
        match self {
            Distro::Debian => "",
            Distro::Alpine => {
                "ip=dhcp alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
            }
            Distro::Fedora => {
                "ip=dhcp rd.neednet=1 inst.text inst.repo=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/"
            }
        }
    }
}

pub struct Artifacts {
    pub kernel: PathBuf,
    pub initramfs: PathBuf,
}

struct Download {
    /// Name under the artifacts cache directory.
    file_name: &'static str,
    url: &'static str,
}

struct NetbootImages {
    kernel: Download,
    initramfs: Download,
}

impl Distro {
    fn name(self) -> &'static str {
        match self {
            Distro::Debian => "debian",
            Distro::Alpine => "alpine",
            Distro::Fedora => "fedora",
        }
    }

    /// Debian and Fedora URLs are pinned to a release (Debian `trixie`, Fedora 44) and must be
    /// bumped by hand; Alpine tracks `latest-stable`.
    fn source(self) -> NetbootImages {
        match self {
            Distro::Debian => NetbootImages {
                kernel: Download {
                    file_name: "linux",
                    url: "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/linux",
                },
                initramfs: Download {
                    file_name: "initrd.gz",
                    url: "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/initrd.gz",
                },
            },
            // boot.alpinelinux.org only serves iPXE images; the netboot files live on the CDN.
            Distro::Alpine => NetbootImages {
                kernel: Download {
                    file_name: "vmlinuz-virt",
                    url: "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/vmlinuz-virt",
                },
                initramfs: Download {
                    file_name: "initramfs-virt",
                    url: "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/initramfs-virt",
                },
            },
            Distro::Fedora => NetbootImages {
                kernel: Download {
                    file_name: "vmlinuz",
                    url: "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/vmlinuz",
                },
                initramfs: Download {
                    file_name: "initrd.img",
                    url: "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/initrd.img",
                },
            },
        }
    }
}

/// Root of all CLI state: `$UEMULATOR_HOME`, else `~/.uemulator`. Holds `quick/<distro>/disk.img`
/// and, unless `UEMULATOR_CACHE` overrides it, the download cache in `cache/`. `HOME` is unix
/// only, so quick mode needs `UEMULATOR_HOME` on Windows.
fn home() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("UEMULATOR_HOME") {
        return Ok(dir.into());
    }
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    Ok(Path::new(&home).join(".uemulator"))
}

/// Downloads live at `<cache>/quick/<distro>/artifacts/`.
fn cache_root() -> Result<PathBuf, String> {
    match std::env::var("UEMULATOR_CACHE") {
        Ok(dir) => Ok(dir.into()),
        Err(_) => Ok(home()?.join("cache")),
    }
}

/// Per-distro scratch disk, created sparse on first use and never resized afterwards: `gib` only
/// matters for the call that creates the file.
pub fn ensure_disk(distro: Distro, gib: u64) -> Result<PathBuf, String> {
    let dir = home()?.join("quick").join(distro.name());
    let path = dir.join("disk.img");
    if !path.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        File::create(&path)
            .and_then(|file| file.set_len(gib << GIB_SHIFT))
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        eprintln!("created {gib} GiB disk {}", path.display());
    }
    Ok(path)
}

pub fn ensure(distro: Distro, update: bool) -> Result<Artifacts, String> {
    let dir = cache_root()?
        .join("quick")
        .join(distro.name())
        .join("artifacts");
    fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let src = distro.source();
    let fetch = |image: Download| -> Result<PathBuf, String> {
        let path = dir.join(image.file_name);
        // A zero-length file counts as missing.
        let cached = fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false);
        if update || !cached {
            download(image.url, &path)?;
        }
        Ok(path)
    };
    Ok(Artifacts {
        kernel: fetch(src.kernel)?,
        initramfs: fetch(src.initramfs)?,
    })
}

/// Streams to a `.part` file and renames it into place, so an interrupted download never leaves a
/// truncated artifact that `ensure` would trust.
fn download(url: &str, dest: &Path) -> Result<(), String> {
    eprintln!("downloading {url}");
    let part = dest.with_extension("part");
    let res = (|| -> io::Result<()> {
        let resp = ureq::get(url)
            .call()
            .map_err(|e| io::Error::other(e.to_string()))?;
        let mut body = resp.into_body().into_reader();
        let mut out = File::create(&part)?;
        let mut buf = vec![0u8; DOWNLOAD_CHUNK_LEN];
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
        }
        out.sync_all()?;
        fs::rename(&part, dest)
    })();
    res.map_err(|e| {
        let _ = fs::remove_file(&part);
        format!("download of {url} failed: {e}")
    })
}
