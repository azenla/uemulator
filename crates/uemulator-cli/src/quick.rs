use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::ValueEnum;

#[derive(Clone, Copy, ValueEnum)]
pub enum Distro {
    Debian,
    Alpine,
    Fedora,
}

pub struct Artifacts {
    pub kernel: PathBuf,
    pub initramfs: PathBuf,
}

struct Source {
    kernel: (&'static str, &'static str),
    initramfs: (&'static str, &'static str),
}

impl Distro {
    fn name(self) -> &'static str {
        match self {
            Distro::Debian => "debian",
            Distro::Alpine => "alpine",
            Distro::Fedora => "fedora",
        }
    }

    // (cached file name, URL)
    fn source(self) -> Source {
        match self {
            Distro::Debian => Source {
                kernel: (
                    "linux",
                    "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/linux",
                ),
                initramfs: (
                    "initrd.gz",
                    "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/initrd.gz",
                ),
            },
            // boot.alpinelinux.org only serves iPXE images; the netboot files live on the CDN.
            Distro::Alpine => Source {
                kernel: (
                    "vmlinuz-virt",
                    "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/vmlinuz-virt",
                ),
                initramfs: (
                    "initramfs-virt",
                    "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/initramfs-virt",
                ),
            },
            Distro::Fedora => Source {
                kernel: (
                    "vmlinuz",
                    "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/vmlinuz",
                ),
                initramfs: (
                    "initrd.img",
                    "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/initrd.img",
                ),
            },
        }
    }
}

pub fn cache_root() -> Result<PathBuf, String> {
    if let Ok(d) = std::env::var("UEMULATOR_CACHE") {
        return Ok(d.into());
    }
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    Ok(Path::new(&home).join(".cache/uemulator"))
}

pub fn ensure(distro: Distro, update: bool) -> Result<Artifacts, String> {
    let dir = cache_root()?
        .join("quick")
        .join(distro.name())
        .join("artifacts");
    fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let src = distro.source();
    let fetch = |(file, url): (&str, &str)| -> Result<PathBuf, String> {
        let path = dir.join(file);
        let cached = fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false);
        if update || !cached {
            download(url, &path)?;
        }
        Ok(path)
    };
    Ok(Artifacts {
        kernel: fetch(src.kernel)?,
        initramfs: fetch(src.initramfs)?,
    })
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    eprintln!("downloading {url}");
    let part = dest.with_extension("part");
    let res = (|| -> io::Result<()> {
        let resp = ureq::get(url)
            .call()
            .map_err(|e| io::Error::other(e.to_string()))?;
        let mut body = resp.into_body().into_reader();
        let mut out = File::create(&part)?;
        let mut buf = vec![0u8; 1 << 16];
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
