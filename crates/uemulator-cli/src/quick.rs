use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::ValueEnum;

use crate::run::DEFAULT_MEMORY_MIB;

/// The library's upper memory bound: RAM must stay below the 0xC000_0000 PCI/MMIO window.
const MAX_MEMORY_MIB: usize = 3072;
const GIB_SHIFT: u32 = 30;
const DOWNLOAD_CHUNK_LEN: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
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
            // Alpine's virt kernel is built without `earlyprintk=`, so with the default hvc0
            // console everything before the virtio-console driver loads (including the banner)
            // would be lost; `earlycon=` is built in and reaches the same UART.
            Distro::Alpine => {
                "earlycon=uart8250,io,0x3f8 ip=dhcp alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
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
    pub fn name(self) -> &'static str {
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
    home_from(env_var)
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn home_from(env: impl Fn(&str) -> Option<String>) -> Result<PathBuf, String> {
    if let Some(dir) = env("UEMULATOR_HOME") {
        return Ok(dir.into());
    }
    let home = env("HOME").ok_or_else(|| "HOME not set".to_string())?;
    Ok(Path::new(&home).join(".uemulator"))
}

/// Downloads live at `<cache>/quick/<distro>/artifacts/`.
fn cache_root() -> Result<PathBuf, String> {
    cache_root_from(env_var)
}

fn cache_root_from(env: impl Fn(&str) -> Option<String>) -> Result<PathBuf, String> {
    match env("UEMULATOR_CACHE") {
        Some(dir) => Ok(dir.into()),
        None => Ok(home_from(env)?.join("cache")),
    }
}

fn disk_path(home: &Path, distro: Distro) -> PathBuf {
    home.join("quick").join(distro.name()).join("disk.img")
}

fn artifacts_dir(cache_root: &Path, distro: Distro) -> PathBuf {
    cache_root
        .join("quick")
        .join(distro.name())
        .join("artifacts")
}

/// Creates `path` (and its parent directories) as a sparse file of `gib` GiB.
fn create_sparse_disk(path: &Path, gib: u64) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    File::create(path)
        .and_then(|file| file.set_len(gib << GIB_SHIFT))
        .map_err(|e| format!("create {}: {e}", path.display()))
}

/// Per-distro scratch disk, created sparse on first use and never resized afterwards: `gib` only
/// matters for the call that creates the file.
pub fn ensure_disk(distro: Distro, gib: u64) -> Result<PathBuf, String> {
    let path = disk_path(&home()?, distro);
    if !path.exists() {
        create_sparse_disk(&path, gib)?;
        eprintln!("created {gib} GiB disk {}", path.display());
    }
    Ok(path)
}

pub fn ensure(distro: Distro, update: bool) -> Result<Artifacts, String> {
    ensure_in(
        &artifacts_dir(&cache_root()?, distro),
        distro,
        update,
        download,
    )
}

/// A zero-length file counts as missing: it is what a crash between create and first write leaves.
fn needs_download(path: &Path, update: bool) -> bool {
    let cached = fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
    update || !cached
}

/// `ensure` with the transfer injected, so the cache logic runs without a network.
fn ensure_in(
    dir: &Path,
    distro: Distro,
    update: bool,
    mut fetch_url: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Artifacts, String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let src = distro.source();
    let mut fetch = |image: Download| -> Result<PathBuf, String> {
        let path = dir.join(image.file_name);
        if needs_download(&path, update) {
            fetch_url(image.url, &path)?;
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
    let res = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(e.to_string()))
        .and_then(|resp| write_atomically(resp.into_body().into_reader(), dest));
    res.map_err(|e| format!("download of {url} failed: {e}"))
}

fn part_path(dest: &Path) -> PathBuf {
    dest.with_extension("part")
}

/// Copies `body` to `dest` through `part_path(dest)`. On any failure the partial file is removed
/// and `dest` is left as it was.
fn write_atomically(mut body: impl Read, dest: &Path) -> io::Result<()> {
    let part = part_path(dest);
    let res = (|| -> io::Result<()> {
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
    if res.is_err() {
        let _ = fs::remove_file(&part);
    }
    res
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Cursor;

    use super::*;
    use crate::test_support::{EnvGuard, TempDir};

    const ALL_DISTROS: [Distro; 3] = [Distro::Debian, Distro::Alpine, Distro::Fedora];
    const GIB: u64 = 1 << 30;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    /// The release component of `.../releases/<n>/...`.
    fn release_in(url: &str) -> &str {
        let after = url
            .split("/releases/")
            .nth(1)
            .expect("no /releases/ in url");
        after.split('/').next().unwrap()
    }

    #[test]
    fn distro_names_are_the_lowercase_command_line_names() {
        assert_eq!(Distro::Debian.name(), "debian");
        assert_eq!(Distro::Alpine.name(), "alpine");
        assert_eq!(Distro::Fedora.name(), "fedora");
    }

    #[test]
    fn only_fedora_gets_more_than_the_default_memory() {
        assert_eq!(Distro::Debian.default_memory_mib(), DEFAULT_MEMORY_MIB);
        assert_eq!(Distro::Alpine.default_memory_mib(), DEFAULT_MEMORY_MIB);
        assert_eq!(Distro::Fedora.default_memory_mib(), MAX_MEMORY_MIB);
        assert!(Distro::Fedora.default_memory_mib() > DEFAULT_MEMORY_MIB);
    }

    #[test]
    fn max_memory_matches_the_limit_the_library_enforces() {
        let library = |mib: usize| match uemulator::EmulatedMachine::new(
            uemulator::MachineConfig::new().memory_mib(mib),
        ) {
            Ok(_) => panic!("an empty kernel must not boot"),
            Err(e) => e.to_string(),
        };
        assert!(library(MAX_MEMORY_MIB + 1).contains(&format!("{MAX_MEMORY_MIB}")));
        assert!(library(MAX_MEMORY_MIB + 1).contains("between"));
        assert_eq!(GIB_SHIFT, 30);
    }

    #[test]
    fn every_source_url_is_https_and_ends_in_its_cache_file_name() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            for image in [&source.kernel, &source.initramfs] {
                assert!(image.url.starts_with("https://"), "{}", image.url);
                assert!(
                    image.url.ends_with(&format!("/{}", image.file_name)),
                    "{} vs {}",
                    image.url,
                    image.file_name
                );
                assert!(!image.file_name.contains('/'));
                assert!(!image.file_name.is_empty());
            }
        }
    }

    #[test]
    fn kernel_and_initramfs_have_different_names_and_urls_per_distro() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            assert_ne!(source.kernel.file_name, source.initramfs.file_name);
            assert_ne!(source.kernel.url, source.initramfs.url);
        }
    }

    #[test]
    fn part_files_of_one_distro_never_collide() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            let dir = Path::new("/cache");
            assert_ne!(
                part_path(&dir.join(source.kernel.file_name)),
                part_path(&dir.join(source.initramfs.file_name))
            );
        }
    }

    #[test]
    fn debian_files_match_the_netboot_layout() {
        let source = Distro::Debian.source();
        assert_eq!(source.kernel.file_name, "linux");
        assert_eq!(source.initramfs.file_name, "initrd.gz");
        assert!(source.kernel.url.contains("/dists/trixie/"));
        assert!(
            source
                .kernel
                .url
                .contains("/installer-amd64/current/images/netboot/")
        );
        assert!(source.kernel.url.starts_with("https://deb.debian.org/"));
    }

    #[test]
    fn alpine_gets_earlycon_because_its_kernel_has_no_earlyprintk() {
        assert!(
            Distro::Alpine
                .cmdline()
                .contains("earlycon=uart8250,io,0x3f8")
        );
        assert!(!Distro::Debian.cmdline().contains("earlycon"));
    }

    #[test]
    fn alpine_tracks_latest_stable_for_kernel_and_modloop_alike() {
        let source = Distro::Alpine.source();
        assert_eq!(source.kernel.file_name, "vmlinuz-virt");
        assert_eq!(source.initramfs.file_name, "initramfs-virt");
        let cmdline = Distro::Alpine.cmdline();
        for url in [source.kernel.url, source.initramfs.url] {
            assert!(
                url.contains("/alpine/latest-stable/releases/x86_64/netboot/"),
                "{url}"
            );
        }
        assert!(
            cmdline
                .contains("alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main")
        );
        assert!(cmdline.contains(
            "modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
        ));
    }

    #[test]
    fn fedora_repo_release_matches_the_cached_kernel_release() {
        let source = Distro::Fedora.source();
        let kernel_release = release_in(source.kernel.url);
        assert_eq!(kernel_release, release_in(source.initramfs.url));
        let cmdline = Distro::Fedora.cmdline();
        let repo = cmdline
            .split_whitespace()
            .find_map(|arg| arg.strip_prefix("inst.repo="))
            .expect("inst.repo");
        assert_eq!(release_in(repo), kernel_release);
    }

    #[test]
    fn fedora_files_match_the_pxeboot_layout() {
        let source = Distro::Fedora.source();
        assert_eq!(source.kernel.file_name, "vmlinuz");
        assert_eq!(source.initramfs.file_name, "initrd.img");
        assert!(source.kernel.url.contains("/images/pxeboot/"));
    }

    #[test]
    fn network_installers_configure_the_nic_with_dhcp_in_the_initramfs() {
        assert!(
            Distro::Alpine
                .cmdline()
                .split_whitespace()
                .any(|a| a == "ip=dhcp")
        );
        assert!(
            Distro::Fedora
                .cmdline()
                .split_whitespace()
                .any(|a| a == "ip=dhcp")
        );
        assert!(Distro::Fedora.cmdline().contains("rd.neednet=1"));
        assert!(Distro::Fedora.cmdline().contains("inst.text"));
    }

    #[test]
    fn debian_needs_no_extra_kernel_arguments() {
        assert_eq!(Distro::Debian.cmdline(), "");
    }

    #[test]
    fn cmdlines_have_no_stray_whitespace() {
        for distro in ALL_DISTROS {
            let cmdline = distro.cmdline();
            assert_eq!(cmdline, cmdline.trim());
            assert!(!cmdline.contains("  "));
        }
    }

    #[test]
    fn home_prefers_uemulator_home() {
        let env = [("UEMULATOR_HOME", "/data/emu"), ("HOME", "/home/me")];
        assert_eq!(home_from(env_of(&env)).unwrap(), PathBuf::from("/data/emu"));
    }

    #[test]
    fn home_falls_back_to_dot_uemulator_under_home() {
        let env = [("HOME", "/home/me")];
        assert_eq!(
            home_from(env_of(&env)).unwrap(),
            PathBuf::from("/home/me/.uemulator")
        );
    }

    #[test]
    fn home_without_any_variable_is_an_error() {
        assert_eq!(home_from(env_of(&[])).unwrap_err(), "HOME not set");
    }

    #[test]
    fn uemulator_home_works_without_home() {
        let env = [("UEMULATOR_HOME", "C:\\emu")];
        assert_eq!(home_from(env_of(&env)).unwrap(), PathBuf::from("C:\\emu"));
    }

    #[test]
    fn cache_defaults_to_the_cache_directory_inside_home() {
        let env = [("UEMULATOR_HOME", "/data/emu")];
        assert_eq!(
            cache_root_from(env_of(&env)).unwrap(),
            PathBuf::from("/data/emu/cache")
        );
        let env = [("HOME", "/home/me")];
        assert_eq!(
            cache_root_from(env_of(&env)).unwrap(),
            PathBuf::from("/home/me/.uemulator/cache")
        );
    }

    #[test]
    fn uemulator_cache_overrides_only_the_cache() {
        let env = [
            ("UEMULATOR_CACHE", "/fast/cache"),
            ("UEMULATOR_HOME", "/data/emu"),
        ];
        assert_eq!(
            cache_root_from(env_of(&env)).unwrap(),
            PathBuf::from("/fast/cache")
        );
        assert_eq!(home_from(env_of(&env)).unwrap(), PathBuf::from("/data/emu"));
    }

    #[test]
    fn uemulator_cache_needs_no_home() {
        let env = [("UEMULATOR_CACHE", "/fast/cache")];
        assert_eq!(
            cache_root_from(env_of(&env)).unwrap(),
            PathBuf::from("/fast/cache")
        );
    }

    #[test]
    fn cache_root_without_any_variable_is_an_error() {
        assert_eq!(cache_root_from(env_of(&[])).unwrap_err(), "HOME not set");
    }

    #[test]
    fn disk_and_artifact_paths_are_per_distro() {
        let home = Path::new("/h");
        assert_eq!(
            disk_path(home, Distro::Debian),
            PathBuf::from("/h/quick/debian/disk.img")
        );
        assert_eq!(
            disk_path(home, Distro::Fedora),
            PathBuf::from("/h/quick/fedora/disk.img")
        );
        let cache = Path::new("/c");
        assert_eq!(
            artifacts_dir(cache, Distro::Alpine),
            PathBuf::from("/c/quick/alpine/artifacts")
        );
        let all: std::collections::HashSet<_> =
            ALL_DISTROS.iter().map(|d| disk_path(home, *d)).collect();
        assert_eq!(all.len(), ALL_DISTROS.len());
    }

    #[test]
    fn real_home_reads_the_process_environment() {
        let _env = EnvGuard::set(&[
            ("UEMULATOR_HOME", Some("/tmp/uemulator-env-home")),
            ("UEMULATOR_CACHE", None),
        ]);
        assert_eq!(home().unwrap(), PathBuf::from("/tmp/uemulator-env-home"));
        assert_eq!(
            cache_root().unwrap(),
            PathBuf::from("/tmp/uemulator-env-home/cache")
        );
    }

    #[test]
    fn real_cache_root_honours_uemulator_cache() {
        let _env = EnvGuard::set(&[
            ("UEMULATOR_HOME", Some("/tmp/uemulator-env-home")),
            ("UEMULATOR_CACHE", Some("/tmp/uemulator-env-cache")),
        ]);
        assert_eq!(
            cache_root().unwrap(),
            PathBuf::from("/tmp/uemulator-env-cache")
        );
        assert_eq!(home().unwrap(), PathBuf::from("/tmp/uemulator-env-home"));
    }

    #[test]
    fn real_home_falls_back_to_home_variable() {
        let _env = EnvGuard::set(&[
            ("UEMULATOR_HOME", None),
            ("UEMULATOR_CACHE", None),
            ("HOME", Some("/tmp/uemulator-fake-home")),
        ]);
        assert_eq!(
            home().unwrap(),
            PathBuf::from("/tmp/uemulator-fake-home/.uemulator")
        );
    }

    #[test]
    fn env_guard_restores_previous_values() {
        {
            let _outer = EnvGuard::set(&[("UEMULATOR_HOME", Some("/tmp/one"))]);
            assert_eq!(std::env::var("UEMULATOR_HOME").unwrap(), "/tmp/one");
        }
        let _again = EnvGuard::set(&[]);
        assert_ne!(
            std::env::var("UEMULATOR_HOME").ok().as_deref(),
            Some("/tmp/one")
        );
    }

    #[test]
    fn sparse_disk_has_the_requested_length_without_allocating_it() {
        let dir = TempDir::new("quick-sparse");
        let path = dir.path().join("nested").join("deeper").join("disk.img");
        create_sparse_disk(&path, 2).unwrap();
        let meta = fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 2 * GIB);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            const BLOCK_BYTES: u64 = 512;
            assert!(meta.blocks() * BLOCK_BYTES < GIB, "file is not sparse");
        }
    }

    #[test]
    fn sparse_disk_of_zero_gib_is_empty() {
        let dir = TempDir::new("quick-sparse0");
        let path = dir.path().join("disk.img");
        create_sparse_disk(&path, 0).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn sparse_disk_creation_reports_the_path_on_failure() {
        let dir = TempDir::new("quick-sparse-fail");
        let blocker = dir.path().join("file");
        fs::write(&blocker, b"x").unwrap();
        let err = create_sparse_disk(&blocker.join("disk.img"), 1).unwrap_err();
        assert!(err.starts_with("create "), "{err}");
    }

    #[test]
    fn ensure_disk_creates_it_under_home_and_never_resizes_it() {
        let dir = TempDir::new("quick-ensure-disk");
        let home = dir.path().to_string_lossy().into_owned();
        let _env = EnvGuard::set(&[("UEMULATOR_HOME", Some(&home))]);
        let path = ensure_disk(Distro::Alpine, 1).unwrap();
        assert_eq!(path, dir.path().join("quick/alpine/disk.img"));
        assert_eq!(fs::metadata(&path).unwrap().len(), GIB);

        fs::write(&path, b"guest data").unwrap();
        let again = ensure_disk(Distro::Alpine, 4).unwrap();
        assert_eq!(again, path);
        assert_eq!(fs::read(&path).unwrap(), b"guest data");
    }

    #[test]
    fn ensure_disk_keeps_distros_apart() {
        let dir = TempDir::new("quick-ensure-disk2");
        let home = dir.path().to_string_lossy().into_owned();
        let _env = EnvGuard::set(&[("UEMULATOR_HOME", Some(&home))]);
        let debian = ensure_disk(Distro::Debian, 1).unwrap();
        let fedora = ensure_disk(Distro::Fedora, 2).unwrap();
        assert_ne!(debian, fedora);
        assert_eq!(fs::metadata(&debian).unwrap().len(), GIB);
        assert_eq!(fs::metadata(&fedora).unwrap().len(), 2 * GIB);
    }

    #[test]
    fn needs_download_when_missing_empty_or_updating() {
        let dir = TempDir::new("quick-needs");
        let missing = dir.path().join("missing");
        let empty = dir.path().join("empty");
        let full = dir.path().join("full");
        fs::write(&empty, b"").unwrap();
        fs::write(&full, b"x").unwrap();
        assert!(needs_download(&missing, false));
        assert!(needs_download(&empty, false));
        assert!(!needs_download(&full, false));
        assert!(needs_download(&full, true));
        assert!(needs_download(&missing, true));
    }

    /// Records every requested URL and writes the URL itself as the file contents.
    struct FakeFetcher {
        urls: RefCell<Vec<String>>,
        fail_on: Option<&'static str>,
    }

    impl FakeFetcher {
        fn new() -> Self {
            FakeFetcher {
                urls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }

        fn fetch(&self, url: &str, dest: &Path) -> Result<(), String> {
            self.urls.borrow_mut().push(url.to_string());
            if self.fail_on.is_some_and(|needle| url.contains(needle)) {
                return Err(format!("download of {url} failed: injected"));
            }
            fs::write(dest, url).map_err(|e| e.to_string())
        }

        fn urls(&self) -> Vec<String> {
            self.urls.borrow().clone()
        }
    }

    #[test]
    fn first_ensure_fetches_kernel_then_initramfs_into_the_artifact_dir() {
        let dir = TempDir::new("quick-ensure");
        let artifacts_dir = dir.path().join("a").join("b");
        let fetcher = FakeFetcher::new();
        let artifacts = ensure_in(&artifacts_dir, Distro::Debian, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        let source = Distro::Debian.source();
        assert_eq!(
            fetcher.urls(),
            vec![source.kernel.url, source.initramfs.url]
        );
        assert_eq!(artifacts.kernel, artifacts_dir.join("linux"));
        assert_eq!(artifacts.initramfs, artifacts_dir.join("initrd.gz"));
        assert_eq!(
            fs::read_to_string(&artifacts.kernel).unwrap(),
            source.kernel.url
        );
    }

    #[test]
    fn cached_artifacts_are_not_fetched_again() {
        let dir = TempDir::new("quick-cached");
        let fetcher = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.urls().len(), 2);
    }

    #[test]
    fn update_refetches_both_artifacts() {
        let dir = TempDir::new("quick-update");
        let fetcher = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Fedora, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        ensure_in(dir.path(), Distro::Fedora, true, |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(fetcher.urls().len(), 4);
    }

    #[test]
    fn a_zero_length_cached_file_is_fetched_again() {
        let dir = TempDir::new("quick-zero");
        let fetcher = FakeFetcher::new();
        fs::write(dir.path().join("vmlinuz-virt"), b"kernel").unwrap();
        fs::write(dir.path().join("initramfs-virt"), b"").unwrap();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.urls(), vec![Distro::Alpine.source().initramfs.url]);
        assert_eq!(
            fs::read(dir.path().join("vmlinuz-virt")).unwrap(),
            b"kernel"
        );
    }

    #[test]
    fn a_failed_kernel_download_stops_before_the_initramfs() {
        let dir = TempDir::new("quick-fail");
        let mut fetcher = FakeFetcher::new();
        fetcher.fail_on = Some("vmlinuz");
        let err = ensure_in(dir.path(), Distro::Fedora, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .err()
        .unwrap();
        assert!(err.contains("injected"), "{err}");
        assert_eq!(fetcher.urls().len(), 1);
    }

    #[test]
    fn a_failed_initramfs_download_keeps_the_kernel_for_next_time() {
        let dir = TempDir::new("quick-fail2");
        let mut fetcher = FakeFetcher::new();
        fetcher.fail_on = Some("initrd.gz");
        assert!(
            ensure_in(dir.path(), Distro::Debian, false, |u, d| fetcher
                .fetch(u, d))
            .is_err()
        );
        assert!(dir.path().join("linux").exists());
        assert!(!dir.path().join("initrd.gz").exists());
        let retry = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Debian, false, |u, d| retry.fetch(u, d)).unwrap();
        assert_eq!(retry.urls(), vec![Distro::Debian.source().initramfs.url]);
    }

    #[test]
    fn ensure_fails_cleanly_when_the_cache_directory_cannot_be_created() {
        let dir = TempDir::new("quick-nodir");
        let blocker = dir.path().join("file");
        fs::write(&blocker, b"x").unwrap();
        let fetcher = FakeFetcher::new();
        let err = ensure_in(&blocker.join("sub"), Distro::Debian, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .err()
        .unwrap();
        assert!(err.starts_with("create "), "{err}");
        assert!(fetcher.urls().is_empty());
    }

    #[test]
    fn ensure_uses_the_cache_override_without_touching_the_network_when_cached() {
        let cache = TempDir::new("quick-real-ensure");
        let cache_path = cache.path().to_string_lossy().into_owned();
        let _env = EnvGuard::set(&[("UEMULATOR_CACHE", Some(&cache_path))]);
        let dir = cache.path().join("quick/alpine/artifacts");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("vmlinuz-virt"), b"k").unwrap();
        fs::write(dir.join("initramfs-virt"), b"i").unwrap();
        let artifacts = ensure(Distro::Alpine, false).unwrap();
        assert_eq!(artifacts.kernel, dir.join("vmlinuz-virt"));
        assert_eq!(artifacts.initramfs, dir.join("initramfs-virt"));
    }

    #[test]
    fn part_file_sits_next_to_the_destination_with_a_part_extension() {
        assert_eq!(
            part_path(Path::new("/c/linux")),
            PathBuf::from("/c/linux.part")
        );
        assert_eq!(
            part_path(Path::new("/c/initrd.gz")),
            PathBuf::from("/c/initrd.part")
        );
        assert_eq!(
            part_path(Path::new("/c/x")).parent(),
            Path::new("/c/x").parent()
        );
    }

    #[test]
    fn atomic_write_copies_everything_and_leaves_no_part_file() {
        let dir = TempDir::new("quick-atomic");
        let dest = dir.path().join("linux");
        let body: Vec<u8> = (0..(3 * DOWNLOAD_CHUNK_LEN + 7))
            .map(|i| (i % 251) as u8)
            .collect();
        write_atomically(Cursor::new(body.clone()), &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn atomic_write_of_an_empty_body_creates_an_empty_file() {
        let dir = TempDir::new("quick-atomic-empty");
        let dest = dir.path().join("empty");
        write_atomically(Cursor::new(Vec::new()), &dest).unwrap();
        assert_eq!(fs::metadata(&dest).unwrap().len(), 0);
        assert!(
            needs_download(&dest, false),
            "an empty result is treated as missing"
        );
    }

    #[test]
    fn atomic_write_replaces_an_existing_file() {
        let dir = TempDir::new("quick-atomic-replace");
        let dest = dir.path().join("linux");
        fs::write(&dest, b"old contents that are longer").unwrap();
        write_atomically(Cursor::new(b"new".to_vec()), &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new");
    }

    struct BreaksAfter {
        remaining: Vec<u8>,
    }

    impl Read for BreaksAfter {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.remaining.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "connection reset",
                ));
            }
            let n = self.remaining.len().min(buf.len());
            buf[..n].copy_from_slice(&self.remaining[..n]);
            self.remaining.drain(..n);
            Ok(n)
        }
    }

    #[test]
    fn interrupted_download_leaves_neither_part_file_nor_destination() {
        let dir = TempDir::new("quick-atomic-fail");
        let dest = dir.path().join("linux");
        let err = write_atomically(
            BreaksAfter {
                remaining: vec![7; 1000],
            },
            &dest,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionReset);
        assert!(!part_path(&dest).exists());
        assert!(!dest.exists(), "ensure must not find a truncated artifact");
    }

    #[test]
    fn interrupted_download_keeps_the_previous_good_file() {
        let dir = TempDir::new("quick-atomic-keep");
        let dest = dir.path().join("linux");
        fs::write(&dest, b"good").unwrap();
        assert!(
            write_atomically(
                BreaksAfter {
                    remaining: vec![7; 10]
                },
                &dest
            )
            .is_err()
        );
        assert_eq!(fs::read(&dest).unwrap(), b"good");
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn atomic_write_into_a_missing_directory_fails() {
        let dir = TempDir::new("quick-atomic-nodir");
        let dest = dir.path().join("missing").join("linux");
        assert!(write_atomically(Cursor::new(b"x".to_vec()), &dest).is_err());
    }

    #[test]
    fn env_helper_ignores_variables_that_are_not_set() {
        let _env = EnvGuard::set(&[("UEMULATOR_CACHE", None)]);
        assert_eq!(env_var("UEMULATOR_CACHE"), None);
    }
}
