mod netgw;
mod quick;
mod run;
mod term;
mod usernet;

use std::path::PathBuf;
use std::process::exit;

use clap::{Args, Parser, Subcommand};

use quick::Distro;

#[derive(Parser)]
#[command(name = "uemulator", about = "Minimal x86_64 system emulator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Boot a kernel (bzImage) and optional initramfs
    Boot {
        /// Linux bzImage
        #[arg(long)]
        kernel: PathBuf,
        /// Initial ramdisk
        #[arg(long)]
        initramfs: Option<PathBuf>,
        #[command(flatten)]
        machine: MachineArgs,
    },
    /// Boot a distro's netboot kernel and initramfs, downloading them on first use
    Quick {
        distro: Distro,
        /// Re-download the cached artifacts
        #[arg(long)]
        update: bool,
        /// Size in GiB of the disk created at ~/.uemulator/quick/<distro>/disk.img
        #[arg(long, default_value_t = 8, value_name = "GIB")]
        disk_size: u64,
        #[command(flatten)]
        machine: MachineArgs,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum ConsoleArg {
    Serial,
    /// hvc0; the host terminal size is forwarded and follows resizes
    Virtio,
}

#[derive(Args)]
pub struct MachineArgs {
    /// Guest memory in MiB [default: 2048, 3072 for `quick fedora`]
    #[arg(long, short = 'm')]
    memory: Option<usize>,
    /// Kernel command line
    #[arg(long)]
    cmdline: Option<String>,
    /// Raw disk image as a virtio-blk device; append ",ro" for read-only
    #[arg(long, value_name = "PATH[,ro]")]
    disk: Option<String>,
    /// Attach a virtio-net device with user-mode NAT to the host network (default for `quick`)
    #[arg(long)]
    net_user: bool,
    /// Append to the kernel command line
    #[arg(long, value_name = "ARGS")]
    append: Option<String>,
    /// Attach a virtio-net device answered by a built-in gateway (DHCP, ARP, ICMP echo)
    #[arg(long)]
    net_loop: bool,
    /// Attach a virtio-net device bridged to a UDP socket pair
    #[arg(long, value_name = "LOCAL,REMOTE")]
    net_udp: Option<String>,
    /// Guest console device
    #[arg(long, value_enum, default_value = "serial")]
    console: ConsoleArg,
    /// Run the interpreter only
    #[arg(long)]
    disable_jit: bool,
    /// Run the JIT in lockstep with the interpreter and stop at the first divergence
    #[arg(long)]
    verify: bool,
    /// Dump CPU state and the guest kernel log on exit
    #[arg(long)]
    debug: bool,
    /// Fix the guest RTC to this Unix time
    #[arg(long, value_name = "UNIX")]
    time: Option<u64>,
    /// Pace guest time to the wall clock even when stdin is not a terminal
    #[arg(long)]
    realtime: bool,
    /// Stop the machine after this many seconds
    #[arg(long, value_name = "SECS")]
    timeout: Option<u64>,
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Boot {
            kernel,
            initramfs,
            machine,
        } => read(&kernel).and_then(|k| {
            let i = initramfs.as_ref().map(read).transpose()?;
            run::run(k, i, machine)
        }),
        Cmd::Quick {
            distro,
            update,
            disk_size,
            mut machine,
        } => quick::ensure(distro, update).and_then(|a| {
            let k = read(&a.kernel)?;
            let i = read(&a.initramfs)?;
            if !machine.net_loop && machine.net_udp.is_none() {
                machine.net_user = true;
            }
            let extra = distro.cmdline();
            if !extra.is_empty() {
                let mut a = extra.to_string();
                if let Some(user) = machine.append.take() {
                    a = format!("{a} {user}");
                }
                machine.append = Some(a);
            }
            machine.memory.get_or_insert(distro.default_memory_mib());
            if machine.disk.is_none() {
                let disk = quick::ensure_disk(distro, disk_size)?;
                machine.disk = Some(disk.to_string_lossy().into_owned());
            }
            run::run(k, Some(i), machine)
        }),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        exit(1);
    }
}
