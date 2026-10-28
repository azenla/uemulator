mod netgw;
mod quick;
mod run;
mod term;
#[cfg(test)]
mod test_support;
mod usernet;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use uemulator::ExitReason;

use quick::Distro;

const DEFAULT_DISK_GIB: u64 = 8;
/// The machine ran and ended in a way that is not a failure of the emulator or the guest.
const EXIT_SUCCESS: i32 = 0;
/// Setup failed before or while starting the machine: bad arguments, unreadable files, failed
/// downloads, configuration the library rejected.
const EXIT_ERROR: i32 = 1;
/// The machine started and then failed: a guest triple fault or an emulator-internal error
/// (including a JIT divergence under `--verify`).
const EXIT_MACHINE_FAILURE: i32 = 2;

const EXIT_CODES_HELP: &str = "\
Exit codes:
  0  the machine ended by itself (guest power-off, or reset with --no-reboot), or was stopped
     with Ctrl-] or --timeout
  1  setup failed (bad arguments, unreadable file, download failure, rejected configuration)
  2  the machine failed: guest triple fault or emulator-internal error (e.g. --verify divergence)";

/// Maps how the machine ended to the process exit status. A power-off is how a guest finishes
/// cleanly; a reset only reaches here with `--no-reboot` and counts as success because Linux
/// guests reboot on panic with `panic=-1` and a clean reboot is how an installer finishes.
/// Stopping the machine (Ctrl-] or `--timeout`) is a deliberate host action, so it is success too;
/// callers that need to tell a timeout from a normal end can run without `--timeout`.
fn exit_code(reason: &ExitReason) -> i32 {
    match reason {
        ExitReason::Shutdown | ExitReason::Reset | ExitReason::Stopped => EXIT_SUCCESS,
        ExitReason::TripleFault | ExitReason::Internal(_) => EXIT_MACHINE_FAILURE,
    }
}

#[derive(Parser)]
#[command(
    name = "uemulator",
    about = "Minimal x86_64 system emulator",
    after_help = EXIT_CODES_HELP
)]
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
    /// Boot a firmware image (a BIOS, or a real-mode program) at the reset vector in real mode
    Firmware {
        /// Image placed so that it ends at 1 MiB; execution starts at F000:FFF0
        image: PathBuf,
        #[command(flatten)]
        machine: MachineArgs,
    },
    /// Boot a distro's netboot kernel and initramfs, downloading them on first use
    Quick {
        distro: Distro,
        /// Re-download the cached artifacts
        #[arg(long)]
        update: bool,
        /// Size in GiB of the sparse disk created at ~/.uemulator/quick/<distro>/disk.img; ignored
        /// if the file exists or `--disk` is given
        #[arg(long, default_value_t = DEFAULT_DISK_GIB, value_name = "GIB")]
        disk_size: u64,
        #[command(flatten)]
        machine: MachineArgs,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum ConsoleArg {
    /// ttyS0 on the 16550; for kernels without a built-in virtio_console driver
    Serial,
    /// hvc0; the host terminal size is forwarded and follows resizes
    Virtio,
}

#[derive(Args)]
pub struct MachineArgs {
    /// Guest memory in MiB, 64 to 3072 [default: 2048, 3072 for `quick fedora`]
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
    /// Attach a virtio-net device answered by a built-in gateway that only replies to ARP and ICMP
    /// echo for 10.0.2.2: no DHCP, nothing is forwarded
    #[arg(long)]
    net_loop: bool,
    /// Attach a virtio-net device bridged to a UDP socket pair
    #[arg(long, value_name = "LOCAL,REMOTE")]
    net_udp: Option<String>,
    /// Guest console device
    #[arg(long, value_enum, default_value = "virtio")]
    console: ConsoleArg,
    /// End the run with "machine reset" when the guest resets instead of rebooting it in place
    #[arg(long)]
    no_reboot: bool,
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

/// Quick mode's defaults for everything the user did not choose.
fn apply_quick_defaults(machine: &mut MachineArgs, distro: Distro) {
    // The installers fetch packages over the network, so default to NAT unless the user picked
    // another backend.
    if !machine.net_loop && machine.net_udp.is_none() {
        machine.net_user = true;
    }
    let distro_args = distro.cmdline();
    if !distro_args.is_empty() {
        let mut append = distro_args.to_string();
        if let Some(user) = machine.append.take() {
            append = format!("{append} {user}");
        }
        machine.append = Some(append);
    }
    machine.memory.get_or_insert(distro.default_memory_mib());
}

fn execute(cli: Cli) -> Result<ExitReason, String> {
    match cli.cmd {
        Cmd::Boot {
            kernel,
            initramfs,
            machine,
        } => {
            let kernel_image = read(&kernel)?;
            let initramfs_image = initramfs.as_ref().map(read).transpose()?;
            run::run(kernel_image, initramfs_image, machine)
        }
        Cmd::Firmware { image, machine } => run::run_firmware(read(&image)?, machine),
        Cmd::Quick {
            distro,
            update,
            disk_size,
            mut machine,
        } => {
            let artifacts = quick::ensure(distro, update)?;
            let kernel_image = read(&artifacts.kernel)?;
            let initramfs_image = read(&artifacts.initramfs)?;
            apply_quick_defaults(&mut machine, distro);
            if machine.disk.is_none() {
                let disk = quick::ensure_disk(distro, disk_size)?;
                machine.disk = Some(disk.to_string_lossy().into_owned());
            }
            run::run(kernel_image, Some(initramfs_image), machine)
        }
    }
}

/// clap exits 2 on a usage error, which would collide with `EXIT_MACHINE_FAILURE`; help requests
/// are not errors.
fn usage_exit_code(error: &clap::Error) -> i32 {
    if error.use_stderr() {
        EXIT_ERROR
    } else {
        EXIT_SUCCESS
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(usage_exit_code(&error) as u8);
        }
    };
    match execute(cli) {
        Ok(reason) => ExitCode::from(exit_code(&reason) as u8),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(EXIT_ERROR as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;
    use clap::error::ErrorKind;

    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["uemulator"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv)
    }

    fn error_kind(args: &[&str]) -> ErrorKind {
        parse(args).err().expect("parse should fail").kind()
    }

    fn boot(args: &[&str]) -> (PathBuf, Option<PathBuf>, MachineArgs) {
        match parse(args).unwrap().cmd {
            Cmd::Boot {
                kernel,
                initramfs,
                machine,
            } => (kernel, initramfs, machine),
            _ => panic!("expected boot"),
        }
    }

    fn quick(args: &[&str]) -> (Distro, bool, u64, MachineArgs) {
        match parse(args).unwrap().cmd {
            Cmd::Quick {
                distro,
                update,
                disk_size,
                machine,
            } => (distro, update, disk_size, machine),
            _ => panic!("expected quick"),
        }
    }

    fn machine_args(extra: &[&str]) -> MachineArgs {
        let mut args = vec!["boot", "--kernel", "k"];
        args.extend_from_slice(extra);
        boot(&args).2
    }

    fn help_of(path: &[&str]) -> String {
        let mut command = Cli::command();
        for name in path {
            command = command.find_subcommand(name).unwrap().clone();
        }
        command.render_long_help().to_string()
    }

    #[test]
    fn exit_code_is_zero_when_the_machine_ends_on_its_own() {
        assert_eq!(exit_code(&ExitReason::Reset), 0);
        assert_eq!(exit_code(&ExitReason::Shutdown), 0);
    }

    #[test]
    fn exit_code_is_zero_when_stopped_from_the_host() {
        assert_eq!(exit_code(&ExitReason::Stopped), 0);
    }

    #[test]
    fn exit_code_of_a_triple_fault_is_two() {
        assert_eq!(exit_code(&ExitReason::TripleFault), 2);
    }

    #[test]
    fn exit_code_of_an_internal_error_is_two() {
        assert_eq!(exit_code(&ExitReason::Internal(String::new())), 2);
        assert_eq!(
            exit_code(&ExitReason::Internal("vcpu thread panicked".into())),
            2
        );
        assert_eq!(
            exit_code(&ExitReason::Internal("jit divergence at 0xfff0".into())),
            2
        );
    }

    #[test]
    fn exit_codes_are_distinct_and_setup_errors_are_one() {
        assert_eq!(EXIT_SUCCESS, 0);
        assert_eq!(EXIT_ERROR, 1);
        assert_eq!(EXIT_MACHINE_FAILURE, 2);
        for reason in [
            ExitReason::Shutdown,
            ExitReason::Reset,
            ExitReason::Stopped,
            ExitReason::TripleFault,
            ExitReason::Internal("x".into()),
        ] {
            assert_ne!(exit_code(&reason), EXIT_ERROR, "{reason:?}");
        }
    }

    #[test]
    fn exit_codes_fit_in_the_process_status_byte() {
        for code in [EXIT_SUCCESS, EXIT_ERROR, EXIT_MACHINE_FAILURE] {
            assert!(u8::try_from(code).is_ok());
        }
    }

    #[test]
    fn exit_codes_are_documented_in_the_help() {
        let help = help_of(&[]);
        assert!(help.contains("Exit codes:"), "{help}");
        for code in [EXIT_SUCCESS, EXIT_ERROR, EXIT_MACHINE_FAILURE] {
            assert!(EXIT_CODES_HELP.contains(&format!("  {code}  ")), "{code}");
        }
        assert!(EXIT_CODES_HELP.contains("triple fault"));
        assert!(EXIT_CODES_HELP.contains("--timeout"));
    }

    #[test]
    fn usage_errors_exit_one_not_clap_default_two() {
        for args in [
            &["boot"][..],
            &["quick", "ubuntu"],
            &["nope"],
            &["boot", "--kernel", "k", "--memory", "x"],
            &[],
        ] {
            let error = parse(args).err().unwrap();
            assert_eq!(usage_exit_code(&error), EXIT_ERROR, "{args:?}");
        }
    }

    #[test]
    fn help_requests_exit_zero() {
        for args in [&["--help"][..], &["boot", "--help"], &["quick", "-h"]] {
            let error = parse(args).err().unwrap();
            assert_eq!(usage_exit_code(&error), EXIT_SUCCESS, "{args:?}");
        }
    }

    #[test]
    fn command_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn firmware_takes_an_image_and_the_machine_options() {
        match parse(&["firmware", "bios.bin", "--memory", "64", "--disable-jit"])
            .unwrap()
            .cmd
        {
            Cmd::Firmware { image, machine } => {
                assert_eq!(image, PathBuf::from("bios.bin"));
                assert_eq!(machine.memory, Some(64));
                assert!(machine.disable_jit);
            }
            _ => panic!("expected firmware"),
        }
        assert_eq!(
            error_kind(&["firmware"]),
            ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn boot_requires_a_kernel() {
        assert_eq!(error_kind(&["boot"]), ErrorKind::MissingRequiredArgument);
        assert_eq!(
            error_kind(&["boot", "--initramfs", "i"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(error_kind(&["boot", "--kernel"]), ErrorKind::InvalidValue);
    }

    #[test]
    fn boot_with_only_a_kernel_uses_the_defaults() {
        let (kernel, initramfs, machine) = boot(&["boot", "--kernel", "bz"]);
        assert_eq!(kernel, PathBuf::from("bz"));
        assert_eq!(initramfs, None);
        assert_eq!(machine.memory, None);
        assert_eq!(machine.cmdline, None);
        assert_eq!(machine.disk, None);
        assert_eq!(machine.append, None);
        assert_eq!(machine.net_udp, None);
        assert_eq!(machine.time, None);
        assert_eq!(machine.timeout, None);
        assert!(matches!(machine.console, ConsoleArg::Virtio));
        for flag in [
            machine.net_user,
            machine.net_loop,
            machine.disable_jit,
            machine.verify,
            machine.debug,
            machine.realtime,
            machine.no_reboot,
        ] {
            assert!(!flag);
        }
    }

    #[test]
    fn no_reboot_is_a_machine_flag_of_boot_and_quick() {
        assert!(machine_args(&["--no-reboot"]).no_reboot);
        let quick = parse(&["quick", "alpine", "--no-reboot"]).unwrap();
        assert!(matches!(quick.cmd, Cmd::Quick { machine, .. } if machine.no_reboot));
    }

    #[test]
    fn boot_takes_an_initramfs() {
        let (_, initramfs, _) = boot(&["boot", "--kernel", "k", "--initramfs", "rd.gz"]);
        assert_eq!(initramfs, Some(PathBuf::from("rd.gz")));
    }

    #[test]
    fn boot_accepts_every_machine_flag_together() {
        let (_, _, m) = boot(&[
            "boot",
            "--kernel",
            "k",
            "-m",
            "256",
            "--cmdline",
            "console=ttyS0 root=/dev/vda",
            "--disk",
            "d.img,ro",
            "--net-user",
            "--append",
            "quiet",
            "--net-loop",
            "--net-udp",
            "127.0.0.1:1,127.0.0.1:2",
            "--console",
            "virtio",
            "--disable-jit",
            "--verify",
            "--debug",
            "--time",
            "1700000000",
            "--realtime",
            "--timeout",
            "30",
        ]);
        assert_eq!(m.memory, Some(256));
        assert_eq!(m.cmdline.as_deref(), Some("console=ttyS0 root=/dev/vda"));
        assert_eq!(m.disk.as_deref(), Some("d.img,ro"));
        assert_eq!(m.append.as_deref(), Some("quiet"));
        assert_eq!(m.net_udp.as_deref(), Some("127.0.0.1:1,127.0.0.1:2"));
        assert!(matches!(m.console, ConsoleArg::Virtio));
        assert_eq!(m.time, Some(1_700_000_000));
        assert_eq!(m.timeout, Some(30));
        assert!(m.net_user && m.net_loop && m.disable_jit && m.verify && m.debug && m.realtime);
    }

    #[test]
    fn memory_has_a_short_and_a_long_flag() {
        assert_eq!(machine_args(&["-m", "128"]).memory, Some(128));
        assert_eq!(machine_args(&["--memory", "128"]).memory, Some(128));
        assert_eq!(machine_args(&["--memory=128"]).memory, Some(128));
    }

    #[test]
    fn memory_must_be_a_non_negative_integer() {
        for bad in ["abc", "-1", "1.5", "", "0x40"] {
            let kind = error_kind(&["boot", "--kernel", "k", "--memory", bad]);
            assert!(
                matches!(
                    kind,
                    ErrorKind::ValueValidation | ErrorKind::UnknownArgument
                ),
                "{bad:?}: {kind:?}"
            );
        }
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "--memory"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn memory_range_is_enforced_by_the_library_not_the_parser() {
        assert_eq!(machine_args(&["--memory", "0"]).memory, Some(0));
        assert_eq!(machine_args(&["--memory", "99999"]).memory, Some(99999));
    }

    #[test]
    fn console_accepts_only_serial_and_virtio() {
        assert!(matches!(
            machine_args(&["--console", "serial"]).console,
            ConsoleArg::Serial
        ));
        assert!(matches!(
            machine_args(&["--console", "virtio"]).console,
            ConsoleArg::Virtio
        ));
        for bad in ["hvc0", "Serial", "VIRTIO", ""] {
            assert_eq!(
                error_kind(&["boot", "--kernel", "k", "--console", bad]),
                ErrorKind::InvalidValue,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn time_and_timeout_must_be_unsigned_integers() {
        assert_eq!(machine_args(&["--time", "0"]).time, Some(0));
        assert_eq!(machine_args(&["--timeout", "0"]).timeout, Some(0));
        assert_eq!(
            machine_args(&["--time", "18446744073709551615"]).time,
            Some(u64::MAX)
        );
        for flag in ["--time", "--timeout"] {
            for bad in ["soon", "1.5", "99999999999999999999999"] {
                assert_eq!(
                    error_kind(&["boot", "--kernel", "k", flag, bad]),
                    ErrorKind::ValueValidation,
                    "{flag} {bad}"
                );
            }
        }
    }

    #[test]
    fn unknown_flags_and_stray_positionals_are_rejected() {
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "--nope"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "extra"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "--net-user=yes"]),
            ErrorKind::TooManyValues
        );
    }

    #[test]
    fn quick_flags_do_not_exist_on_boot() {
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "--update"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["boot", "--kernel", "k", "--disk-size", "4"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn boot_flags_do_not_exist_on_quick() {
        assert_eq!(
            error_kind(&["quick", "debian", "--kernel", "k"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["quick", "debian", "--initramfs", "i"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn quick_requires_a_known_distro() {
        assert_eq!(error_kind(&["quick"]), ErrorKind::MissingRequiredArgument);
        for bad in ["ubuntu", "Debian", "DEBIAN", "arch", ""] {
            assert_eq!(
                error_kind(&["quick", bad]),
                ErrorKind::InvalidValue,
                "{bad:?}"
            );
        }
        assert_eq!(
            error_kind(&["quick", "debian", "alpine"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn quick_accepts_each_distro_with_defaults() {
        for (name, want) in [
            ("debian", Distro::Debian),
            ("alpine", Distro::Alpine),
            ("fedora", Distro::Fedora),
        ] {
            let (distro, update, disk_size, machine) = quick(&["quick", name]);
            assert_eq!(distro, want, "{name}");
            assert!(!update);
            assert_eq!(disk_size, DEFAULT_DISK_GIB);
            assert_eq!(machine.memory, None);
        }
    }

    #[test]
    fn quick_takes_update_disk_size_and_machine_flags_in_any_order() {
        let (_, update, disk_size, machine) = quick(&[
            "quick",
            "--memory",
            "512",
            "alpine",
            "--update",
            "--disk-size",
            "20",
            "--net-loop",
            "--timeout",
            "5",
        ]);
        assert!(update);
        assert_eq!(disk_size, 20);
        assert_eq!(machine.memory, Some(512));
        assert!(machine.net_loop);
        assert_eq!(machine.timeout, Some(5));
    }

    #[test]
    fn quick_disk_size_must_be_an_unsigned_integer() {
        for bad in ["-1", "big", "1.5"] {
            let kind = error_kind(&["quick", "debian", "--disk-size", bad]);
            assert!(
                matches!(
                    kind,
                    ErrorKind::ValueValidation | ErrorKind::UnknownArgument
                ),
                "{bad}: {kind:?}"
            );
        }
        assert_eq!(
            error_kind(&["quick", "debian", "--disk-size"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn missing_or_unknown_subcommands_are_errors() {
        assert_eq!(
            error_kind(&[]),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(error_kind(&["run"]), ErrorKind::InvalidSubcommand);
        assert_eq!(error_kind(&["--kernel", "k"]), ErrorKind::UnknownArgument);
    }

    #[test]
    fn help_is_available_for_every_command() {
        for args in [&["--help"][..], &["boot", "--help"], &["quick", "--help"]] {
            assert_eq!(error_kind(args), ErrorKind::DisplayHelp, "{args:?}");
        }
    }

    #[test]
    fn there_is_no_version_flag() {
        assert_eq!(error_kind(&["--version"]), ErrorKind::UnknownArgument);
    }

    #[test]
    fn distro_names_match_the_command_line_names() {
        use clap::ValueEnum;
        for distro in Distro::value_variants() {
            let value = distro.to_possible_value().unwrap();
            assert_eq!(value.get_name(), distro.name());
        }
    }

    #[test]
    fn net_loop_help_describes_only_what_the_gateway_answers() {
        let help = help_of(&["boot"]);
        assert!(help.contains("ARP"), "{help}");
        assert!(help.contains("ICMP echo"), "{help}");
        assert!(help.contains("10.0.2.2"), "{help}");
        assert!(!help.contains("DHCP, ARP"), "{help}");
        assert!(help.contains("no DHCP"), "{help}");
    }

    #[test]
    fn memory_help_states_the_range_the_library_enforces() {
        use crate::run::DEFAULT_MEMORY_MIB;
        let help = help_of(&["boot"]);
        assert!(help.contains("64 to 3072"), "{help}");
        assert!(
            help.contains(&format!("default: {DEFAULT_MEMORY_MIB}")),
            "{help}"
        );
        assert!(
            help.contains(&format!(
                "{} for `quick fedora`",
                Distro::Fedora.default_memory_mib()
            )),
            "{help}"
        );
    }

    #[test]
    fn memory_help_range_matches_the_library_error() {
        // The library does not export its limits, so read them from the rejection it produces.
        let rejected =
            uemulator::EmulatedMachine::new(uemulator::MachineConfig::new().memory_mib(0));
        let message = match rejected {
            Ok(_) => panic!("0 MiB must be rejected"),
            Err(e) => e.to_string(),
        };
        let numbers: Vec<usize> = message
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|word| word.parse().ok())
            .collect();
        assert_eq!(numbers.len(), 2, "{message}");
        assert!(
            help_of(&["boot"]).contains(&format!("{} to {}", numbers[0], numbers[1])),
            "{message}"
        );
    }

    #[test]
    fn quick_help_mentions_the_disk_location() {
        let help = help_of(&["quick"]);
        assert!(
            help.contains("~/.uemulator/quick/<distro>/disk.img"),
            "{help}"
        );
        assert!(help.contains(&format!("{DEFAULT_DISK_GIB}")), "{help}");
    }

    #[test]
    fn quick_defaults_pick_user_networking_when_no_backend_was_chosen() {
        let mut machine = machine_args(&[]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert!(machine.net_user);
    }

    #[test]
    fn quick_defaults_respect_another_network_backend() {
        let mut machine = machine_args(&["--net-loop"]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert!(!machine.net_user);
        let mut machine = machine_args(&["--net-udp", "a,b"]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert!(!machine.net_user);
    }

    #[test]
    fn quick_defaults_keep_an_explicit_net_user() {
        let mut machine = machine_args(&["--net-user", "--net-loop"]);
        apply_quick_defaults(&mut machine, Distro::Alpine);
        assert!(machine.net_user && machine.net_loop);
    }

    #[test]
    fn quick_defaults_set_distro_memory_only_when_unset() {
        let mut machine = machine_args(&[]);
        apply_quick_defaults(&mut machine, Distro::Fedora);
        assert_eq!(machine.memory, Some(Distro::Fedora.default_memory_mib()));
        let mut machine = machine_args(&["-m", "1024"]);
        apply_quick_defaults(&mut machine, Distro::Fedora);
        assert_eq!(machine.memory, Some(1024));
        let mut machine = machine_args(&[]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert_eq!(machine.memory, Some(Distro::Debian.default_memory_mib()));
    }

    #[test]
    fn quick_defaults_leave_the_cmdline_alone_for_debian() {
        let mut machine = machine_args(&[]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert_eq!(machine.append, None);
        let mut machine = machine_args(&["--append", "auto=true"]);
        apply_quick_defaults(&mut machine, Distro::Debian);
        assert_eq!(machine.append.as_deref(), Some("auto=true"));
    }

    #[test]
    fn quick_defaults_put_distro_arguments_before_the_users() {
        let mut machine = machine_args(&[]);
        apply_quick_defaults(&mut machine, Distro::Alpine);
        assert_eq!(machine.append.as_deref(), Some(Distro::Alpine.cmdline()));

        let mut machine = machine_args(&["--append", "console=ttyS0"]);
        apply_quick_defaults(&mut machine, Distro::Fedora);
        assert_eq!(
            machine.append,
            Some(format!("{} console=ttyS0", Distro::Fedora.cmdline()))
        );
    }

    #[test]
    fn quick_defaults_do_not_touch_the_explicit_cmdline_or_disk() {
        let mut machine = machine_args(&["--cmdline", "root=/dev/vda", "--disk", "x.img"]);
        apply_quick_defaults(&mut machine, Distro::Alpine);
        assert_eq!(machine.cmdline.as_deref(), Some("root=/dev/vda"));
        assert_eq!(machine.disk.as_deref(), Some("x.img"));
    }

    #[test]
    fn read_error_names_the_path() {
        let missing = PathBuf::from("/nonexistent-uemulator-dir/kernel");
        let err = read(&missing).unwrap_err();
        assert!(
            err.starts_with("read /nonexistent-uemulator-dir/kernel: "),
            "{err}"
        );
    }

    #[test]
    fn read_returns_the_file_contents() {
        let dir = crate::test_support::TempDir::new("main-read");
        let path = dir.path().join("blob");
        std::fs::write(&path, [1u8, 2, 3, 0, 255]).unwrap();
        assert_eq!(read(&path).unwrap(), vec![1, 2, 3, 0, 255]);
    }
}
