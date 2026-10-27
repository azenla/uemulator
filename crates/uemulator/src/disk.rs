use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// Storage behind the virtio-blk device. Requests are served synchronously on the vCPU thread
/// when the guest kicks the queue, so a slow backend stalls the guest; hence `Send` is the only
/// thread requirement. Offsets and lengths are in bytes; the guest sees `len() / 512` sectors.
pub trait BlockBackend: Send {
    fn len(&self) -> u64;
    /// Must fill `buf` completely or fail.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    /// Must not extend the disk: the guest sees a fixed capacity.
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    /// Advertised to the guest (virtio-blk RO feature), and the device refuses writes itself.
    fn read_only(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Disk {
    pub(crate) backend: Box<dyn BlockBackend>,
}

struct FileDisk {
    file: File,
    /// Captured at open; a file that grows or shrinks afterwards does not change the capacity
    /// the guest was told.
    len: u64,
    read_only: bool,
}

struct MemDisk(Vec<u8>);

// Positional I/O (no shared seek offset). Unix has it natively; Windows `seek_*` can return
// short counts and moves the file pointer, so loop to match `read_exact_at`/`write_all_at`.
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(unix)]
fn write_all_at(file: &File, data: &[u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, data, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let count = file.seek_read(buf, offset)?;
        if count == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[count..];
        offset += count as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, mut data: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        let count = file.seek_write(data, offset)?;
        if count == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        data = &data[count..];
        offset += count as u64;
    }
    Ok(())
}

impl BlockBackend for FileDisk {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        read_exact_at(&self.file, buf, offset)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        write_all_at(&self.file, data, offset)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.read_only {
            Ok(())
        } else {
            self.file.sync_data()
        }
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

/// `offset..offset + length` as an index range, or an error if it does not fit in `capacity`.
fn checked_range(
    capacity: usize,
    offset: u64,
    length: usize,
) -> io::Result<std::ops::Range<usize>> {
    let start =
        usize::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    match start.checked_add(length) {
        Some(end) if end <= capacity => Ok(start..end),
        _ => Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
    }
}

impl BlockBackend for MemDisk {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let range = checked_range(self.0.len(), offset, buf.len())?;
        buf.copy_from_slice(&self.0[range]);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        let range = checked_range(self.0.len(), offset, data.len())?;
        self.0[range].copy_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Disk {
    /// Raw image file, opened for writing unless `read_only`. Guest writes go straight to the file;
    /// `flush` is `sync_data`.
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> io::Result<Disk> {
        let file = OpenOptions::new().read(true).write(!read_only).open(path)?;
        let len = file.metadata()?.len();
        Ok(Disk {
            backend: Box::new(FileDisk {
                file,
                len,
                read_only,
            }),
        })
    }

    /// Volatile in-memory disk; contents are lost when the machine is dropped.
    pub fn memory(data: Vec<u8>) -> Disk {
        Disk {
            backend: Box::new(MemDisk(data)),
        }
    }

    pub fn from_backend(backend: impl BlockBackend + 'static) -> Disk {
        Disk {
            backend: Box::new(backend),
        }
    }
}
