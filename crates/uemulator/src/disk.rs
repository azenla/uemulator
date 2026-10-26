use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

pub trait BlockBackend: Send {
    fn len(&self) -> u64;
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
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
    len: u64,
    ro: bool,
}

struct MemDisk(Vec<u8>);

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(unix)]
fn write_all_at(f: &File, data: &[u8], off: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(f, data, off)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(f: &File, mut data: &[u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        let n = f.seek_write(data, off)?;
        if n == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        data = &data[n..];
        off += n as u64;
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
        if self.ro {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        write_all_at(&self.file, data, offset)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.ro {
            Ok(())
        } else {
            self.file.sync_data()
        }
    }
    fn read_only(&self) -> bool {
        self.ro
    }
}

fn range(len: usize, offset: u64, n: usize) -> io::Result<std::ops::Range<usize>> {
    let s = usize::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    match s.checked_add(n) {
        Some(e) if e <= len => Ok(s..e),
        _ => Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
    }
}

impl BlockBackend for MemDisk {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let r = range(self.0.len(), offset, buf.len())?;
        buf.copy_from_slice(&self.0[r]);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        let r = range(self.0.len(), offset, data.len())?;
        self.0[r].copy_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Disk {
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> io::Result<Disk> {
        let file = OpenOptions::new().read(true).write(!read_only).open(path)?;
        let len = file.metadata()?.len();
        Ok(Disk {
            backend: Box::new(FileDisk {
                file,
                len,
                ro: read_only,
            }),
        })
    }

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
