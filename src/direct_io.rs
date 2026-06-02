use anyhow::Result;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::ptr::NonNull;

const DIRECT_IO_ALIGNMENT: usize = 4096;

pub(crate) struct DirectFileWriter {
    file: File,
    block: AlignedBlock,
    used: usize,
    logical_len: u64,
}

impl DirectFileWriter {
    pub(crate) fn create_append_new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: open_direct_append_new(path)?,
            block: AlignedBlock::new(DIRECT_IO_ALIGNMENT)?,
            used: 0,
            logical_len: 0,
        })
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.flush_block()?;
        self.file.set_len(self.logical_len)?;
        self.file.sync_all()?;
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.used == 0 {
            return Ok(());
        }
        self.block.zero_from(self.used);
        self.file.write_all(self.block.as_slice())?;
        self.logical_len += self.used as u64;
        self.used = 0;
        Ok(())
    }
}

impl Write for DirectFileWriter {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let written = buf.len();
        while !buf.is_empty() {
            let take = (self.block.len() - self.used).min(buf.len());
            self.block.as_mut_slice()[self.used..self.used + take]
                .copy_from_slice(&buf[..take]);
            self.used += take;
            buf = &buf[take..];
            if self.used == self.block.len() {
                self.flush_block()?;
            }
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct AlignedBlock {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBlock {
    fn new(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGNMENT)?;
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| anyhow::anyhow!("unable to allocate direct I/O block"))?;
        Ok(Self { ptr, layout })
    }

    fn len(&self) -> usize {
        self.layout.size()
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    fn zero_from(&mut self, start: usize) {
        self.as_mut_slice()[start..].fill(0);
    }
}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_direct_append_new(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    const O_DIRECT: i32 = 0o40000;
    Ok(OpenOptions::new()
        .create_new(true)
        .append(true)
        .custom_flags(O_DIRECT)
        .open(path)?)
}

#[cfg(target_os = "macos")]
fn open_direct_append_new(path: &Path) -> Result<File> {
    use std::os::fd::AsRawFd;

    const F_NOCACHE: i32 = 48;

    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }

    let file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(path)?;
    let rc = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
    if rc == -1 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(file)
}

#[cfg(windows)]
fn open_direct_append_new(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_WRITE_THROUGH: u32 = 0x80000000;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x20000000;

    Ok(OpenOptions::new()
        .create_new(true)
        .append(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH)
        .open(path)?)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos", windows)))]
fn open_direct_append_new(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().create_new(true).append(true).open(path)?)
}
