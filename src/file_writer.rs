use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub(crate) struct NewFileWriter {
    file: File,
}

impl NewFileWriter {
    pub(crate) fn create_new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?,
        })
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }
}

impl Write for NewFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
