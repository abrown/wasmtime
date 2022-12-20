use crate::clocks::WasiClocks;
use crate::dir::{DirCaps, DirEntry, WasiDir};
use crate::file::{FileCaps, FileEntry, WasiFile};
use crate::sched::WasiSched;
use crate::string_array::{StringArray, StringArrayError};
use crate::table::Table;
use crate::Error;
use cap_rand::RngCore;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct WasiCtx {
    pub args: StringArray,
    pub env: StringArray,
    pub random: Box<dyn RngCore + Send + Sync>,
    pub clocks: WasiClocks,
    pub sched: Box<dyn WasiSched>,
    pub table: Table,
}

impl WasiCtx {
    pub fn new(
        random: Box<dyn RngCore + Send + Sync>,
        clocks: WasiClocks,
        sched: Box<dyn WasiSched>,
        table: Table,
    ) -> Self {
        let s = WasiCtx {
            args: StringArray::new(),
            env: StringArray::new(),
            random,
            clocks,
            sched,
            table,
        };
        s.set_stdin(Arc::new(crate::pipe::ReadPipe::new(std::io::empty())));
        s.set_stdout(Arc::new(crate::pipe::WritePipe::new(std::io::sink())));
        s.set_stderr(Arc::new(crate::pipe::WritePipe::new(std::io::sink())));
        s
    }

    pub fn insert_file(&self, fd: u32, file: Arc<dyn WasiFile>, caps: FileCaps) {
        self.table()
            .insert_at(fd, Arc::new(FileEntry::new(caps, file)));
    }

    pub fn push_file(&self, file: Arc<dyn WasiFile>, caps: FileCaps) -> Result<u32, Error> {
        self.table().push(Arc::new(FileEntry::new(caps, file)))
    }

    pub fn insert_dir(
        &self,
        fd: u32,
        dir: Arc<dyn WasiDir>,
        caps: DirCaps,
        file_caps: FileCaps,
        path: PathBuf,
    ) {
        self.table().insert_at(
            fd,
            Arc::new(DirEntry::new(caps, file_caps, Some(path), dir)),
        );
    }

    pub fn push_dir(
        &self,
        dir: Arc<dyn WasiDir>,
        caps: DirCaps,
        file_caps: FileCaps,
        path: PathBuf,
    ) -> Result<u32, Error> {
        self.table()
            .push(Arc::new(DirEntry::new(caps, file_caps, Some(path), dir)))
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn push_arg(&mut self, arg: &str) -> Result<(), StringArrayError> {
        self.args.push(arg.to_owned())
    }

    pub fn push_env(&mut self, var: &str, value: &str) -> Result<(), StringArrayError> {
        self.env.push(format!("{}={}", var, value))?;
        Ok(())
    }

    pub fn set_stdin(&self, f: Arc<dyn WasiFile>) {
        let rights = Self::stdio_rights(f.clone());
        self.insert_file(0, f, rights);
    }

    pub fn set_stdout(&self, f: Arc<dyn WasiFile>) {
        let rights = Self::stdio_rights(f.clone());
        self.insert_file(1, f, rights);
    }

    pub fn set_stderr(&self, f: Arc<dyn WasiFile>) {
        let rights = Self::stdio_rights(f.clone());
        self.insert_file(2, f, rights);
    }

    fn stdio_rights(f: Arc<dyn WasiFile>) -> FileCaps {
        let mut rights = FileCaps::all();

        // If `f` is a tty, restrict the `tell` and `seek` capabilities, so
        // that wasi-libc's `isatty` correctly detects the file descriptor
        // as a tty.
        if f.isatty() {
            rights &= !(FileCaps::TELL | FileCaps::SEEK);
        }

        rights
    }

    pub fn push_preopened_dir(
        &self,
        dir: Arc<dyn WasiDir>,
        path: impl AsRef<Path>,
    ) -> Result<(), Error> {
        let caps = DirCaps::all();
        let file_caps = FileCaps::all();
        self.table().push(Arc::new(DirEntry::new(
            caps,
            file_caps,
            Some(path.as_ref().to_owned()),
            dir,
        )))?;
        Ok(())
    }
}
