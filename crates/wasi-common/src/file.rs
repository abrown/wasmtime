use crate::{Error, ErrorExt, SystemTimeSpec};
use bitflags::bitflags;
use std::any::Any;
use std::sync::{Arc, RwLock};

#[cfg(unix)]
use cap_std::io_lifetimes::{AsFd, BorrowedFd};

#[cfg(windows)]
use io_extras::os::windows::{AsRawHandleOrSocket, RawHandleOrSocket};

#[wiggle::async_trait]
pub trait WasiFile: Send + Sync {
    fn as_any(self: Arc<Self>) -> Arc<dyn Any>;
    async fn get_filetype(self: Arc<Self>) -> Result<FileType, Error>;

    #[cfg(unix)]
    fn pollable(self: Arc<Self>) -> Option<Arc<dyn AsFd>> {
        None
    }

    #[cfg(windows)]
    fn pollable(self: Arc<Self>) -> Option<Arc<dyn AsRawHandleOrSocket>> {
        None
    }

    fn isatty(self: Arc<Self>) -> bool {
        false
    }

    async fn sock_accept(self: Arc<Self>, _fdflags: FdFlags) -> Result<Arc<dyn WasiFile>, Error> {
        Err(Error::badf())
    }

    async fn sock_recv<'a>(
        self: Arc<Self>,
        _ri_data: &mut [std::io::IoSliceMut<'a>],
        _ri_flags: RiFlags,
    ) -> Result<(u64, RoFlags), Error> {
        Err(Error::badf())
    }

    async fn sock_send<'a>(
        self: Arc<Self>,
        _si_data: &[std::io::IoSlice<'a>],
        _si_flags: SiFlags,
    ) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn sock_shutdown(self: Arc<Self>, _how: SdFlags) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn datasync(self: Arc<Self>) -> Result<(), Error> {
        Ok(())
    }

    async fn sync(self: Arc<Self>) -> Result<(), Error> {
        Ok(())
    }

    async fn get_fdflags(self: Arc<Self>) -> Result<FdFlags, Error> {
        Ok(FdFlags::empty())
    }

    async fn set_fdflags(self: Arc<Self>, _flags: FdFlags) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn get_filestat(self: Arc<Self>) -> Result<Filestat, Error> {
        Ok(Filestat {
            device_id: 0,
            inode: 0,
            filetype: self.get_filetype().await?,
            nlink: 0,
            size: 0, // XXX no way to get a size out of a Read :(
            atim: None,
            mtim: None,
            ctim: None,
        })
    }

    async fn set_filestat_size(self: Arc<Self>, _size: u64) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn advise(
        self: Arc<Self>,
        _offset: u64,
        _len: u64,
        _advice: Advice,
    ) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn allocate(self: Arc<Self>, _offset: u64, _len: u64) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn set_times(
        self: Arc<Self>,
        _atime: Option<SystemTimeSpec>,
        _mtime: Option<SystemTimeSpec>,
    ) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn read_vectored<'a>(
        self: Arc<Self>,
        _bufs: &mut [std::io::IoSliceMut<'a>],
    ) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn read_vectored_at<'a>(
        self: Arc<Self>,
        _bufs: &mut [std::io::IoSliceMut<'a>],
        _offset: u64,
    ) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn write_vectored<'a>(
        self: Arc<Self>,
        _bufs: &[std::io::IoSlice<'a>],
    ) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn write_vectored_at<'a>(
        self: Arc<Self>,
        _bufs: &[std::io::IoSlice<'a>],
        _offset: u64,
    ) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn seek(self: Arc<Self>, _pos: std::io::SeekFrom) -> Result<u64, Error> {
        Err(Error::badf())
    }

    async fn peek(self: Arc<Self>, _buf: &mut [u8]) -> Result<u64, Error> {
        Err(Error::badf())
    }

    fn num_ready_bytes(self: Arc<Self>) -> Result<u64, Error> {
        Ok(0)
    }

    async fn readable(self: Arc<Self>) -> Result<(), Error> {
        Err(Error::badf())
    }

    async fn writable(self: Arc<Self>) -> Result<(), Error> {
        Err(Error::badf())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FileType {
    Unknown,
    BlockDevice,
    CharacterDevice,
    Directory,
    RegularFile,
    SocketDgram,
    SocketStream,
    SymbolicLink,
    Pipe,
}

bitflags! {
    pub struct FdFlags: u32 {
        const APPEND   = 0b1;
        const DSYNC    = 0b10;
        const NONBLOCK = 0b100;
        const RSYNC    = 0b1000;
        const SYNC     = 0b10000;
    }
}

bitflags! {
    pub struct SdFlags: u32 {
        const RD = 0b1;
        const WR = 0b10;
    }
}

bitflags! {
    pub struct SiFlags: u32 {
    }
}

bitflags! {
    pub struct RiFlags: u32 {
        const RECV_PEEK    = 0b1;
        const RECV_WAITALL = 0b10;
    }
}

bitflags! {
    pub struct RoFlags: u32 {
        const RECV_DATA_TRUNCATED = 0b1;
    }
}

bitflags! {
    pub struct OFlags: u32 {
        const CREATE    = 0b1;
        const DIRECTORY = 0b10;
        const EXCLUSIVE = 0b100;
        const TRUNCATE  = 0b1000;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filestat {
    pub device_id: u64,
    pub inode: u64,
    pub filetype: FileType,
    pub nlink: u64,
    pub size: u64, // this is a read field, the rest are file fields
    pub atim: Option<std::time::SystemTime>,
    pub mtim: Option<std::time::SystemTime>,
    pub ctim: Option<std::time::SystemTime>,
}

pub(crate) trait TableFileExt {
    fn get_file(&self, fd: u32) -> Result<Arc<FileEntry>, Error>;
}
impl TableFileExt for crate::table::Table {
    fn get_file(&self, fd: u32) -> Result<Arc<FileEntry>, Error> {
        self.get(fd)
    }
}

pub(crate) struct FileEntry {
    caps: RwLock<FileCaps>,
    file: Arc<dyn WasiFile>,
}

impl FileEntry {
    pub fn new(caps: FileCaps, file: Arc<dyn WasiFile>) -> Self {
        FileEntry {
            caps: RwLock::new(caps),
            file,
        }
    }

    pub fn capable_of(&self, caps: FileCaps) -> Result<(), Error> {
        if self.caps.read().unwrap().contains(caps) {
            Ok(())
        } else {
            let missing = caps & !(*self.caps.read().unwrap());
            let err = if missing.intersects(FileCaps::READ | FileCaps::WRITE) {
                // `EBADF` is a little surprising here because it's also used
                // for unknown-file-descriptor errors, but it's what POSIX uses
                // in this situation.
                Error::badf()
            } else {
                Error::perm()
            };
            Err(err.context(format!("desired rights {:?}, has {:?}", caps, self.caps)))
        }
    }

    pub fn drop_caps_to(&self, caps: FileCaps) -> Result<(), Error> {
        self.capable_of(caps)?;
        *self.caps.write().unwrap() = caps;
        Ok(())
    }

    pub async fn get_fdstat(&self) -> Result<FdStat, Error> {
        let caps = self.caps.read().unwrap().clone();
        Ok(FdStat {
            filetype: self.file.clone().get_filetype().await?,
            caps,
            flags: self.file.clone().get_fdflags().await?,
        })
    }
}

pub trait FileEntryExt {
    fn get_cap(&self, caps: FileCaps) -> Result<Arc<dyn WasiFile>, Error>;
}

impl FileEntryExt for FileEntry {
    fn get_cap(&self, caps: FileCaps) -> Result<Arc<dyn WasiFile>, Error> {
        self.capable_of(caps)?;
        Ok(self.file.clone())
    }
}

bitflags! {
    pub struct FileCaps : u32 {
        const DATASYNC           = 0b1;
        const READ               = 0b10;
        const SEEK               = 0b100;
        const FDSTAT_SET_FLAGS   = 0b1000;
        const SYNC               = 0b10000;
        const TELL               = 0b100000;
        const WRITE              = 0b1000000;
        const ADVISE             = 0b10000000;
        const ALLOCATE           = 0b100000000;
        const FILESTAT_GET       = 0b1000000000;
        const FILESTAT_SET_SIZE  = 0b10000000000;
        const FILESTAT_SET_TIMES = 0b100000000000;
        const POLL_READWRITE     = 0b1000000000000;
    }
}

#[derive(Debug, Clone)]
pub struct FdStat {
    pub filetype: FileType,
    pub caps: FileCaps,
    pub flags: FdFlags,
}

#[derive(Debug, Clone)]
pub enum Advice {
    Normal,
    Sequential,
    Random,
    WillNeed,
    DontNeed,
    NoReuse,
}

#[cfg(unix)]
pub struct BorrowedAsFd<'a, T: AsFd>(&'a T);

#[cfg(unix)]
impl<'a, T: AsFd> BorrowedAsFd<'a, T> {
    pub fn new(t: &'a T) -> Self {
        BorrowedAsFd(t)
    }
}

#[cfg(unix)]
impl<T: AsFd> AsFd for BorrowedAsFd<'_, T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd {
        self.0.as_fd()
    }
}

#[cfg(windows)]
pub struct BorrowedAsRawHandleOrSocket<'a, T: AsRawHandleOrSocket>(&'a T);

#[cfg(windows)]
impl<'a, T: AsRawHandleOrSocket> BorrowedAsRawHandleOrSocket<'a, T> {
    pub fn new(t: &'a T) -> Self {
        BorrowedAsRawHandleOrSocket(t)
    }
}

#[cfg(windows)]
impl<T: AsRawHandleOrSocket> AsRawHandleOrSocket for BorrowedAsRawHandleOrSocket<'_, T> {
    #[inline]
    fn as_raw_handle_or_socket(&self) -> RawHandleOrSocket {
        self.0.as_raw_handle_or_socket()
    }
}
