# Win32 compatibility provenance

The POSIX shadow headers, `posix.cpp`, `spawn.cpp`, and HIP `cmath` workaround
were imported from `thomas9120/gufo-adapted` branch `windows-port` at
`ab4d98189a6d8103fce29d2537e4113da3da2420`. Their original compatibility commit
is `edb2f8b0629005f2852d4543a3c24786b56979a5`.

Stage-one additions are `directory.cpp`/`dirent.h`, `mkdirat`, a narrow
no-replace rename operation, and 64-bit device/inode types used by the new
upstream cache. The imported path resolver handles `openat(fd, ".", ...)`, and
failed flushes no longer return success. Other inherited limitations remain;
in particular, descriptor-relative paths resolve through names, and POSIX
ownership/modes do not prove Windows ACL privacy. Both disk stores and persistent
identity caching remain disabled pending stage-two qualification.

The companion socket helper and root vcpkg manifest come from the same pinned
fork revision. The new CMake module and build/check scripts adapt those ideas
to the current official engine; no fork model changes, GUI, release workflow,
or tuning options were imported. See [Windows status](../../docs/WINDOWS.md)
for reproducible commands and validated scope.
