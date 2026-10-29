# Third-party code in Merkurius

Files copied from other projects, as the design document requires (Reuse from
LLVM and Flang, "Rules for copied code"). Each keeps its licence header and
names its origin and upstream commit.

All files below come from [llvm-project](https://github.com/llvm/llvm-project),
licensed under Apache 2.0 with the LLVM exception, at upstream commit
`d81bd0216be61d9689fc04e74272b73bc6d8a20b`.

| Local path | Upstream path | Changes |
| --- | --- | --- |
| `include/merkurius/Common/enum-class.h` | `flang/include/flang/Common/enum-class.h` | Namespace and include guard renamed. Needed by `idioms.h` |
| `include/merkurius/Common/idioms.h` | `flang/include/flang/Common/idioms.h` | Namespace and include guard renamed; includes `<variant>` instead of Flang's runtime `variant.h` |
| `include/merkurius/Common/indirection.h` | `flang/include/flang/Common/indirection.h` | Namespace and include guard renamed |
| `include/merkurius/Common/visit.h` | `flang/include/flang/Common/visit.h` | Namespace, include guard and `FLANG_USE_STD_VISIT` renamed; Flang runtime attributes (`api-attrs.h`) and `variant.h` removed |
| `lib/Common/idioms.cpp` | `flang/lib/Support/idioms.cpp` | Namespace and include path renamed |
