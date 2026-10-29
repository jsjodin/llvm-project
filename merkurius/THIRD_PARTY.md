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
| `include/merkurius/Common/interval.h` | `flang/include/flang/Common/interval.h` | Namespace and include guard renamed |
| `include/merkurius/Common/reference.h` | `flang/include/flang/Common/reference.h` | Namespace and include guard renamed |
| `include/merkurius/Common/visit.h` | `flang/include/flang/Common/visit.h` | Namespace, include guard and `FLANG_USE_STD_VISIT` renamed; Flang runtime attributes (`api-attrs.h`) and `variant.h` removed |
| `lib/Common/idioms.cpp` | `flang/lib/Support/idioms.cpp` | Namespace and include path renamed |
| `include/merkurius/Parser/char-block.h` | `flang/include/flang/Parser/char-block.h` | Namespace and include guard renamed |
| `lib/Parser/char-block.cpp` | `flang/lib/Parser/char-block.cpp` | Namespace and include path renamed |
| `include/merkurius/Parser/source.h` | `flang/include/flang/Parser/source.h` | Namespace and include guard renamed; bytes kept exactly as read (no carriage-return removal, no added final newline); positions valid at the end of the file; `Encoding` and `LocateSourceFileAll` removed; `SetBuffer` and `GetLine` added |
| `lib/Parser/source.cpp` | `flang/lib/Parser/source.cpp` | As for `source.h`; a later line directive for the same line replaces an earlier one |
| `include/merkurius/Parser/provenance.h` | `flang/include/flang/Parser/provenance.h` | Namespace and include guard renamed; reduced to `Provenance`, `ProvenanceRange` and `AllSources` with source-file and compiler-insertion origins; macro, module and `INCLUDE` origins, `CookedSource`, `AllCookedSources` and the offset maps removed; opening a file gives it its provenances, with one for its end; `CharBlock` lookups added |
| `lib/Parser/provenance.cpp` | `flang/lib/Parser/provenance.cpp` | As for `provenance.h`; the caret under an echoed source line allows for multi-byte UTF-8 characters |
