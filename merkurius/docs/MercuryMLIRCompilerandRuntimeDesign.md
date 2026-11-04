# Mercury MLIR Compiler and Runtime Design

Oct 26, 2026 · @Jan Leyonberg

## Overview and goals

We will build a clean-room Mercury compiler on MLIR and LLVM, adding OpenMP-style parallel loops and Erlang-style processes with per-process heaps. Mercury already runs deterministic code near C speed, so the work is a new back end and runtime, not a faster language.

**Goals**

- Compile Mercury, as defined by the 22.01 reference manual, to native code through MLIR and LLVM.
- Keep deterministic code at C speed. Baseline measured with Melbourne Mercury 22.01.9 (`hlc.gc`): fib 0.9× C, float loop 1.0×, 12-queens 1.4×, naive reverse 3.5× (Experiments, E1).
- Data parallelism through user-promised loop annotations lowered to OpenMP. A hand-patched `omp parallel for` in mmc's generated C gave 2× on 2 cores with identical output (Experiments, E2).
- Concurrency through lightweight processes: typed mailboxes, non-blocking send, blocking selective receive with timeout.
- Per-process heaps with precise garbage collection, so no collection ever stops every process.

**Design principle: pay only for what you use.** An advanced feature may cost something only in code that uses it. A program that never uses open terms or parallel loops runs as if those features did not exist, and preemption adds nothing to code that allocates. The base design still has costs every program pays: precise stack maps for the moving collector, and a check on the back edge of each loop that does not allocate. Later sections say how each feature meets this rule; open terms are its main test (Language extensions).

**Non-goals for the first versions:** Java and C# back ends, the Mercury debugger and deep profiler, tabling, trailing and solver types, partially instantiated data (until open types arrive; Language extensions), hot code loading, and binary compatibility with Melbourne Mercury.

**Licensing stance:** the compiler and runtime are written from Melbourne Mercury's documentation, from published papers and from experiments on the code mmc generates, never from its source code (Clean-room rules). That lets them use Apache 2.0 with the LLVM exception. The standard library is reimplemented too (Standard library section). This is an engineering position, not legal advice.

**Decisions so far**

| Area | Decision | Reason |
| --- | --- | --- |
| Front end | New, clean-room implementation | Licence freedom (Clean-room rules); MLIR-native data structures |
| Implementation language | C++ for the compiler and runtime | Direct use of MLIR, the LLVM data structures and TableGen |
| Command line | Clang/GCC style: `-c`, `-o`, `-O2`, `-g`, `-fopenmp` | Fits Make, Ninja and CMake, and the habits of C, C++ and Fortran users (Driver) |
| Source text | Read unchanged, with no prescanner; the lexer recognizes `%$omp` directives and line number directives | Both depend on token context: strings may span lines, and line number directives may occur anywhere a token may (Front end, Source manager) |
| Parser | Recursive descent in two layers: a term reader with precedence climbing, then a converter per context | The manual defines items and goals as terms over a fixed operator table |
| Front-end representation | Flang-style parse tree plus a small desugared semantic form; per-procedure tables for mode results | No separate IR; modes need the source structure and have no SSA form before analysis |
| Parallel annotations | `%$omp` comment sentinel, attached to the next goal | The same source compiles with mmc; mirrors Fortran's `!$omp` |
| IR | `merk` dialect, lowered through upstream dialects to LLVM | Reuse `scf`, `omp` and LLVM back ends |
| Parallel loops | User promises independence; no dependence analysis | Same contract as OpenMP and Mercury's `promise_*` |
| Processes | Erlang model with mailboxes typed by message type | Protocols checked at compile time |
| Send | Non-blocking, deep copy into the receiver | Keeps heaps independent |
| Receive | Blocking and selective, over one mailbox or several joined by a selector; the oldest matching message wins, across mailboxes; timeout returns `maybe`; a `receive` goal, with one `select` arm per message pattern, as sugar | Erlang semantics with static types; priority by Erlang's two-step receive |
| Kinds of message | One protocol type per mailbox, a discriminated union; more mailboxes for types the process does not own (replies, `down` messages, other modules' protocols) | Constructors cost nothing; a selector maps each mailbox's type into one result type |
| Unknown messages | Impossible by typing; `det` loops must cover every constructor | No catch-all `_ -> true` needed |
| Request/reply | Fresh reply mailbox per request, and a monitor on the server for the call | Late replies go to a dead mailbox and are dropped; a dead server ends the wait at once |
| Generic dispatch | Existential envelopes; `sendable(M)` required where a mailbox is created | Type-safe forwarding without knowing payload types |
| Process failure | Links and monitors, delivered with processes | Supervision depends on them |
| Heaps | One heap per process, precise moving GC | No global pauses; whole heap freed on exit |
| Heap cells | Headerless, traced by type with `type_info`s from the stack maps | A header word costs up to 50% on large data (E7) |
| Cost rule | Advanced features cost nothing in code that does not use them | Ground, single-process code keeps the speed measured in E1 |
| Prolog-style variables | Opt-in `open(T)` types with their own layout; raw int, float and char fields boxed there (option 1) | Ground code is untouched; no alias analysis in the mode checker |
| Open-term bindings | WAM-style reference chains | Simplest: one write and one trail entry per binding |
| Open terms in messages | Copied with fresh variables, as `copy_term` | Partial structures cross processes without `from_open` |
| Standard library | Reimplemented in full from the library reference manual | Full control; no Melbourne code shipped |
| Distribution | Designed now, built in M6 | Handle and wire formats are hard to change later |
| Wire format | Self-describing and pointer-free, after Erlang's external term format; constructors by type and functor number | Portable, validated on receipt, no atom table |

## Clean-room rules: what we use from Melbourne Mercury

Our compiler and runtime are written without reading Melbourne Mercury's source code. Its compiler is GPL-2, and not reading it is what lets our code use Apache 2.0 with the LLVM exception. The rule covers every directory of source code in the repository, including the LGPL-2 runtime and library. Nothing in this design may depend on how Melbourne implements something, only on what its documents say and on what its compiler produces when we run it. The code mmc generates from our own programs is output, not source, and may be read.

**Allowed**

| Material | Use | Used in this document for |
| --- | --- | --- |
| `LIMITATIONS.md` | Read | What Melbourne does not support, such as definite aliasing |
| Reference manual, on mercurylang.org or as the Texinfo source in `doc/` | Read; it is the language definition | Syntax, types, modes and insts, determinism, solver types, `any`, negated contexts |
| User's guide | Read | Grades such as `.tr` and `.par`; options such as `--optimize-constructor-last-call` |
| Library reference manual, as published | Read for the behaviour of library predicates | Standard library |
| `README.md`, `NEWS.md`, `RELEASE_NOTES`, `BUGS`, `LICENSE`, `COPYING`, `COPYING.LIB`, and `Documentation/` (old release notes and platform READMEs) | Read | Licence terms; what changed in each release |
| Developer notes in `compiler/notes/` | Read for background only, skipping any quoted code | Nothing so far |
| Papers and theses, on mercurylang.org and in the literature | Read, skipping any quoted compiler code | The type system, mode analysis, compilation model, compile-time GC, parallel Mercury |
| Mailing-list posts | Read, skipping any patch or code excerpt from the compiler, runtime or library | Benchmark reports (E6) |
| Commit metadata: dates, authors, messages and changed file names | Counted and classified, with no diffs or file contents | Project activity (E4) |
| mmc as a black box | Built with `configure` and `make`, run on our own programs, its messages read | All experiments; differential tests |
| C that mmc generates from our own programs | Read exactly as mmc writes it | Term representation, nondeterminism, last-call-modulo-constructors (E2, E4) |
| mmc's dumps of its internal form of our own programs, from `--dump-hlds` (user's guide, "Debugging") | Read exactly as mmc writes them, for our programs only | Switch detection, common subexpression elimination and the determinism of goals (pass 5), compared by `utils/type-oracle/compare-switches.py` |

**Not read**

| Material | Licence | Rule |
| --- | --- | --- |
| `compiler/` source | GPL-2 | Never read |
| `runtime/` source, including the headers that generated C includes | LGPL-2 | Never read. Runtime macros seen in generated C are understood from the documents and from behaviour. Generated C is never preprocessed, because that would paste the headers in |
| `library/` source | LGPL-2 | Never read; the library reference manual takes its place |
| Files mmc derives from the library: installed `.int`, `.int2`, `.int3`, `.opt` and `.trans_opt` files, and the C generated for the library | LGPL-2 | Not read. They are copies or translations of library source; mmc reads them, we do not |
| All other code: `tests/`, `extras/`, `benchmarks/`, `samples/`, `tools/`, `trace/`, `browser/`, `mdbcomp/`, `deep_profiler/`, `profiler/`, `scripts/`, `java/` and the rest | GPL-2 or LGPL-2 | Never read. Our test suite is written from the reference manual |
| `boehm_gc/` | Its own licence | Not needed; our runtime does not use Boehm |

**Working rules**

- History statistics come from `git log --no-renames --format=…` or `--name-only`, on a clone made with `--filter=blob:none` and no checkout. Options that need file contents, such as `--stat`, `--numstat`, `-p` and rename detection, make git fetch the files, and are not used.
- Contributors record whether they have read Melbourne's compiler, runtime or standard library source code. Anyone who has does not write compiler, runtime or standard library code for this project without counsel's agreement.

## Reuse from LLVM and Flang

LLVM and Flang are licensed, like this project, under Apache 2.0 with the LLVM exception, so their code may be used and, unlike Melbourne Mercury's, read freely (Clean-room rules). This section records what is reused and how, so that it stays traceable.

**Depend on, don't copy.** Upstream libraries we link anyway, which get fixes for free:

| Library | Used for |
| --- | --- |
| LLVM Support: `ConvertUTF.h` (`convertUTF8Sequence`, `ConvertCodePointToUTF8`, `isLegalUTF8String`, `getNumBytesForUTF8`) | UTF-8 decoding, validation and encoding in strings, quoted names, character literals and comments |
| LLVM Support: `Unicode.h` (`columnWidthUTF8`, `isPrintable`) | Caret positions in diagnostics on lines with multi-byte characters |
| LLVM Support: `BumpPtrAllocator` | The `ASTContext` arena and the term reader's per-item arena |
| `llvm/Frontend/OpenMP`: `OMP.td`, the generated `llvm::omp::isAllowedClauseForDirective`, and `OMPDescriptors.h` | OpenMP directive and clause kinds shared with Clang and Flang; which clauses and modifiers each directive and clause allows, by OpenMP version. This replaces Flang's own clause checks |
| MLIR OpenMP dialect: `OpenMPClauseOperands.h` | Clause operand structures used to build `omp.parallel`, `omp.wsloop` and `omp.loop_nest` for a parallel loop |
| MLIR and its upstream dialects, LLVM code generation | Everything below `merk` (MLIR dialects; Lowering and code generation) |
| LLVM Option: `OptTable`, with options written in TableGen | The driver and `-fc1` command lines (Driver) |
| LLVM TargetParser, Target and Passes: `Triple`, host detection, `TargetMachine`, `PassBuilder` | Targets and CPUs, and the `-O` pipelines |
| `clang`, `clang-linker-wrapper` and `llvm-offload-binary`, run as programs, not linked | Linking, and packaging and linking device code for offloading (Driver, Linking and offloading) |

**Copy and adapt.** Language-neutral Flang infrastructure that is not available as a library:

| Upstream file (llvm-project) | Gives us | Status |
| --- | --- | --- |
| `flang/include/flang/Common/indirection.h` | An owning pointer for recursive variant types in the parse tree | Copied |
| `flang/include/flang/Common/visit.h` | `common::visit`, which compiles faster than `std::visit` | Copied |
| `flang/include/flang/Common/idioms.h` | Boilerplate for parse-tree classes | Copied |
| `flang/include/flang/Common/enum-class.h` | `ENUM_CLASS`, enums that know their names; needed by `idioms.h` | Copied |
| `flang/include/flang/Common/interval.h`, `flang/include/flang/Common/reference.h` | Half-open intervals, for provenance ranges; a non-null, assignable reference | Copied |
| `flang/include/flang/Parser/char-block.h`, `flang/lib/Parser/char-block.cpp` | Ranges of characters within a source file | Copied |
| `flang/include/flang/Parser/source.h`, `flang/lib/Parser/source.cpp` | `SourceFile`: a file's bytes, line starts and line number directives. Changed to keep the bytes exactly as read | Adapted |
| `flang/include/flang/Parser/provenance.h`, `flang/lib/Parser/provenance.cpp` | Provenances and `AllSources`, which maps them to files, lines and columns (Front end, Source manager). Reduced to source files and compiler insertions; the cooked-source maps are left out | Adapted |
| `flang/include/flang/Parser/message.h`, `flang/lib/Parser/message.cpp` | Diagnostics located by `CharBlock` or provenance, with attached notes and contexts. Fortran's language-feature warnings and the merged "expected" texts of Flang's character-level parser are left out | Adapted |
| `flang/include/flang/Common/reference-counted.h`, `flang/include/flang/Common/restorer.h` | Reference counting for attached messages; scoped restoration of a variable, for message contexts | Copied |
| `flang/include/flang/Parser/parse-tree-visitor.h` | The generic `walk` over parse-tree types | Planned |
| `flang/include/flang/Parser/openmp-utils.h` and `flang/lib/Parser/openmp-utils.cpp`: the `descriptor` namespace, `addr_if` and `WithSource` only, about 70 lines | Version-aware queries over LLVM's OpenMP descriptors: the clauses allowed on a directive, as its own clauses plus those of its clause sets; the modifiers allowed on a clause; and the OpenMP versions in which a clause is allowed on a directive. They give directive checking messages such as "allowed only from OpenMP 5.0". `WithSource` pairs a value with its source range | Planned |
| `flang/include/flang/Support/OpenMP-utils.h` and `flang/lib/Support/OpenMP-utils.cpp` | `EntryBlockArgs` and `genEntryBlock`, which create the entry block of an `omp` operation's region with block arguments for clause-defined values such as reductions and private copies. About 100 lines, depending only on MLIR and LLVM ADT, so it is language-neutral | Planned |
| `flang/include/flang/Utils/OpenMP.h` (`createMapInfoOp` only) | Building `omp.map.info` for `map` clauses. Its other helpers depend on Flang's FIR dialect | Deferred, until device offloading |

**Imitate the design, write our own.** Their structure carries over, but their content is Fortran:

| Upstream file | Why not copied |
| --- | --- |
| `flang/include/flang/Parser/characters.h` and `flang/lib/Parser/characters.cpp` | Mostly Fortran: case folding, Fortran token characters, Latin-1. The escapes differ from Mercury's, which need `\a`, hex and octal escapes closed by a backslash, `\u` and `\U` escapes up to `\U0010FFFF`, and deletion of a backslash before a newline. UTF-8 comes from LLVM Support instead. We keep the shape: constexpr predicates that do not depend on `<cctype>`, and a decoder returning a code point and the bytes consumed. |
| `flang/lib/Parser/prescan.h` | Fixed form, `INCLUDE` and preprocessing are Fortran's. Merkurius has no prescanner: Mercury's directives depend on token context, so the lexer handles them (Front end, Source manager) |
| `flang/lib/Parser/basic-parsers.h` | Its combinators read characters and tokens; ours match term shapes (Front end, Converter), though they compose the same way |
| `flang/lib/Parser/unparse.cpp` | Fortran's printing rules; our unparser has exact, canonical and fully parenthesized modes |
| `flang/include/flang/Parser/openmp-utils.h`, everything except the `descriptor` helpers | It works on Fortran's OpenMP parse-tree nodes: object designators, common blocks, iteration through `BLOCK` and `DO` constructs to find a loop nest, `allocate`, and combiner expressions. Our directive nodes are much smaller, and the associated goal is found by the converter |
| `flang/include/flang/Semantics/openmp-utils.h` | Semantic checks tied to Fortran symbols and scopes; clause validity comes from `OMP.td` instead |
| `flang/lib/Lower/OpenMP/Utils.h` | Lowering helpers tied to Fortran's parse tree, symbols and FIR; how it organizes clause processing is still a useful model for lowering parallel loops |
| `flang/tools/flang-driver/driver.cpp` and `fc1_main.cpp` | Built on Clang's driver library in its Flang mode, which a project outside llvm-project cannot extend. We keep the shape: one binary that hands off to `-fc1` |
| `flang/include/flang/Frontend/` and `flang/lib/FrontendTool/` | Tied to Fortran's options and actions, and uses Clang's `DiagnosticsEngine`. We keep `CompilerInvocation`, `CompilerInstance`, the layered `FrontendAction` classes and the `.def` option lists (Driver) |
| `clang/lib/Driver/ToolChains/Flang.cpp` (`Flang::addOffloadOptions`) | Part of Clang's driver library; it is the recipe for the offload job sequence |
| ClangIR's target records: `cir.triple` and the DLTI spec on the module (`clang/lib/CIR/CodeGen/CIRGenModule.cpp`, `CIRDataLayoutSpec.cpp`), read back by `createLowerModule` (`clang/lib/CIR/Dialect/Transforms/TargetLowering/LowerModule.cpp`) | It rebuilds Clang's `TargetInfo`, from Clang's Basic library, which we do not link. We keep the shape: a module records its target, so that it lowers alone (Lowering and code generation, "Target information") |
| Flang's `CompilerInstance::setUpTargetMachine` and `fir::support::setMLIRDataLayout` | Tied to Flang's options and FIR. We make the `TargetMachine` and the DLTI spec the same way |
| The Toy tutorial's `toy.func` (`mlir/examples/toy`) | A tutorial; `merk.proc` is defined the same way, with `FunctionOpInterface` and its parser and printer |

**Rules for copied code**

- Keep the LLVM licence header in every copied file.
- Add a line naming the original path and the llvm-project commit it came from, and stating that the file was modified. Apache 2.0 requires modified files to say so.
- List every copied file with its upstream commit and local path in `THIRD_PARTY.md`, and move its status here from Planned to Copied, then Adapted, so upstream fixes can be diffed in later.
- Prefer depending on an LLVM library to copying from it.

This is an engineering position, not legal advice.

## Experiments

Measurements made on 2026-10-26 and 2026-10-27 set four expectations. Deterministic Mercury already runs at C speed; allocation is the main gap, at 3.5× C on naive reverse; an OpenMP pragma on generated code ran 2× faster on 2 cores; and in a C model, open terms cost about 1.2× in the code that uses them and nothing elsewhere. Nothing was measured beyond 2 cores, and nothing yet on our own compiler. E7, a C model of heap cell layouts made on 2026-11-04 on another machine, decided that cells stay headerless.

**Environment:** a 2-vCPU cloud VM (Intel Xeon at 2.80 GHz, Linux x86-64). gcc 13.3.0 at `-O2`. Melbourne Mercury 22.01.9 built with configure and make from the source distribution, which is not otherwise used (Clean-room rules), with only the `hlc.gc` library grade, at default optimization. SWI-Prolog 9.0.4 run with `-O`, and CPython 3.13.17 without its JIT. Times are wall-clock, including process start.

### E1. Sequential speed against C, SWI-Prolog and Python

Each benchmark uses the same algorithm in all four languages. All four print identical results.

| Benchmark | Size | What it stresses | Result |
| --- | --- | --- | --- |
| fib | fib(40), naive recursion | Calls, integer arithmetic | 102334155 |
| nrev | Naive reverse of a 400-element list, 3001 rounds | Allocation: about 240 million cons cells | head 400 |
| queens | 12-queens, all solutions, by backtracking over lists | Nondeterminism in Mercury and Prolog | 14200 |
| floats | 5,000,000 values × 50 dependent multiply-adds | A floating-point loop | 1.2499372516e+13 |

&#91;embedded content: measured 2026-10-26 · 2-vCPU VM · best of 3 runs · seconds in the chart's rows\]

A first run at one fifth to one tenth of these sizes (best of 5) gave the same ratios to within 0.1×. In C, nrev uses a bump-pointer arena, the best case for C. The Python and Prolog programs deliberately follow the same algorithms, for example linked lists made of tuples in Python, so they measure implementation speed rather than idiomatic code.

### E2. OpenMP pragma on mmc's generated C

The benchmark maps 200 dependent multiply-adds over 2,000,000 floats in a tail-recursive Mercury loop. mmc compiled the loop to `while (MR_TRUE) { … continue; }` with the threaded array copied through temporaries. That loop was rewritten by hand as a canonical `for` loop under `#pragma omp parallel for schedule(static)` and compiled with `gcc -fopenmp`.

| Binary | Time, 3 runs | Checksum |
| --- | --- | --- |
| mmc's generated C, unchanged | 0.99–1.01 s | 1.999599040e+12 |
| Patched, `OMP_NUM_THREADS=1` | 0.98–1.04 s | Identical |
| Patched, `OMP_NUM_THREADS=2` | 0.53–0.57 s | Identical |

- The loop body never allocates. Boehm's allocation counter, read just before and just after the same loop, showed 0 bytes for 2,000,000 elements (`allocprobe.m`), so the non-thread-safe Boehm heap was not used inside the parallel region.
- `array.unsafe_set` updates in place and returns the same pointer, so ignoring the chain of array versions needed no extra work.
- With `--intermodule-optimization`, mmc specialized `int.fold_up` into a local loop that calls the body directly, with array accesses inlined as `Array->elements[Index]`. That is the shape loop formation needs (Optimizations section).

### E3. Allocation and memory

Bytes allocated by the Boehm collector for 1,000,000 `int`s, measured with `GC_get_total_bytes`:

| Operation | Bytes per element |
| --- | --- |
| Build a list directly | 16 |
| Build a cord by `snoc` or by `cons` | 48 |
| `cord.list` of that cord | 48 |

- Boehm's smallest allocation on 64-bit is 16 bytes, so a one-word cell costs as much as a cons cell.
- For nrev, a 1 GB initial heap changed nothing (3.33 s against 3.34 s), and the collector's statistics showed almost no live data. The gap to C is the cost of each free-list allocation and sweep, not the number of collections.
- Consequence: the design bets on bump-pointer nursery allocation. A header word per cell would grow a list cell from 16 to 24 bytes; E7 measured it, and the cells stay headerless.

### E4. Code generation and checking observations

| Experiment | Finding | Used in |
| --- | --- | --- |
| One value of each kind of type, compiled, with the generated C inspected | Constants are integers, functor cells have no header, wrappers vanish, small enums are packed, ground terms are static | Lowering: type representation |
| A nondet `pick` and `pair_sum` in three grades | Low-level C uses nondet frames with a `redoip` slot; high-level C uses continuations; heap marks appear only without GC; trail tickets only in `.tr` grades | Lowering: nondeterminism |
| `double_all`, with and without `--optimize-constructor-last-call` | Off by default; with it, the cell is allocated first and its open field's address passed down | Lowering: last-call-modulo-constructors |
| A Prolog difference list, translated literally | Rejected: free-free unification is a mode error | Semantic analysis: mode analysis |
| Difference-list clauses with the tail as an input, and a DCG | Accepted; the compiler reorders the body and builds the list back to front | Semantic analysis: mode analysis |
| Actor API sketch and counter client | Pass type, mode and determinism checks; deleting the `stop` case is a determinism error | Language extensions |
| Polymorphic predicates, a type class and a `type_spec` pragma (`poly.m`) | One C function per predicate, with a hidden type-info argument per type variable and elements passed as one-word boxes; a `list(float)` costs 16 bytes per element; type classes are passed as dictionaries; `type_spec` adds a copy that takes `int` and compares with `X == Y` instead of generic unify | Language extensions: generic processes |
| A generic router with an existential envelope (`router.m`, `routed.m`) | Passes type, mode and determinism checks; sending the hidden payload to a `mailbox(int)`, or wrapping a string for a `mailbox(int)`, is a type error; the envelope cell holds the type info, the destination and the payload | Language extensions: generic processes |
| Melbourne commit metadata (no file contents) | 3,991 commits in 5 years, about 99% by 3 people; LLDS `.par` grades disabled on 2026-03-26 | Risks; why parallelism is a gap |

### E5. Open-term representations (C model)

No compiler for open terms exists yet, so this experiment models in C the code each representation would produce, compiled with `gcc -O2`. Allocation is a bump pointer into an arena that is reset each round, like a nursery. Each variant builds a list of 1,000,000 ints with an open tail, closes the tail, and sums the list, 200 times. Times are the median of three runs, each the best of five, and all variants print the same checksum except the static-box row, explained below.

| Variant | Time | Words per element | Relative |
| --- | --- | --- | --- |
| Ground `list(int)`, built back to front | 0.607 s | 2 | 1.00× |
| Option 1: open layout, ints boxed | 0.714 s | 3 | 1.18× |
| Option 1, small ints from a static box table | 0.623 s | 2 | 1.03× |
| Option 2: small ints tagged | 0.617 s | 2 | 1.02× |
| Option 3: header bitmap, raw ints | 0.826 s | 3 | 1.36× |
| Ground `list(int)`, built front to back by destination passing | 0.975 s | 2 | 1.00×, baseline for the next two rows |
| Option 1, front to back through a hole bound at every step | 2.219 s | 4 | 2.28× |
| The same, with a trail entry per binding | 2.285 s | 4 | 2.34× |

- The ground rows are the same code under every option, because no option changes the ordinary layout. That is the cost rule's main requirement, and it holds by construction.
- Option 1's cost is the box: one more word per int and about 18% more time. Static boxes for small values remove nearly all of it, and so does option 2. The static-box row stores `i mod 1024` so that every box comes from the table, which is its best case.
- Building through a hole costs 2.3× the ground front-to-back build. Each step allocates a variable cell and a box, and the reader dereferences every tail. Trailing every binding added 3%, within the run-to-run spread of about 10%: the variables are the expensive part, not the trail.
- In this model a list is slower to read when it was built front to back, however it was built (0.975 s against 0.607 s for the same ground list), because the traversal starts at the cells written longest ago, which have left the cache. That is why the last two rows are compared with the front-to-back ground build.
- The program is a model, not our compiler; the ratios carry over, not the times. Source: `openrepn.c`.

### E7. Heap cell layouts (C model)

Measured on 2026-11-04, on another machine than E1 to E5: an AMD Ryzen Threadripper PRO 5965WX (32 KiB L1d and 512 KiB L2 per core, 32 MiB L3 per 8 cores), one pinned core, the build's clang at `-O2 -march=native`, medians of 5 runs. A C model of three layouts, each with 3-bit primary tags and bump allocation with a limit check, runs nrev at E1's sizes, monomorphic and polymorphic, and builds, sums and copies, as a minor collection would, lists, binary trees and lists of pairs from 1,000 to 16 million elements, so that the working set passes through every cache level. The program, its script and the full results are in `experiments/layout`; the decision is under Runtime system, "Memory management".

| | A: headerless | B: header word | C: pages of one shape |
| --- | --- | --- | --- |
| Bytes per list, tree, pair-list element | 16, 24, 32 | 24, 32, 48 | 16, 24, 32 |
| nrev, ns per cell, monomorphic | 1.95 | 2.17 (+11%) | 2.03 (+4%) |
| nrev, polymorphic | 1.97 | 2.31 (+17%) | 2.68 (+36%) |
| Summing, data in the caches | 1.22–1.60 ns per element | 0 to +21% | 0 to +12% |
| Summing, data beyond L3: list, tree, pair list | 1.25, 5.04, 1.82 | +16%, +50%, +45% | +1%, +4%, −22% |
| Building beyond L3 | 5.7–11.7 | +25% to +50% | −6% to +8% |

- B costs what its size predicts once the data leaves a cache, and little inside the caches.
- C keeps A's size, and its segregation can beat A: a list's spine apart from its elements packs 4 cons cells to a cache line. But polymorphic code chooses the shape, and so the bump pointer, at run time, and each heap needs a page for every shape it uses: 32.8 bytes per list element at 1,000 elements.
- The order of cells after a copy is the copier's, not the layout's: after a breadth-first copy, summing the 16M-node tree took 1.55 ns per node, against 6.09 after a depth-first one.
- Not modelled: real roots and stack maps, write barriers, our own code generation, and `send`'s copy, which must not overwrite the sender's cells.

### E6. Expectations for the design

- **Deterministic code:** match C. mmc already does, through gcc.
- **Allocation-heavy code:** mmc with Boehm is 3.5× C on nrev, which sets the bar. A nursery allocator should do better; that is unmeasured.
- **Parallel loops:** near-linear scaling for allocation-free bodies. Measured: 2× on 2 cores. Scaling beyond 2 cores is unmeasured.
- **Search code:** queens at 1.4× C with real backtracking, about 29× faster than SWI-Prolog.
- **Cords:** about 3× the memory of the list they hold, while the cord is kept alive.
- **Open terms:** about 1.2× ground speed for `int`-heavy open data under option 1, and nearly 1.0× with static boxes or tagged small ints. Building through holes costs about 2.3× the ground equivalent. Ground code pays nothing.

Published data points agree. Ralph Becket's port of the SPEC95 LZW compressor to Mercury ran 30–50% slower than C ([2001](https://lists.mercurylang.org/archives/users/2001-April/000898.html)). Fabrice Nicol's 2021 N-queens timings put Mercury between two naive C++ programs and about 2.5× faster than GNU Prolog ([mercury-reviews](https://lists.mercurylang.org/archives/reviews/2021-January/021741.html)).

## Architecture

The compiler is a straight pipeline of seven stages, and only the front end reads other modules, through interface files. All Mercury-specific checking happens before MLIR is generated.

&#91;embedded content: compiler pipeline · 7 stages, interface files and runtime\]

Semantic analysis writes each module's interface file, which the front end reads when compiling modules that import it; the runtime library joins only at link time.

## Front end

The front end turns source text into the program's AST, which has two parts: a parse tree that records the program exactly as written, and a semantic form that desugars each clause into a few goal kinds. Its structure follows Flang's front end more than Clang's. Source files are read unchanged; the parse tree's types mirror the grammar one to one and are immutable once built; a generic walker traverses them; and an unparser prints them back as source. Semantic results attach to the semantic form, which points back into the parse tree, much as Flang keeps symbols and typed expressions beside its parse tree. There is no lowering IR between the AST and MLIR (Semantic analysis). All behaviour follows the 22.01 reference manual.

Flang is licensed, like this project, under Apache 2.0 with the LLVM exception, so its design, and its code with attribution, may be reused. The clean-room rule covers Melbourne Mercury only (Clean-room rules).

| Stage | Input | Output |
| --- | --- | --- |
| Source manager | Source files | Each file's bytes, unchanged, with a provenance for every byte and for the end of the file |
| Lexer | One file's bytes | Tokens, with comments kept as trivia, one `directive` token per OpenMP directive, and line number directives recorded |
| Term reader | The tokens of one item | One term tree per item, in a per-item arena |
| Converter | One term | Parse-tree nodes, built by combinators over term shapes |
| Desugaring and quantification | Parse-tree clauses | The semantic form of each clause |
| Module loader | `import_module` and `use_module` | Imported declarations, as parse-tree declaration nodes |

### Source manager

- **Files are read unchanged.** Each file's bytes are kept exactly as read: carriage returns stay, since Mercury treats them as whitespace, and no final newline is added. A UTF-8 byte order mark is recognized and kept out of the text the lexer reads. Printing the text back therefore gives the file, byte for byte.
- **Provenance,** as in Flang's `AllSources`. Every byte of every source file, and every piece of text the compiler inserts, has a provenance: an offset into an unmaterialized concatenation of them all. Each file also has one provenance for its end, so that an error at the end of a file, even an empty one, has a location. A `CharBlock` points into a file's text, and `AllSources` maps between the two. Diagnostics and MLIR locations (`FileLineColLoc`) go through it to report the file, line and column. Line number directives and `pragma source_file` are recorded per file, from the line they apply to.
- **No prescanner.** Flang rewrites its input into a cooked stream first, because Fortran's directive lines, continuation lines and `#` lines can be found line by line. Mercury's cannot. Strings may contain raw newlines ("String literals may also contain embedded newlines", reference manual, Tokens), so a line inside a string can begin with `%$omp`; the same holds inside `/* … */` comments and quoted names. Line number directives "may occur anywhere a token may occur". And `0'"` is a character code, not the start of a string. Finding directives therefore needs most of the lexer, so the lexer finds them (below), and nothing rewrites the text.

### Lexer

- **Which manual.** The lexical rules are the 22.01 reference manual's (Tokens). Where 22.01 says nothing, the current manual's Syntax chapter fills the gap, because it describes what 22.01 already accepts: comments (2.4), graphic names and `;` as a name (2.7), the backquote token (2.9), and "letters" and "digits" meaning Basic Latin ones (2.2). Where the manual and mmc 22.01.9 disagree, mmc is followed, so that test programs compile with both; experiments with mmc settled the cases below. Other changes in the current manual are not adopted (Standard library, compatibility and testing, "Language version").
- **Implementation.** Hand-written, working on one source file's bytes with one character of lookahead. A token records its kind, the trivia before it and its spelling, as character ranges in the file; values are computed from the spelling by the term reader.
- **Token kinds** are those of the reference manual: names (alphanumeric, graphic such as `-->`, and quoted), variables, integers (with base prefixes, `0'c` character codes, underscores and size suffixes such as `7i8` and `42u32`), floats, strings with escapes, the backquote, `open`, `open_ct`, `close`, list and curly brackets, `|`, `,`, `end`, `eof`, and implementation-defined literals such as `$file`, `$line` and `$pred`.
- **Whitespace decides two tokens.** A `(` with no whitespace before it is `open_ct`, which makes `f(X)` a compound term, while `- (X)` is the prefix operator `-` applied to a parenthesized term. A `.` followed by whitespace, a `%` comment or end of file is the `end` token; any other `.` is the module-qualification operator. A graphic name continues through `/*`, as in mmc, so `+/*` and `./*` are each one name, and mmc reports `1 +/* c */ 2` as a syntax error at `+/*`.
- **Comments,** `%` to end of line and `/* … */`, are not tokens. A block comment ends at the first `*/`, so block comments do not nest, as mmc confirms. Each token records its exact spelling and the whitespace and comments before it, and the `eof` token holds whatever follows the last token. Printing every token's trivia and spelling in order reproduces the file byte for byte (Unparser and round-trip tests). This holds for directives too, because their continuation lines are joined only in the directive token's sub-tokens, never in the text.
- **The OpenMP sentinel.** When OpenMP is enabled (`-fopenmp`), a `%` comment that is the first non-blank text of its line, and that begins `%$omp` followed by whitespace, is a directive. Any whitespace but a newline counts as blank, including a form feed, so a sentinel is recognized whatever the indentation. Each following line whose first non-blank text is `%$omp&` continues it. The directive becomes one `directive` token holding its own sub-tokens, lexed with the normal rules. Without `-fopenmp`, these lines stay comments. A `%$omp` inside a string, quoted name or block comment is never seen, because the lexer is already inside that token.
- **Line number directives.** `#` followed by a positive integer and a newline, wherever a token may occur, sets the number of the next line (reference manual, Tokens). The lexer records it in the source file's line table and keeps the directive as trivia, so round trips still reproduce it. A carriage return before the newline is accepted, so files with CRLF line endings work.
- **Reserved:** lines beginning `%$` and a space, for conditional compilation like Fortran's `!$` lines, meaning code compiled only when OpenMP is enabled. They are not in the first version.

### Term reader

- **Output.** The reader reads one item, from its first token to the `end` token, into a term tree (`include/merkurius/Parser/term.h`). There is one node kind per way of writing a term: variables, integers, floats, strings, names, implementation-defined literals such as `$line`, compound terms (written `f(Args)` or with a prefix or infix operator), backquoted operators, list and tuple terms, higher-order applications (`Term(Args)`), parenthesized terms, and directive-annotated terms. Notation is kept for the exact unparser; the meaning is given by a canonical printer, in which lists are `'[|]'(H, T)`, tuples `'{}'(Args)`, a higher-order term `''(Term, Args…)` as the manual specifies, and parentheses vanish, so two terms mean the same when they print the same (round-trip test 4). Each node records its first and last token. Nodes live in an arena that is reset after each item is converted.
- **Variables** are numbered per item through a table from names to numbers. Each `_` is a fresh variable, and singleton warnings follow the manual's rules for names beginning with `_`.
- **Precedence climbing, hand-written** (`lib/Parser/term-reader.cpp`). `Parse(minPriority)` reads a primary term, then extends it with infix operators whose priority is at least `minPriority` (no builtin operator is postfix). An `x` argument must have a strictly higher priority than its operator, and a `y` argument a higher or equal one. Arguments of compound terms and list elements are read at the item priority, 300, with `,` as a separator rather than an operator, so `p(int::in, X)` needs no parentheses although `::` (325) binds less tightly than `,` (500). Inside parentheses `,` is an operator again. As in mmc, a tuple's contents are a single term in which `,` is an operator, and the elements are the operands of its top-level commas, looking through parentheses: `{a :- b, c}` has one element and `{a, (b, c)}` three. Nesting is limited to 10000 levels, counting brackets, arguments and operands, so `a, b, c` is three; a deeper item is an error, and `merkc -fc1` runs on a 64 MB stack so that the limit, not the stack, is what stops it. Flang writes Fortran's expression levels as a chain of combinators, one per level; Mercury's table has many more levels, so a table-driven reader is simpler and faster here.
- **Prefix operators.** A name that is a prefix operator is applied as one when the next token can begin a term; otherwise it is an ordinary name, so `-` alone is a name and `- X` is an operator term. The `fxy` operators, such as `some`, `trace`, `try` and `require_complete_switch`, take two arguments, as in `some [X] Goal`. A name immediately followed by `open_ct` is always a compound term, operator or not.
- **Operators as constants.** An operator used as a constant must be parenthesized or begin an argument, element, parenthesized term or item, and there be followed by a token that ends it, by `,` as an operator, or, for an operator with no prefix form, by an infix operator: `f(+)`, `X = (:)`, `- , a` and `; -> a` are terms, while `X = ':'`, `a = ; + b` and `- = a` are errors. The manual's examples show the first rule; mmc showed the rest.
- **Signs.** As in mmc, `-` before a number is the number's sign, whether `-` is quoted or not and whatever trivia separates them, so `- 1` is the integer `-1` and `- 1 ^ 2` is `(-1) ^ 2`; `-(1)` and `- (1)` are `'-'(1)`. An unsigned literal has no sign, so `-1u` is `'-'(1u)`, a negated zero is zero, and a negative number cannot be applied to arguments.
- **Higher-order terms and backquotes.** As in mmc, any primary term followed by `open_ct` is a higher-order term, `1(X)` and `"s"(X)` included, except that `[](X)` and `{}(X)` are the compound terms `'[]'(X)` and `'{}'(X)`. Backquoted operators are infix at priority 1380; a backquote is an operator only when `Name`, `Module.Name` or `Variable` and a closing backquote follow it, and where a term begins it is otherwise the name `` '`' ``.
- **The operator table** is data compiled into the front end. Its operators are 22.01's, numbered as in the current manual, where a higher priority binds more tightly. The current numbering is 1500 minus 22.01's, so the two describe the same parses; the current one is used because it is what Mercury programmers now read. The table, `lib/Parser/operators.def`, was generated from the 22.01 manual's table; all 131 of its rows have exactly the current manual's priority under that rule, and the current manual adds only `<<u` and `>>u`.
- **Literal values** are computed by the reader: escapes and doubled quotes are replaced in names and strings, and integers are read with arbitrary precision. As in mmc, the reader does not check that an integer fits its type; the converter reports `128i8` or `2^64`, through `IntegerTerm::InRange`. A float too large for a double is infinity, which mmc accepts.
- **Recovery.** On a syntax error the reader reports the token it expected and skips to the next `end` token, so one bad item does not hide errors in later ones. A language server uses the same recovery. This is a deliberate difference from mmc after a lexical error: mmc ends the item at the bad token and starts a new one after it, so `x = 1.5_ .` gives it two errors where we give one.

### Converter: combinators over terms

The converter is a small parser-combinator library in the spirit of Flang's, whose primitives match the shapes of terms instead of sequences of tokens:

- `functor("then", A, B)` matches a compound term with that functor and arity, and applies `A` and `B` to its arguments.
- `name`, `variable`, `literal`, `listOf(P)` and `anyOf(P1, P2, …)` match the obvious shapes.
- `construct<T>(…)` builds a parse-tree node of type `T` from the results and records the term's first and last token on it.
- `withValue(v, P)` records which alternative matched, as in `withValue(Section::Interface, functor("interface"))`, and `expect("a module name", P)` names a rule in error messages.
- `commaListOf(P)` reads the operands of `,` terms, as in `:- import_module io, list`; as in mmc, both operands may be lists, so `(a, b), c` has three elements.
- **Commitment, not backtracking** (`include/merkurius/Parser/term-combinators.h`). Each rule has a test on the term's shape, its top functor and arity, and a conversion that runs only on a term that passes it. Once a rule applies, every error below it is reported there, so a message always comes from the rule that applies and nothing is converted twice. Parentheses do not change a term's meaning, so every rule sees the term inside them.

A rule then reads like the manual's definition. In this example, `if C then T else E` is the term `else(if(then(C, T)), E)`:

```cpp
constexpr auto ifThenElse =
    sourced(construct<IfThenElse>(
        functor("else", functor("if", functor("then", goal, goal)), goal)));
```

- **One rule set per category:** items, goals, DCG goals, data-terms, types, insts and modes, each following its chapter of the manual. Declarations are in `lib/Parser/convert.cpp`, clauses, goals and data-terms in `lib/Parser/convert-goals.cpp`, and pragmas in `lib/Parser/convert-pragmas.cpp`. The rules they share are in `lib/Parser/convert-rules.h`. `anyOf` tries its alternatives' shape tests in order, which is enough for the declarations. Nearly every goal and special data-term is known by its top functor and arity, and any other term is a call or a functor term. So each of these categories is one function that tests the functor in turn. If profiling shows that this costs too much, the function can dispatch through a table.
- **Recursive categories,** types, insts and modes, are small classes declared before the rules that use them and defined after, because a `constexpr` rule cannot contain itself. `appliedName(P)` reads a possibly qualified name with its arguments, so `list.list(int)`, which is the term `'.'(list, list(int))`, is the name `list.list` applied to `int`.
- **Predicate, function and mode declarations** are read through their layers, outermost first: `all [Vs]`, `some [Vs]`, `<= Constraints`, `=> Constraints`, `impure` or `semipure`, then `pred`, `func` or `mode`, then `is Det`, `` `with_inst` ``, `` `with_type` `` (or `: Type`), and for a function `= Result`. mmc accepts the layers in this order only. `=>` binds less tightly than `<=`, which is why the manual's example of both is parenthesized. `<= c(T) is det` makes `c(T) is det` the constraint, as it does in mmc. The converter reports a declaration where some but not all of the arguments have modes, as mmc does. In a mode position a name is a mode, so `int::bound(a ; b)` is the mode `bound` applied to an inst, which mmc too reports as an undefined inst `;`/2.
- **Type, inst and mode definitions.** A discriminated union's constructors are the operands of its `;` terms, and `where` binds more tightly than `;`, so its attributes are taken from the last constructor. A constructor in braces is taken as written, so `{ int ; int }` is the constructor `;`/2 (reference manual, "Discriminated unions"). `Inst ---> Functors` is read as `bound(Functors)`. As in mmc, the function symbols of a bound inst may be numbers and strings, and the parameters of a definition must be distinct variables. Also as in mmc, an inst may be named `[]`, as the library's `exception` module's is. List syntax in a bound inst is term syntax: `[I]` is `[I | []]`, whose tail is the inst named `[]`. A longer list would need an inst named `'[|]'`, and is an error. Integer literals are checked against their type here, where they become `Literal` nodes, since the reader accepts one of any size.
- **Type classes.** A typeclass declaration's methods are converted by the same rules as `pred`, `func` and `mode` declarations, since a method is one of those without `:-`. Its constraints are superclasses and functional dependencies, `(A, B -> C)`, whose variables must be the class's parameters, as mmc requires. An instance's methods are `pred(Name/Arity) is Implementation` or clauses; the clauses are converted with the clauses of items. Without `where [...]` either declaration is abstract.
- **Clauses, goals and data-terms** are kept as written. DCG rules, state variables (`!X`, `!.X`, `!:X`), record syntax and `call`/`apply` are expanded later, by the transformations the manual defines over these forms. A clause is a `FactOrRule`, for a predicate or, with `Head = Result`, a function, or a `DcgRule` with a DCG goal for its body. Head arguments may carry modes, as in `f(X::in) = (Y::out)`. A DCG lambda's last two arguments are modes, as in `pred(X::in, in, out) is det --> G`, and are kept as its `dcgModes`. As in mmc, in `pred(!.S::in, !:S::out) is det --> G` the two are read as modes named `'::'`, which semantic analysis will report as undefined, as mmc does. As in mmc, either all of them have modes or none do. A function's head, in a clause or a `func` or `mode` declaration, may be written in field access syntax (reference manual, "User-supplied field access function declarations"). The converter gives it the function's own name and arguments: `T ^ f(A) = V` is `f(A, T) = V`, and `(T ^ f(A) := V) = R` is `'f :='(A, T, V) = R`. As in mmc, the field must be a name or a compound term, so the clause `X ^ Y = Z` defines `'^'/2`. The manual's mode declaration example `in ^ in = out` is, by this rule and in mmc, a mode for a function named `in`. Conjunctions and disjunctions are flattened through parentheses, and `C -> T ; E` within a disjunction is an if-then-else. These cases follow mmc's reading:
  - **Calls.** Any other non-variable goal is a call, including `` X `p` Y `` and a tuple, which calls `'{}'`. `try [] G` without `then` calls `try/2`.
  - **Errors.**
    - `->` without `;`, and `if ... then` without `else`.
    - The DCG-only forms `=(T)`, `:=(T)`, `T =^ F` and `^ F := T` in an ordinary goal, and `T ^ F := V` unless `T` is a state variable `!S`, which makes it a goal that updates the state variable's field (reference manual, "State variables").
    - `X = Y` and `X := Y` in a DCG body.
    - A `try` goal without `then`.
    - Trace and try parameters that are not of their documented forms, with a state variable for `io(!S)`.
    - Malformed lambda heads, such as `pred(X) :- G` without a determinism.
  - **Lambdas.** `pred(...)` or `func(...)` that is neither a lambda nor malformed is an ordinary functor.
  - **Left to semantic analysis.** mmc reports a goal construct used as a data-term, such as `X = (a, b)`, during type checking, so here it stays a `FunctorData`.
  - **Data-terms.** The special data-terms are their own nodes: conditional expressions, `@`, field selection and update, lambdas, `: Type`, `` `with_type` `` and `coerce`.
  - **DCG goals.** `{ G1, G2 }`, which the reader reads as a tuple, holds the conjunction of the goals. `&` is a DCG parallel conjunction, which mmc accepts.
  - **OpenMP.** A goal with a `%$omp` directive is an `OmpGoal` (OpenMP directives). A directive anywhere else draws a warning and is ignored.
  - **Atomic goals.** The reference manual lists `atomic` only as an operator, so `atomic [Params] G` follows what mmc 22.01.9 accepts: `outer` and `inner` are required, each `!S` or a pair `S0, S`, with an optional `vars([Vars])`, and none may be repeated. `G1 or_else G2` is an operator of goals only at the top of the goal, where the alternatives become a list; elsewhere it is a call, as in mmc.
- **Pragmas.** The pragma's name decides the forms of its arguments.
  - **Node kinds:**
    - one node kind per family;
    - `PredPragma` for the ten whose one argument names predicates or functions (`inline`, `promise_pure`, `terminates`, and so on);
    - `TablingPragma` for `memo`, `loop_check` and `minimal_model`;
    - one node each for the foreign language interface's pragmas.
  - **How predicates are named:** `p/1`, `pred(p/1)` or `func(f/1)`. A mode is named `p(in, out)` or `f(in) = out`; `type_spec`, the tabling pragmas, `obsolete_proc` and `foreign_export` take this form.
  - **Foreign code** is kept as uninterpreted text with its source range, or as an `include_file` path. The language is a `ForeignLanguage`: `"C"`, `"C#"` or `"Java"`, or the names `c`, `csharp` and `java`.
  - **Errors** follow the ones mmc reports while reading a pragma:
    - an unknown pragma name;
    - the wrong number of arguments;
    - attributes that are not in the manual's lists;
    - conflicting `foreign_proc` attributes, such as `may_call_mercury` with `will_not_call_mercury` (repeating one attribute is allowed, as in mmc);
    - a tabling or `foreign_export_enum` attribute given twice;
    - `disable_warning_if_ignored` outside `memo`;
    - `foreign_proc` arguments that are not `Var::Mode`, or a variable that appears more than once;
    - `where` on anything but `foreign_type`.
  - **mmc's undocumented forms.** The `size_limit` tabling attribute and the hidden-argument form of `specified` are accepted.
  - **Not yet implemented:** `require_tail_recursion`, `mode_check_clauses` and `oisu`. mmc accepts these, but the reference manual does not describe them.
- **Module directives, promises and solver types.**
  - `initialise` and `finalise`, also spelled `initialize` and `finalize`, name a predicate as `Name/Arity` only. As in mmc, `pred(p/2)` is an error.
  - **`mutable`** takes four or five arguments: the name, the type, the initial value as a data-term, the inst, and an optional list of attributes. As in mmc, these are errors:
    - an unknown attribute or a repeated one;
    - two foreign names for the same language;
    - the conflicting pairs `trailed` and `untrailed`, `constant` and `attach_to_io_state`, `constant` and `trailed`, `thread_local` and `trailed`, and `thread_local` and `constant` (the last one is from the manual).
  - **Promises.** `promise Goal` is a `PromiseDecl`. `all [Vars] promise_exclusive Disjunction` is a `PromiseDisjunctionDecl`, and so are the same forms with `promise_exhaustive` and `promise_exclusive_exhaustive`. The quantifier must be `all`, and the goal must be a disjunction.
  - **Solver types.** `solver type t` is an `AbstractTypeDecl` with `solver` set. `solver type t where ...` is a `SolverTypeDefn`, whose attributes must come in the manual's order. As in mmc, only `representation` is required, although the manual also requires `constraint_store`. A constraint store is one `mutable` or a list of them. Constructors and `==` are errors.
- **Trace conditions** are typed. At compile time a condition is `flag("Name")`, `tracelevel(shallow)` or `tracelevel(deep)`, or `grade(Test)`; at run time it is `env("Name")`. Either kind may be combined with `and`, `or` and `not`. The grade tests are the manual's, plus `rbmm`, which mmc also accepts. As in mmc, `compile_time`, `run_time` and `io` may each be given only once.
- **Names** are `SymName` nodes: a name and its module qualifiers, from `'.'(Qualifier, Name)` or from `__` within a name (reference manual, "The module system"), so `a.b__c` is `a.b.c`. A name that begins or ends with `__` is not split.
- **Errors.** A term that matches nothing is reported with the forms expected in its context and the start of its source, as in "expected a module name, but found '"string"'". An item with an error is left out of the tree, and conversion continues with the next item. Forms the converter does not handle yet, namely three pragmas, are reported as not yet implemented.
- **Items** convert one at a time: the reader's terms for an item live in an arena that is reset once it is converted, and the strings the parse tree keeps are copied into the parse tree's own arena. An item with a lexical error is not converted.
- **Kept small.** Rules are `constexpr` objects built from a few templates. Heavy template parsers are known for slow builds and high compiler memory, so the library stays minimal and nodes are allocated in an arena rather than moved around as values.

### Parse tree

- **Types mirror the grammar.** There is one C++ type per construct of the reference manual: each declaration kind, each goal form (`IfThenElse`, `Disjunction`, `SomeGoal`, `TraceGoal`, …), each data-term form (`FunctorTerm`, `FieldAccess`, `Lambda`, `StateVarRef`, …), and the type, inst and mode languages. A category with alternatives is a `std::variant` of its forms, so `std::visit` over it is checked for exhaustiveness, much as a `det` switch is.
- **Immutable.** Nodes are created once by the converter in the `ASTContext` arena and never change. The only fields filled later are annotation slots on names, as Flang's `parser::Name` carries a symbol pointer.
- **Generic walker.** `Walk(node, visitor)` visits every node, calling the visitor's `Pre` and `Post` for the node types it has them for, so a pass handles only the node types it cares about. Each node lists its fields once, in a `ForEachField` member, and the walker and the dump are written over that, so a new node type needs nothing else.
- **Unparser.** It prints a parse tree back as Mercury source, either exactly as written or in a canonical layout (Unparser and round-trip tests). Every node records its first and last token, literals keep their spelling beside their value, and parentheses, list notation and operator notation are recorded as written, so nothing needed to print the source back is lost.
- **Debug dumps.** Options print the parse tree and the semantic form, as Flang's do.

### Unparser and round-trip tests

The unparser is the first component to build, together with the source manager, lexer, term reader and converter it tests. Its round-trip tests then guard every later change to them. It has these modes:

- **Exact** (`-fdebug-unparse-exact`, `lib/Parser/unparse.cpp`). Prints the parse tree using the tokens it recorded, so every spelling, parenthesis, comment and space comes from the tree and the output equals the input, byte for byte. It is one generic walker over `ForEachField`: each node prints the tokens of its range, passing its children's subranges to them and printing the tokens between them itself. Every token is printed whatever the tree, so the output alone would prove nothing. What the walker checks is where the value is. It reports, as internal errors:
  - a node whose tokens are not within its parent's;
  - a node whose tokens overlap those of a node before it;
  - a variable or literal token that a node prints itself, but that none of its fields holds. A variable needs a string field equal to its name. A literal needs a number or string field, or, for a string such as `"C"`, an enum field such as a `ForeignLanguage`. Names and punctuation are a node's own syntax and are not checked.

  The last check catches a converter that drops part of a goal or an argument, if the dropped part has a variable or a literal. The tokens between items are checked only when no item was left out for an error. The first checks found that every `SymName` held the whole term it was read from, arguments included. A name's tokens are now its qualifiers and its functor's token, or the opening bracket of `{}` and `[|]`. For a qualified infix operator, as in `m.(X + Y)`, they are the qualifiers alone.
- **Minimal layout** (`-fdebug-unparse-exact-minimal-layout`). The exact mode without comments or line number directives, and with each token's layout reduced to one newline if it had one, one space if it had other layout, and nothing if it had none. It stood in for the canonical mode in test 4 before that mode existed, and remains a debugging aid.
- **Canonical** (`-fdebug-unparse-no-sema`, `lib/Parser/unparse-canonical.cpp`). Prints the same tree in a standard layout. This mode is the formatter. The tree does not record literal spellings, quoting or parentheses, so the printer takes each token's spelling from the tokens the tree recorded, and decides only the layout between tokens, from the tree:
  - four-space indentation, and one goal per line in a clause body, a disjunct or the branch of an if-then-else;
  - if-then-else, `else if` chains, `C -> T ; E` and disjunctions laid out as in the reference manual, and constructors aligned under `--->`;
  - a goal in parentheses on lines of its own when it is the goal of a scope, such as `trace` or `promise_pure`;
  - spaces around infix operators, except `::` and the `/` of `Name/Arity`;
  - line breaking by a Wadler-style pretty-printer, within 79 columns where possible: a bracketed list that does not fit puts each element on a line of its own;
  - at most one blank line between items, and none within an item.

  The layout cannot change how the text lexes:
  - nothing goes between a term and an open_ct, or after a `.` name;
  - an open keeps the layout before it;
  - two tokens whose spellings would run together, such as two graphic names, are kept apart;
  - within a name, variable or literal, such as `io.write`, `!.X` or `-1`, and after a graphic prefix operator, the layout is the source's.

  Comments keep their text, and stay attached to their token. A comment on the same line as the token before it stays at the end of that line; a comment on a line of its own precedes its token, at its indentation. Line number directives are dropped. An item that was left out for an error is printed as written.
- **Fully parenthesized** (`-fdebug-unparse-parenthesized-no-sema`). The canonical layout with every data-term, goal, type, inst and mode that is written as an operator term in parentheses. It shows how the reader grouped each operator, and test 4 checks it too.

Tests, run on every file of every corpus:

1. **Lexer round trip.** Printing every token's trivia and spelling gives back the file, byte for byte.
2. **Exact round trip.** Parsing a file and unparsing it exactly gives back the file, byte for byte, and the unparser reports nothing. `utils/round-trip/round-trip.py` runs this test and tests 3 and 4 on the test files that convert without errors (`test/Parser/unparse/round_trip.test`). The fuzz target runs them on every input that converts without errors, and this test on the rest too (`test/Parser/fuzz_corpus.test`). `test/Parser/unparse/canonical.test` checks the canonical layout of a module laid out carelessly against the expected one.
3. **Canonical fixed point.** Parse the file and unparse it canonically, giving C1; parse C1 and unparse it canonically, giving C2. C1 and C2 must be identical.
4. **Meaning preserved.** The parse trees of the file, of C1 and of its fully parenthesized form must be equal once locations and trivia are removed. Test 3 alone would pass a printer that consistently dropped something; this test catches it. The trees are compared through `-fdebug-dump-parse-tree-no-sema`, which prints neither. Writing the tests found that the converter read `t ---> a ; b where ...` as one constructor named `;`, because `where` binds less tightly than `;`.
5. **Against mmc's term parser.** A term oracle (`utils/term-oracle/term_oracle.m`) reads items with the Mercury library's term parser, through its documented interface, and prints them in our canonical notation; `compare-terms.py` requires it to agree with `merkc -fc1 -fdebug-dump-terms`, item by item. The lit tests under `test/Parser/oracle` run it on hand-written cases, on every pair of builtin operators (about 16000 items), and on random items from a grammar of the surface syntax, when mmc is found (`MMC`, or the `PATH`). Every rule above marked "as in mmc" came from a difference it found. Such a difference is a fact about mmc, settled by running it, never by reading its source.
6. **Against mmc's printer: not run.** Tests 1 to 4 compare our parser with itself, so a parser that is consistently wrong about an operator's priority would pass them all. The plan was to compare our parse of a file with our parse of mmc's `--convert-to-mercury` output, which parenthesizes every operator term. That output is too far from a reprint of the source to compare:
   - it expands DCG rules;
   - it renames `_` to `V_1`;
   - it module-qualifies field names;
   - it splits declarations and imports;
   - it even prints invalid syntax for some trace goals.

   Test 5 already catches what test 6 was for: it compares our reader with Mercury's own term parser, item by item, on every pair of builtin operators among other cases.
7. **Fuzzing.** `tools/merkurius-term-fuzzer` checks, for any input, that the lexer, reader, converter and desugaring do not crash, that the tokens unparse to the input, that tests 2 to 4 hold, with and without `-fopenmp`, and that each item's canonical term reads back as itself; `utils/fuzz/run-term-fuzzer.sh` runs it under libFuzzer with the address and undefined behavior sanitizers. Inputs that once failed go in `test/Parser/Inputs/fuzz`, and a lit test replays them and every Mercury file in the tests.

   **Running the fuzzer.** `utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]` builds the target with the build tree's clang and fuzzes for `SECONDS` (default 180) in `JOBS` processes (default 32). Each piece of work is fuzzed with the defaults before it is reported, which takes about 3 minutes in all.
   - **Building:** the script writes a ninja file, so the sources compile in parallel and only those that changed are recompiled: about 45 seconds from scratch, a few seconds after a typical change.
   - **Fork mode:** the fuzzer runs with libFuzzer's `-fork`, in which one parent process keeps the corpus and starts short fuzzing jobs. An input is added to the corpus only when it covers something new, so the corpus stays at a few thousand files. Earlier runs used `-jobs`, where every process added its inputs to the shared corpus and replayed all of them at startup; the corpus grew to over 250,000 files, and replaying it took most of each run.
   - **Where:** the fuzzer, corpus, log and crash files are in `build-merk/fuzz/term`. Each run first copies every `.m` file in `test/` and `utils/` into the corpus.
   - **Reading the results:** the run's output is in `fuzz.log`. Its last `#N:` line gives the number of inputs run and the coverage, and it ends `INFO: exiting: 0` for a clean run. A failing check prints a line starting `merkurius-term-fuzzer:` and stops the run, which then exits with status 77, and the input that failed is saved in `artifacts/`, which should stay empty.
   - **Checking the run is fresh:** `fuzz.log` must be newer than the build. A build that fails leaves the previous run's log in place, so a run that silently did not start can look like a clean one.

**Corpora:** the reference manual's examples, our own test programs, and generated programs. Melbourne's library and test sources are not used, because nobody may read them to diagnose a failure (Clean-room rules).

### Semantic form

The semantic form is the desugared body of each clause, built from a small set of node kinds. Mode analysis and MLIR generation work only on these:

| Node | Meaning |
| --- | --- |
| `Conj`, `ParConj` | Conjunction, parallel conjunction |
| `Disj` | Disjunction |
| `IfThenElse`, `Not` | If-then-else, negation |
| `Unify` | `X = Y`, or `X = f(Y1, …, Yn)` with exactly one functor, including a closure built from a lambda |
| `Call`, `HOCall` | Call to a named predicate, call of a closure |
| `Scope` | One goal wrapped in a quantifier, `promise_*`, `require_*`, `trace`, a purity annotation or an OpenMP directive |
| `Try` | A `try` goal: the goal, its then and else parts, and each catch branch's pattern, as unifications with the exception's variable, and goal |
| `Atomic` | An `atomic` goal: the outer state's versions before and after it, and each `or_else` alternative with its inner state's first and last versions |

Each node points back to the parse-tree node it came from, and diagnostics use that pointer, so an error found in the semantic form is reported at the construct the programmer wrote. Together, the parse tree and the semantic form are the AST.

Desugaring builds the semantic form in this order:

1. State variables (`!X`, `!.X`, `!:X`) become numbered variables, following the manual's scoping rules.
2. DCG rules become clauses with two extra arguments.
3. Field access and update become calls to the field functions; function applications and unification expressions become unifications with fresh variables.
4. Lambdas become closure constructions, with their captured variables made explicit.
5. `:- mutable` and `:- process` declarations become ordinary predicates, added as implicit declarations (Language extensions).
6. Bodies are put in superhomogeneous form: each unification holds at most one functor, nested functors become chains of unifications through fresh variables, and head arguments become distinct variables unified in the body.
7. Quantification computes each goal's nonlocal variables by the manual's implicit quantification rules, and renames apart variables that are local to different branches.

An instance's method clauses are desugared in the same way, into the form's method clauses, apart from the module's clauses, and a method defined by name becomes the clause that calls its implementation (see "Type classes" under type checking).

Desugaring also numbers the goals and unifications of each semantic form densely, once. Semantic analysis indexes its per-procedure tables by these numbers.

**Implementation** (`lib/Semantics`, `semantic-form.h` and `desugar.cpp`). Each clause is desugared on its own, in one pass over its parse tree:

- **Variables** are numbered densely per clause in a table of `VarInfo`s: program variables by name, each `_` as an anonymous variable of its own, and the fresh variables desugaring makes. Goals refer to variables by number.
- **Heads.** An argument that is a variable not already in the head stays as it is. Any other argument, or a repeated variable, becomes a fresh head variable, unified with the argument at the start of the body. A function's result is the last head variable.
- **Data-terms** are flattened into unifications with at most one functor. The dump quotes a name that is not a plain identifier, as Mercury does, so the character `'0'` and the integer `0` differ. A functor is a name, which type checking resolves to a constructor or a function, `[|]`, `[]`, a tuple, a literal, an implementation-defined literal such as `$line`, or `coerce`. Nested functors become chains through fresh variables, listed outermost first, as written. `X @ Y` unifies both sides with one variable, and `Term : Type` records a type qualification against the variable that holds `Term`.
- **Higher-order calls.** `P(X)`, unqualified `call(P, X)`, `Y = F(X)` and unqualified `Y = apply(F, X)` are `HOCall`s; a function's result is its last argument.
- **Goals.** `true` is the empty conjunction and `fail` the empty disjunction. A sequential conjunction absorbs the conjunctions inside it; a parallel one keeps its conjuncts. Following the manual's state variable rules, `G1 => G2` is `not (G1, not G2)`, `G1 <= G2` is `G2 => G1`, `G1 <=> G2` is both, and `all Vs G` is `not some Vs not G`; `X \= Y` is `not X = Y`. As in mmc, the negation of a negation is the goal inside, and the negation of a conjunction of negations is the disjunction of their goals, so `\+ (\+ A, \+ B)` is `(A ; B)` and `all [X] (A => B)` is `not some [X] (A, not B)`; the goal may then bind variables used after it, which the manual's `if Goal then fail else true` does not (see the register). The numbers of the dropped negations are left unused. `event p(...)` is a call marked as an event.
- **Scopes** keep their parse-tree goal, which holds their parameters, such as a trace goal's conditions or an OpenMP directive.
- **State variables** are threaded through the clause as it is desugared. This gives the result of the manual's `transform_goal` and `transform_clause` without their chains of `X1 = X2`. Each state variable in scope has a current version, numbered from 0 within the clause and printed as `S!0`, `S!1`, ….
  - **Calls.** In a call's arguments `!X` is `!.X, !:X`. An atomic goal that names `!:X` makes the next version, which is current after it; its `!.X` is the version before.
  - **Branches.** The arms of a disjunction or if-then-else that leave a state variable at different versions each end by unifying theirs with a new one. An if-then-else's condition updates are seen in its then part only, and a negation's updates are not seen after it.
  - **Scopes.** The head's `!X`, `!.X` and `!:X` bring it into scope for the clause. Its first version is the head's `!.X`, and its last becomes the head's `!:X`, with a unification only when the body leaves it at a version already in the head. `some [!X]` and a trace goal's `io(!X)` and `state(M, !X)` bring a new one into scope inside the goal, hiding any of the same name; for a trace goal, its first and last versions are the scope's variables.
  - **Errors,** as mmc reports them: a state variable not in scope, `!X` as a side of a unification, `!X` anywhere but a call's or head's argument (mmc reads `f(!X)` as the functor `!` applied to `X`), `!X` as a function's result, and `!.X` or `!:X` among a quantifier's variables. Reading `!.X` before `X` has a value is a warning, as in mmc.
- **Lambda expressions** become the functor of a unification, `F = pred(X::in, Y::out) is det`, with the body desugared like a clause's, under it. The head's arguments become distinct variables, unified at the start of the body when they are not, and each keeps the mode it was written with. The variables in the arguments are local to the lambda (reference manual, "Lambda expressions"), so the dump prints such a variable with its number, as `X$2`, when the clause has another `X`. Variables of the result and the body that are not in the arguments may be shared with the clause; quantification decides which.
  - **State variables.** The state variables in scope outside a lambda may only be read in it, as `!.X`. `!:X`, or `!X` as an argument, is an error, as in mmc ("cannot use !:S here due to the surrounding lambda expression"). The head's `!.X` and `!:X` name state variables of the lambda's own, which hide those outside, as does `some [!X]` in its body. `!X` cannot be a lambda's argument, since a lambda has no syntax for the modes of its two parts.
  - **DCG lambdas.** `pred(X::in, in, out) is det --> G` has the DCG transformation applied to `G`, and its last two arguments, whose modes are the head's last two, are the first and last versions of the DCG argument.
- **Conditional expressions** follow mmc: `X = (if C then T else E)` is the goal `if C then X = T else X = E`, with state variables threaded as in that goal, so an update in `C` is seen in `T` and merged after the expression. The manual allows only `!.X` in a conditional expression; see "Where mmc and the manual disagree". Within a term, the expression is a fresh variable that the if-then-else binds.
- **Field access** follows the manual's transformation ("Record syntax"). `T ^ f(A) ^ g` is `g(f(A, T))`, as a chain of unifications with field access functions. `T ^ f(A) := V` is `'f :='(A, T, V)`. With more fields, `T0 ^ f ^ Rest := V` is `Old = f(T0)`, then `New` is `Old ^ Rest := V`, and the result is `'f :='(T0, New)`. The update function's name is the field's with ` :=` added, keeping its module qualifiers. `!S ^ Fields := V` is `!:S = !.S ^ Fields := V`. The DCG goal `T =^ Fields` is `T = V_in ^ Fields`, which leaves the DCG argument as it is, and `^ Fields := T` is `V_out = V_in ^ Fields := T`. Type checking later resolves each name to a field access function, generated or written by hand, or to another function.
- **DCG rules** thread their two implicit arguments as the versions of a hidden state variable, named `dcg`, which no Mercury variable can be, since a variable's name starts with a capital letter or `_`. This gives the result of the manual's DCG-transform: `Head --> Body` is `Head(V_in, V_out) :- Body`, with `V_in` and `V_out` the first and last versions. `[T1, T2]` is `V_in = [T1, T2 | V_out]`, `[]` and `{ G }` leave the version as it is, `=(T)` is `T = V_in`, `:=(T)` is `V_out = T`, and a call, a higher-order call, `call(P, …)` or a variable gets the two arguments appended. Disjunctions, if-then-elses, negations and quantifiers follow the rules for state variables.
- **Back pointers.** Each node records the tokens of the parse-tree node it came from, which is what diagnostics need; scopes also point to their parse-tree goal.
- **`try` goals** (reference manual, "Exception handling") keep their structure, since an exception is not a failure. The goal's updates of state variables are seen in the then part only, as in an if-then-else: a run of a test program shows that in mmc a catch branch sees the version from before the goal. An `io(!IO)` parameter's state is the exception, since the goal's I/O cannot be undone: it is threaded through the goal, and every branch continues from the version after it. A catch's pattern is unified with a variable for the exception, apart from its goal, because a pattern that fails to match chooses the next branch, while a goal that fails makes the branch fail.
- **`atomic` goals,** transactions of mmc's software transactional memory, follow mmc 22.01.9, since the manual lists `atomic` only as an operator. `outer(!IO)` reads the current version before the goal and makes the next one after it, and, as in mmc, `IO` is not visible inside. `inner(!S)` is a new state variable in each `or_else` alternative, with its own first and last versions; `outer(S0, S)` and `inner(S0, S)` name the versions as plain variables. The alternatives' updates of other state variables are merged as a disjunction's are.
- **A scope's list of variables** may name state variables: as in mmc, `!.S` in the list of `promise_equivalent_solutions` or `require_complete_switch` is the version before the goal, `!:S` the one after it, and `!S` both.
- **Quantification** runs on each clause once it is desugared, and records each goal's nonlocal variables in a table of the clause, indexed by goal number. A goal's nonlocals are the variables that occur in it and outside it. "Outside" leaves out the goals parallel to it, as the manual defines them ("Implicit quantification"): the other disjuncts, the else part for the condition and then part and the other way round, and the goals of other lambda expressions. The other branches of a `try` or an `atomic` goal are parallel too. A variable that occurs in a disjunct, an if-then-else's parts, a negation or a lambda expression's body, and not outside it, is local to it. Where it is local to two parallel goals, it is two variables: desugaring gives each name one variable, so quantification gives the later goal a copy, which the dump prints as `X$N`. mmc does the same: the program `( X = 2, ... ; X = "s", ... )` type-checks, with an `int` X and a `string` X.
  - **Lambda expressions.** Whether a lambda expression captures a variable is decided from the occurrences outside all lambda expressions, since two lambda expressions' goals are parallel. A goal beside a lambda expression then counts the variables it captures as occurrences, so `X = 1` before a closure that reads `X` has `X` as a nonlocal.
  - **`some`** quantifies new variables. Where the name is used outside the scope too, the variable is renamed within it and, as in mmc, reported with "variable X has overlapping scopes". The same warning applies to a lambda argument that has the name of a variable outside the lambda expression. At the top of an if-then-else's condition, `some` quantifies over the condition and the then part (manual, "Goals"), so the then part sees its variables.
  - **State.** The I/O state versions of a `try` and the outer versions of an `atomic` goal are occurrences at that goal. The state after a `try` goal is therefore shared with the catch branches, not renamed apart.
  - **Variable lists** of promises, `require_*` scopes and `atomic`'s `vars` are references, not occurrences, so they do not make a variable nonlocal. They are renamed with the goal.
  - `-fdebug-dump-nonlocals` prints the semantic form with each goal's nonlocals after `~`.
- **Not yet desugared:** `mutable` declarations, which are reported as not yet implemented and left out, and `process` declarations, which come with processes.

### Where mmc and the manual disagree

Where mmc 22.01.9 and the reference manual disagree, mmc is followed, so that test programs compile with both. The manual is more often the one that is wrong, since the compiler is what programs are tested against and the manual's examples are not; `in ^ in` below is one. The exceptions are mmc bugs, which we must not copy: an obscure corner case is the likeliest place for one. So each case is recorded here with its evidence, which is an experiment with mmc, never its source, and the side taken. Before a case is settled for mmc, the public reports are searched for the behaviour: the issues of the GitHub repository Mercury-Language/mercury (for instance with `gh search issues --repo Mercury-Language/mercury`), and the archives of the mercury-users and mercury-developers lists at lists.mercurylang.org. Only the descriptions and discussion are read, not any compiler code or patches quoted in them; the mercury-reviews list, which is mostly patches, is not used. The Mantis tracker at bugs.mercurylang.org that Mercury's `BUGS` file names refuses outside access, as of 2026-10-30. A reported bug reverses the choice.

| Case | Manual | mmc 22.01.9, and the evidence | Followed |
| --- | --- | --- | --- |
| State variables in a conditional expression | Those outside may only be read, as `!.X` ("State variables") | `!:X` and `!X` are allowed in the condition. `Y = (if q(!S) then 10 else 20), !:S = !.S + Y` threads `S` as the if-then-else goal does: run with `S` = 1, where `q` adds 1, it gives 12, so the update is seen after the expression | mmc, pending a check of the tracker |
| `:- mode in ^ in = out is semidet` | Given as the mode of the field access function `elem` ("User-supplied field access function declarations") | A mode for a function named `in`, by the rule that the field is the right operand of `^` | mmc; the manual's example is wrong, and `in ^ elem(in) = out` is what it means |
| Lexical rules | See "Lexer" | | mmc, each case settled by an experiment |
| An `atomic` goal's alternative that does not use its inner state, as in `... or_else Y = 0` | Not described | A mode error: the inner state's final version is left free in that alternative, though it is the alternative's first. No report found among the GitHub issues | Neither side's text decides it; we accept it, since desugaring gives such an alternative an inner state that it leaves as it was. Check again when the STM is implemented |
| `require_switch_arms_<det> [X] Goal` whose goal is a switch on another variable | An error: "If GOAL is not a switch on VAR ... the compiler is required to generate an error message" ("Goals") | No such error; the arms of the switch on the other variable are checked against `<det>`, and their errors name that variable. `require_complete_switch` reports it. No issue on GitHub mentions it | mmc |
| A `promise_equivalent_solution_sets` goal without `arbitrary` | It must have at least one `arbitrary` goal inside it ("Goals": "of which there should be at least one") | No error or warning; the goal is as its own goal, with no cast. No issue on GitHub mentions it | mmc |
| A function mode with more than one solution | Any mode of a function whose arguments are all input can only be `det`, `semidet`, `erroneous` or `failure` ("Determinism categories") | Only the primary mode, `f(in, ...) = out`, is checked, and by its inferred determinism, not the declared one. A body with more than one solution is "Error: invalid determinism for \`f'(in) = out: the primary mode of a function cannot be \`multi'", naming the inferred determinism (`nondet` for a body that can also fail), whether the mode is declared in the `func` declaration or by a `mode` declaration. The declaration is then checked as usual. `:- mode g(in) = out is multi.` with a `det` body has only the warning that the declaration could be tighter. Other modes whose arguments are all input are not checked. No issue on GitHub mentions it | mmc |
| The type `character` | The primitive character type is `char` ("Character type") | `character` is the primitive type, of no module: `:- pred p(character::in)` compiles without imports, and the library's `char` module defines `char` as `character`. `char` itself is not primitive: without `import_module char`, `:- pred p(char::in)` is "undefined type `char'/0" | mmc: `character` is the primitive type, and `char` a type only where `char` is imported (`resolve/undefined.m`) |
| Errors during type inference | A clause is in error when it has no valid type assignment ("Predicate and function type declarations"); an inferred type is the most general valid one ("Type inference" in the user's guide), so only the final types decide | mmc reports the type errors of the first pass of inference that has any, anywhere in the module, and stops there. An error that only later passes would find is not reported: with `q(Y) :- Y = "a"` inferred, `p(X) :- X = 1, q(X)` draws no error when some other clause has one. An "error" that only an early pass sees is reported: with `mymap` undeclared, `lengths(Ss) = mymap(string.length, Ss)` is "ambiguous overloading", because in the first pass `mymap`'s argument may still be `length/1` or `length/2`. Declaring `mymap` removes it. The `Inferred` lines it then prints are first-pass guesses, such as `func sum(T2) = T2`. merkc infers to the fixpoint quietly and then checks every clause by the final types. No report found among the GitHub issues | The manual: an mmc bug. The type oracle (utils/type-oracle) allows for it |
| Inferred declarations | `Inferred` lines are declarations ("Type inference" in the user's guide) | They name each type variable after its type's declaration, so two can print alike: `p(_, Y) :- Y = []` and `p(X, _) :- X = leaf`, with `tree(T) ---> leaf ; ...`, print as `pred p(tree(T), list(T))`, though callers may use `p` at `tree(int)` and `list(char)`, which declaring the printed line forbids | mmc's typing, which is the manual's; merkc names the variables T1, T2, ... |
| A fixpoint of type inference | Inference repeats until the types do not change ("Type inference" in the user's guide) | `f(H1, _, H3) = H4 :- H4 = f(H3, {H1}, H1)` exceeds the iteration limit, at any limit, though the signature, `func f(T1, T2, T3) = T4`, is the same in each pass and mmc accepts it declared. With two arguments, as in `f(H1, _) = H4 :- H4 = f(H1, {H1})`, it converges. No report found among the GitHub issues | The manual: an mmc bug. The type oracle allows for it |
| Ground constraints | Each constraint on a type class, an instance or a predicate must contain at least one type variable ("Type class constraints on predicates and functions", "... on type class declarations", "... on instance declarations") | `c(T) <= d(int)`, `instance d(s(T)) <= d(int)` and `:- pred p(int) <= c(int)` are accepted | mmc: the restriction has no evident purpose, and a ground constraint is merely checked where it is used |
| An instance that duplicates an imported one | "A program may not contain more than one instance declaration for a particular type ... and typeclass" ("Instance declarations") | With `classes` declaring `instance shown(int)`, a module that imports it and declares `instance shown(int)` again passes `--errorcheck-only`. Linking the two fails: "multiple definition of `<instance declaration for classes.shown(int/0)>'". An instance with other constraints than the imported one is reported, as "incompatible" constraints | The manual: the error is reported when the module is compiled |
| Instances that overlap through a repeated variable | The restrictions on instance types "ensure that there are no overlapping instance declarations" ("Instance declarations"), but `bintree(T, T)` is allowed, and overlaps `bintree(K, V)` | `instance d(p(A, B))` and `instance d(p(A, A))` pass `--errorcheck-only`; the C compiler then rejects mmc's code for the two methods ("conflicting types for `ovl__ClassMethod_for_ovl__d____ovl__p__arity2______ovl__name_1_1_f_0'"). mmc reports only instances that are the same up to renaming. No report found among the GitHub issues | The manual: an mmc bug. Instances overlap if their type constructors are the same |
| A constructor as a method's implementation | `func(M/N) is F`: "The FUNCNAME must name a ... function" ("Instance declarations") | mmc checks the definition as `M(X1, ..., XN) = F(X1, ..., XN)`, so a constructor of the right type is accepted, and compiled: `func(wrap/1) is box`, with `box(T) ---> box(T)`. No GitHub issue | mmc, since the constructor is a function of its fields' types (`test/Semantics/types/typeclasses/methods.m`) |
| Inferred constraints and functional dependencies | A constraint's variables need only be in the closure of the signature's under the functional dependencies ("Functional dependencies"), and inference gives the most general valid type ("Type inference" in the user's guide) | Inference keeps only constraints on the head's type variables: `hidden(X) = 0 :- _ = baz(X)`, with `baz(A, B) <= (A -> B)`, is reported as an unsatisfied `baz(A, B)`, though `:- func hidden(T) = int <= baz(T, U)` is accepted. No GitHub issue | mmc, since what merkc accepts should compile with mmc (`test/Semantics/types/typeclasses/improvement-undetermined.m`) |
| A crash in determinism analysis | Not described | `f = P :- P = ((func(X) = X) - 1), P = ((func(Y) = Y) - 2)`, declared or not, is "Software Error: ... det_infer_unify ... complicated_unify but no fail context". It should be the mode error for unifying higher-order values. No report found among the GitHub issues | Not followed. The type oracle compares only merkc's own checks for such a module |
| A closure of an existentially typed predicate or function | Not described ("Creating higher-order terms", "Existential types") | `P = e_foo`, with `:- some [T] pred e_foo(T::out)`, and `P = e_bar(1)` are "error: undefined symbol `e_foo'/0", declared or not. No GitHub issue | mmc, with the reason given: only a call determines the existential types, which the closure's type would need (`test/Semantics/types/existential/predicate-errors.m`) |
| Inferred existential declarations | `Inferred` lines are declarations ("Type inference" in the user's guide) | Two existential variables can print alike: `pairs(X, Y) :- e_foo(X), ec(Y)` is `some [T, T] (pred pairs(T, T) => (sh(T)))`, and that, declared, crashes mmc ("map.det_insert: key already present"). No GitHub issue | Not followed: merkc names them apart, `some [T, T_2]`, and reports a variable quantified twice as an error (`test/Semantics/types/existential/quantified-twice.m`). The type oracle settles such a difference by merkc's checks |
| Unifying values of a type with existential constraints | A unification of two bound values is a test ("Unifications") | With `box ---> some [T] (box(T) => showable(T))`, `B = 'new box'(42), C = B` is "error: implicit determinism declaration not satisfied" in mmc's own `__Unify__` for `box`, and in a larger module the same crashes mode checking ("modecheck_queued_proc ... found detism error"). No GitHub issue | Not a type error, so it does not concern merkc's checker; the type oracle's typed version does not qualify variables of such types, since its `V = (V : T)` goals set it off |
| An abstract type as a supertype | The supertype's definition must be "in scope where the subtype definition occurs, by normal module visibility rules" ("Subtypes") | `:- type public =< secret ---> secret.`, where `secret` is abstract in the imported interface, is accepted, and compiles, checked against the definition the module cannot see; an abstractly exported subtype is rejected ("cannot be a supertype because it is an abstract type"). No GitHub issue | The manual: an abstract type's constructors are not visible, so it cannot be a supertype (`test/Semantics/types/subtypes/abstract-supertype.m`) |
| Coercing an abstract type | Both types must be discriminated union types ("Type conversions") | A coerce of an abstract type that the module imports, such as `secret(citrus)` to `secret(fruit)`, or to itself, crashes mmc ("build_type_param_variance_restrictions ... Unexpected: not du type"); one of the library's `map`, which is abstract, is accepted. An abstractly exported subtype is rejected. No GitHub issue | The manual: an abstract type is not a discriminated union type here, so it cannot be coerced (`test/Semantics/types/coerce/abstract.m`); merkc's stub library has no `map` |
| A coerce through a tuple parameter | A tuple is not among the places where a parameter must be the same ("Type conversions") | `in_tuple(T) ---> in_tuple({T, int})`, coerced from `in_tuple(citrus)` to `in_tuple(fruit)`, is type-correct, but mmc's mode analysis of it crashes ("modecheck_coerce_from_ground_make_bound_functor ... missing constructor for result type"). No GitHub issue | Type-correct, as both say; mode analysis will decide (`test/Semantics/types/coerce/tuple-parameter.m`) |
| A method and a predicate of one name and arity | A predicate or function is declared once ("Predicate and function type declarations") | Both report the later declaration as "multiply defined". When the method comes first, mmc also drops the method, so an instance that defines it is told "the type class has no predicate method named `mm'/1"; when the predicate comes first, both drop the method | Keep the first declaration, so a method declared first stays the class's, without mmc's cascading error; not compared |
| A pure declaration promised semipure | A purity pragma promises the declared purity ("Promising that a predicate is pure") | `:- pred p.` with `:- pragma promise_semipure(p/0).` is an error worded "declared pure but promised pure". No GitHub issue | The same error, worded "declared pure, but promised semipure" (`test/Semantics/purity/pragmas.m`) |
| An unannotated impure_apply | "In addition to using an `impure` or `semipure` operator on the unification which invokes the higher-order function application, you must also use `impure_apply` or `semipure_apply`" ("Purity annotations on higher-order calls") | `Y = impure_apply(F, X)` without `impure` is accepted; the application still makes the clause impure. No GitHub issue | mmc: no annotation needed, and the clause is impure (`test/Semantics/purity/lambdas.m`) |
| A conditional expression in a term assigned to an unused variable | `_R = wrapper(if C then 0 else J + 1)` is a valid goal; the expression's variable appears outside the if-then-else, so it is not local to its branches ("Implicit quantification") | A mode error, "the variable `V_10` is ground in some branches but not others", though the same goal with a used `R` compiles. GitHub issue 120, reproduced with 22.01.9; the reporter closed it without an answer | The manual: an mmc bug. The desugared form binds the variable in both branches, and mode analysis must accept the program |

| Conflicting constraints on an inst parameter after `<=` | Not described ("Constrained polymorphic modes") | `:- (mode q(in(I), out(I)) is det) <= (I =< ground, I =< unique)` is accepted, and the first constraint is used: with `unique` first, a call with a ground argument is a mode error. The same conflict within the modes, `in(I =< ground), out(I =< any)`, is "inconsistent constraints on inst variable I". No GitHub issue | mmc (`test/Semantics/modes/constraints.m`) |
| A partially instantiated term unified with a bound one | Mode-correct: abstract unification binds the free parts ("Modes") | When an implied mode, or a test such as `V3 = V1` with `V1` of inst `unique(tb(free))`, unifies such a term with a ground one, mmc reports a mode error in its own `'__Unify__'` procedure, at the goal ("in argument 1 of unsafe_type_cast", or "mode error in unification of `HeadVar__1'"), or a determinism error in it, for a term of a bound inst such as `unique(ta)` ("implicit determinism declaration not satisfied"). GitHub issue 74, "Mode error in semidet list pattern matching without intermediate variable", open, shows the same message | The manual: an mmc bug. merkc accepts the programs, and `compare-types.py --error-lines` ignores mmc's messages about `'__Unify__'` (`test/Semantics/modes/random-modes-2.test`) |
| A unification of two partially instantiated terms after branches | A mode error, since their free parts would be aliased, and mmc does not track aliases (Overton, 3.2) | mmc's mode analysis crashes: "Software Error: predicate `hlds.instmap.merge_instmapping_delta_vars'/9: Unexpected: error merging var 7", on a module reduced from a generated one, where an if-then-else leaves a variable with free parts in both branches and a disjunction then unifies it with another such term. No GitHub issue | The manual: an mmc bug. merkc reports the mode error (`test/Semantics/modes/merge-crash.m`), and `compare-types.py --error-lines` notes the crash and goes on |
| A declaration with inconsistent inst constraints | One error for the declaration ("Constrained polymorphic modes") | After "inconsistent constraints on inst variable I", mmc drops the declaration, so each clause is also "clause for predicate `p'/2 without corresponding `:- pred' declaration" | The error only: merkc keeps the declaration, with each variable's first constraint |
| Modes that differ only in inst variables | Not described ("Constrained polymorphic modes"); `in(I)` gives a call more precise insts than `in` | Duplicates: `:- mode r(in(I), out(I))` with `:- mode r(in, out)`, or with `r(in(J), out(J))`, is "duplicate mode declaration", as if each inst variable were its constraint. No GitHub issue | mmc (`test/Semantics/modes/procedures.m`) |
| A higher-order call whose inst has inst variables | The inst variables of the called term's modes are bound by its arguments, as at a call (Overton, 4.2.5, HO-CALL) | They are taken to be the caller's own inst variables of the same number, whatever their names, which the arguments do not bind: with `P = p0`, `:- pred p0(int::in(I), int::in)`, `P(1, X)` is a mode error in a predicate with an inst variable of its own (`in(J)`), and accepted in one without. A first-order call renames them apart. No GitHub issue | mmc (`test/Semantics/modes/polymorphic-mode-errors.m`) |
| An inst variable narrowed by a unification, passed to an inst variable | The argument is a sub-inst of the constraint, so the call binds the callee's variable to it ("Constrained polymorphic modes"; Overton, 4.2.4) | After `X = a`, with `X::in(I)`, `id(X, Y)`, `:- pred id(T::in(I), T::out(I))`, is a mode error: "arguments `X, Y' have the following insts: ( I =< bound(a) ), free, which does not match any of the modes". `X` unnarrowed, or a plain `bound(a)`, is accepted. No GitHub issue | mmc (`test/Semantics/modes/polymorphic-mode-errors.m`) |
| A construction's argument used after | Both the term and the argument are shared when both are used after (Overton, definition 3.2.20); `unique` carries "the additional constraint that there can only be one reference to the corresponding value" ("Destructive update") | `X = [1], L = [X], d(X), usel(L)`, with `d(list(int)::di)`, is accepted: the argument keeps its uniqueness, and only the term's copy is shared, so `d` may clobber what `L` refers to; so are `P = p(T0, t(2)), updu(1, T0, T), gp(P)`, with `T0::di`, and `X = [1], L = [X]` with `X::uo` and `L::out`. mmc does not reuse such values ("Limitations of the current implementation"), so no program goes wrong yet; arrays, which it does update in place, are not checked for uniqueness at all, as `uniq_array` is `ground` (library manual, `array`). An assignment, `T1 = T0`, shares both copies, and a field taken out of a term used after is shared. No GitHub issue | Not mmc: unsound. By default, a live argument of a construction whose term is live too is shared, as the thesis has it and as an assignment's copies are (`uniqueness/aliased.m`); as with an assignment, `P = p(T0, t(2)), gp(P), updu(1, T0, T)` is then rejected too. mmc's behaviour is behind `-fmmc-uniqueness`, which the oracles use (`compare-types.py --modes` too). Until 2026-11-01 merkc followed mmc here |
| Goals after a call that infers a new mode | A call of a predicate whose modes are inferred continues with the final insts of the mode inferred for it (Overton, 3.4.2, step 7) | Without `--infer-modes`, mmc does not check them in any later pass: in `p(Z) :- q(1, Y), r(Z, Y)`, with `q` and `r` undeclared, `r` gets no mode, and mmc exits with status 1 without a message, even with `-E`. With `--infer-modes` it checks them in a later pass and infers `r(uo, di)`. No GitHub issue | Follow the thesis, as mmc does with `--infer-modes`, for predicates without a `:- pred` declaration too |
| A mode error in a declared procedure during mode inference | Not described: the thesis's fixpoint repeats until the inferred modes no longer change (Overton, 3.4.2) | mmc ends inference after the pass that finds it, so a call that made a mode in that pass leaves its caller's later goals unchecked, and their errors unreported. No GitHub issue | Follow mmc |
| Partial switches | "A disjunction is a _switch_ if each disjunct has near its start a unification that tests the same bound variable" ("Switches") | `( X = a, Y = 1 ; q(1, M), X = b, Y = M ; X = b, Y = 3 )` becomes a disjunction of a switch on `X`, with the first and third disjuncts, and the second; the HLDS dump after switch detection shows it. No GitHub issue | mmc, since it decides the determinism of the parts (`test/Semantics/determinism/switches.m`) |
| A nested disjunction within a nested disjunction | The test may occur inside a nested disjunction, "preceded only by other unifications, both inside the nested disjunction and before" ("Switches"), at any depth | mmc splits a disjunct at a disjunction one level down only: with a disjunction in a branch of the nested one, whose branches all test the variable, the disjunct is not part of the switch. No GitHub issue | mmc (`test/Semantics/determinism/random-switches-1.test`) |
| A variable that every disjunct tests against one functor | A switch on any variable each disjunct tests against a different functor ("Switches") | `( W = a, X = b, Y = 1 ; W = c(_), X = b, Y = 2 )` is no switch at first: mmc moves `X = b` out of the disjunction, and then switches on `W`. Without such a move, as when the functor's arguments are partly free, there is no switch; one on an `int`, or one that cannot fail, is still made. No GitHub issue | mmc, since the switch usually comes after the move (`test/Semantics/determinism/switches.m`) |
| The start of determinism inference | "The least fixpoint of the transformation which, given an initial assignment of the determinism `det` to all such procedures, applies those rules" ("Determinism checking and inference") | `loop(X) :- loop(X)`, undeclared, is inferred `erroneous` (the HLDS dump after determinism analysis says "mode number 0 (erroneous)"), which a start from `erroneous` gives; from `det` it would be `det`. No GitHub issue | mmc: a recursion that never returns has no solutions |
| An if-then-else whose condition has no solutions | "An if-then-else can fail if either the 'then' part or the 'else' part can fail ... It can succeed more than once if any one of the condition, the 'then' part and the 'else' part can succeed more than once" ("Determinism checking and inference"), with no exception for a then part that cannot run | `( if fl(X, Z) then nd(Z, Y) else Y = X )`, where `fl` is `failure` and `nd` `nondet`, is `det` in mmc, and so is the procedure; by the manual it would be `nondet`. Where mode analysis finds the condition cannot succeed, by declared determinisms, mmc leaves the then part out; where only inference does, mmc keeps it, with its errors, but the if-then-else is still its else part. No GitHub issue | mmc: the then part cannot run |
| A negation of negations | "`\+ Goal` ... equivalent to `if Goal then fail else true`" ("Goals"), so a negation binds nothing after it | `\+ \+ G` is `G`, and `\+ (\+ A, \+ B)` is `(A ; B)`: `p(Y) :- \+ \+ X = 1, Y = X` is accepted as `det`, `\+ (\+ X = 1, \+ X = 2), Y = X` is a `multi` disjunction ("Disjunction has multiple clauses with solutions"), and `\+ X \= 1` binds `X`. Mode errors in `G` then count what is used after the negation as live. No GitHub issue | mmc, which accepts more |
| A value an if-then-else's condition makes from one the else part needs | "A value is not considered `unique` if it might be needed on backtracking" ("Unique modes") | In the condition, the variables the else part uses are `mostly_unique`, but not what an assignment makes from them: `( if T4 = T0, updu(1, T4, T5), 1 > 2 then T = T5 else T = T0 )`, with `T0::di`, is accepted, though the else part uses `T0` after `updu` clobbered it. After a goal that can succeed more than once, an assignment does pass it on. No GitHub issue | Not mmc: unsound. By default, the variable is `mostly_unique` while the condition runs if the condition clobbers it or what it makes from it, and so is what it makes from it; then, since the else part is not run once the condition has succeeded, the then part starts from the condition checked without that (accepting `( if peek(T0, X), X > 1 then updu(1, T0, T) else T = T0 )`, which mmc does too, `peek` taking `T0::ui`). mmc's behaviour is behind `-fmmc-uniqueness`, which the oracle uses |
| A value used after an if-then-else | "A value is not considered `unique` if it might be needed on backtracking" ("Unique modes"); backtracking into a condition or a then part that can succeed more than once runs what follows the if-then-else again | Only what the then part uses counts as used after the condition, and nothing as used after the then and else parts: `( if m(X), X > 1 then Y = X else Y = 0 ), updu(Y, T0, T)` and `( if 1 > 2 then m(Y) else Y = 2 ), updu(Y, T0, T)`, with `m` `multi` and `T0::di`, in a `multi` procedure, are accepted, though backtracking into `m` runs `updu` on `T0` again; the same with a disjunction for the if-then-else is an error. No GitHub issue | Not mmc: unsound. By default, what is used after the if-then-else is used after each of its parts. mmc's behaviour is behind `-fmmc-uniqueness`, which the oracle uses |
| The determinism of a procedure with I/O state arguments | No rule: `io.state` "represents the state of the world", and predicates that do I/O are passed its only reference ("The state-of-the-world type"); "Unique modes" restricts only where unique values may be used | `:- pred p(io::di, io::uo) is semidet` (or `nondet`, `multi`, `failure`, `cc_nondet`), declared or inferred, is "error: invalid determinism for a predicate with I/O state arguments", for exactly two `io.state` arguments, one bound at the start and one free, of any uniqueness, `in` and `out` too; `det`, `cc_multi` and `erroneous` are accepted. No GitHub issue | mmc, which rejects more |
| Nested unique modes | "It is not possible to use nested unique modes, i.e. modes in which anything but the top level of a variable is unique. If you do, you will get unique mode errors when you try to get a unique field of a unique data structure" ("Limitations of the current implementation") | With `:- inst up == unique(p(unique, unique))`, a predicate taking `up >> dead` that takes out a unique field and updates it with a `di` call, and builds a new `up` from the result, is accepted, and so is taking out a unique field as `uo`. No GitHub issue | mmc, which is what the manual wants |
| A named unique inst passed to an `in` argument | "A value is not considered `unique` if it might be needed" again; `in` is `ground >> ground` ("Unique modes") | With `:- inst up == unique(p(unique, unique))`, a variable `P0` of inst `up` stays `up` after a call that takes it `in`, so `cp(P0, Q), pdi(P0, P)`, with `cp(P::in, Q::out) :- Q = P` and `pdi(p::di, p::uo)`, is accepted, though `pdi` destroys what `Q` refers to. `upn == unique(p(ut, ut))`, an unnamed `unique(p(unique, unique))` and `w(unique)` with `w(I) == unique(p(I, I))` are shared, as is a named `unique`. No GitHub issue | Not mmc: unsound. The value is shared after the call whatever its inst (`modes/shared-named-inst.m`, not compared with mmc); the program generator avoids the case |
| A lambda expression's mostly clobbered argument | `mostly_clobbered` is "not referenced anywhere, but needed on backtracking" ("Backtrackable destructive update"); a value that `di` clobbered is not available on backtracking | mmc does not check a lambda expression's head variables' final insts when they are clobbered: `P = (pred(X::(unique >> mostly_clobbered), Y::out) is det :- updu(2, X, Y))` is accepted, though `updu` may destroy what the caller keeps for backtracking. Nor does it treat the head variables as used at the end, which is harmless only because one whose modes do not clobber it is shared in the body. No run-time effect yet, as mmc does not reuse such values | Not mmc: unsound, and mmc's rule is not clear (`X::in` clobbered by `ground >> clobbered`, then read, is also accepted). merkc checks the final insts with and without `-fmmc-uniqueness`, and the generator's `--array` lambda expressions take arrays `array_di` (`test/Semantics/uniqueness/lambda-clobbered.m`) |
| Arrays' uniqueness | Arrays are updated destructively ("Destructive update"; library manual, `array`: `set` takes `array_di`) | The library declares `uniq_array(I) == array(I) == ground`, "XXX work-around", so `array_di` takes any array: `A1 = A0, array.set(0, 5, A0, A)` compiles, and the program then prints 5 for `array.lookup(A1, 0)`, which is 0. Documented in the library itself, so not a bug to report | Not mmc: unsound. By default merkc loads a sound stub, with the insts the library's comments intend and `array_ui` modes for the predicates that read arrays (Standard library); mmc's declarations are behind `-fmmc-uniqueness` |
| Unique-input modes and fields of copied values | "It is also not possible to use unique-input modes; only destructive-input and unique-output modes work" ("Limitations of the current implementation"); a unique value has "only one reference" ("Destructive update"), which a copied value always has | A field taken out of a unique term and used after is shared in the term whatever its type, so the term no longer matches `unique`: `T = t(X)`, with `T::ui` and `X::out`, is "argument 1 did not get sufficiently instantiated. Final instantiatedness of `T' was `unique(t(ground))'", and `T0 = t(X), updu(X, T0, T)`, with `T0::di`, is an error at `updu`, for `int`, `float`, `char` and enumeration fields too; so a `ui` argument can only be passed on. No GitHub issue | Not mmc, which only rejects more. By default, a value of a type whose values are copied, a number, a character or an enumeration, matches a unique inst when it is shared or `mostly_unique`, not when clobbered, so reading such fields of a `ui` or `di` term keeps it unique (`uniqueness/copied.m`); a field of another type, as a string, which `string.set_char` may update in place, or a term, still makes the term shared in that part. mmc's behaviour is behind `-fmmc-uniqueness`, which the oracles use |
| `mostly_unique(...)` | Only `unique(...)` is described, "like `bound(...)`" ("Destructive update"), and `mostly_unique` "equivalent to `unique`" for backtracking ("Backtrackable destructive update") | `mostly_unique(p(mostly_unique, unique))` is accepted, and checked as a bound inst whose top level is `mostly_unique`: a field taken out of it as `uo` is an error. No GitHub issue | mmc |
| A `mostly_unique` bound inst with a bound argument | `mostly_unique` is `ground(mostly_unique)` of any value ("Unique modes") | `T6 = t(3), ( T = T6 ; T = T6 )`, with `T::muo`, is an error: "Final instantiatedness of `T` was `mostly_unique(t(mostly_unique(3)))`, expected final instantiatedness was `mostly_unique`". A constant, `T6 = e1`, is accepted, and so is such an inst matched to `ground` (`in`), or of a variable that was an argument of a call before, which mmc has found ground then. No GitHub issue | mmc, which only rejects more |
| A constructed unique value passed `in` after a `ui` call | "A value is not considered `unique` if it might be needed" ("Unique modes"); a callee may keep an `in` argument in an output | A term built in the clause becomes shared when it is passed to an `in` argument, but not after it has been passed to a `unique >> unique` (`ui`) argument: `T0 = t(1), ui(T0), wrap(T0, L), T = T0`, with `wrap(X, [X])`, `T::uo` and `L::out`, is accepted, though `L` holds `T`. Without the `ui` call, or with a `unique(t(ground)) >> unique(t(ground))` or `unique >> ground` one, it is an error. No GitHub issue | Not mmc: unsound, and mmc's rule is not clear. The value is shared after an `in` call either way (`uniqueness/ui-then-in.m`, not compared with mmc); `uisweep/s4/u304_21.m` differs from mmc for this reason |
| A negated call of an `any_pred` closure | "if any non-local solver type variable or higher-order variable with inst `any` is used in a negated context, the goal must be placed inside a ... promise" ("Solver types and negated contexts") | `P = (any_pred is semidet :- r(X)), not P`, with `X::ia`, is accepted with no promise. No GitHub issue | Follow mmc: only a nonlocal whose inst has an `any` part counts, not an `any_pred` closure |
| A type variable whose first constraint has it inside a type | A constraint is `TYPECLASS(TYPE, ...)`, with any type ("Type class constraints on predicates and functions") | For `cons(X, N) <= sh(list(T))`, mmc takes T's type_info from the typeclass_info's place for `list(T)`, and so passes `list(T)`'s as T's: `cons(42, S)` with `S = type_name(type_of(X))` prints `list.list(int)`. It does so even when a later constraint has T itself, `<= (sh(list(T)), sh(T))`. No GitHub issue | Sound: the same constraint as mmc, and T's type_info is the argument of `list(T)`'s (`polymorphism/constrained-types.m`); not compared with mmc |
| Several clauses of a predicate with an existential constraint | Each clause gives the existentially quantified variables the same types, and satisfies the existential constraints ("Semantics of type quantifiers", "Existential class constraints"); a predicate may have several clauses | `:- some [T] pred multi(ab::in, T::out) is det => sh(T).` with `multi(a, 1). multi(b, 2).` is a mode error in mmc: "mode mismatch in disjunction. The variable `TypeClassInfo_for_sh' is ground in some branches but not others". mmc makes the typeclass_info in each clause, and gives it back once after their switch, under a variable that only one branch binds. One clause, an if-then-else in one clause, or no constraint (the type_info is made in every branch) is accepted. No GitHub issue | Not followed: merkc accepts it, and each clause gives the typeclass_info for its own proof (`polymorphism/existential-clauses.m`); not compared with mmc. The generator gives existentially typed predicates one clause |
| The quotient of the least integer by -1 | `//` and `rem` are `det` and throw only for a zero divisor; `X rem Y = X - (X // Y) * Y` (library manual, `int`) | With the operands known, mmc folds `min_int // -1` to `min_int` and `min_int rem -1` to 0. With them computed, as from a `no_inline` predicate, the program dies of SIGFPE, the x86-64 division's trap, for `//`, `rem`, `div` and `mod`, and for `int8` and the other signed types alike. No report found among the GitHub issues | The manual: it wraps, to `min_int` and 0, as mmc folds it, wherever the operands come from (`test/CodeGen/arithmetic.m`) |
| Negating a float zero | `- X` is unary minus (library manual, `float`), which in IEEE arithmetic makes 0.0 into -0.0 | `-Z`, with `Z` = 0.0, is 0.0, while `Z * -1.0` is -0.0: mmc's negation is `0.0 - X`. `max` and `min` decide -0.0 and NaN as `X >= Y` and `X =< Y` do, `round(X)` is `floor(X + 0.5)`, and `float.pow` multiplies by squaring, rounding as that does | mmc, as these are what programs print (`test/CodeGen/float-math.m`) |
| Mode inference of a unique array argument | Modes are inferred for a predicate without mode declarations with `--infer-modes` (user's guide); nothing restricts inferring unique modes | `zero(I, !A) :- ( if I < 0 then true else !A ^ unsafe_elem(I) := 0, zero(I - 1, !A) ).`, declared `:- pred zero(int, array(int), array(int)).`, called with an array from `array.init`, is "unique-mode error: the called procedure would clobber its argument, but variable `STATE_VARIABLE_IO_0' is still live", in a clause for `zero(di, (ground >> not_reached), out(not_reached))`: the array's inst at the call is `ground`, and the variable named is not one of that clause's. The same for the translation of `saxpy`'s loop written by hand. No GitHub issue | Not mmc: merkc infers `zero(di, di, uo)`, which is what a loop that updates an array in place translates to (Loops, "Translation"). The message's variable is an mmc bug either way |

Reports checked and found not to be mmc bugs: GitHub issue 31, "Lambda scoping error", which was the debugger's query translation; the same code compiles in a module. It shows a warning that quantification should give too, "variable `W` has overlapping scopes", when a lambda's argument has the name of a variable outside it.

### mmc bugs to report

The mmc 22.01.9 bugs found so far, to report to the Mercury developers once each has a small reproducer. The register above has the evidence for each; a row's name is given in quotes. Each was searched for among the GitHub issues when it was found. Those marked *test* have a reproducer in merkc's tests already; for the others, a small one is still to be written. Add each new one here when it is recorded in the register.

**Unsound: programs that are accepted, but are wrong**

- **Uniqueness of a value an if-then-else's condition makes from one its else part needs** ("A value an if-then-else's condition makes from one the else part needs"). `( if T4 = T0, updu(1, T4, T5), 1 > 2 then T = T5 else T = T0 )`, with `T0::di`, is accepted, though the else part uses `T0` after `updu` clobbered it. Not reported. *test*: `Semantics/uniqueness/sound.m`.
- **Uniqueness of values used after an if-then-else** ("A value used after an if-then-else"). Backtracking into a condition or a then part that can succeed more than once runs what follows again, but `( if m(X), X > 1 then Y = X else Y = 0 ), updu(Y, T0, T)` is accepted in a `multi` procedure. Not reported. *test*: `Semantics/uniqueness/sound.m`.
- **A named unique inst kept unique across an `in` call** ("A named unique inst passed to an `in` argument"). `cp(P0, Q), pdi(P0, P)` is accepted for `P0` of a named inst `up == unique(p(unique, unique))`, though `pdi` destroys what `Q` refers to. Not reported. *test*: `Semantics/modes/shared-named-inst.m`.
- **A unique value put in a term stays unique** ("A construction's argument used after"). `P = p(T0, t(2)), updu(1, T0, T), gp(P)` is accepted with `T0::di`. No run-time effect yet, as mmc does not reuse such values. Not reported. *test*: `Semantics/uniqueness/aliased.m`.

- **A lambda expression clobbers what its caller keeps for backtracking** ("A lambda expression's mostly clobbered argument"). `P = (pred(X::(unique >> mostly_clobbered), Y::out) is det :- updu(2, X, Y)), P(A, C)` is accepted. No run-time effect yet, as mmc does not reuse such values. Not reported. *test*: `Semantics/uniqueness/lambda-clobbered.m`.

- **A constructed unique value stays unique across an `in` call after a `ui` call** ("A constructed unique value passed `in` after a `ui` call"). `T0 = t(1), ui(T0), wrap(T0, L), T = T0`, with `T::uo`, is accepted, though `L` holds the same value. Not reported. *test*: `Semantics/uniqueness/ui-then-in.m`.

**Crashes**

- **Determinism analysis of a unification of higher-order values** ("A crash in determinism analysis"): "det_infer_unify ... complicated_unify but no fail context". Not reported.
- **Coercing an imported abstract type** ("Coercing an abstract type"): "build_type_param_variance_restrictions ... Unexpected: not du type". Not reported.
- **Mode analysis of a coerce through a tuple parameter** ("A coerce through a tuple parameter"): "modecheck_coerce_from_ground_make_bound_functor ... missing constructor for result type". Not reported. *test*: `Semantics/types/coerce/tuple-parameter.m`.
- **Merging branches that leave a variable partly free** ("A unification of two partially instantiated terms after branches"): "merge_instmapping_delta_vars ... error merging var 7". Not reported. *test*: `Semantics/modes/merge-crash.m`.
- **Mode analysis with constrained insts**: "check_hlds.modecheck_util.add_constrained_inst/4: Unexpected: SubInst differs", on `modes_25_24.m` from `generate-mode-programs.py --seed 25`. Not in the register yet, and not reduced. Not reported.
- **Unifying values of a type with existential constraints** ("Unifying values of a type with existential constraints"): a determinism error in mmc's own `__Unify__`, and in a larger module a crash in "modecheck_queued_proc ... found detism error". Not reported.

**Wrong errors or wrong code**

- **Mode inference loses an array's uniqueness** ("Mode inference of a unique array argument"): a recursive predicate that updates an array in place, with its types declared and its modes inferred, is a unique-mode error, which names a variable, `STATE_VARIABLE_IO_0`, that is not in the clause. Not reported.

- **Type inference reports the first pass's errors** ("Errors during type inference"), and **does not reach a fixpoint** ("A fixpoint of type inference") for `f(H1, _, H3) = H4 :- H4 = f(H3, {H1}, H1)`. Not reported.
- **Overlapping instances through a repeated variable** ("Instances that overlap through a repeated variable"): `d(p(A, B))` and `d(p(A, A))` pass mmc's checks, and the C compiler then rejects the generated code. Not reported.
- **An abstract imported type as a supertype** ("An abstract type as a supertype") is accepted. Not reported. *test*: `Semantics/types/subtypes/abstract-supertype.m`.
- **A conditional expression in a term assigned to an unused variable** ("A conditional expression in a term assigned to an unused variable"): a spurious mode error. Reported as GitHub issue 120, closed by its reporter without an answer.
- **A partially instantiated term unified with a bound one** ("A partially instantiated term unified with a bound one"): a mode error in mmc's own `__Unify__`. Reported as GitHub issue 74, open.
- **Several clauses with an existential constraint** ("Several clauses of a predicate with an existential constraint"): `multi(a, 1). multi(b, 2).`, with `=> sh(T)`, is "mode mismatch in disjunction" for mmc's own `TypeClassInfo_for_sh`. Not reported. *test*: `Semantics/polymorphism/existential-clauses.m` (merkc's version); the reproducer is that test's `multi`.
- **The type_info of a type variable constrained inside a type** ("A type variable whose first constraint has it inside a type"): with `<= sh(list(T))`, T's type_info is `list(T)`'s, so `type_of(X)` for an `int` is `list.list(int)`. Not reported. *test*: `Semantics/polymorphism/constrained-types.m` (merkc's sound version); the reproducer is that test with `type_name(type_of(X))`.

- **The quotient of the least integer by -1** ("The quotient of the least integer by -1"): `min_int // -1`, `rem`, `div` and `mod` kill the program with SIGFPE when the operands are computed, though mmc folds them to `min_int` and 0 when they are known, and the manual makes them `det`. Not reported. *test*: `CodeGen/arithmetic.m` has merkc's result.

**The reference manual**

- `:- mode in ^ in = out is semidet` ("`:- mode in ^ in = out is semidet`"): the example is wrong; `in ^ elem(in) = out` is what it means.
- "Limitations of the current implementation" says nested unique modes cannot be used ("Nested unique modes"), but mmc accepts them.
- `mostly_unique(...)` is accepted by mmc, but not described ("`mostly_unique(...)`").

### OpenMP directives

- **Definitions from LLVM.** Directive and clause kinds, and which clauses each directive allows, come from `llvm/Frontend/OpenMP/OMP.td`, which Clang and Flang share. Our directives therefore use the same `llvm::omp` kinds that the `omp` dialect's lowering already understands.
- **Syntax.** `%$omp` is followed by a directive in its Fortran spelling, such as `parallel do`, and then its clauses, which may be separated by commas, as in Fortran. Each continuation line begins `%$omp&`. A directive's last name may take arguments, as `critical(Name)` does. A clause is a name, perhaps applied to arguments, as in `nowait`, `private(X, Y)` or `schedule(dynamic, Chunk)`. Arguments are Mercury terms, read by the term reader, so `:` separates modifiers from a list as in Fortran: `map(always, to: A)`. A map clause may map a section of an array, `A(Lo .. Hi)`, with Mercury's inclusive range operator (Language extensions, "Target regions"). As in mmc, an operator before an infix operator must be in parentheses, so a reduction with a list is written `reduction((+) : X)`. Over a fold, the accumulator is the reduction's variable, so `reduction(+)` is the usual form. Two clauses of our own, `index(I)` and `upper(N)`, name the induction variable and the inclusive bound of a self-recursive counted loop (Language extensions). The directive covers the goal that follows it.
- **Fortran's spellings.** The directive is the longest run of names that OMP.td knows, as in Flang. A directive that OMP.td has only for C, such as `parallel for`, is an error that names the Fortran form, `parallel do`.
- **Attachment.** The term reader attaches a directive token to the term after it, read as one conjunct. The converter accepts the annotation only where that term is a goal. There it builds an `OmpGoal`, which holds the directive and its goal, as Clang's `OMPExecutableDirective` holds its associated statement; desugaring turns it into a directive `Scope`. Anywhere else, such as before a clause head, inside a data-term or in a DCG body, the directive draws a warning and is ignored.
- **Parse tree.** `convert-omp.cpp` reads the directive token's own tokens into an `OmpDirective` node, which holds the directive's kind, its arguments and its clauses. Flang has a parse-tree class for each clause, generated from OMP.td, whose fields are Fortran's expressions and object lists. We follow it for the clauses that later passes use: `schedule`, `reduction` and `map` have nodes of their own, with their modifiers, kinds and map types as enums, and a map clause's objects are `OmpMapObject`s, each a variable with the bounds of its section, if it has one. Every other clause of OMP.td is an `OmpGenericClause`, with its kind, its modifiers and its arguments as data-terms, for semantic analysis to interpret. A clause gets a node of its own when a pass needs one.
  - **Tokens.** A directive's nodes index the directive's own tokens, not the file's. The exact unparser prints a directive by unparsing its node over those tokens, so test 2 checks directives as it checks the rest of the file. The canonical unparser prints a directive as written, on a line of its own at the indentation of its goal, with its continuation lines at the same indentation.
  - **Variables.** A directive's variables are numbered as the item's, so `index(I)` names the same variable as the `I` of its goal.
- **Checking** is part of semantic analysis: the clauses must be allowed on the directive in the OpenMP version in use (`isAllowedClauseForDirective`), the associated goal must be a loop the compiler can form, and the clauses must make sense for it. For now, MLIRGen checks a `parallel do` loop's form, since it has the loop's translation and determinism at hand (Language extensions, "Loops", Translation).
- **Decomposition** (implemented). MLIRGen breaks each directive into its leaf constructs, `parallel do` into `parallel` and `do`, each with the clauses that apply to it, by LLVM's `ConstructDecompositionT`, as Flang (`flang/lib/Lower/OpenMP/Decomposer.cpp`) and ClangIR (`CIRGenOpenMPConstructDecomposition.h`) do. `lib/CodeGen/Decomposer.cpp` follows ClangIR's: a clause is converted only as far as the algorithm reads it, its objects are variables by name and its expressions data-terms, and each leaf keeps the clauses of the directive it got, so a leaf's generator reads them from the parse tree; the clauses the decomposition adds, which OpenMP implies, such as `shared` on `parallel` for a reduction's variable, are listed apart. A loop's counter is the decomposition's iteration variable. A clause that no leaf allows in the OpenMP version in use is an error, "the map clause is not allowed on the parallel do directive", the check of `isAllowedClauseForDirective`. Merkurius's own `index` and `upper` take no part. Each leaf but the innermost is `omp.combined`, as Flang marks it. Of the clauses, only a loop's `reduction` is generated, on the `do` and `teams` leaves the decomposition puts it on, with the `map` it implies on `target` (Language extensions, "Loops", Parallel do, *Reductions*); every other clause is reported as not yet implemented, with its leaf, as "the schedule clause of the do directive", and an added `shared` needs nothing, since the variable is shared already.
- **Without `-fopenmp`** the lexer leaves directive lines as comments, so the program's meaning is that of the sequential program, which is also what mmc compiles.

### Modules and interface files

- **Modules.** Interface and implementation sections, nested and separate submodules, and the 22.01 visibility rules for `import_module` and `use_module`.
- **Our own interface-file format,** not Melbourne's `.int`/`.opt`. It serializes the exported declarations' parse-tree nodes, with the signatures semantic analysis inferred or checked: types and their representation, insts, modes, predicate and function signatures with modes and determinism, type classes and instances. The module loader reads them back into the same parse-tree types, so imported and local declarations look alike to semantic analysis.
- **Type fingerprints.** Each exported type gets a fingerprint of its definition, stored in the interface file. The same fingerprints identify types in the distribution handshake (Distribution and wire format).
- **Optimization section.** An optional section carries the semantic forms of selected bodies, for intermodule inlining and specialization, and analysis results.
- **Build driver.** It computes module dependencies from imports and rebuilds only what changed, like `mmc --make` (Driver, Build mode).
- **Implementation so far.** Until there are interface files, the module loader reads an imported module's interface from its source. Name resolution needs only the declarations, as parse-tree nodes, and the source's interface section has them all; the interface-file format becomes necessary for what is not in the source, such as type representations, fingerprints and the optimization section, and for compiling against a library that ships without its source.
  - **Finding a module.** `name.m` is looked for in the importing module's directory, then the `-I` directories, then the library directory. A file whose `:- module` names another module is an error.
  - **What is loaded.** The interface section's items, except its imports, and, transitively, the modules that section imports and uses, since an imported module's declarations are resolved in its own scope. Its implementation's imports are not loaded. Every module but `builtin` imports `builtin` implicitly. Nested submodules are not loaded yet.
  - **The library** is declaration-only stubs of 13 modules (`library/`): `builtin`, `int`, `float`, `char`, `string`, `list`, `io`, `bool`, `maybe`, `pair`, `require`, `exception` and `univ`. Their declarations are those of the 22.01 library reference manual, which a module's interface must match for programs written for mmc to compile. The manual's comments are not copied. Declarations that need modules not stubbed yet are left out. `stm_builtin`, which mmc provides for `atomic` goals, is missing, because the manual does not document it.

### Running example

This fragment is followed through every stage below. mmc compiles and runs it, printing 30 for the array `[1, 2, 3, 4]`; to mmc the `%$omp` line is a comment.

```mercury
:- pred sum_squares(array(int)::in, int::in, int::out) is det.

sum_squares(A, N, Sum) :-
    %$omp parallel do reduction(+)
    int.fold_up(add_square(A), 0, N, 0, Sum).

:- pred add_square(array(int)::in, int::in, int::in, int::out) is det.

add_square(A, I, !Acc) :-
    array.unsafe_lookup(A, I, X),
    !:Acc = !.Acc + X * X.
```

**1. Source manager.** The file is read unchanged and given one provenance per byte, plus one for its end. `AllSources` maps each of them back to the file, line and column.

**2. Lexer.** The `add_square` clause becomes these tokens, one source line per row. With `-fopenmp`, the comment on line 4 is a directive, and becomes one token holding its own, with the text `parallel do reduction(+)`; without it, the line stays a comment. The `.` after `array` is followed by a name, so it is the qualification operator; the last `.` is followed by a newline, so it is the `end` token.

```text
name add_square · open_ct · variable A · comma · variable I · comma · name ! · variable Acc · close · name :-
name array · name . · name unsafe_lookup · open_ct · variable A · comma · variable I · comma · variable X · close · comma
name !: · variable Acc · name = · name !. · variable Acc · name + · variable X · name * · variable X · end

directive [ name parallel · name do · name reduction · open_ct · name + · close ]
```

**3. Term reader.** Precedence climbing gives one term per item: `:-` (300) is outermost, then `,` (500), `=` (800), `+` (1000), `*` (1100), and the state-variable prefixes `!`, `!.` and `!:` (1460). Variables are numbered in order of appearance, and the directive annotates the term that follows it.

```text
':-'( add_square(A, I, '!'(Acc)),
      ','( array.unsafe_lookup(A, I, X),
           '='( '!:'(Acc), '+'( '!.'(Acc), '*'(X, X) ) ) ) )
variables: A = 0, I = 1, Acc = 2, X = 3

':-'( sum_squares(A, N, Sum),
      <directive> int.fold_up(add_square(A), 0, N, 0, Sum) )
```

**4. Converter: parse tree.** Combinators turn each term into typed parse-tree nodes. Nothing is resolved yet: `+` and `*` are just functor terms, and `add_square(A)` could still be a constructor, a function call or a partial application.

```text
Clause add_square(A, I, !Acc)
├─ head  PredHead add_square [VarTerm A, VarTerm I, StateVarRef !Acc]
└─ body  Conjunction
         ├─ CallGoal array.unsafe_lookup [VarTerm A, VarTerm I, VarTerm X]
         └─ UnifyGoal
            ├─ StateVarRef !:Acc
            └─ FunctorTerm + [StateVarRef !.Acc, FunctorTerm * [VarTerm X, VarTerm X]]

Clause sum_squares(A, N, Sum)
└─ body  OmpGoal
         ├─ OmpDirective parallel do [OmpReductionClause +]
         └─ CallGoal int.fold_up [FunctorTerm add_square [VarTerm A], Literal 0,
                                  VarTerm N, Literal 0, VarTerm Sum]
```

**Round trip.** Unparsing this parse tree exactly gives back the source above, including the `%$omp` line and the blank lines. The fragment already follows the canonical layout, so canonical mode prints the same text. The fully parenthesized form prints `add_square` as follows, and parsing that gives the same tree:

```mercury
add_square(A, I, (!Acc)) :-
    (array.unsafe_lookup(A, I, X),
    ((!:Acc) = ((!.Acc) + (X * X)))).
```

**5. Desugaring: semantic form.** State variables become numbered variables (desugaring step 1). The nested arithmetic becomes one-functor unifications through a fresh variable, and constants become unifications too (steps 3 and 6). Quantification then records each goal's nonlocal variables (step 7). Nodes are numbered for `ProcInfo`, and each points back to the parse-tree node it came from.

```text
add_square(A, I, Acc_0, Acc_1) :-               came from
  #0 Conj                                       Conjunction
  #1   Call  array.unsafe_lookup(A, I, X)       CallGoal
  #2   Unify Acc_1 = +(Acc_0, V_1)              UnifyGoal
  #3   Unify V_1 = *(X, X)                      FunctorTerm *(X, X)

sum_squares(A, N, Sum) :-
  #0 Scope omp parallel do, reduction(+)        OmpGoal
  #1   Conj                                     CallGoal
  #2     Unify V_1 = add_square(A)              FunctorTerm add_square(A)
  #3     Unify V_2 = 0                          Literal 0 (first)
  #4     Unify V_3 = 0                          Literal 0 (second)
  #5     Call  int.fold_up(V_1, V_2, N, V_3, Sum)  CallGoal
```

**6. The two trees after name resolution.** Semantic analysis adds the "refers to" links beside the "came from" links. Neither tree contains the other: the parse tree is the trunk that holds every declaration and clause, and each clause's semantic form hangs off it with its own leaves, which are variables and inline literals.

```text
Parse tree (trunk)                                  Semantic form of add_square's clause
------------------                                  ------------------------------------
PredDecl array.unsafe_lookup (imported) <-refers to-- #1 Call  array.unsafe_lookup(A, I, X)
PredDecl int.+ (a function, imported)   <-refers to-- #2 Unify Acc_1 = +(Acc_0, V_1)
PredDecl int.* (a function, imported)   <-refers to-- #3 Unify V_1 = *(X, X)
Clause add_square (as written)
  ├─ CallGoal …                        <-came from-- #1
  ├─ UnifyGoal …                       <-came from-- #2
  └─ FunctorTerm *(X, X)               <-came from-- #3
```

**7. What happens next, for orientation** (Semantic analysis):

- **Types.** `A` has type `array(int)`, and every other variable `int`. In `sum_squares`, `#2` resolves to a closure of `add_square/4` with one argument supplied, and `#5` to `int.fold_up`.
- **Modes.** In `add_square`'s only mode, `#2` needs `V_1`, which `#3` produces, so the `ProcInfo` order is `#1, #3, #2`. `#3` and `#2` are calls of the functions `int.*` and `int.+`, and the procedure is `det`.
- **MLIR.** `#1` becomes `merk.call` with a type-info operand for `int`, `#3` becomes `arith.muli`, `#2` becomes `arith.addi`, and `Acc_1` is the procedure's result. In `sum_squares`, `#2` becomes `merk.closure`, and the directive scope becomes `omp.parallel` around `omp.wsloop` over `omp.loop_nest`, with `reduction(+)`.

## Semantic analysis

Semantic analysis decides what the program means: what each name refers to, every type, which way each unification runs, which procedures exist, and the determinism of every goal. It runs on the AST, resolving names in the parse tree and analysing the semantic form, and ends where MLIR generation begins. ClangIR is the model for that boundary, with `merk` in the role of CIR; Flang is the model for keeping results beside the parse tree instead of rewriting it. Mode and determinism analysis belong here rather than in an MLIR dialect, for three reasons:

- **No SSA before modes.** Before mode analysis, a unification has no direction, so values cannot be in SSA form. A dialect for this stage would hold variables as memory slots, like `alloca` before mem2reg, and would amount to the AST in another syntax.
- **Diagnostics.** Mode errors are Mercury's hardest diagnostics to explain, and they are best reported against the source structure.
- **Inference.** Inferring modes and determinism for local predicates iterates over the call graph, which is simplest on the AST.

### What semantic analysis writes

Semantic analysis never changes the structure of the parse tree or of the semantic form. One semantic form serves every procedure of a predicate, so it cannot be rewritten for any one of them: `X = f(Y)` can construct in one mode and deconstruct in another. Results go to three places instead: annotation slots on AST nodes, for results that hold in every mode; the predicate's declaration, for what is inferred about the predicate as a whole; and one `ProcInfo` per procedure, for results that differ between modes.

| Written to | What | Written by | Used by MLIR generation for |
| --- | --- | --- | --- |
| Names in the parse tree | The declaration each name resolves to | Name resolution | Nothing; it serves diagnostics and tooling, such as go-to-definition |
| `Unify` and `Call` nodes of the semantic form | The constructor, function or predicate chosen by overload resolution | Type checking | The constructor of `merk.construct`, `merk.deconstruct` and `merk.switch` arms; the callee of `merk.call` |
| Variables of the semantic form | Their types | Type checking | The MLIR type of each value |
| Goals of the semantic form | Purity | Purity checking | An `impure` attribute on calls, so optimizations neither move nor delete them |
| `PredDecl` | Inferred types, modes and determinism, and the list of the predicate's procedures | Type, mode and determinism inference | One `merk.proc` per procedure |
| New implicit declarations | Each type's unify and compare procedures, field selection and update functions, and the predicates behind `:- mutable` and `:- process` | Type checking and desugaring | Emitted like user predicates |
| `ProcInfo`, one per procedure | Conjunction order, unification kinds, the procedure chosen for each call, implied-mode tests, the variables each branching goal binds, switches, determinism, commits, type-info sources, and which arguments are consumed (`di`) | Mode, switch, determinism, uniqueness and polymorphism analysis | Operation order, operation kinds, branch results and attributes |

- **`ProcInfo` layout.** A `ProcInfo` is a handful of flat arrays indexed by the goal and unification numbers that desugaring assigned, plus the procedure's argument modes and determinism. Adding a mode to a predicate costs one more `ProcInfo`, not a copy of its body. Per-procedure additions, such as the test after an implied-mode call, are entries in these arrays, not new nodes.
- **Insts.** Instantiation states (`free`, `ground` with a uniqueness, `bound(…)` trees, `any`, higher-order insts and named insts) are hash-consed in the `ASTContext`, so equal insts are the same pointer. The operations mode analysis performs constantly, matching, merging at branch joins and abstract unification, are memoized on those pointers.
- **Types** are interned the same way: type constructors applied to arguments, type variables, higher-order types, and the opaque types that calls give their callees' existentially quantified variables.

### Passes, in order

Analysis takes one strongly connected component (SCC) of the call graph at a time, bottom up, so that every callee outside the SCC is finished first. Inference iterates within an SCC until it reaches a fixpoint, and independent SCCs can be analysed in parallel.

1. **Name resolution:** module-qualify every name, and record on its AST node the candidate declarations of each overloaded predicate, function or constructor (below).
2. **Type checking and inference:** the manual's type assignments, computed Hindley–Milner style with overloading, type classes (multi-parameter, functional dependencies, superclasses), existential types and equivalence types. Overload resolution picks one candidate per node. Types of local predicates without declarations are inferred, as mmc's `--infer-types` does; exported predicates must be declared.
3. **Purity checking:** `pure`, `semipure` and `impure` goals and `promise_pure` scopes. A predicate's purity is inferred from its body and compared with its declaration.
4. **Mode analysis,** per procedure (below).
5. **Switch detection and determinism analysis:** disjunctions whose arms unify the same input variable with different constructors become switches. Each goal's determinism is then inferred by the manual's rules (`det`, `semidet`, `multi`, `nondet`, `cc_multi`, `cc_nondet`, `erroneous`, `failure`). Local procedures without annotations get a fixpoint of those rules, starting from `erroneous`, as in mmc, where the manual says `det` (see the register). Incomplete switches in `det` code are errors, which is what makes catch-all receive clauses unnecessary. Commits are inserted where solutions are indistinguishable.
6. **Uniqueness checking** (below): enforce `di`/`uo` and `mostly_unique` modes, including for arrays, which needs to know where backtracking can happen, so it runs after determinism analysis. The reference manual says unique modes are not yet fully implemented in mmc; we implement them fully, because parallel loops and move-on-send depend on them. The first step matches mmc's second phase of unique mode analysis; nested unique modes come after it.
7. **Polymorphism analysis:** for each procedure, decide where every `type_info` and `typeclass_info` comes from: a hidden argument, a field of an existential constructor, a superclass of another dictionary, or a constant built at the call site. This runs after mode analysis, because the source can depend on the mode: a deconstruction yields an existential's type info, while a construction must supply one. MLIR generation materializes the arguments; the AST is not rewritten. Done: `type_info`s and `typeclass_info`s, existential types and closures included, for all but instance methods with existential types (below).
8. **Sendability analysis (new):** decide per type whether values may be sent, and check `sendable(M)` where mailboxes are created (Language extensions).
9. **OpenMP directive checks (new):** the associated goal must be a call to `int.fold_up`, `int.fold_down`, `array.map` or `array.generate`, or a self-recursive counted loop whose `index` and `upper` clauses name its induction variable and bound. Its body must be `det`, pure and free of `io`, and a `reduction` must apply to the loop's accumulator. Violations are errors, never a silent sequential fallback.

### Name resolution

Name resolution (`resolve-names.cpp`) finds, for each name, every declaration it may refer to, and records them on the `SymName`, in its `candidates` slot. Choosing among them is type checking's job.

- **Symbols.** Each module's declarations become `Symbol`s, identified by kind, module, name and arity, and a constructor or field by its type too, since types of one module may share constructors: types, constructors, insts, modes, type classes, predicates and functions, and each field as two functions, `f/1` to select it and `'f :='/2` to update it. A class method is a predicate or function of the module, with its class as its owner, as are a constructor's and a field's type. A type's abstract declaration in the interface and its definition in the implementation are one symbol, which is exported. The primitive types (`int`, the sized integers, `float`, `char`, `string`, and `character`; see the register) belong to no module, as in mmc, where `builtin.int` is undefined. With `char` imported, `char` therefore has two candidates, the primitive type and `char.char`, which are one type once the equivalence `char.char == character` is expanded.
- **Scopes,** by the manual's rules ("The module system"). A module's implementation sees all its own symbols and the exported symbols of every module it imports or uses; its interface sees only what it exports and the modules its interface section imports or uses. An imported module's declarations are resolved in its own interface's scope. A qualified name looks in the visible modules it names, as a suffix of their full names; an unqualified one in the module itself, the modules imported with `import_module`, and, for a type, the primitive types. A module imported with `use_module` is visible only qualified. `builtin` is imported by every module.
- **Uses.** A name is resolved by what it is used as: a type, inst, mode or class, the predicate of a call, or the function symbol of a data-term. A data-term `f(X1, ..., Xn)` may be a constructor or field of arity n, a function of arity n, or a closure of a predicate or function with n or more arguments, so `foldl(F, Xs, 0)` has both `list.foldl/3` and `list.foldl/4` as candidates, as mmc lists them. A clause's head and a mode declaration name a predicate or function of the module's own. A clause for a predicate or function without a declaration declares a local one, whose types are to be inferred (mmc's `--infer-all`).
- **Undefined names** are errors in mmc's words ("undefined predicate", "undefined symbol" for a data-term, "undefined type"), with a note for mmc's cases: the module named was not imported, or not in the interface; a declaration of the module's own that is not exported is used in the interface; a name of a module imported with `use_module` is written unqualified; and a name that exists with another arity.
- **Duplicate declarations** of the module's own are reported as mmc reports them, and the first is kept (`Duplicate`): a predicate or function declared twice, whatever its types, modes or `with_type`, a method and a predicate of one name and arity among them; a type, inst or mode defined twice, and a type as both a discriminated union type and an equivalence type, reported at the former. A type declared abstractly twice is a warning. A type's abstract declaration and a `foreign_type` pragma go with its definition, which becomes the symbol's declaration and source. A duplicate type's constructors are not its type's, and a duplicate type class's methods are reported with the class only. Imported modules' duplicates are not reported.
- **Instance methods.** A method clause's name resolves to its type class's method of that kind, name and arity, and its body is resolved as other clauses are. A clause for a method the class does not have is not resolved further; type checking reports it.
- **Not resolved yet:** functors in `bound` insts, pragmas, and `__` as a qualifier.
- **List syntax and characters.** `[X | Xs]` and `[]` resolve to the `list` module's constructors `'[|]'/2` and `'[]'/0`, so `list` must be imported to use them, as in mmc ("undefined symbol `[|]'/2"). A name of one character with no arguments, such as `a`, may be a character constant; it is not undefined even if no constructor has the name.
- **`with_type` declarations.** `:- pred p(int) : foldl_pred(T, U)` has the arguments of the higher-order type that its type is, through equivalences, so its arity is known only once the types are; such a declaration's symbol is added after the module's others.

The resolver found gaps in the library stubs, which are now fixed: types whose definitions mention the sized integers had been dropped, such as `string.poly_type`, and `bool` and `exception` had not imported modules their interfaces use.

### Type checking

Type checking (`check-types.cpp`) follows the manual's definition ("Predicate and function type declarations") literally. A type assignment gives a type to every variable of a clause and chooses a declaration for every name in it. A clause is type-correct if it has a unique valid assignment, up to renaming.

The type system is the one Jeffery's thesis defines (Sources), which describes the one in mmc. Its base, section 2.3, is a higher-order version of the Mycroft–O'Keefe type system, which carries Hindley–Milner over to logic programs, and the checker follows its rules. A call instantiates its callee's universally quantified signature and unifies it with the arguments (MO-CALL). A unification unifies its sides' types (MO-UNIFY). Higher-order values are monomorphic within a clause (MO-HO-CALL). A predicate's declaration gives its clauses' types (MO-PRED). It differs from Hindley–Milner's Algorithm W in three ways:

- **No `let`-polymorphism.** A clause's variables are monomorphic, and types are generalized only for a whole predicate or function, by its declaration or, for an undeclared one, by inference.
- **Declarations are checked.** Callers use a declared signature, so a declared predicate may call itself at another type, which Hindley–Milner does not allow. Inferring the types of undeclared ones must iterate to a fixpoint over each strongly connected component of the call graph.
- **Overloading.** The thesis's rules give each name one type. Mercury allows several declarations with the same name and arity in different modules, and the manual's definition chooses among them. The assignment sets below do this; when every name has one candidate, there is one assignment, and the checker is the thesis's unification traversal.

Its chapter 3, with the typing rules of figures 3.3 to 3.10, is the reference for type classes, and chapter 4 for existential types, including how they combine with type classes. The thesis implements type classes by a source-to-source transformation that passes dictionaries. Here that transformation is not done to the AST: polymorphism analysis decides where each dictionary comes from, after mode analysis, and MLIR generation passes them (Passes, in order). Subtypes and `coerce` came after the thesis.

- **Types** are interned in a `TypeContext`: variables, type constructors applied to arguments, tuples, and higher-order types, with their purity. Equivalence types are expanded when a type is made, so `io.io` is `io.state` and `char.char` is the primitive `char`, which messages call `character`, as mmc does (see the register). A circular equivalence is an error.
- **Signatures.** Each predicate and function, constructor, field selector (`f/1`) and field update (`'f :='/2`) has a signature, made from its declaration. The module's own are made, and its equivalence types expanded, before inference starts, so that their errors are reported; an imported module's on first use. A function's result is its last argument. A `with_type` declaration adds its higher-order type's arguments. A signature has the declaration's class constraints (`<= c(T)`), and each of their variables must occur in the argument types. A method's signature has its type class's parameters as its first type variables, and its type class's constraint first, as if it were declared `<= c(T)`.
- **Assignment sets.** The checker keeps the set of assignments that are still possible. It starts from one, in which the head has the declaration's types, and takes the goals in order. Each goal keeps the assignments it is valid in. An overloaded name splits each assignment into one per declaration it may refer to. Name resolution gave the candidates: a data-term `f(X1, ..., Xn)` may be a constructor, a field, a function of arity n, or a closure of a predicate or function with more arguments. A unification of types binds type variables, with an occurs check. This is mmc's method too, as its "highly ambiguous overloading" warning shows, and it reports the manual's errors exactly. More than 1000 assignments is an error that asks for a type qualification.
- **Rigid variables.** The head must match the declaration exactly, so the declaration's type variables are rigid: they unify only with themselves and unbound variables. A type qualification's variable with one of their names is that variable. Its other variables are the clause's own, shared by all its qualifications, as in mmc.
- **Errors.** A goal that leaves no assignment is a type error. It is reported at the term, with the types involved and, for an overloaded name, each declaration it may refer to. The goal is then skipped, so one clause can report several errors. Assignments left that chose different declarations for a name are mmc's "ambiguous overloading causes type ambiguity". A variable whose type nothing determines draws mmc's "unresolved polymorphism" warning; its variables will be bound to `void`. The warning lists every such variable, a term's by the term, as `[]` in `length([])`. After an error or an undefined name in a clause, neither an ambiguity nor the warning is reported for it, since either may only follow from the error.
- **Results.** For each clause, `ClauseTypes` holds each variable's type and, by goal number, the declaration chosen for each name. These are side tables, as for `ProcInfo`, so the semantic form is not changed. To give errors a position, desugaring gives every unification with a functor the tokens of its term, `f(...)`, rather than of the goal `X = f(...)`.
- **Type inference.** The types of predicates and functions without declarations are inferred by default, as mmc does with `--infer-types`. `-fno-infer-types` is a pedantic mode that reports each one in mmc's words ("clause for predicate ... without corresponding `:- pred' declaration"), and still infers it. Inference takes the call graph's strongly connected components, callees first. Each member's signature starts most general, a type variable per argument. Each pass checks the member's clauses with free head types and calls by the last pass's signatures. A clause's remaining assignments may give its head more than one typing, by choosing different declarations for a name. Each combination of one typing per clause, renamed apart and unified, is a candidate, and is generalized. A clause none of whose typings fits the others' is left out. The passes repeat until no signature changes, so a member may call itself at other types, as a declared one may.
  - **Ambiguity.** Clauses with more than one candidate signature at the fixpoint have more than one type assignment, even when one candidate is an instance of another. That is an error (reference manual, "Predicate and function type declarations"), as in mmc. `f(Xs) = list.append(Xs, [1])` may be the function `list.append/2` or a closure of the predicate `list.append/3`. `length(item) = 1` and `item = length(3)`, with `length/1` also a constructor of the type `item`, have two typings that are both fixpoints. Such a predicate or function is then left without a signature, so that its clauses report the ambiguity, as a declared one's would, and its callers are not checked by it. Until the fixpoint, an ambiguous member's next signature is the least general generalization of its candidates.
  - **Limit.** At most 60 passes, as mmc's `--type-inference-iteration-limit`, which `-ftype-inference-iteration-limit=N` sets. A component that does not settle, such as `p(X, Y) :- p([X], Y0), Y = [Y0]`, is an error, as in mmc. It is then left without signatures, so its last guess draws no more errors. A signature can also double in size in every pass, as for `p(_, _, S0, S) :- S0 = p, S = S0`, where `p` is a closure of itself. Inference therefore also stops when an argument type has more than 1000 nodes as a tree.
  - **Shared types.** Interned types share their parts, so a type can be exponentially larger as a tree than as a graph: `X1 = {X0, X0}, X2 = {X1, X1}, ...` doubles in each goal. Substitution, the occurs check, unification and size counting therefore visit each shared part once, and a message prints a type of more than 200 nodes as a placeholder.
  - **Errors.** The passes' messages are dropped. Afterwards every clause is checked once more by the final signatures, declared or inferred, and reports its errors as a declared one would.
  - **Mode and determinism inference** are not part of this; mode analysis infers modes later.
- **Testing against mmc.** `utils/type-oracle/compare-types.py` checks modules by merkc and by `mmc --errorcheck-only --infer-all`. mmc's inference is little used, and has bugs (see the register), so its checking of declared types settles differences:
  - **Errors** must agree by predicate. Where only one of them reports one, mmc checks the module again with merkc's other inferred signatures declared, and, for a recursive predicate, with its own too. With its own declared too when an error of mmc's first pass, in a caller, hides a constraint that mmc reports only without other errors.
  - **Signatures** must agree up to renaming. Where mmc's differs, mmc must reject its own, declared, and merkc's must be at least as general.
  - **Consistency.** With merkc's inferred signatures declared, merkc must give every variable the type that inference gave it.
  - **Soundness.** A typed version of the module must pass both checkers. It declares merkc's inferred signatures, qualifies each clause variable with its type (`X = (X : T)`) and each overloaded name with the module of its chosen declaration.

  With `--error-lines`, for modules of declaration errors, only the lines with errors and notes are compared, since mmc stops at the first phase with errors. With `--include DIR`, a module may import those in `DIR`. The lit tests under `test/Semantics/types/typeclasses` use both, each for one phase of mmc's checks; mmc reports constraints on undetermined types only in a module without other type errors, so those have tests of their own, and the oracle leaves merkc's out where merkc reports other errors. mmc may report such a constraint at the declaration, with the goal in a note, or with no goal; the oracle takes the goal's line, or the first clause's. The lit tests under `test/Semantics/types/oracle` run it on modules by kind of term and goal, on ambiguities and type errors, and on random modules from `generate-random-programs.py`. That generator gives predicates random types, some declared, and builds clauses that fit them, with a mutation in some; with `--classes`, the modules have type classes and instances, the types have constraints, and the clauses call the methods on types that satisfy them; with `--dependencies`, a type class with a functional dependency too, whose method's result only improvement determines; with `--existentials`, existentially typed functions, whose results only their existential constraints and polymorphic callees accept; with `--existential-constructors`, an existentially typed constructor too; with `--coerce`, a subtype and coerce expressions.
- **Type classes** (reference manual, "Type classes"). Before the clauses, the module's typeclass and instance declarations are checked by the manual's rules:
  - **Typeclasses.** The parameters are distinct variables (the parser checks this). A superclass constraint's arguments are type variables among the parameters, or ground types. The superclass relation is acyclic, and each cycle is reported once, with its classes. A type class is defined once, and an abstract declaration has a definition with the same superclass constraints, up to renaming of the parameters.
  - **Instance types.** Each, after equivalence types are expanded, is a type constructor or tuple applied to type variables, which may repeat, as in `p(T, T)`.
  - **Instance constraints** constrain type variables of the instance's types, or ground types.
  - **Overlap.** Since instance types' arguments are variables, two instances overlap if their type constructors are the same position by position. The module's instances must not overlap each other or those of the interfaces it loads; mmc reports the second only when linking, and misses some of the first (see the register).
  - **Abstract instances.** A non-abstract instance is not allowed in the interface section. An abstract one, in either section, needs a definition in the module with the same types and constraints, up to renaming. Variables are numbered in the order they occur, so these compare as interned types do.
  - **Methods.** Each method is defined once, by clauses or by naming a predicate or function, and each definition is of a method of the type class, with its kind, name and arity. A type class whose declaration is abstract where the instance is has no visible methods, so it can have no instances there.
  - **Method definitions** (reference manual, "Instance declarations"). Each of a valid instance's methods is an auxiliary predicate or function, a `MethodDefinition` of the result: its signature is the method's, with the instance's types in place of the class's parameters, the instance's type variables numbered first and the method's own after them. Its constraints are the instance's and the method's others; the method's first, its class's on the instance's types, is what the instance gives. The instance's method clauses are checked by it, as a declared predicate's are, with its constraints assumed. Desugaring makes `pred(M/N) is P` the clause `M(X1, ..., XN) :- P(X1, ..., XN)`, and `func(M/N) is F` the clause `M(X1, ..., XN) = Y :- Y = F(X1, ..., XN)`, whose goal has F's tokens. As in mmc, P or F must then be at least as general as the method; F is a data-term, so a predicate there is a closure, and a constructor is accepted (see the register). `-fdebug-dump-types` prints a method clause with its instance and signature, after the module's other clauses, and `-fdebug-dump-semantic-form` prints the desugared method clauses.
  - **Functional dependencies** (reference manual, "Functional dependencies"). The parser checks that a dependency's variables are the class's parameters; an abstract declaration has its definition's dependencies, in any order. In each of the module's instances, the domain's type variables, closed under the dependencies that the instance's constraints induce, include the range's; and two instances whose domains unify, renamed apart, have the same range by the unifier, each of the module's against the earlier ones (an overlap is reported as one). The constraints of a signature induce dependencies, with those of their superclasses, recursively, and a constraint's variables need only be in the closure of the argument types' under them, so a method need not use a parameter that a dependency determines.
  - **Improvement.** Once overloading is resolved, as in mmc, which does not use improvement to choose among overloadings, each assignment's constraints are improved to a fixpoint: a constraint with the domain of an assumed one, or of another wanted one, takes its range; one whose domain an instance's types match takes the instance's range, with the instance's other variables fresh, and the instance's constraints, so instantiated, are improved in turn. A unification that fails, or binds none of the assignment's variables, is undone, and the constraint's check then reports it, as mmc does: with `baz(int, string)`, `baz(1)` declared `int` is an unsatisfiable `baz(int, int)`. Inference keeps constraints on the head's variables only (see the register).
  - **Constraint satisfaction** (thesis, section 3.1.4). Applying a signature, in a call, a function application or a closure, instantiates its constraints with its types, and the assignment keeps them with the goal. They choose among overloadings only by ruling out: as mmc does, an assignment is dropped, if others are left, when one of its constraints reduces by the instances to a ground one that neither an instance nor the declaration gives, so `show(string.length(S))` takes the function, not the closure of the predicate. Once the clause has its unique assignment, each is resolved and entailed, in one of three ways:
    - **Assumed:** it is one of the declaration's constraints, or a superclass constraint of one of them, on its arguments.
    - **Instance:** an instance's types match it. Since instances do not overlap, at most one does. The instance's constraints, on the matched types, are then entailed in turn. A constraint that would need itself fails.
    - **Otherwise** it is an error at the goal, once per clause and constraint. A constraint on a type variable of the clause that nothing determines is mmc's "unsatisfied typeclass constraint", alongside the unresolved polymorphism warning. Any other is mmc's "unsatisfiable typeclass constraint": no instance fits, or the declaration does not assume it. When an instance reduced it, as `show([1.5])` to `show(float)`, the note names the constraint it came from.

    `ClauseTypes::constraints` records how each goal's constraint was satisfied, with the instances' premises below it, for polymorphism analysis; `-fdebug-dump-types` prints them. The instances are those of every interface loaded, as in mmc: an instance in a module that is loaded but not imported counts. An instance's superclass constraints, on its types, are entailed in the same way, with its own constraints assumed.
  - **Constraint inference** (thesis, section 3.2), for undeclared predicates and functions, with their types and as mmc does. A clause's typing has, with its head's types, its goals' constraints reduced by the instances: those that no instance matches and that have a type variable are kept, as `show(w(T))`, and a ground one is left to be checked. Of these, the constraints on the head's type variables are the typing's; the others, which nothing determines, are reported when the clause is checked. Repeats are removed, but a superclass constraint is kept beside its subclass's, as mmc keeps them. Combining the clauses' typings unifies their heads, and the constraints are reduced again; as in a clause, a combination with a refuted constraint is dropped if others fit, so one clause's constraint may decide another's overloading. The passes reach a fixpoint over the constraints as over the types: a pass changes nothing when each signature has the same arguments and the same set of constraints.
- **Existentially typed predicates and functions** (reference manual, "Existential types"; thesis, chapter 4).
  - **Declarations.** A signature numbers its existentially quantified variables last, after the universal ones, with its existential constraints (`=>`) apart from its universal ones (`<=`). An existential constraint may constrain only existential variables, and a universal one only universal ones; one that breaks the rule is reported and left out. A variable quantified twice is an error (mmc crashes on it; see the register). A quantified variable need not occur in the arguments, as in mmc.
  - **Callers.** A call, or a function application, gives each of its callee's existential variables a type of its own, an `Existential` type: an opaque type that unifies only with itself, so the caller cannot bind it, and two calls' types differ. The callee's existential constraints, on those types, are given to the assignment, which assumes them as it does the declaration's. The types are the same in each assignment, by goal, callee and variable, so that assignments that choose alike agree. No closure of an existentially typed predicate or function can be made, as in mmc (see the register).
  - **Callees.** In the callee's own clauses the existential variables are not rigid: the clause determines them, and must satisfy the existential constraints on what it determines, by instances or by its calls' existential constraints. Each clause must give them the same types (reference manual, "Semantics of type quantifiers"): the first clause's types, if determined, are bound in the later ones, so `bad_e_bar2("blah")` after `bad_e_bar2(42)` is a type error there, as in mmc. An instance's existentially typed method has its existential variables last, after the instance's and the method's own.
  - **Inference** gives an existential type only from a call's: the `Existential` types in an inferred head become its existential variables, named as their callees name them, with the existential constraints given on them. So `chained(X) :- constrained(X)` is existential too when `constrained` is only inferred to be, as the manual's "directly or indirectly" says. Signatures print as declarations: `some [T] pred p(T) => (c(T))`, or `some [T] (pred p(T, U) => (c(T))) <= (d(U))` with both kinds of constraint.
- **Existentially typed constructors** (reference manual, "Existentially typed data types").
  - **Definitions.** A constructor's existential variables are numbered after its type's parameters, and its constraints are existential constraints. No variable is both a parameter and existentially quantified; each existential variable occurs in the arguments or the constraints; and a constraint's variables are existentially quantified, and determined by the arguments' types, closed under functional dependencies. A constraint on a parameter is reported and left out.
  - **Deconstruction and construction.** A constructor without `new` is a deconstruction, and acts as an existentially typed function: each occurrence gives its existential variables opaque types, with its constraints given. Name resolution resolves `'new f'`, or `m.'new f'`, to the existentially typed constructors `f` of its arity; `new` on another constructor is an error, and no closure of `'new f'` can be made, since a constructor's arity is exact. Its signature is the universal version, whose constraints the construction must satisfy (`NewSignature`). Construction and deconstruction need no modes to tell apart.
  - **Fields.** A field's selection is a deconstruction. Its update constructs the field's existential variables, whose constraints it must satisfy, and keeps the others, so a field cannot be updated if another field shares one of its existential variables, as in mmc ("Field `a' cannot be updated").
- **Subtypes** (reference manual, "Subtypes"). A subtype is a type of its own: its constructors are its own symbols, which overload its supertype's, and its values are not its supertype's without a `coerce`. Its definition's rules:
  - **The supertype**, with equivalences expanded, is a discriminated union type whose definition is visible: not a type variable, a primitive, or an abstract type, even an abstractly exported subtype. mmc accepts an abstract type that is not a subtype (see the register). The chain of supertypes ends at a base type; each type on a circular chain is reported, with the chain.
  - **Constructors.** Each of the subtype's is one of its supertype's, by name and arity, with each argument's type a subtype of the supertype's, whose parameters are the supertype's arguments. A constructor defined twice in any discriminated union type is an error. Constructors in a different order from the supertype's draw a warning, as in mmc.
  - **Existential constructors** have the supertype's existential variables, one to one, and the same constraints.
  - **No equality or comparison** of its own.
  - **The relation** `S =< T` (`IsSubtype`): equal types; the same discriminated union type constructor, with each argument a subtype; or S's supertype, with S's arguments, a subtype of T. Any other types, higher-order ones among them, are subtypes only of themselves. `coerce` uses it.
- **Type conversions** (reference manual, "Type conversions"). `coerce(X)` binds no types: the clause records it, and once the clause's types are known its argument's and result's types are checked, as in mmc.
  - **Type-correct** when both are discriminated union types whose definitions are visible, whose base types (`BaseOf`, up the chain of supertypes, with the arguments) have the same type constructor; and for each of its parameters, the two arguments are the same, or, unless the parameter is invariant, one is a subtype of the other.
  - **Invariant parameters** (`InvariantIn`): those that occur in the base type's definition in a higher-order type, in another type constructor's arguments (a discriminated union type other than the base type's own exact recursion `base(B1, ..., Bn)`, or an abstract, foreign or primitive type), with equivalences expanded. A tuple's elements are not such a place, as neither the manual nor mmc has them.
  - **Undetermined types.** A coerce whose argument's or result's type nothing else determines is an error, as in mmc, which infers nothing through it. A rigid type variable is not a discriminated union type.
  - **Overloading.** Assignments with an invalid coerce are dropped at the end of the clause if others are left, as mmc does, so in `X = k, Y = coerce(X)` the result's type chooses `k`. Otherwise each invalid coerce is reported with the reason, and an ambiguity hides them.
  - **Modes.** A coerce to a subtype, or between sibling subtypes, is type-correct; whether the argument's inst allows it is for mode analysis.
- **Not yet:** the states of `atomic` goals, since `stm_builtin` is not stubbed. Instance methods' purity, modes and determinism are checked by purity checking, mode analysis and determinism analysis.

### Purity checking

Purity checking (`check-purity.cpp`) runs after type checking, by the callees it chose; a goal with a type error is not checked. It follows the reference manual's "Impurity declarations" and mmc. See also Becket, "Unclean! Unclean! or Purity Issues in Declarative Constraint Logic Programming" (2006), on purity and solver types.

- **Annotations.** A call of an impure or semipure predicate is annotated so. An impure or semipure function is applied in a unification of its own, so annotated: the unification whose result is a variable the program names, since a nested application's result is a fresh variable; nested pure functions are fine. A missing annotation, or `semipure` where `impure` is needed, is an error; one more impure than the callee is a warning. An annotation on a compound goal is an error, and its goals are taken to have it; one on a unification that applies no function is an error, except between two variables, as in mmc.
- **Goals' purity** is the least of their parts': a call's or function application's is its callee's declared purity, annotated or not; a `promise_pure`, `promise_semipure` or `promise_impure` scope's is the one it promises; a trace goal's is pure.
- **Predicates and functions.** One's purity is the least of its clauses'. Declared purer than that is an error, at the declaration, and less pure a warning. One without a declaration is pure, since mmc infers no purity, so its impurity is an error at its first clause. A `promise_pure` or `promise_semipure` pragma names a predicate or function of the module, a function by its arity without the result; it must promise the declared purity, and stands for the clauses' purity; one that is not needed is a warning.
- **Higher-order calls** (reference manual, "Purity annotations on higher-order calls"). A higher-order predicate call's annotation is the purity of the closure it needs, which desugaring records on the `HOCall`, and type checking makes the closure's type's, so a missing, unnecessary or too weak annotation is a type error, as in mmc. `apply` and `F(X)` apply pure functions, and `impure_apply` and `semipure_apply` impure and semipure ones; mmc needs no annotation on `impure_apply`'s unification (see the register). Making a closure, of an impure predicate or function too, is pure; a unification applies a function only with all its arguments.
- **Lambda expressions.** A lambda expression's body is at most as impure as its annotation, and an annotation less pure than the body is fine, as in mmc. The body does not count towards its clause's purity, and its goals are not under the annotation of the goal the lambda expression is in.
- **Instance methods** are at least as pure as their declarations in the type class, whether defined by clauses or by name; the call that a definition by name stands for has the method's declared purity, so it is reported unannotated only when the method is declared purer.
- **Type errors.** A predicate or function with a type error in a clause is not checked, as mmc checks purity only after type checking succeeds.
- **Clauses for one mode** (reference manual, "Different clauses for different modes"). A predicate or function with more than one mode, and a clause whose head arguments have modes, is impure unless it is promised pure or `promise_equivalent_clauses`; the latter does not promise the clauses' own goals pure. With one mode, mode annotations change nothing, as in mmc. Desugaring records the clause's `modeSpecific`.
- **`foreign_proc`** (reference manual, "Foreign code attributes"). A `foreign_proc` is impure unless its `promise_pure` or `promise_semipure` attribute says otherwise, and its purity must be the declared one, as in mmc, which reports both the predicate and the `foreign_proc`; a pragma promise stands for the attribute. Only `foreign_proc`s for C count, as merkc's foreign code is C.
- **Goals with `any` nonlocals in negated contexts** need a purity promise; mode analysis checks this, since it needs the insts (Mode analysis, "The `any` inst").

### Mode analysis

Mode analysis runs per procedure, starting from the instmap that the argument modes give. It follows chapter 3 of Overton's thesis, mmc's system, and chapter 4, constrained polymorphic modes (Overton, "Precise and Expressive Mode Systems for Typed Logic Programming Languages", 2003); the constraint-based analysis of chapter 6 is not used. So far `check-modes.cpp` reads the insts, modes and procedures, reports the errors in their declarations, and checks the clauses made of conjunctions, disjunctions, if-then-elses, negations, unifications, calls, including function applications and field accesses and updates, higher-order terms and calls, constrained polymorphic modes, the first phase of unique modes, the `any` inst, type conversions, and the clauses of instance methods, and infers the modes of predicates without mode declarations. `-fdebug-dump-modes` runs type, purity and mode checking and prints each inst definition and procedure, and under each procedure its clauses' goals in execution order, each with what it does (a construction, deconstruction, assignment, test, call or higher-order call, with the procedure called, or for a higher-order term made from a name, its procedure), a branched goal's branches and a lambda expression's body indented under it, and the head's final insts; or "mode error"; or "not checked" with the reason; then the procedures inferred from calls, by predicate, as `pred m.p/2 inferred mode 0: p(in, uo)`, a rejected one with its final insts `not_reached`, as mmc writes them `...`; and then each instance method, by instance, in each of the method's modes, as `method m.c(m.t) pred m.p/1 mode 0`, the same way. A procedure is not checked yet when its predicate has type errors or mode-specific clauses, or a clause has a parallel conjunction, `trace` scope, `try` or `atomic` goal, or a field access function's name as a higher-order term, or a lambda expression whose inst variables have inconsistent constraints, or calls an inferred procedure that is not checked, or makes a higher-order term of a predicate whose modes or determinism are to be inferred, since neither mmc nor merkc infers higher-order modes. `utils/type-oracle/generate-mode-programs.py` writes random modules for `compare-types.py --modes --error-lines`, which compares the lines of mode errors with mmc's, and the inferred and rejected modes (`test/Semantics/modes/random-modes-*.test`); with `--undeclared`, some predicates have no declarations.

- **Insts** (`insts.h`) are interned, so equal insts are one pointer. Each inst but `free` has a uniqueness, from `unique` to `clobbered` (thesis, 3.2.3). A bound inst's functors are kept by name, sorted, and resolved against a variable's type when an operation needs their argument types, so `bound(a ; d)` is accepted where `d` is no constructor, as in mmc. A bound inst is `bound(...)`, or with a uniqueness `unique(...)` or `bound_unique(...)` (reference manual, "Destructive update") or `mostly_unique(...)`, which mmc has though the manual does not, and the insts below its top level may be unique too (see the register). A builtin inst with arguments it does not take, as `ground(a)` or `unique(a, b)`, is mmc's "the builtin inst ground should only be used with arity zero", and leaves the definition it is in undefined, as in mmc; mmc also drops a `:- pred` or `:- mode` declaration that has one, so its clauses are without a declaration, which merkc does not do. A named inst stays named and is expanded one level when an operation needs its body, so recursive insts such as `list_skel(I)` are finite. An inst parameter carries its constraint, `ground` if none is given, and the declaration it belongs to, so that a callee's are not its caller's. A variable's inst may be a constrained inst, `(I, J =< bound(a))` as mmc prints it: an inst known to be compatible with every inst that each of a set of inst parameters stands for (thesis, 4.2.3); an inst parameter is the constrained inst of itself and its constraint.
- **Operations.** Matching (⊑, thesis definition 3.2.15) and instantiatedness (≼, 3.2.14) are compared coinductively on recursive insts; merging is the least upper bound under ⊑, and abstract unification the greatest lower bound under ≼, with a recursive result tied as a generated inst. `free` does not match `any`. `ground` stands for every constructor of the variable's type where a bound inst is compared with it; for a type without a known set of constructors it does not. Two different higher-order insts merge to `ground`, and do not unify. Liveness and clobbering are left to the caller.
- **Procedures.** A predicate or function has one procedure for each mode: from a declaration that combines types and modes, from each separate mode declaration, including a type class's for its methods, with the arguments of a `with_inst` inst added, or, for a function without either, the default mode `f(in, ...) = out is det`. A predicate without a `:- pred` declaration has its modes inferred from its calls (Mode inference), as mmc does even without `--infer-modes`; with `-finfer-modes`, mmc's `--infer-modes`, so does a declared predicate without mode declarations, which is otherwise an error. A function without a declaration has the default mode, as in mmc. `-fmode-inference-iteration-limit=N` (default 30) is mmc's `--mode-inference-iteration-limit`.
- **Errors in declarations,** in mmc's cases: a mode declaration with no predicate or function of its name, arity and kind, with the arities there are, and a hint when it is a function's mode written without `= Mode` or the other way; a mode declared twice, whatever the determinism, at the first, with a note at the other; modes both in a `:- pred` declaration and separately; a mode declaration in the other section from its predicate's; an exported mode without a determinism, which a `with_inst` inst can give; a declared predicate with no modes, which is reported for an exported one even with mode inference; an inst parameter with two different constraints in one declaration (in a `<=` list the first is kept, as in mmc; see the register); and an inst declared `for` one type constructor applied to values of another, at an argument or within the expansion of a named inst there, once per argument.
- **Errors in definitions:** reserved inst and mode names (`ground`, `>>` and the others); an inst variable on the right-hand side that is not a parameter, which leaves the definition undefined, so its uses are "undefined inst" or "undefined mode", as in mmc; and an inst or mode that expands to itself, with the next definition of the cycle as a note.

- **Scheduling.** A conjunction's conjuncts, flattened, are tried in source order. One that cannot run yet, because a variable it needs is not bound enough, waits, on the variables whose change could let it run: for a call with one mode, the first argument that does not match, with and without an implied mode; for a call with several, every argument not yet bound; for a unification, its variables with free parts and its left side. When a goal runs, the waiting goals whose variables it changed are woken and run next, in the order they waited, before the other goals; one that still cannot run waits again, at the end of the queue. When no goal is left, every waiting goal is retried once, in order, since the goals that ran may have left its variables unused, and again while a retry runs one. A branched goal or negation waits as a whole, for what the goal inside that failed waits for, and a conjunction inside one for what any of its waiting goals waits for; a mode mismatch waits for its variable. A lambda expression with a mode error in it waits for no variable, so that only the retries try it again. After a goal in which some insts are not reached, as in a branch, condition or negation that cannot succeed, every waiting goal is woken, as if each of its variables had changed: in the order of the first variable each waits for, by number, those that wait for none last, and one of the latter that still cannot run is then not retried at the end, so that it stays first. A variable changes when its inst does, or when a construction puts it into a term, free or not, or when it is aliased to a free variable that is not used again; the last two can make a goal's variables dead, so that it can run. A goal that waits because it would clobber a variable used after it is also woken when any goal that uses the variable runs, as an assignment of it to another, even if its inst stays the same, as a `ground` one's does, but no other goal is (`modes/assignment-wakes.m`). After a goal that cannot succeed, every variable is `not_reached`, and the rest of the conjunction is not checked, so that goal is the last run, with mmc's warning that the unification cannot succeed. If goals are left waiting, that is a mode error, reported at the first waiting goal, with its own error; mmc's verbose message (`-E`) lists every waiting goal, in that order. This is mmc's order, found by experiment, with mmc's `--debug-modes` trace. It is not exact: about one generated module in a hundred has a goal that mmc does not wake when a construction puts its variable, free, into a term, where merkc wakes it, so that a different waiting goal is reported. A rule that tells the cases apart has not been found; not waking at all disagrees more often. Later, a waiting goal that needs a free variable of an open type will make that variable a fresh unbound variable, and scheduling will continue (Language extensions, "Open terms").
- **Liveness.** A head variable is live after the clause unless its final inst is `clobbered`, as for `di`. Another variable is live after a goal if a goal of an enclosing conjunction that has not run uses it.
- **Unique modes, the first phase** (Overton, 3.2.1 and 3.2.3; reference manual, "Unique modes"). Liveness decides sharing: a test of two variables both used after makes both shared (definition 3.2.19); an assignment shares the value if its source and its target are both used after (one to a variable not used again mmc deletes); a deconstruction keeps the term's uniqueness, and a part that both the term and its variable use after is shared, in both (3.2.20). The term's copy of a construction's argument is shared if the argument is used after, or if the term has more than one copy of it, as `p(A, A)` does; if the term is used after too, the argument is shared as well, as in the thesis, where mmc keeps its inst, unsoundly (see the register; `-fmmc-uniqueness`). A unique term's fields may be unique, and these rules apply to them as to any value: a field taken out of a term that is used after is shared in both, one passed to an `in` argument is shared after, and a term rebuilt from a shared field is unique only at its top, `unique(p(ground, unique))`. Passing a unique value to an `in` argument makes it shared after, since the final inst is `ground`, whatever its inst; mmc keeps some named insts unique there, unsoundly (see the register). A value of a type whose values are copied, a number, a character or an enumeration, has no other reference however it is used, so a shared or `mostly_unique` one matches `unique`: taking a number out of a unique term keeps the term unique, which makes `ui` arguments usable; mmc compares them as any other, and rejects that (see the register; `-fmmc-uniqueness`). A named inst is shared as its body is, unique parts below its top level too, and keeps its name if that changes nothing; a recursive one, as `ul == unique([] ; [unique | ul])`, becomes a generated recursive inst, printed unfolded once. A call whose mode clobbers an argument that is used after is "unique-mode error: the called procedure would clobber its argument, but `X` is still live", for a higher-order call too; with one procedure, mmc reports it before an argument's inst that does not match. `mostly_unique` and `mostly_clobbered`, and values needed on backtracking, are the second phase, after determinism analysis (pass 6); mmc reports those as mode errors too, so a test of this phase uses `det` code, or keeps a unique value's construction and its clobbering call free of choice points between them.
- **The `any` inst** (reference manual, "Solver types and negated contexts"). Negation as failure is unsound if a negated goal can constrain a solver variable, so a negation, or an if-then-else's condition, with a nonlocal whose inst has an `any` part, of a solver type or not, is a "purity error" unless it is inside a `promise_pure`, `promise_semipure` or `promise_impure` scope; as in mmc, a `promise_pure` pragma or an `impure` declaration does not count. The rule is checked after the goal inside, so an inner negation is reported first, as in mmc. A `pred` or `func` lambda expression is ground, so a nonlocal whose inst contains `any` is a purity error; an `any_pred` or `any_func` one, which the parser now accepts, may have them. As in mmc, a negated call of an `any_pred` closure needs no promise, though the manual asks for one (see the register).
- **Type conversions** (reference manual, "Type conversions", "Mode checking"). `Y = coerce(X)` waits until `X` is bound, and gives `Y` the inst the manual's traversal builds: a recursive inst node needs its type to be at most the result's; a bound node keeps its functors, each of which the result's type must have; a ground node of the same type stays, and one of a subtype becomes the bound inst of the subtype's constructors, down to a recursive type node, which is ground. A failure is mmc's message, with the path to the subterm ("in the first argument of function symbol `p`") and the reason: a functor the type lacks, or a ground value whose type is not a subtype. The result keeps the argument's uniqueness, and as for an assignment both are shared if `X` is used after. With `Y` bound, as in mmc, the converted value is unified with it, with a warning if that cannot succeed.
- **Mode inference** (Overton, 3.4.2; user's guide, `--infer-modes`; the reference manual does not describe it). A call of a predicate whose modes are inferred uses an inferred procedure whose initial insts its arguments match, as any call does, but not one that would clobber an argument used after, nor a rejected one; otherwise it makes a procedure. Its initial insts are the arguments', normalised: a ground inst becomes ground of the least unique uniqueness of its parts (definition 3.4.1), while a partly free inst or an inst variable stays, since mmc does not normalise below a bound inst either. A unique argument not used after the call is clobbered, `di`; the other final insts are those of the clauses, merged and normalised, so `p(X, Y) :- Y = X + 1` called with a shared `X` is `p(in, uo)`, and `id(X, X)` called with a unique list used after is `id((unique >> ground), out)`, as in mmc. The module is checked in passes, as mmc does with `--infer-modes`: each checks the declared procedures and then the inferred ones, those it made included, and a call of an inferred procedure not yet checked leaves the goals after it not reached in that pass, even when its final insts are known. Passes repeat until none makes a procedure or changes one's final insts, or, as in mmc, one finds a mode error in a procedure not inferred, which ends inference with that pass's errors; the iteration limit is an error, as mmc's "Mode analysis iteration limit exceeded", which a recursive mode with partly free insts can reach, since they grow with each pass, in mmc too (thesis, "Limitations of Mode Inference"). A procedure with a mode error is rejected, mmc's "REJECTED :- mode p(out(...))", and its error is not reported by itself: a call whose insts, normalised, are a rejected mode's waits, as a call that matches no mode does, so `bad(Y), Y = 3` makes `bad((unique >> ground))` instead; if nothing comes, the call is the error, with the rejected mode's error as a note, at the predicate's first clause, or in a lambda expression at it, as in mmc. Without `--infer-modes`, mmc checks a caller's goals after a call that makes a mode in no later pass, and may fail without a message; merkc does what `--infer-modes` does (see the register). Determinism is not inferred yet, so an inferred procedure has none, which pass 5 will give it.
- **Instance methods.** An instance's clauses for a method, and the clause that `pred(m/2) is p` desugars to, `m(HeadVar__1, HeadVar__2) :- p(HeadVar__1, HeadVar__2)`, are checked in each of the method's modes, as a predicate's clauses are, with the instance's types; an error says "in the clause for the method `m(in, out)` of instance `m.c(m.t)`", after mmc's "In clause for type class method implementation for `m(in, out)`".
- **Unifications** are classified by the insts of their two sides: construction when the left side is free, deconstruction when it is bound and the right side has free arguments, assignment, a test when both are bound, or a call to the type's unify procedure for tests on structured values. In a deconstruction or test, a bound argument is tested against a fresh variable for the bound variable's own argument, as mmc splits it, so the variable learns only the functor there: after `M = 1, S = c(M)`, with `S::in`, `S` is `c(ground)`, and `S = c(0)` after it can still succeed (`determinism/case-splits.m`). Without alias tracking, a value with free parts may not be copied while both copies stay live: unifying two free variables that are both live is an error, and so is assigning, or putting into a live term, a live value with free parts, as in mmc; a construction whose term is not used again may have free arguments. A constructed term is `unique`, and a shared copy of a live value is `bound`. A live argument of a construction whose term is live too is shared as well, as an assignment's copies are, which mmc does not do (see the register; `-fmmc-uniqueness` keeps mmc's rule). All of this is except at open types (Language extensions). At open types, a unification of an `any` value with a functor is a `get_structure`, and one of two `any` values is a full unification.
- **Calls** choose a procedure of the callee whose initial insts the arguments match, the most specific if several do, leaving out, as mmc does, one that would clobber an argument used after the call if another does not: `append(["a"], L1, L2)` with `L1` used after it is `append(in, in, out)`, run in place, not `append(di, di, uo)`, delayed until `L1` is no longer used (`modes/clobber-choice.m`). If none does, a bound argument may match a free initial inst: the call writes to a fresh variable and an implicit unification with the original follows; this is an implied mode, which mmc uses for arguments with free parts too. With one procedure, the error is at the first argument that an implied mode cannot fix, as in mmc. An argument that was free where the mode's initial inst is free gets the final inst; another is unified with it. A head variable's final inst must match the declared one; as in mmc the error says it "did not get sufficiently instantiated" if it has free parts, "became too instantiated" if the declared inst has, and otherwise "had the wrong inst", and it is reported at the last goal run, or at the clause for a fact. Only the first mode error of a procedure is reported, as by mmc, though its other clauses are still checked for the modes their calls infer. A function without mode declarations gets the default mode the manual defines: all arguments `in`, result `out`.
- **Branches.** Each branch of a disjunction starts from the instmap before it, as does the else part of an if-then-else, and its then part from the condition's; the branches' insts of the goal's nonlocals are merged after it (Overton, 3.1.4). `fail` is not reached after, and a branch not reached merges as nothing. A variable whose insts do not merge, because it is free in one branch and bound in another, or part of it is, is a "mode mismatch", reported at the goal run last, with a note at each branch, at the branch's first goal, and every other such variable in notes after them, as mmc reports each; with three or more branches, counting those not reached, where the variable is ground in all but one and free in that one, mmc says it is "ground in 2 out of 3 branches" and notes only that one. A variable that is free in some branch and not used after is not merged, as in mmc, but one with a bound inst that does not merge is still an error, even when not used after.
- **Negation and conditions.** While a negation, or the condition of an if-then-else, is checked, the nonlocals of the negation or of the if-then-else are locked: a unification or call that binds one that is free is a "scope error", at that goal, which waits, as a mode error does, for the variable to be bound outside. Quantification makes a variable of the condition and the else part two variables, so the condition may bind what it shares only with the then part. A goal may bind the free parts of a locked variable that is partially instantiated, as in mmc. A negation binds nothing after it. Open terms will be allowed to bind inside a promise scope, as the manual's rule for solver types requires.
- **Clauses** of a predicate with several are the branches of a disjunction, as in mmc. Each is checked with new head variables, `HeadVar__N`, and head unifications `HeadVar__N = X` at the arguments, before the body; so an argument's variable is free at first, and used only until its head unification runs. A head unification that waits is reported only if no other goal waits, as mmc does, and so is the unification of a single clause's head argument that is not a variable, as `p(A, A)`, which desugaring puts at the start of the body. The head's insts at the end of the clauses are merged, with a mismatch noted at each clause's `:-`, and then compared with the final insts, at the last goal run in the last clause. A `true` that is a branch counts as a goal run, at its position, but one in a conjunction or as a clause's body does not.
- **Constrained polymorphic modes** (reference manual, "Constrained polymorphic modes"; Overton, 4.2). The operations on constrained insts are the thesis's: an inst matches one with inst variables if it is known to be compatible with each of them; a unification keeps the inst variables of each side whose inst the result is a sub-inst of, one that matches it with the same uniqueness, so `X = Y` of `(I =< ground)` and `(J =< ground)` is `(I, J =< ground)`, and `X = a` narrows `(I =< ground)` to `(I =< bound(a))`; and a merge keeps those of every branch. A clause starts from the head's insts with the inst parameters in them, and a final inst with an inst variable needs a value that has it; mmc says one without it "became too instantiated". At a call, the callee's inst parameters are bound by the arguments' insts at their occurrences in the initial insts, to the greatest lower bound of the argument's inst and the constraint, if it matches the constraint; as in mmc, an earlier argument's binding is kept, and only occurrences in one argument are merged, so `app([A], [B], L)` of `bound(a)` and `bound(b)` is a mode error, though the thesis merges them all. An argument whose inst there is an inst variable narrowed by a unification binds nothing, as in mmc (see the register). The modes, with the bindings, are then checked and passed as without inst variables; a callee's inst parameter left unbound stays in the outputs. A lambda expression's inst variables are its own. A higher-order term of a name has the procedure's own modes, since mmc makes it a lambda expression that calls it: its remaining arguments are passed from their initial insts, with what the curried ones bound, and must get their final insts, so `P = id(A)` is a mode error. A higher-order call binds the term's inst variables as a call does, but, as in mmc, those numbered as the procedure's own are its own (see the register). As in mmc, two modes that differ only in inst variables are the same mode.
- **Higher-order code** (reference manual, "Higher-order"; Overton, 3.2.5). A higher-order inst carries the modes and determinism of the term, and a higher-order call is checked against them as a call is against a procedure's, implied modes included; the closure must have a higher-order inst of its kind and arity, or for a function no higher-order inst, which means the default `func(in, ..., in) = out is det`. So a function's higher-order inst matches `ground` only if it is the default, and an error that expects `ground` there prints the default, as mmc's does. One higher-order inst matches another of the same kind, arity and determinism if its modes accept the other's, as in mmc (the manual does not say): for each argument the expected initial inst matches the closure's, and the closure's final inst matches the expected one. So `pred(di(uniq_array), out(uniq_array))`, a named inst that expands to `unique`, is passed for `pred(di, uo)`, and so is `pred(ground >> clobbered, uo)`, which asks for less; `pred(in, out)`, which gives less, is not, nor is `pred(di, uo)` for `pred(in, out)` (`test/Semantics/modes/closure-variance.m`). Unifying two higher-order terms, or a lambda expression or higher-order term with a bound variable, is an error ("attempt at higher-order unification").
  - A lambda expression waits until its nonlocals are ground, as mmc requires. Its body is then checked as a clause is, from its arguments' modes and the nonlocals' insts made shared, since the closure may be called more than once, with the nonlocals not live, as in mmc, so that clobbering one in it is an error of its inst ("T0 has inst ground, but unique is expected"), and its arguments' final insts must match; an error is reported at the goal in it, or at the lambda expression for a final inst, naming the argument `LambdaHeadVar__N`. Its term gets the higher-order inst of its modes, unique, and the nonlocals are shared. A function's lambda expression without modes has the default ones.
  - A predicate's or function's name, with its first arguments, makes a higher-order term of one of its procedures. With several procedures, the one is chosen whose initial insts the arguments of higher-order types match, as the manual says; mmc does not look at the others, so `P = m(1)` of a predicate `m(in, out)`, `m(out, in)` is an error, which says that the modes match more than one. The arguments are passed as a call's are, implied modes included, and must then be ground, like a lambda expression's nonlocals; the term gets the higher-order inst of the remaining modes.

### Determinism analysis

Determinism analysis (`check-determinism.cpp`) works on each procedure's body in the form mmc gives it after mode analysis, which `-fdebug-dump-determinism` prints. The procedure has one set of variables; a procedure of several clauses is a disjunction of them over head variables `HeadVar__N`, with an argument's variable the head variable itself when desugaring made it for a pattern; conjunctions are flattened and in the order mode analysis ran them; and a unification of a bound `X` with `f(Y1, ..., Yn)` deconstructs `X` into new variables, each followed by a test against its `Yi` if that was bound, as mmc writes `U = u(V, A), Z = V`. As mmc's mode analysis does, the form leaves out a construction or assignment of a variable not used after it, replaces a unification that cannot succeed by `fail` and leaves out the goals written after it (the head unifications, which come first, are written before the body), leaves out a branch of a disjunction, or a clause, that is `fail`, makes a disjunction in a disjunction its branches, and a disjunction of one branch that branch. So far the pass detects switches, eliminates common subexpressions, infers the determinism of goals, checks declared determinisms, checks committed choice, determinism casts and determinism requirements, and infers the determinisms of procedures declared without one. It runs only if mode analysis found no errors, as mmc stops after them.

- **Switches** (reference manual, "Switches"; Henderson, Somogyi and Conway, "Determinism analysis in the Mercury compiler", 1996, section 3). The candidates of a disjunction are its nonlocals that are bound before it, in variable order. A disjunct tests a candidate if, after only unifications other than calls, a deconstruction tests it or a variable unified with it, either way round; a `some` scope is looked into, and a nested disjunction whose every branch, followed by the goals after it, tests the candidate is split into one disjunct for each, with those goals copied; mmc does not split a disjunction inside such a branch. A candidate that every disjunct tests, against at least two functors, makes a switch; if none does, one that some disjuncts test against at least two functors makes a partial switch, the others following it as a disjunction, in which switches are detected again. The disjuncts that test a functor are its case, a disjunction if there are several, those that test it directly first, in order, then the parts of split ones, the last split first; cases are in order of functor name and arity, integers by value. A disjunct that starts with a disjunction whose branches each only deconstruct the candidate itself, with arguments not used after, is in mmc the goals after it in the case of each of those functors, without the tests, and the cases that have only it are one case for all of their functors (`case b/0 or c/1`); one with goals before it, a `some` scope, an alias or any other goal in a branch is split as above. A disjunction is split on the candidate and the variables unified with it outside `some` scopes; a test after a scope may be of a variable unified with the candidate in it. A switch can fail if the variable's inst before it allows a functor of its type that has no case, as `bound(a ; b)` does not; one on a type without constructors, such as `int`, can always fail. Inside a case, a test of its variable against a constant is `true` in mmc, so it does not count for another switch, nor is a disjunction after it split, though a test after it counts (`determinism/case-splits.m`), and a switch on the same variable has only that case, which cannot fail, or is `fail`; a test of a variable unified with it is kept.
- **Choosing a candidate**, as mmc does, which the manual leaves open. Of full switches, the first that cannot fail, else the first on a type without constructors, else the first; of partial ones, the first that cannot fail, else the one with the most functors, else the first. If every disjunct tests some candidate against the same functor, there is no switch but a full one that cannot fail or is on a type without constructors, and if there is none the disjuncts are left as they are, for common subexpression elimination to take the test out (see the register).
- **Common subexpression elimination** (ibid., section 4). In a disjunction, or the cases of a switch, the first candidate that every branch deconstructs with the same functor, after only unifications, through aliases and passing over the tests of a switch's variable against constants (but not those of a variable unified with it, which mmc keeps: in a case of the switch on `X`, `Z = X, Z = 0, ...` and `V = X, V = 0, ...` both test `X` against 0, which moves out), is deconstructed before it instead, into new variables, which each branch assigns to its own; a variable whose inst has free parts in the functor's arguments is left alone. If anything moved, the body is simplified as mmc's mode analysis does when it runs again: knowing the functor of each switch's variable in its cases, and of variables deconstructed and assigned, it makes a deconstruction against another functor `fail`, with the rest of its conjunction, and deletes the goals of a case that cannot be reached; it deletes constructions and assignments of variables not used after them, where a compound goal after one uses the variables it had before its own goals were deleted, and a `fail` replacing a unification uses its variables, and it deletes the tests of a switch's variable against constants, and branches that are `fail`, and a conjunction left in a conjunction, as a disjunction with one branch, is part of it; then switches are detected again in the whole body, until nothing moves. A single-functor type, `u ---> u(t, int)`, is so switched on its arguments.
- **Goal determinism** (reference manual, "Determinism checking and inference"; Henderson, Somogyi and Conway, section 5.1). A determinism is whether a goal can fail and whether it has zero, one or more solutions. A call has its callee's declared or inferred determinism, a higher-order call its term's inst's, and a function application its function's. An assignment and a construction are det, and so is a lambda expression, whose body is inferred with the arguments its modes make inputs bound and checked against its declaration; a test of two variables is semidet, even of a type with one value. A deconstruction's argument bound before it is a test of the argument, unless the variable's argument is free, which the unification binds, as when mode analysis built the term first from free variables: `P = p(T3, mk(2)), P = p(T4, _)` with `T3` still free is det, as in mmc. A deconstruction is det if its variable's inst allows no other functor (an inst `bound(...)` of an `int` counts), if it is the test of a switch's case, or if a deconstruction before it, or a switch around it, gives that functor, passed on by assignments; that last only when common subexpression elimination changed the body, after which mmc checks its modes again, and so knows more than the first mode analysis did: without that, `Z = S` before `S`'s case test does not tell mmc `Z`'s functor. A construction gives none. A conjunction can fail if a conjunct can before one with no solutions; a disjunction can fail if all its disjuncts can, and has more than one solution if one disjunct has, or two have any; a switch takes the most of its cases and can fail if a case or the switch itself can; an if-then-else whose condition cannot fail is its condition and then part, else as both parts; a negation is det if its goal is failure, failure if det or multi, erroneous if erroneous, else semidet. A call of an erroneous or failure procedure makes the rest unreachable for mode analysis, as in mmc, so `( X > 0, Y = X ; error("no") )` binds `Y` in every branch that can succeed; mode analysis then leaves out the goals after it, and the then part of an if-then-else whose condition it is, so their errors are not reported, though their variables count as used until mmc quantifies the body again, as for a unification replaced with `fail`. An if-then-else whose condition has no solutions is its else part, as in mmc, where the manual counts the then part too (register). The negation of `true`, or of a conjunction of `true`s, is `fail` for mode analysis too, as in mmc, so the goals after it are not reached; a negation whose goal fails only by inference is not. A goal that fails only by inference cuts nothing: the goals after it are checked, and still give the reasons that a goal can fail, but not that it can succeed more than once, unless no solution is allowed.
- **After mode errors.** As in mmc, determinism analysis runs on the procedures without mode errors when other procedures have them; only errors before mode analysis, such as type errors, stop it (`CheckDeterminism`'s `analyse`). A procedure without a declared determinism that has a mode error is `erroneous` for its callers, since it stays at the fixpoint's start, as in mmc.
- **No outputs.** A goal with more than one solution, but no variable that it binds and that is used outside it, has at most one, unless it is impure (reference manual, "Interfacing nondeterministic code with the real world"). Used outside goes by quantification after switch detection: the head, the other conjuncts and the parent's, the condition and then part of an if-then-else for each other, but not the other branches of a disjunction or switch, so a goal that switch detection copied into several cases is counted once in each. The innermost such goal is pruned, as mmc does for disjunctions. mmc infers a conjunct other than a disjunction that has no outputs, such as a switch, in a context where only its first solution is wanted, and so marks it and the disjunctions inside it `cc_multi` or `cc_nondet` instead of pruning them; the procedure's determinism, and its errors, are the same.
- **Committed choice.** A committed choice call is allowed only where just its first solution is wanted (reference manual, "Committed choice nondeterminism"). That is the body of a `cc_multi` or `cc_nondet` procedure or lambda expression, and any goal with no outputs (as above), including a negated goal. Where a conjunction is in such a context, so is its last conjunct, and so is each conjunct before conjuncts that cannot fail. Where an if-then-else is, so are its then and else parts, and so is its condition if the then part cannot fail. Before a goal that mode analysis finds `erroneous`, by declared determinisms, only the first solution is wanted, as nothing after it asks for another; the manual says only that a goal after a committed choice call may not fail, and mmc does not count a goal `erroneous` only by inference. The parts of disjunctions, switches and scopes inherit the context. A committed choice call, higher-order call or function application anywhere else is an error at the call, as in mmc ("call to predicate \`p'/1 with determinism \`cc_multi' occurs in a context which requires all solutions"). Its notes are the goals after it that can fail, in the conjunctions around it and in an if-then-else's then part, found as for a declaration's notes; of one after it that cannot fail, those in its disjuncts, a switch's cases, an if-then-else's then and else parts (not its condition) and a scope's goal, as mmc records them. A goal with no outputs, where only the first solution is wanted, keeps the goals after it as notes, for a call in it that a goal in it makes need all its solutions, and so does a determinism cast's goal, but not a negated goal (`determinism/no-outputs.m`). The call then counts as having more than one solution. Where only the first solution is wanted, a goal with more than one is committed choice, as mmc marks it: a call if its callee is, and a compound goal if each of its parts with more than one solution is in such a context too. A condition outside such a context, whose then part can fail, makes its if-then-else `nondet` inside a `cc_nondet` procedure, for example. The procedure's own determinism is the committed choice one if its body has more than one solution. A parallel conjunct must be `det` or `cc_multi` (ibid., "Goals"), but mode analysis does not check parallel conjunctions yet, so neither does this pass.
- **Determinism casts** (reference manual, "Goals"). `promise_equivalent_solutions Vars Goal` and `arbitrary Vars Goal` succeed at most once where their goal can succeed more than once, so `multi` and `cc_multi` become `det`, and `nondet` and `cc_nondet` become `semidet`; only the first solution of their goal is wanted, so it may make committed choice calls. `promise_equivalent_solution_sets Vars Goal` is as its goal; the `arbitrary` scopes in it are the casts. Each lists the variables its goal binds: those whose insts have free parts before it and that are used outside it, as mmc's quantification finds them after mode analysis, none if mode analysis finds that the goal's end cannot be reached (after a call or a unification that cannot succeed, in every branch). Mode analysis deletes a construction or an assignment of a variable used nowhere after it, so a variable used only there is not used outside (`R1 = M` in `( arbitrary [M] m(M), R1 = M ; R1 = 3 ), R = 0`). Which ones it deletes goes by its liveness, from before the deletions, as in mmc: an assignment of a variable that only a deleted one uses is kept, so in `( promise_equivalent_solutions [C] ( 3 > 1, C = A ), D = C ; D = A ), promise_equivalent_solutions [E] ( E = D )`, with `E` not used after, only `E` is extra (`determinism/before-deletion.m`). It turns a unification that cannot succeed into `fail`, but quantification still finds its variables, for the cast and for whether a goal has outputs, until mmc quantifies the body again: after it deletes an assignment (not a construction), or checks the modes again after common subexpression elimination. As in mmc, one that binds variables it does not list is an error ("the \`promise_equivalent_solutions' goal binds a variable that is not listed: Y"), and so is one that lists others, bound before, local to the goal, or not in it ("lists an extra variable: X"), each with the variables in their order. An `arbitrary` scope outside a `promise_equivalent_solution_sets` scope is an error, where a lambda expression starts afresh and `promise_equivalent_solutions` does not count, as is one that lists a variable of the scope it is in, and a `promise_equivalent_solution_sets` scope in another, with a note at the outer scope; the casts apply all the same. The errors do not keep the pass from checking the procedure's declared determinism, as in mmc. Where a cast's goal can succeed more than once, the reasons given inside it for a declaration not satisfied are only the goals that make it fail, as in mmc; where it cannot, they are as elsewhere. Switch detection and common subexpression elimination look into scopes of every kind, not only `some`, as mmc does, and stop at a call in one. Where no solution is allowed, a negation that can succeed is a reason too ("the negated goal can fail", in mmc's words). A scope that switch detection copies is checked once. `generate-switch-programs.py --casts` adds casts to the generated modules (`random-casts-*.test`, `casts.m`). Of 2,400 generated modules (1,600 with `--committed --casts`, 800 with `--casts`), every procedure's determinism and every error line agree with mmc; 58 differ only in the labels of their goals, as do 51 of the 2,400 modules without casts. (Two procedures whose determinism differed, c27_11 p1 and c3_19 p2, came from a test's bound argument and a split after a case's constant test, now fixed.) (A committed choice call in an if-then-else condition that mmc reported and merkc did not, c28_21, came from deleting assignments transitively, now fixed.) Not handled: mmc's common subexpression elimination of a case's own functor tests in a disjunction inside the case, which merkc splits instead, and so does not quantify the body again.
- **Determinism requirements** (reference manual, "Goals"). The goal of `require_<det> Goal` must have that determinism exactly, a tighter one being an error too ("the goal in this scope is required to be det, but its determinism is semidet"), with the reasons of a declaration that is not satisfied. A goal with more than one solution is committed choice where only its first solution is wanted, as in mmc, so `require_cc_multi m(Y)` in a `cc_multi` procedure is met though `m` is `multi`. The goal of `require_complete_switch [X] Goal` must be a switch on `X`, not on a variable unified with it, nor a conjunction or a single disjunct ("the goal inside the require_complete_switch [X] scope is not a switch on X"), with a case for each functor `X` can be bound to there, by its inst ("the switch on X is required to be complete, but it does not cover c/1"; for a type without constructors, "but it is not"). The arms of the switch in `require_switch_arms_<det> [X] Goal` must make the promises `<det>` makes, as a declaration must; one with more than one solution is acceptable in a committed choice context only if only its first solution is wanted. Each arm that is not is an error at the arm, with its reasons; the switch need not be complete. As in mmc, a switch on another variable has its arms checked so too (register). A scope that switch detection copies is checked once. `generate-switch-programs.py --requires` adds the scopes to the generated modules (`random-requires-*.test`, `requires.m`). Of 1,600 generated modules (800 with `--requires`, 800 with `--committed --casts --requires`), every procedure's determinism and every error line agree with mmc but for one, r14_26, the gap noted under determinism casts: mmc's common subexpression elimination in a case, which merkc splits, so that merkc does not quantify the body again; 42 differ only in the labels of their goals.
- **Determinism inference** (reference manual, "Determinism checking and inference"). The determinisms of the procedures declared without one, all local to the module, are the least fixpoint of the rules above, starting from `erroneous`, as in mmc, where the manual says `det` (register), so that `loop(X) :- loop(X)` is `erroneous`. Each step infers every such procedure's body with the others' determinisms so far, joined with its own last one in the lattice, so that the steps end; then every body is checked once more, with its messages. Such a body is in a context where all its solutions are wanted, so a committed choice call in it is an error and makes it `multi` or `nondet`, as in mmc. A function's default mode is declared `det`. An exported procedure must have a declaration, an error of mode analysis, and so must the procedure of a higher-order term, as mmc makes the term a lambda expression with its callee's declared determinism before inference ("the predicate \`p' has no declared determinism, so a curried call to it may not be used as a lambda expression"; it reports this before mode analysis, and so without the mode errors elsewhere in the module). A procedure whose modes were not checked has no determinism, and neither has one that calls it without its being declared (`DeterminismUnchecked`). `generate-switch-programs.py --inferred` declares some generated predicates without a determinism, and adds predicates so declared whose bodies call the others and each other (`random-inferred-*.test`, `inference.m`, and `unreachable.m` for what the sweeps found). Of 3,200 generated modules (1,600 with `--inferred`, 1,600 with `--committed --casts --requires --inferred`), every procedure's determinism and every error line agree with mmc; 178 differ only in the labels of their goals, most of them the labelling of goals with no outputs above, or if-then-elses whose then part has no solutions, which mmc labels by the manual's rule though it infers the procedure's determinism by its own. The rules the sweeps brought in left the 6,400 earlier modules as they were, but for d50_30, which now agrees.
- **Checking a declaration.** An inferred determinism must be within the declared one in the lattice; if it is tighter, there is a warning that the declaration could be tighter (not for a function's default mode). An error is at the declaration, with a note at each goal that makes the difference (ibid., section 5.3): a call that can fail, succeed more than once, or both, or, with no solutions declared, succeed at all; a unification that can fail; a switch that does not cover functors of its variable's type, or can fail, for `int`; a disjunction with more than one disjunct with solutions, and the goals of each disjunct that can fail if the disjunction cannot; each conjunct; the condition of an if-then-else for its solutions and both parts unless the condition cannot fail; a negation that can succeed; goals in a case of a switch, prefixed with the case. A procedure that cannot succeed but is declared to has no notes, as in mmc. A disjunction made of a switch's case is where its first disjunct in the source is. A mode of a function whose arguments are all `in` and whose result is `out`, the primary mode, is an error if its inferred determinism has more than one solution, as in mmc, which goes by the inferred determinism and not the declared one; the declaration is then checked as any other (see the register).
- **Testing.** `utils/type-oracle/compare-switches.py` compares the structure of each declared procedure with mmc's after common subexpression elimination (`--dump-hlds 45`): its switches, with variable, whether they can fail and cases, disjunctions, if-then-elses and negations; `generate-switch-programs.py` makes random modules for it (`test/Semantics/determinism/random-switches-*.test`). Of the probes and about 1900 generated procedures (seeds 1 to 16), four differ, all from seeds 13 and 15, and all in a case of a switch where its variable is tested again against conflicting functors (`S = c(K1)` then `S = c(K2)` or `S = b` then `S = c(K)`): there mmc's order of deleting the goals that cannot be reached and detecting switches again gives a different nesting from merkc's. With `--determinism` it compares mmc's dump after determinism analysis (`--dump-hlds 50 --dump-hlds-options d`): each procedure's inferred determinism, each switch's, disjunction's, if-then-else's and negation's, committed choice ones included, and the lines of mmc's errors, which must be those of merkc's errors and notes; `generate-switch-programs.py --determinism` declares the predicates and lambda expressions with random determinisms and calls predicates of each determinism, some without outputs (`random-determinism-*.test`, `goals.m`, `goal-errors.m`). Of 800 generated modules (seeds 1 to 20) and the probes, every procedure's determinism and every error line agrees; 18 procedures differ only in the determinism of goals inside them, or in nesting: 6 are nesting differences like those above, 11 are disjunctions that mmc marks `cc_multi` or `cc_nondet` inside a goal with no outputs (see "No outputs"), and one is both. `generate-switch-programs.py --determinism --committed` also declares and calls committed choice predicates (`random-committed-*.test`, `committed.m`). Of 1600 such modules (seeds 1 to 40) and 800 without `--committed` (seeds 21 to 40), every procedure's determinism and every error line agrees; 55 procedures differ only in the labels of goals inside them or in nesting, as above. Two differences in determinism were in switch detection, and are fixed: tests through aliases that common subexpression elimination did not move out, and a disjunction left with one branch that was not part of the conjunction around it, so that switches in it were not found (`alias_common` and `one_left` in `switches.m`).

### Uniqueness checking

Uniqueness checking (`check-uniqueness.cpp`) is the second phase of unique mode analysis, which needs determinism analysis's bodies and goal determinisms (Overton, 3.2.3; reference manual, "Unique modes": "a value is not considered `unique` if it might be needed on backtracking"). It walks each body whose determinism was inferred, in execution order, with the variables bound so far and those that are `mostly_unique`, and checks each call's arguments, from their insts before it with the marks applied, and the head's final insts. As in mmc, only the first error of a procedure is reported, and a first-phase error in it comes first and suppresses its determinism checking. The rules, found by probing mmc, are mmc's but for two of the if-then-else's, where mmc misses values that are needed on backtracking; `-fmmc-uniqueness` has mmc's for those too, and the oracle uses it:

- **After a goal that can succeed more than once,** the variables bound before it and used after it are `mostly_unique`, since backtracking into it runs what follows again, and so are the outputs of a call that can, since the callee may need them for its next solution. A committed choice call is never backtracked into, nor, where only the first solution is wanted (a `cc_multi` or `cc_nondet` procedure or lambda expression, a negated goal), a goal with no goal after it there that can fail.
- **Commits.** A `promise_equivalent_solutions` or `arbitrary` scope, and a goal with no outputs, whatever its determinism (mmc infers such a goal where only its first solution is wanted), commit to their first solution: inside them, a goal is backtracked into only if a goal after it there can fail, and what follows them is not run again, so does not count as used. A `promise_equivalent_solution_sets` scope does not commit; the `arbitrary` scopes inside it do. A `require_<det>` scope changes nothing.
- **A disjunction** that can succeed more than once makes the variables bound before it that it uses `mostly_unique`, in every disjunct, the last one too. One where only the first solution is wanted, or that has at most one because its outputs are not used, runs a later disjunct only after an earlier one fails, so only a disjunct that can fail makes the values it uses `mostly_unique`, and only in it; the merge passes them on. Disjuncts whose end mode analysis does not reach are left out, and a disjunction with one left is a conjunction.
- **An if-then-else's condition** makes a variable bound before it that the else part uses too `mostly_unique` while it runs, if it clobbers it, or what an assignment, construction or deconstruction in it makes from it, and so what it makes from it is `mostly_unique` too. Once the condition has succeeded the else part is not run, so the condition is checked again without them for what the then part starts from. mmc looks only for a clobber of the variable itself, does not pass the mark on to what is made from it, and keeps it in the then part (see the register); with `-fmmc-uniqueness`, merkc does the same. A condition whose end mode analysis does not reach makes the if-then-else its else part.
- **After an if-then-else,** what is used is used after its condition, then part and else part, since backtracking into one of them runs it again. mmc counts only what the then part uses as used after the condition, and nothing as used after the then and else parts, so nothing used after an if-then-else is `mostly_unique` because of backtracking into it (see the register); `-fmmc-uniqueness` does the same.
- **A lambda expression's body** is checked where the closure is made, as mmc does, as a procedure of its own: its head variables are bound as their modes have them, those its modes do not clobber are used at its end, a committed choice one wants only its first solution, and its head's final insts are checked, from its modes' (mode analysis has checked that its own match them). A closure made from a name is a lambda expression whose body calls it, so a closure of a `multi` or `nondet` predicate with a `uo` argument is an error at the closure. A higher-order call's arguments are checked against its term's modes, as a call's against its callee's.
- **A negation's goal** has the variables used after the negation `mostly_unique`, and only its first solution is wanted. In the first phase, inside a negation nothing outside counts as live, as for mmc's `if G then fail else true`; the values used after it are this phase's.
- **What is made from a `mostly_unique` value,** by an assignment, a construction or a deconstruction, is `mostly_unique` too, in the parts made from it: a term built from a `mostly_unique` value and a unique one is `unique(p(mostly_unique, unique))`, as in mmc, and a field taken out of a term is `mostly_unique` if its part is. So a unique term's fields are as any value: a field taken out of a term before a goal that can succeed more than once, or of one bound before it after it, is `mostly_unique` after it, a term rebuilt after it from fields taken before it is `mostly_unique` with them, and the if-then-else rules find a field that the condition clobbers and the else part needs through its term, which mmc's rules (`-fmmc-uniqueness`) miss (`nested.m`). A head variable is used at the end unless its mode clobbers it.
- **I/O state arguments.** A procedure with exactly two arguments of type `io.state`, one bound at the start and one free, whatever their uniqueness (`di` and `uo`, `mdi` and `muo`, or `in` and `out`), must be `det`, `cc_multi` or `erroneous`, by its declared determinism, or the inferred one if it has none: "invalid determinism for p(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi", at the declaration, as in mmc ("invalid determinism for a predicate with I/O state arguments"). One with a single such argument, two inputs, two outputs or two pairs is not checked, nor a lambda expression, nor a procedure with mode errors; an instance method is. The manual has no such rule (see the register).
- **Instance methods.** As in mmc, a determinism error of an instance method is reported at its definition in the instance, its clause or its `pred(...) is` item, not at the type class's method declaration.
- **Arrays** are unique by default, through the sound array stub (Standard library, "Unique arrays"): `array_di` and `array_ui` take a unique array, so an array shared by an assignment, a construction or an `in` argument of the program's own cannot then be updated; the library's own predicates that read an array keep it unique. With `-fmmc-uniqueness`, the manual's stub makes them all `ground`, as in mmc. On realistic array code (`test/Semantics/uniqueness/arrays.m`: a sieve, a histogram, a bubble sort, folds over an index and over the array, closures, committed choices and field syntax) the sound default rejects what mmc accepts in two ways only: an array read by a predicate of the program's own that takes it `in`, which shares it, and then updated (the fix is to declare the reader `array_ui`); and an array updated after a choice point, which mmc lets every solution see, as `solutions` of `( X = 1 ; X = 2 ), array.set(0, X, A1, A)` gives two arrays whose element is 2. It accepts what mmc rejects where a closure over a unique array is folded by `array.foldl`, or a term of a number and a unique array is built by `di`/`uo`, since the array is then unique. `generate-unique-programs.py --array` makes random modules whose value is an array, which the oracle compares with mmc under `-fmmc-uniqueness`, where only clobbering is checked (`random-array-1.test`, `random-array-2.test`).
- **Messages.** A call's argument is "mode error in argument 2 of the call to pred m.p/3: T has inst mostly_unique, but unique is expected", and a head variable "mode error: argument 2, T, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected", or "had the wrong inst", at the last goal run. As in mmc, a bound inst made `mostly_unique` that has a bound argument, `mostly_unique(t(mostly_unique(3)))`, does not match `ground(mostly_unique)`, though a constant does (see the register).

`utils/type-oracle/generate-unique-programs.py` makes random modules whose predicates thread a unique value through updates that clobber it (`di`, `uo`) or keep it for backtracking (`mdi`, `muo`), constructions, assignments and calls of each other, between goals that can succeed more than once or fail, for `compare-switches.py --determinism`, which compares every error line and determinism with mmc's (`test/Semantics/uniqueness/random-unique-*.test`; `backtracking.m` for the rules, and `determinism/deleted.m` for what the sweeps changed in pass 5). With `--scopes`, its goals may also be lambda expressions of each determinism, closures of helpers with unique arguments, `arbitrary` scopes in `promise_equivalent_solution_sets` scopes, `require_<det>` scopes and `all` goals (`random-scopes-*.test`; `scopes.m` for the rules). With `--io`, the value is the I/O state, and the helpers write or keep it (`random-io-*.test`; `io_state.m` for the rules). With `--ui`, the helper that reads the value takes it `ui`, so that it must still be unique there. With `--nested`, a goal may also put the value in a unique pair, `unique(p(unique, unique))`, with a new value, and take it out again after up to two goals on the pair: helpers that update its first field, keeping it for backtracking or not, or swap its fields, a deconstruction and a rebuild, or goals that can succeed more than once or fail (`random-nested-*.test`; `nested.m` for the rules). No pair is passed to an `in` argument (see the register). `sound.m` has the two if-then-else rules where merkc's default differs from mmc's. Of 430 generated modules, 800 with `--scopes` and 400 with `--io`, every error line and every procedure's determinism agree with mmc; 14 differ only in the labels of their goals, most in procedures with an error of this phase, where mmc's labels are not those its own rules give (`multi` and `failure` in a conjunction labelled `nondet`), or inside goals with no outputs. Of 480 with `--ui`, three differ: one reports the same final-inst error at another line, one has a determinism message mmc gives after a mode error and merkc does not, and in one mmc accepts a constructed value that a `ui` call has read and then a nested condition clobbers, but `-fmmc-uniqueness` rejects it, as mmc does the same with a head variable (merkc's default accepts both, rightly). Of 480 with `--nested` (seeds 401 to 412, a third with `--scopes` and a quarter with `--ui`), every error line and every procedure's determinism agree with mmc, after two fixes the first sweep found: the second phase now marks only the parts of a term made from a `mostly_unique` value, where it had marked the whole term, which reported a final inst error that mmc does not; and a construction wakes the goals that wait for any of its arguments, not only its free ones, which changed the waiting goal reported first.

### Polymorphism analysis

Polymorphism analysis (`polymorphism.cpp`) decides, for each procedure, which `type_info`s and `typeclass_info`s it takes as hidden arguments and which each of its goals passes, and where each comes from. It follows mmc's order and kinds of source, so that its dump (`-fdebug-dump-polymorphism`) can be compared with mmc's HLDS after its polymorphism pass; MLIR generation will build the type_infos from it. It is the representation of the manual's polymorphism by run-time type information, which the manual leaves to the implementation; the calling convention is mmc's (Jeffery's thesis, chapter 3 for `typeclass_info`s and chapter 4 for existential types).

- **Hidden arguments.** A procedure takes one `type_info` for each type variable of its signature that no class constraint has, before its other arguments, a function's result included, in the order of their first appearance in its declaration, so `rev(U, T)` takes U's first, whether its body uses them or not, and then one `typeclass_info` for each constraint, in their order: `mix(T, U, V) <= sh(U)` takes T's, V's and `sh(U)`'s. Equivalence types are expanded first, so `e(T) == pair(T, T)` gives one for T. A predicate whose signature is inferred takes them in the order of the inferred signature's variables, as mmc's. An instance method takes the instance's, for its type variables and its constraints, and then the method's: for `c(list(T)) <= sh(T)` and a method `m(T, U, V) <= sh(V)`, `sh(T)`, U's and `sh(V)`'s.
- **Constrained type variables.** A type variable that a constraint has has its `type_info` in the `typeclass_info` of the first constraint that has it, at the place of the type that has it, numbered from 1 after the type class's superclasses, as mmc numbers them: T's is the second of `sub(T)` for `sub(T) <= sh(T)`. If that type is not the variable itself, as in `sh(list(T))`, T's is that type's argument; mmc passes the type's as T's (see the register).
- **Typeclass_infos at a call,** one for each constraint of the callee, as the type checker proved it (Jeffery's thesis, 3.1.4): the caller's own; a superclass's, taken out of one of the caller's through as many superclasses as it takes, breadth first; a constant for a constraint without the caller's type variables, `sh(list(int))`; or a cell of an instance's, with the `type_info`s of the instance's type variables that its constraints do not have, the `typeclass_info`s of its constraints, proved in turn, those of its type class's superclasses on its types, which the type checker did not need and which merkc proves the same way, and the `type_info`s of its types. A method call is a call of the method's signature, whose type class's constraint is its first: `m(X, Y, Z, N)` passes U's, `c(T)`'s and `sh(V)`'s.
- **At a call,** the callee's type variables are bound by matching its argument types with those of the arguments, and for each, in order, the goal passes the caller's own hidden argument for one of its type variables, a constant for a type without them (`int`, `list(int)`), a cell built from its arguments' type_infos for another (`list(T)`), and `void` for a type variable nothing determines, which type checking reported as unresolved polymorphism; a type with only such variables is a constant, `list(void)`. A function application passes them for its result too, as does a field access or update, and a call in a lambda expression's body passes them from the enclosing procedure's.
- **Unifications.** A test of two values whose type has the procedure's type variables needs that type's type_info, to call its unification procedure; an assignment, construction or deconstruction needs none. mmc's HLDS after polymorphism shows no type_info at such a test, so these are merkc's own and are not compared. `builtin.compare` is a call like any other, and is.
- **Existentially typed procedures** (Jeffery's thesis, chapter 4) give their callers, after the hidden arguments they take, a `type_info` for each existentially quantified type variable that no existential constraint has, in the order of their first appearance, and then a `typeclass_info` for each existential constraint, in theirs: `all [U, W] some [T, V] (mix(T, W, V, U) => (two(T, V), sh(T))) <= sh(W)` takes U's and `sh(W)`'s, and gives `two(T, V)`'s and `sh(T)`'s. Each clause gives them for the types it chose, which are the same in every clause, as a call passes them: a constant, the clause's own hidden argument (`bound(U, U)` gives U's), a cell, or what one of its goals gave. The type checker proves the existential constraints at the goal number after the clause's goals; mode analysis's copy of a clause of a predicate with several, with its `HeadVar__N` unifications, keeps those proofs there, past the unifications. The dump prints what each clause gives as `results:`; mmc gives them once, after the switch of the clauses, so the comparison takes the different ones.
- **What goals give.** A call of an existentially typed predicate or function gets those `type_info`s and `typeclass_info`s, as outputs, for the opaque types the type checker gave its existential variables, and the caller then uses them as it does its own hidden arguments: a type's `type_info` is the one the call gave, or, for a constrained type, taken out of the `typeclass_info` it gave; the caller's constraints on such a type, the type checker's proofs from the call's existential constraints, are the `typeclass_info` it gave or a superclass's taken out of it, and an instance's cell can have it. The dump names such a type `E<n>`, or by the procedure's own existential variable when a clause gives it as that. A construction with an existentially typed constructor, `'new f'(...)`, passes the `type_info`s of the constructor's existentially quantified variables, in their declared order, and the `typeclass_info`s of its constraints, as a call passes them, but none for its type's parameters; a deconstruction gets them, as a call does, in a switch's case as elsewhere. mmc deletes a construction whose result nothing uses, and merkc leaves it out too. When common subexpression elimination moves the same deconstruction out of a switch's arms, the deconstruction it moved has the first arm's types, and each arm assigns its own variables from its fresh ones: such an assignment makes the arm's given types the moved deconstruction's, whose `type_info`s and `typeclass_info`s the arm then uses, as mmc's arms use the moved one's.
- **Closures.** mmc makes a closure of a predicate's or function's name, `P = two(T)`, a lambda expression whose body calls it with the curried arguments and the lambda's own, `P = (pred(V1::in, V2::out) is det :- two(T, V1, V2))`, and its polymorphism pass then makes the type_infos and typeclass_infos in that body, as for any call there. So a closure passes what that call would, its callee's type variables bound by the curried arguments' types and then by the closure type's arguments and result: `P = use` at `pred(list(U), int)` passes `sh(list(U))`'s instance cell. That holds in lambda expressions and instance methods, of methods, and for types that goals gave; a higher-order call passes nothing, since the closure has them. The dump prints such a goal as `closure`. A closure of a field access function needs a declared mode (reference manual, "Field selection"), as in mmc.
- **Not analysed,** and so left out of the dump and the comparison: instance methods with existential types.
- **What is not compared.** Where mmc makes the type_infos, and whether it reuses one made earlier in the procedure (within a conjunction, not across branches), are its own choices; only the hidden arguments and the types each call's type_infos describe are.

`utils/type-oracle/generate-polymorphic-programs.py` makes random modules whose polymorphic predicates and functions call each other and library predicates at types that bind the callees' type variables to the callers' own, to types without them, to types built from both and, now and then, to nothing, and test and compare values of type variables in if-then-elses. With `--classes`, the modules also have type classes, some with superclasses and one now and then with two parameters, instances for `int`, `string`, lists, maybes and pairs, with constraints on their type variables or not, whose methods call the method or a superclass's on their parts, and constraints on the predicates' type variables; bodies then call methods and constrained predicates at types that the caller's constraints or the instances satisfy. With `--closures`, which goes with the others, a call is now and then made through a closure, with some of its arguments curried, or through a lambda expression whose body makes it. With `--existentials`, the modules also have existentially typed predicates and functions, half with an existential constraint, and existentially typed constructors, half with a constraint; bodies call the former and take apart inputs made with the latter, use what they get as their own type variables' values, give it back as their own existentially typed outputs, and construct with `'new'`. `compare-polymorphism.py` compares each procedure's hidden arguments, outputs among them, where its constrained type variables' type_infos are, what its clauses give, and each call's, construction's and deconstruction's type_infos and typeclass_infos, each by what it describes and where it comes from, with those of mmc's HLDS dump after determinism analysis (`--dump-hlds 50 --dump-hlds-options vx`), instance methods by their instance, with the type variables named by their places among the hidden arguments, since inferred signatures and instance methods name them differently; `--infer` adds `--infer-all` and `-finfer-modes` (`test/Semantics/polymorphism/`: `type-infos.m`, `typeclass-infos.m` and `existential-types.m` for the rules, `constrained-types.m` and `existential-clauses.m`, which are not compared, `closures.m`, `random-polymorphism-*.test`, `random-typeclasses-*.test`, `random-existentials-*.test`, `random-closures-*.test`). They agree on 400 modules without type classes, 1,786 procedures, after one fix to mode analysis that the sweep found (the clobbering choice under Calls, above), and on 400 with them, 4,596 procedures, of which 143 calls take a superclass's typeclass_info and 86 build an instance's with others in it. The type class sweep also found that mmc does not warn that an instance method's determinism could be tighter, since its declaration is its type class's; merkc now does not either. With `--existentials`, they agree on 4,617 of 4,618 procedures in 400 modules, with 687 clause results, 1,491 calls that give type_infos or typeclass_infos, 167 deconstructions and 471 constructions with existentially typed constructors; the other passes, for a superclass in an instance's cell, the same `typeclass_info` that mmc does, `c0(T)`, but taken straight out of the caller's `c2(T)`, where mmc takes it out of `c1(T)`, which it took out of `c2(T)` for the instance's premise, a choice of path that is not compared further. That sweep found the renumbering of the existential constraints' proofs under several clauses, and the assignments that common subexpression elimination makes between given types (above). With `--closures`, alone and with each of the others, they agree on all 3,383 procedures of 400 modules, with 842 closures, 221 of them passing typeclass_infos, and 584 lambda expressions.

### Diagnostics

- **Format.** Modelled on Clang's: a severity, a location, highlighted ranges and attached notes, printed with the source line and a caret. Each message has an identifier, so warnings can be enabled, disabled or made errors one by one.
- **Reported against the parse tree.** A mode error inside a DCG rule, for example, points at the rule as written, not at the clause it expands to.
- **Analysis continues after errors.** A procedure with errors is excluded from MLIR generation, but other procedures and other SCCs are still checked, so one run reports as many problems as possible.
- **Warnings** follow the same passes: singleton variables, unused imports, suspected occurs-check failures, unifications that cannot succeed, and misplaced `%$omp` directives.

### Mapping to `merk`

The output of semantic analysis is the annotated AST, one `ProcInfo` per procedure, and the module's interface file; there is no other intermediate form. MLIR generation reads them directly, one procedure at a time, as ClangIR's code generator reads Clang's AST:

- **Values.** Once modes are known, each variable is produced exactly once on each path. So a map from variables to SSA values, updated at each producer and read at each consumer, is all MLIR generation needs, and the output is in SSA form from the start.
- **Locations.** Each operation carries the location of the parse-tree node behind its semantic node, and variable names are recorded as `NameLoc`s for debug information.
- **Types.** `int`, `float`, `char` and the sized integers become builtin MLIR types such as `i64` and `f64`. Other types become `!merk.term<…>`, closures `!merk.closure<…>`, and type infos `!merk.typeinfo`. `io.state` has no values to pass, so its arguments are left out.
- **Done so far:** the procedures, goals and constants listed in Lowering and code generation, "What is lowered so far". A string is `!merk.string`.

| Semantic form, as the `ProcInfo` classifies it | `merk` output |
| --- | --- |
| Procedure | `merk.proc` with a determinism attribute. Type-info and dictionary parameters come first, then the inputs as block arguments; outputs are results |
| Conjunction | The conjuncts' operations, in the `ProcInfo`'s order |
| Construction `X = f(A, B)` | `merk.construct` |
| Deconstruction that cannot fail | `merk.deconstruct` |
| Deconstruction that can fail | A one-arm `merk.switch` whose default fails |
| Assignment `X = Y` | No operation: `X` maps to `Y`'s value |
| Test of two integers, floats, characters or enum constants | An `arith` comparison |
| Test of two other values | `merk.unify_test`, which calls the type's unify procedure |
| Call | `merk.call` to the chosen procedure, with type-info operands first and consumed operands marked |
| Unification with a function application, `X = f(Y1, …)` where `f` resolves to a function | A call of the function's chosen procedure, as for a call; builtin arithmetic such as `int.+` becomes the matching `arith` operation |
| Implied-mode test after a call | `merk.unify_test` of the call's fresh result against the original value |
| Closure construction, including a partial application such as `add_square(A)`; higher-order call | `merk.closure`; `merk.call_closure` |
| Switch | `merk.switch`, one region per arm |
| If-then-else, negation, disjunction | `merk.ite`, `merk.not`, `merk.disj` |
| Commit | `merk.commit` around the committed goal |
| Parallel conjunction | `merk.par_conj` |
| OpenMP directive scope | The `omp` dialect's operations, built inside the `merk.proc`: `omp.target`, and `omp.parallel`, `omp.wsloop` and `omp.loop_nest` for a loop, with the clauses as their operands and attributes. There is no `merk` operation for them (Language extensions, "Loops") |
| `try`, `throw` | `merk.try`, `merk.throw` |
| Quantification, `promise_*` and `require_*` scopes | No operation; they only affect checking |
| `trace` scope | Its goal, if its compile-time condition holds; otherwise nothing |
| Unification of open values | `merk.open_unify` and the other open-term operations |

- **Branch results.** A switch, if-then-else or disjunction produces the variables its arms bind: those the `ProcInfo` records as going from free to bound across it. They become the operation's results, and each arm yields them in the same order, as `scf.if` does. This is why mode analysis keeps instantiation states at branch points.
- **Polymorphism.** Type-info and dictionary operands are materialized where the `ProcInfo` says they come from: a parameter, a field of a deconstructed existential, a superclass of another dictionary, or a constant type-info value.
- **Uniqueness.** Consumed (`di`) operands carry an attribute, so later passes can update in place, or move a message instead of copying it.

## Language extensions

Five extensions are added: processes with typed mailboxes, loops, parallel loop annotations, a sendability rule, and open terms. A program that uses none of them is plain Mercury and pays nothing for them.

### Processes and mailboxes

The API is a library module. This interface, with stub bodies, passes Melbourne Mercury 22.01.9's type, mode and determinism checks, together with a client that uses every part of it: a counter server with a selector over two mailboxes, the two-step priority receive, `request` and `send_request`. The `receive` goal (below) is sugar over it.

```mercury
:- type mailbox(M).                         % accepts messages of type M
:- type timeout ---> after(int) ; infinity. % milliseconds; after(0) polls

:- pred spawn(pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
    mailbox(M)::out, io::di, io::uo) is det.
:- pred new_mailbox(mailbox(M)::out, io::di, io::uo) is det.
:- pred close_mailbox(mailbox(M)::in, io::di, io::uo) is det.
:- pred send(mailbox(M)::in, M::in, io::di, io::uo) is det.

    % Waiting on one or more mailboxes.
:- type selector(R).
:- func new_selector = selector(R).
:- func selecting(mailbox(M), func(M) = R, selector(R)) = selector(R).
:- func selecting_matching(mailbox(M)::in,
    pred(M, R)::in(pred(in, out) is semidet), selector(R)::in)
    = (selector(R)::out) is det.
:- pred receive(selector(R)::in, timeout::in, maybe(R)::out,
    io::di, io::uo) is det.

    % One mailbox.
:- pred receive(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    timeout::in, maybe(R)::out, io::di, io::uo) is det.
:- pred any(M::in, M::out) is semidet.      % the selector that takes anything

    % Monitors.
:- type monitor_ref.
:- type down_reason.
:- type down_info ---> down_info(monitor_ref, down_reason).
:- pred monitor(mailbox(T)::in, mailbox(M)::in, (func(down_info) = M)::in,
    monitor_ref::out, io::di, io::uo) is det.
:- pred demonitor(monitor_ref::in, io::di, io::uo) is det.
:- func reason(down_info) = down_reason.

    % Request and reply.
:- type reply(Rep) ---> reply(Rep) ; timeout ; down(down_reason).
:- pred send_request(mailbox(Req)::in, (func(mailbox(Rep)) = Req)::in,
    mailbox(Rep)::out, io::di, io::uo) is det.
:- pred request(mailbox(Req)::in, (func(mailbox(Rep)) = Req)::in,
    timeout::in, reply(Rep)::out, io::di, io::uo) is det.
```

- **Send** never blocks. It deep-copies the message for the receiver and returns.
- **Receive** blocks until a message matches or the timeout expires, and returns `yes(R)` or `no`. It takes the oldest message for which the `semidet` selector succeeds; the rest stay queued in order. Erlang's `receive {reply, Ref, S} -> …` becomes the selector `(pred(M::in, S::out) is semidet :- M = reply(Ref, S))`. `receive/6`, on one mailbox, is `receive/5` over a one-mailbox selector.
- **Order is Erlang's.** Erlang's `receive` takes the first message in the queue that matches any clause; the order of the clauses only decides which clause handles a message that matches several (Erlang reference manual, Expressions, "Receive"; checked with OTP 24). Typed mailboxes never hold an unknown message, so a mailbox is a plain FIFO, and a selector keeps this rule across all its mailboxes: it takes the matching message that arrived first.
- **Exhaustiveness:** a receive loop declared `det` must handle every message constructor. Removing one case is a compile error: "the switch on Msg does not cover \`stop'/0".
- **No backtracking over messages:** send and receive thread the unique `io` state, so they cannot appear in nondeterministic code.
- **`any` is `det`** in fact, and mmc warns that its `semidet` declaration could be tighter. It is declared `semidet` only to fit the selector's mode; a `det` `receive_any` would avoid the warning.

**Kinds of message.** The kinds of message a process handles are the constructors of one discriminated union, its protocol, as Erlang's tagged tuples are:

```mercury
:- type counter_msg
    --->    increment(int)
    ;       get(mailbox(int))
    ;       stop.
```

A constructor costs a tag in the pointer bits, as Erlang's atom tag costs a word in a tuple. A process cannot add constructors to a type it does not own, so it uses more mailboxes for three kinds of message: replies to its requests, `down` messages from monitors, and messages in another module's protocol. A selector waits on them together and maps each mailbox's type into one result type, as Gleam's `Selector` does:

```mercury
:- type server_event
    --->    message(counter_msg)
    ;       worker_down(down_reason).

    Sel = selecting(Self, (func(M) = message(M)),
        selecting(Downs, (func(D) = worker_down(reason(D))), new_selector)),
    receive(Sel, infinity, MaybeEvent, !IO)
```

The mapping happens in the receiver, so senders know nothing of it. Separate mailboxes also limit what each sender may say: a client that holds only `Self` can send `counter_msg` and nothing else. A selector is a value, so a loop can build it once; the compiler can build constant selectors statically.

**Priority** is Erlang's two-step receive: look for a high-priority message with `after(0)`, then wait for whichever message is oldest.

```mercury
    receive(Downs, any, after(0), MaybeDown, !IO),
    (
        MaybeDown = yes(D),
        MaybeEvent = yes(worker_down(reason(D)))
    ;
        MaybeDown = no,
        receive(Sel, infinity, MaybeEvent, !IO)
    )
```

A selector option for strict priority between its cases can be added later if programs need it. Strict priority can starve the lower cases, which is why it is not the default.

**The `receive` goal.** Building a selector, a result type to map every mailbox into, and a switch on `yes(message(...))` is boilerplate. The `receive` goal writes the same loop directly. The goal receives one message; each arm selects the messages it wants, by mailbox and pattern:

```mercury
loop(Count, Self, Downs, !IO) :-
    receive [timeout(after(5000)), io(!IO)] (
        select(Self, increment(N)) ->
            loop(Count + N, Self, Downs, !IO)
    ;   select(Self, get(ReplyTo)) ->
            send(ReplyTo, Count, !IO),
            loop(Count, Self, Downs, !IO)
    ;   select(Self, stop) ->
            true
    ;   select(Downs, _) ->
            io.write_string("worker died\n", !IO),
            loop(Count, Self, Downs, !IO)
    ;   timeout ->
            io.write_string("idle\n", !IO),
            loop(Count, Self, Downs, !IO)
    ).
```

- **Syntax.** `receive Params Arms`, shaped like `trace` and `try`. The parameters are `timeout(T)`, with a `timeout` from the library, and `io(!IO)`, the state variable that the wait and the arms thread. Each arm is `select(Mailbox, Pattern) -> Goal`, or `timeout -> Goal`, and the arms are joined by `;`. `select` is not an operator, so the arms are ordinary terms; the term reader needs only `receive` as a binary prefix operator, with `trace`'s priority. `receive(...)` written as a call, with no layout before the parenthesis, stays an ordinary call, so the library predicates keep their name.
- **The option.** The operator departs from mmc's operator table, so it exists only with `-freceive-goal`. Without it, a file reads exactly as mmc reads it, and test 5 (Unparser and round-trip tests) stays exact, as `-fopenmp` keeps plain files plain.
- **Conversion.** The converter reads the arms into a `ReceiveGoal` node with one child per arm, instead of an if-then-else chain. An arm that is neither form, or a `timeout` arm without a `timeout` parameter or the other way round, is an error. With neither, the timeout is `infinity`.
- **Meaning.** The goal is the selector code above:
  - Arms for different mailboxes may be written in any order, mixed freely. The compiler groups them by mailbox, keeping the order of each mailbox's arms, which is the only order that matters: a message is in one mailbox, so only that mailbox's arms compete for it. Each mailbox becomes one `selecting_matching` case, whose `semidet` predicate tries that mailbox's patterns in order and returns the arm and the pattern's variables, as a constructor of a hidden result type. A mailbox whose first arm takes anything, as `select(Downs, _)` does, becomes a `selecting` case, which never scans.
  - `receive/5` takes the oldest message, across the mailboxes, that some arm selects. A message that several arms select goes to the first of them, as in Erlang's clauses.
  - A switch on the result runs the arm's goal, with the pattern's variables bound. The goal is `det` if every arm's goal is.
- **Selective arms.** The arms need not cover a mailbox's type. A message that no arm selects stays queued for a later `receive`, as in Erlang. Since that can leave messages piling up, a mailbox whose arms leave a constructor out draws the warning `unselected-message`, unless some other `receive` in the same predicate selects it. `-Wno-unselected-message` turns it off.
- **Priority** is still the two-step receive: a `receive` with `timeout(after(0))` and only the urgent arms, whose `timeout` arm is the ordinary `receive`.

**Request and reply.** `request(Counter, (func(R) = get(R)), after(1000), Reply, !IO)` asks the counter for its value:

1. It monitors the server, delivering any `down` message to a private mailbox. A server that is already dead delivers one at once.
2. It creates a fresh `mailbox(Rep)`, here `mailbox(int)`, builds the request by applying the function to it, here `get(ReplyTo)`, and sends it. These two steps are `send_request`.
3. It receives from the reply mailbox and the `down` mailbox, with the timeout.
4. It removes the monitor and closes both mailboxes, then returns `reply(N)`, `timeout` or `down(Reason)`.

The server answers with an ordinary `send(ReplyTo, N, !IO)`. A fresh mailbox per request has four effects:

- **Each request has its own reply type.** A process can call servers that reply with different types without adding them to its own protocol.
- **No scanning.** The caller waits on a short queue of its own, and its other messages stay queued, untouched, until `request` returns.
- **No stale replies.** A reply after the timeout, or a second reply, goes to a closed mailbox and is dropped, so it can never surprise a later receive.
- **A dead server is noticed at once,** through the monitor, rather than after the timeout.

Erlang's `gen_server:call` works the same way: it creates a monitor that is also an alias, a temporary address for the caller, sends the request with it, and selectively receives the reply that carries it. The alias is deactivated afterwards, so late replies are dropped; Erlang added aliases in OTP 24 for exactly this. A fresh reply mailbox is a typed alias.

While a process waits in `request`, it handles no other messages, so two servers that call each other synchronously wait until one times out, as with `gen_server`. `send_request` avoids this: it returns the reply mailbox, which the caller adds to its own selector, and the caller closes that mailbox with `close_mailbox` when it is done with it (Distribution and wire format, "Remote calls").

A `:- process` declaration removes the boilerplate. It expands like `:- mutable` does today:

```mercury
:- process counter(int) receives counter_msg.

% expands to
:- pred counter(int::in, mailbox(counter_msg)::in, io::di, io::uo) is det.
:- pred spawn_counter(int::in, mailbox(counter_msg)::out, io::di, io::uo) is det.
:- pred spawn_counter_on(node::in, int::in,
    maybe_error(mailbox(counter_msg), spawn_error)::out, io::di, io::uo) is det.
```

The declaration is the process kind's contract: what it is spawned with and what it receives. A module may put it in its interface section and keep the definition of `counter` in its implementation, or in another program altogether (Distribution and wire format, "Remote spawn").

- **Several mailboxes.** `receives` may list more than one message type, each a mailbox of its own, in order. `:- process server(config) receives control_msg, data_msg` gives `server` and `spawn_server` one `mailbox(control_msg)` and one `mailbox(data_msg)` each, after the arguments. Clients can then be given only the mailbox they need: one that holds only the data mailbox cannot send `stop`. A process could create the second mailbox itself and send it out in a message, but every client would then have to receive it before using it, so the declaration says it directly.
- **No `sends` clause.** A process can send only to mailboxes it holds: those it was spawned with, those in messages it receives, and those it creates, which it can hand out only inside messages. Every type it can send is therefore reachable from its argument and message types, through the `mailbox(T)` types inside them, and the compiler computes that set from the declaration. A list of sent types would repeat it, and could not be complete, since the process may send to any mailbox it is given later. Two cases stay outside the set by design: an existential envelope's payload, which travels opaque (Opaque payloads at relays), and a mailbox found through a registry, whose type is in the registry's own declaration.

A module defines process kinds; it is not itself a process. Exporting the mailbox as an abstract type and keeping the message constructors private forces other modules through the module's API predicates, as Erlang's gen\_server does by convention.

Failure handling follows Erlang. An uncaught exception ends the process with that exception as its exit reason. A monitor delivers a `down_info` message to a mailbox the watcher names, through a function into that mailbox's message type, so a process with one protocol needs no second mailbox for it. Links deliver exit messages the same way. Both are required and ship with processes in M4; their semantics are in Runtime system, "Exceptions, links and monitors".

### Loops

Writing a small recursive predicate for every loop is tedious, and a counted loop that OpenMP can take is easier to see written as a loop than recovered from a recursion. Loops are Schimpf's logical loops from ECLiPSe (J. Schimpf, "Logical Loops", ICLP 2002), adapted to Mercury's types, modes and quantification. Like his, a loop means its translation into a recursive predicate, so it needs no semantics of its own and no declarations: the translation's types come from the clause that holds the loop, and its modes and determinism are inferred. Two other designs were considered. Zhou's `foreach` for B-Prolog (N.-F. Zhou, "Declarative Loops and List Comprehensions for Prolog", 2010) nests its iterators and adds accumulators and list comprehensions, but Mercury's state variables already are accumulators, and its iterators skip elements that do not match their pattern, which a typed language does not want. SAC's with-loops (S.-B. Scholz, "With-Loop-Folding in SAC", 1997) build arrays whose elements are independent by construction, which makes parallel loops safe and fusion possible, but they cover only arrays. Instead, as in C and Fortran, a parallel loop is an ordinary loop under an OpenMP directive, which is the programmer's promise (Parallel loop annotations, below): in general a compiler cannot prove such loops safe, which is why OpenMP exists.

```mercury
:- pred sum_list(list(int)::in, int::in, int::out) is det.
sum_list(Xs, !Sum) :-
    ( foreach(X, Xs) do !:Sum = !.Sum + X ).

:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
saxpy(A, X, !Y) :-
    N = array.size(X),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !Y ^ unsafe_elem(I) := A * X ^ unsafe_elem(I) + !.Y ^ unsafe_elem(I)
    ).
```

- **Syntax.** `( Iterators do Body )`, where `Iterators` is a conjunction of iterators and `Body` is a goal. `do` is an infix operator, `xfx` with `;`'s priority, 400 (1100 in 22.01's numbering), so a body that is a disjunction or an if-then-else is parenthesized. `do(...)` written as a call, with no layout before the parenthesis, stays a call. As an operand, `do` is then written `(do)`, as other operators are: mmc rejects `X = mod` too.
- **The option.** mmc has no `do` operator (it reports "Syntax error at token 'do'"), so a program with loops is not a Mercury program, and the operator exists only with `-fdo-loops`. Without it, a file reads exactly as mmc reads it, as with `-freceive-goal`.
- **Parsing** (implemented). The term reader knows `do` only with the option (`ReaderOptions`), and the converter makes a `do` written as an operator into a `LoopGoal`, which holds its `LoopIterator`s, each with its kind and its arguments as written, and its body. The converter checks what needs no types: each iterator is one of the forms below, a loop has at most one `for`, the counter of a `for` or a `count` is a variable, and a `for`'s step is a nonzero `int` literal. A `param` or a `foreacharg` gets an error that says why it is left out, and a loop in a DCG body is an error, since its state would be threaded invisibly. A `%$omp` directive before a loop applies to it, as to any goal. The canonical unparser lays a loop out as above: the iterators on the line of its parenthesis, after a space, and its body on the lines below.
- **Desugaring and quantification** (implemented). A loop becomes a `Loop` goal of the semantic form: its iterators, each with a variable for each argument, the state variables it threads, and its body.
  - *Scopes.* Before a clause is desugared, a walk of its parse tree finds each loop's variables and the regions around it, by the rule of Variables, below; a variable of the loop that occurs in no region around it gets a new variable there, so loops one after another have different variables of the same name (printed `I$10`), and a quantifier around the loops, `some [X]`, makes `X` occur around them, so they share it.
  - *Arguments.* The arguments around the loop, `Lo`, `Hi`, the list, `First` and `Last`, are flattened into goals before it; the patterns of the loop's own arguments, such as `foreach(K - V, Pairs)` or an `Out` of `KS + K`, are unifications at the start of its body. A `fromto` whose `Last` has no variables is marked as ending the loop.
  - *State variables.* Each one in scope that the body updates with `!S` or `!:S`, outside lambda expressions, is threaded: versions before the loop, at the start and the end of an iteration, and after it. One the body only reads is shared. Threading one with no value yet draws mmc's warning for an uninitialized state variable.
  - *Quantification* treats the body as it treats a lambda expression's, with the iterators' and the states' variables as its arguments, so the loop's nonlocal variables are those around it and the variables of the body only are local.
  - *Warnings*, both from desugaring: an iterator's variable that also occurs around the loop, and two `fromto`s with a variable in the same place.
- **Types and purity** (implemented). A loop is typed in its clause, as a lambda expression is, by the constraints its translation would give: a counter and its bounds are `int`s, a `foreach`'s list is a `list` of its element's type, a `fromto`'s four arguments have one type, and a threaded state variable has one type before the loop, in each iteration and after it. The loop's own types are not generalized, but a predicate whose types are inferred may hold a loop and is generalized as usual. The types then choose the overloads, so `fromto(0.0, S0, S1, Sum)` makes `S1 = S0 + Y` float's `+`. A type error names the iterator or the loop, as in "type error in for(I, 1, N): variable N has type float, but a counter and its bounds have type int". `foreach` needs the list module, as list syntax does; without it, "foreach(X, L) needs the list module, which is not imported". A loop is as impure as its body, with no annotation of its own, as an if-then-else is; the calls in its body are annotated as anywhere, and mmc reports the same errors for the translation. Mode checking takes a loop's translation (below); a loop it never sees, in a predicate with a type error, is reported as "not yet implemented: mode checking loops".
  - *Tests.* `Semantics/types/loops.m` has typed loops and `Inputs/loops_translation.m` their translation, with the types merkc gives declared, which mmc accepts as `det` and infers the same signatures for (`loops-translation.test`); `loop-errors.m`, `resolve/loops.m` and `purity/loops.m`, with its translation, have the errors.
- **Translation** (implemented). After type and purity checking, each loop becomes a call of its *loop predicate*, `p__loop_L_C`, named after the predicate and the loop's line and column, which calls its *body predicate*, `p__body_L_C`, in each iteration. Both are local to the module, with the clause's types as their signatures and the loop's purity, and no modes: mode analysis infers their modes, and determinism analysis their determinism, as for any predicate without declarations. Their clauses share the clause's variables and goal numbers, with the new ones after them, so the clause's types are theirs.
  - *The loop predicate* takes each iterator's state, a `for`'s counter and remaining trip count, a `count`'s counter and `Hi`, a `foreach`'s remaining list, a `fromto`'s accumulator and `Last`; then each threaded state variable's version in the iteration and after the loop; then the shared variables the body uses. With a `for` it is `( if Count = 0 then Ends else Steps, Body, Next, Recurse )`, and without one `( Ends ; Steps, Body, Next, Recurse )`. `Ends` unifies each list with `[]`, each `fromto`'s `Last` with its accumulator, each `count`'s `Hi` with its counter minus one, and each state variable's version after the loop with its version in the iteration; `Steps` unifies each list with `[X | T]`. A `fromto` whose `Last` has no variables puts `Last`'s functor in `Ends` and takes `In`'s pattern from the start of the body into `Steps`, so that the disjunction is a switch on the accumulator, as for Schimpf's `fromto(L, [X | Xs], Xs, [])`.
  - *The trip count* of `for(I, Lo, Hi, S)` is computed before the call: 0 if `Hi < Lo`, or `Lo < Hi` for a negative step, and otherwise, with `M = |S|` and `Small` and `Big` the lesser and greater bound, `Big - Small + 1` for `M = 1`, which wraps only for the whole range of `int`, and `Big div M - Small div M`, plus one unless `Big mod M < Small mod M`, for a larger stride, which cannot overflow. `I` advances by the step beside it; its last advance may wrap, but its value is not used.
  - *The body predicate*'s arguments are those of the iterators' own variables, the variables of the patterns taken into `Steps`, the state variables' versions in the iteration, and the shared variables, that the body uses.
  - *Modules.* `int`, whose arithmetic the counters use, is loaded when a loop has a `for` or a `count`, without being visible to the module's names, as `builtin` is. `-fdebug-dump-loops` lists each translated clause and its loops' predicates, with their signatures.
  - *Not yet.* The software error for lists of different lengths in lockstep: the translation cannot tell a list bound before the loop from one it builds, which mode analysis decides, so `L = []` in `Ends` is a test that can fail, and a loop over two bound lists is `semidet` (`loops/determinism.m`). It needs a unification that throws when its test fails, which determinism analysis counts as one that cannot fail. Nor the error that nothing ends a loop: such a loop is `multi`, and reported as a determinism error of the predicate. Errors found in a loop's predicates name them and their variables, not yet the loop's.
  - *Tests.* `Semantics/loops/translate.m` (the translation), `loops/determinism.m`, `types/loops.m` (the inferred modes) and `CodeGen/loops.m`, which runs loops to `max_int`, with a negative step and with strides, and prints what mmc prints for their translation written by hand. mmc's mode inference rejects `saxpy`'s translation written by hand, which merkc accepts (register, "Mode inference of a unique array argument").
  - *Constructors.* Mode analysis needs the signatures of all of a type's constructors to see that a switch is complete, and type checking makes a constructor's signature where the program uses it. So at a `foreach`, type checking makes those of all of `list`'s constructors, `[]` among them, which only the translation uses (`loops/foreach-nil.m`).
- **Parallel do** (implemented for a loop without clauses, step 5). MLIRGen finds the loop's translation under the directive's scope: the goals of the bounds, the trip count's computation and the call of the loop predicate, which `TranslateLoops` describes for each loop (`TranslatedLoop`: its predicates, its step, the trip count's test, whose arguments are the bounds, and the argument positions of each threaded value). It generates `omp.parallel`, the bounds' goals in it, as Flang computes the bounds, and `omp.wsloop` over an `omp.loop_nest` from `Lo` to `Hi`, `inclusive`, by the step, whose body calls the body procedure with the counter, the threaded arrays as given and the shared variables; the trip count and the loop procedure are not used, and after the loop each threaded array is the one given. A negative step is the nest's own, with `Lo` above `Hi`. Lowering needs nothing new: it flattens the goals of every region of an `omp` operation, nested ones too.
  - *Checks,* each an error that leaves the procedure out: the directive is before a loop; the loop has a `for` and no `foreach`, `count` or `fromto` that ends it; each value it carries from one iteration to the next, a `fromto`'s or a state variable's, is an array or named in a `reduction` clause, and not the io state; its body procedure is `det` and pure. They are made in MLIRGen, which has the translation and its determinism at hand, rather than in a pass of semantic analysis (OpenMP directives, "Checking").
  - *Reductions* (implemented, step 7). Each value a `reduction` clause names, by the name of a `fromto`'s `Last` or of a state variable, or the one scalar the loop carries, for a clause that names none, is a slot of the procedure. It holds `First`, or the state variable's value before the loop, stored before `omp.parallel`, whose goals, `First`'s among them, are generated before it too. The slot is the `omp.wsloop`'s reduction variable, by value, as Flang's scalar is: the nest loads the copy, passes it to the body procedure as the value's `In`, and stores the body's result for it, the output that the loop predicate passes to its own call in `In`'s position. After the loop the slot is loaded as `Last`. Each operator and type has a module-level `omp.declare_reduction`, made once, whose names, identities and combiners are Flang's (`ReductionProcessor`): `add_reduction_i64` for `+`, `multiply_reduction_` for `*`, `max_` and `min_`, with the least and greatest value as identities, and `iand_`, `ior_` and `ieor_` for `/\`, `\/` and `xor`. A float sum depends on the order it is added in, and so on the schedule (Language extensions, "The promise").
    - *Checks:* each name is a value the loop carries, named in one clause; a clause without variables needs exactly one scalar; the value is an `int` or a `float`, and the bitwise operators are ints'; a qualified operator is the type's own, as `int.max`. Not yet: other operators, such as a function the programmer promises is associative, which would be a `declare_reduction` that calls it; other types; and modifiers.
    - *Tests.* `CodeGen/omp-parallel-do-reduction.m` checks the operations and runs π, a state variable, an unnamed clause, `min` and `max` beside an array updated in place, `*`, `xor` and `int.max` on four threads and as the sequential program; `omp-parallel-do-errors.m` has the checks.
  - *Not yet.* Clauses other than `reduction` (`collapse`, `schedule` and the rest), which are decomposed onto their leaves but not generated (OpenMP directives, "Decomposition"), `parallel do` in a target region, and removing the loop's predicates, which are generated though not called.
  - *Running.* `-fopenmp` links `libomp` (Driver, "Linking"). `CodeGen/omp-parallel-do.m` checks the operations, and `omp-parallel-do-run.test` runs it on four threads, `saxpy` and loops with a negative step, a stride and no iterations, and checks that it prints what the sequential program prints; `omp-parallel-do-errors.m` has the checks.
- **Target teams distribute parallel do** (implemented for a loop whose clauses are reductions, steps 6a, 6b and 7). The same loop, with the same checks, in a target region that maps what it uses as any target region does (Target regions), so an array the loop updates in place is mapped `tofrom` and one it reads `to`. The region holds `omp.teams` around `omp.parallel`, `omp.distribute` and `omp.wsloop` over one `omp.loop_nest`, as Flang lowers `!$omp target teams distribute parallel do`: `omp.target` and `omp.teams` are combined, and `omp.parallel`, `omp.distribute` and `omp.wsloop`, the parts of the composite `distribute parallel do`, are composite. The kernel is SPMD, as Flang makes it: the host computes the bounds and the step before `omp.target` and gives them to it as `host_eval` values, from which the host computes the trip count and so the teams and threads to launch, and the host's nest runs over them. The device compile has no `host_eval`, as Flang's has none: its nest runs over the bounds computed in the region. So the bounds' goals are generated in the region on both sides, and on the host before it too: the host and the device then map the same values, such as `N` in `for(I, 0, N - 1)`, which only the device's nest reads. The host's copy in the region is not used.
  - *Which kernel.* MLIRGen decides from the leaves, by Flang's patterns (`KernelTypeVisitor` and `HostEvalVisitor`), as ClangIR's `isTargetSPMD` decides from the directive: after `target`, `teams distribute parallel do` and `parallel do` are SPMD, with the bounds on the host; `teams distribute` is generic, with the bounds on the host too, from which the host computes the number of teams; anything else, such as a plain `target`, is generic. Flang also follows a target region whose body is a single directive, as a combined directive; that waits for directives in target regions.
  - *Where the bounds are computed* in the region. In `omp.parallel`, where the region also reads the values it is given, as Flang places them when they are not evaluated on the host. OpenMP allows nothing in the target region but `teams`, and on an AMD GPU a generic kernel traps when the parallel in its `teams` uses a value computed outside the parallel, in `teams` or before it. The OMPIRBuilder passes such a value through `__kmpc_alloc_shared` memory that the teams function allocates and the kernel frees after it returns; why that traps is not known yet, and a reduced case for the OMPIRBuilder is to be made. Flang never makes one, since its values are in memory and loaded where they are used.
  - *Reductions.* The decomposition puts a `reduction` on `teams` and on the `do`, and implies a `map` of its variable on `target`, as Flang has them: the slot is mapped `tofrom` by reference, the target region's argument for it is the `teams`' reduction variable, and the `teams`' copy is the `do`'s. The device combines the threads' and then the teams' partial results through the runtime (`__kmpc_nvptx_parallel_reduce_nowait_v2` and `__kmpc_gpu_xteam_reduce_nowait`), as Flang's does. `CodeGen/omp-target-teams-distribute-parallel-do-reduction.m` checks both sides and runs π, the `normalize` example below and a `min` and `max` of ints on the host's threads and sequentially; its `-gpu.test` runs them on the GPU and checks the maps, `!A` `to` in `normalize`'s first loop and `tofrom` in its second, which reads `Sum` `to`.
  - *Not yet.* Its other clauses (an explicit `map`, `num_teams`, `thread_limit`, `dist_schedule` and the rest), reported for the leaf they apply to, of which `num_teams`, `thread_limit` and `num_threads` would be `host_eval` values too, and the directive in a target region.
  - *Running.* `CodeGen/omp-target-teams-distribute-parallel-do.m` checks the operations and the device's LLVM IR, and runs the program on the host's threads and as the sequential program; `omp-target-teams-distribute-parallel-do-gpu.test` runs it on the GPU, `saxpy` over 100000 elements, more than a team has threads, two arrays in one loop, and loops with a negative step, a stride and no iterations, and checks the maps and that each kernel is launched in SPMD mode, with one team for a short loop; `omp-parallel-do-errors.m` has its checks too.
- **Iterators.** All the iterators of a loop advance together, in lockstep; a nest is a loop in a loop's body.
  - `for(I, Lo, Hi)` and `for(I, Lo, Hi, Step)`: `I` takes the integers from `Lo` to `Hi`, inclusive, `Step` apart. `Step` is a nonzero integer literal, so the direction is known; a negative step counts down to `Hi`. `Lo` and `Hi` are evaluated once, before the loop. A loop has at most one `for`.
  - `foreach(X, List)`: `X` takes the elements of `List` in order. A `List` bound before the loop is traversed; a free one is built, one element per iteration, which mode analysis decides, as it decides any unification.
  - `count(I, Lo, Hi)`: `I` counts the iterations from `Lo`, and the last value is unified with `Hi`, so `Hi` may be free. It never ends the loop.
  - `fromto(First, In, Out, Last)`: Schimpf's most general iterator, an accumulator, `In` in each iteration and `Out` for the next, from `First` to `Last`. When `Last` is a ground term, such as `[]` or `stop`, it ends the loop, as in `fromto(L, [X | Xs], Xs, [])`, which traverses a list; when it has variables, it is the accumulator's final value, as in `fromto(0, S0, S1, Sum)`. Schimpf's loop that ends on a condition, `fromto(continue, _, C, stop)`, is written `fromto(continue, continue, C, stop)`, so that the end and a step are exclusive (Meaning, below).
  - Schimpf's `foreacharg`, over a term's arguments, is left out, since a term's arguments have different types in Mercury, and so is `param` (Variables, below).
- **State variables** are the usual accumulators. A state variable of the clause that the body updates, with `!S` or `!:S`, is threaded through the iterations: the first gets the value of `!.S` before the loop, each later one the value the previous one gave, and `!.S` after the loop is the last one's. A body that only reads `!.S` shares it, as any variable from outside. `fromto` is for an accumulator local to a goal, whose final value is an ordinary variable, as in the inner loop of the matrix multiplication below, and for one that ends the loop.
- **Variables.** A variable of the body is shared by all iterations when it also occurs in the goal around the loop outside every loop's body: in the clause's head, in a goal before or after the loop, or in an outer argument of an iterator, `Lo`, `Hi`, `List`, `First` or `Last`, which belong to the goal around the loop. A shared variable must be bound before the loop; binding it in the body is a mode error that says so. Every other variable of the body is local to each iteration, so loops one after another may use the same names, each for its own. This is Zhou's rule, not Schimpf's, whose variables are local unless named by `param`: his rule keeps a loop's meaning independent of its context for meta-calls, which Mercury has none of, and quantification already finds the shared variables. The iterators' own variables, `I`, `X`, `In` and `Out`, are local to their loop; one that also occurs around it, outside every loop, is a different variable, with a warning, as the head variables of a lambda expression are. A value leaves a loop through an iterator, `foreach`'s built list, `count`'s `Hi` or `fromto`'s `Last`, or through a state variable.
- **Meaning.** A loop is replaced by a call of a new predicate, local to the module and named after the loop's position, as in Schimpf's Figure 1. Its arguments are each iterator's state, the threaded state variables and the shared variables. Its types are inferred with the clause that holds the loop, as a lambda expression's are, and are not generalized: inferred on its own, the translation of `sum_list`'s loop is ambiguous between `int` and `float`, as mmc reports for it. Its modes and determinism are inferred as for a predicate without declarations, so a loop may have several modes. Mercury has no cut, so where his auxiliary predicate's first clause commits to the end, ours makes the end and a step exclusive:
  - With a `for`, it is an if-then-else: `( if I > Hi then Ends else Steps, Body, Recurse )`, with `<` for a negative step. `Ends` unifies each list with `[]`, each `fromto`'s accumulator with `Last` and each `count` with `Hi`; `Steps` unifies each list with `[X | T]`, advances the counters and binds the accumulators. This is the meaning; the generated test counts down a trip count computed before the loop, so that a loop to `max_int` ends ("From a nest to `omp.loop_nest`", below).
  - Without one, it is the disjunction `( Ends ; Steps, Body, Recurse )`, which must be a switch on a list bound before the loop or on a `fromto`'s accumulator, by `Last` against `In`'s pattern, as in `fromto(L, [X | Xs], Xs, [])`. A loop whose disjunction is no switch has nothing that ends it, which is an error. This keeps the number of iterations determined, which is what Schimpf's cut was for, without committing to a choice: the loop has the solutions of its translation.
  - Lists of different lengths in lockstep are a software error that names the loop, as `list.map_corresponding` throws "mismatched list lengths". The other iterators that could end the loop, lists bound before it and `fromto`s with a ground `Last`, are matched by switches whose other arm throws it, both in `Ends`, for one not at its end, and in `Steps`, for one already at it. In ECLiPSe such a loop fails, which would make every loop over two lists `semidet`; here a loop is `det` when its body is.
- **Sequential code** is the translation: the recursive call is a last call, which LLVM makes into a branch.
- **Testing against mmc.** Each loop test has its translation beside it, written by hand as plain Mercury, with declarations; merkc's output for the loop must be mmc's for the translation.
- **Errors** name the loop by its position and the user's variables, never the translation's predicate or its extra arguments. A variable that is unbound because it is local says so: "`Sum` is local to each iteration of the loop at line 12; to give a value out of a loop, use `fromto(First, In, Out, Sum)` or a state variable".
- **Shared iterator variables draw a warning.** Two `fromto`s of one loop that have a variable in the same place, as `fromto(0, S0, S1, Sum), fromto(0, S0, S1, Count)` have `S0` and `S1` when one `fromto` is copied from the other, are one accumulator, so their bodies must agree, and `S1 = S0 + I, S1 = S0 + 1` fails unless `I = 1`. That is legal but almost always a mistake, so the warning names the variable and both iterators. Two accumulators need their own names, as in `fromto(0, S0, S1, Sum), fromto(0, C0, C1, Count)`. Other iterators that share a variable draw none: `foreach(K - V, Pairs), foreach(K, Keys)` builds the list of keys and `foreach(X, Xs), foreach(X, Ys)` copies a list. Nor does a variable in different places, which passes a value from one iterator to the other: in `fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last)`, after Schimpf's idiom for the last element of a nonempty list, `X` is the first `fromto`'s element and the second's `Out`.
- **Open questions:**
  - an iterator over an array's elements, and over a range of a two-dimensional array, with `array2d`'s `A ^ unsafe_elem(I, J)` instead of computing row-major indices by hand;
  - leaving a loop early: a `while(Goal)` iterator, which ends the loop before an iteration in which `Goal` fails, would replace the `fromto(continue, continue, C, stop)` idiom.

Scalars passed from one loop to the next, with no arrays and no directive. The first loop sums 1 to `N`, and the second counts the `I` whose square is below the sum:

```mercury
:- pred below_sum(int::in, int::out) is det.
below_sum(N, Count) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do
        S1 = S0 + I
    ),
    ( for(I, 1, N), fromto(0, C0, C1, Count) do
        ( if I * I < Sum then C1 = C0 + 1 else C1 = C0 )
    ).
```

Each loop has its own `I`, and `S0`, `S1`, `C0` and `C1` are local to each iteration. `Sum` occurs in the first loop's `fromto`, so it belongs to the clause, and the second loop shares it, as an input. The translation is two predicates, the second taking `Sum` as an argument; written by hand with declarations, mmc accepts them as `det`, and `below_sum(4, C)` gives 3:

```mercury
below_sum(N, Count) :-
    loop_1(1, N, 0, Sum),
    loop_2(1, N, Sum, 0, Count).

loop_1(I, Hi, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        S1 = S0 + I,
        loop_1(I + 1, Hi, S1, Sum)
    ).

loop_2(I, Hi, Sum, C0, Count) :-
    ( if I > Hi then
        Count = C0
    else
        ( if I * I < Sum then C1 = C0 + 1 else C1 = C0 ),
        loop_2(I + 1, Hi, Sum, C1, Count)
    ).
```

**Reading `fromto`, and when it is needed.** `fromto(First, In, Out, Last)` is a value carried from each iteration to the next. In `( for(I, 1, N), fromto(0, S0, S1, Sum) do S1 = S0 + I )`, `S0` is the value coming into an iteration and `S1` the value it gives the next: the first iteration's `S0` is `0`, each later one's is the previous `S1`, and after the last iteration `Sum` is the last `S1`, or `0` if there were none. It is C's `sum = 0; for (i = 1; i <= n; i++) sum = sum + i;`, with `S0` and `S1` the values of `sum` before and after the assignment, each a variable of its own since a Mercury variable has one value.

- **No carried value, no `fromto`.** A loop that tests each element, `( foreach(X, Xs) do X > 0 )`, builds a list, `( for(I, 1, N), foreach(Y, Ys) do Y = I * I )`, or updates an array in place, as `saxpy` does, needs only `for` or `foreach`.
- **A state variable instead.** A value that a state variable of the clause holds is threaded without a `fromto`, as in `sum_list`; and so are an array updated in place and `!IO`. A `fromto` is for a value local to a goal, whose result is an ordinary variable such as `Sum`.
- **Ending a loop.** A `fromto` whose `Last` has no variables ends the loop when its value is `Last`: a loop can walk a structure by its own pattern, as here a list by its tail, or end on a condition, as `fromto(continue, continue, C, stop)` does. Two `fromto`s give the last element of a list, or `0` for an empty one; the first walks the list, ending at `[]`, and the second carries the element, `X`, which is its `Out`, so that `Last` is the last iteration's `X`:

  ```mercury
  :- pred last_or_zero(list(int)::in, int::out) is det.
  last_or_zero(L, Last) :-
      ( fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last) do true ).
  ```

  Its translation has no `for`, so it is a disjunction, a switch on the list. Written by hand with declarations, mmc accepts it as `det`, and it gives 3 for `[1, 2, 3]` and 0 for `[]`:

  ```mercury
  last_or_zero(L, Last) :-
      loop(L, 0, Last).

  loop(L0, A0, Last) :-
      (
          L0 = [],
          Last = A0
      ;
          L0 = [X | Xs],
          loop(Xs, X, Last)
      ).
  ```

- **Under a parallel directive** a `fromto` is a scalar carried from one iteration to the next, which the iterations, running at the same time, cannot share, so it must be named in a `reduction` clause, as `Sum` is in the π example below; and one that ends the loop is an error, since the loop has to be counted.

Under a parallel directive, a loop is compiled as a loop, not as its translation:

- **Form.** The loop's one iterator is a `for`, which gives the induction variable, the bounds and the step of an `omp.loop_nest`, with `inclusive` bounds. Its other values are its state variables and `fromto`s with a variable `Last`. Each of them is named in a `reduction` clause, or is an array, `array_di` to `array_uo`, that every iteration updates in place and which is shared, as an array is in C. Lists, `count` and a `fromto` that ends the loop are sequential by nature, so they are errors. With `collapse(N)`, the body of each of the outer `N - 1` loops is exactly the next loop, and the inner bounds do not use the outer induction variables.
- **The promise.** The iterations may run in any order and at the same time, each giving back the array it was given, updated in place. If they write the same element, or one reads what another writes, the result is undefined, as in OpenMP. The compiler checks only what it needs to build the loop: the form above, and the rules for every parallel loop, below. The sequential meaning is checked as without the directive: types, modes, determinism and uniqueness.
- **Reductions.** `reduction((+) : Sum)` names the `fromto` whose `Last` is `Sum`, or the state variable `!Sum` by its name, as a `map` clause does (Target regions, "Array sections"); with one threaded scalar, `reduction(+)` names it. `First`, or the state variable's value before the loop, is the value the partial results are combined with. The promise covers the body too: it combines `In`, or `!.Sum`, with the iteration's own value by the clause's operator. Each operator becomes an `omp.declare_reduction` of its type, as Flang declares them.
- **Lowering.** MLIRGen builds the `omp` dialect's operations straight into the `merk.proc`, as it builds `omp.target` for a target region: `omp.parallel` around `omp.wsloop` over an `omp.loop_nest`, as Flang lowers `!$omp parallel do`. A `merk` operation for parallel loops would only be translated into these one for one, so there is none (the example below follows a nest down). Under `target teams distribute parallel do`, the bounds and the step go to `omp.target` as `host_eval` operands, and the loop nest is wrapped in `omp.teams`, `omp.parallel`, `omp.distribute` and `omp.wsloop`, as Flang wraps a composite construct. This makes the kernel SPMD. A threaded array is mapped `tofrom`, and after the region it is the array given: the iterations updated it in place, so none of the comparisons of a goal that gives an array of its own are needed (Lowering and code generation, "What is lowered so far").

**Examples.** Temporaries in the body: the midpoint rule for π, the usual first OpenMP program.

```mercury
:- pred pi(int::in, float::out) is det.
pi(N, Pi) :-
    Step = 1.0 / float(N),
    %$omp parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y
    ),
    Pi = Step * Sum.
```

- **Temporaries.** `X` and `Y` occur only in the body, so they are local to each iteration, what OpenMP calls `private`, with nothing to declare. A `private` clause naming such a variable is redundant and draws a warning; one naming a shared variable is an error, since a Mercury variable has one value. A variable bound in an if-then-else is bound in both branches, as anywhere in Mercury.
- **Shared.** `Step` occurs before the loop, so it is shared and read by every iteration. Giving a temporary the name of a variable of the clause makes it shared too, and binding it in the body is then the mode error of a shared variable (Variables, above); C avoids this by declaring temporaries in the loop's block.
- **Order.** Mode analysis schedules the body, as any conjunction, so the steps may be written in any order.
- **The translation** keeps `X` and `Y` inside the loop's predicate and passes only `Step` to it. Written by hand with declarations, mmc accepts it as `det`, and with `N = 1000000` it gives 3.1415926535897643.

A matrix multiplication, with row-major N×N matrices:

```mercury
:- pred matmul(int::in, array(float)::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
matmul(N, A, B, !C) :-
    %$omp parallel do collapse(2)
    ( for(I, 0, N - 1) do
        ( for(J, 0, N - 1) do
            ( for(K, 0, N - 1), fromto(0.0, S0, S1, Sum) do
                S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J)
            ),
            !C ^ unsafe_elem(I * N + J) := Sum
        )
    ).
```

- **`unsafe_elem`.** The `array` module's `unsafe_elem` and `'unsafe_elem :='` are `elem` and `'elem :='` without the bounds check, as an array subscript in C or Fortran is. The parallel examples use them, as OpenMP code usually does: a failed check on a device is a trap, not an exception (Lowering and code generation, "What is lowered so far"), and the bounds are part of what the loop's author promises. Field syntax covers both, so they need no syntax of their own.
- **The nest.** `collapse(2)` applies because the body of the `I` loop is exactly the `J` loop, whose bounds do not use `I`.
- **The inner loop** is sequential, inside each `(I, J)` iteration. Its accumulator is a `fromto`, whose `Sum` occurs in the `J` loop's body and is therefore local to each `(I, J)` iteration, so it needs no `reduction` clause.
- **The array.** `!C` is threaded through both parallel loops, updated in place and shared. That each `(I, J)` iteration writes only element `I * N + J` is the directive's promise; the compiler does not check it.
- **Without the directive** this is a sequential triple loop, whose translation is three recursive predicates, one for each loop. Written by hand with declarations, mmc accepts them as `det`.

A Jacobi step on a device, over the inner points of an M×N grid:

```mercury
:- pred jacobi_step(int::in, int::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
jacobi_step(M, N, A, !B) :-
    %$omp target teams distribute parallel do collapse(2)
    %$omp&    map(to: A) map(tofrom: B)
    ( for(I, 1, M - 2) do
        ( for(J, 1, N - 2) do
            !B ^ unsafe_elem(I * N + J) := 0.25 *
                ( A ^ unsafe_elem((I - 1) * N + J) + A ^ unsafe_elem((I + 1) * N + J)
                + A ^ unsafe_elem(I * N + J - 1) + A ^ unsafe_elem(I * N + J + 1) )
        )
    ).
```

- **The map clauses** are what the modes give anyway, `to` for the `in` array `A` and `tofrom` for the threaded `!B`, so they may be left out. `B` names the state variable `!B`.
- **Lowering.** `omp.target` takes the bounds and steps of both loops as `host_eval` operands, and holds `omp.teams`, `omp.parallel`, `omp.distribute` and `omp.wsloop` around one `omp.loop_nest` with two induction variables and `inclusive` bounds.

A reduction, whose result a second loop reads:

```mercury
:- pred normalize(array(float)::array_di, array(float)::array_uo) is det.
normalize(!A) :-
    N = array.size(!.A),
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + !.A ^ unsafe_elem(I)
    ),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := !.A ^ unsafe_elem(I) / Sum
    ).
```

- **`Sum`** is the first loop's `Last`, so it belongs to the clause: the first loop gives it, combining the partial sums with `0.0`, and the second reads it as a shared `in` scalar, mapped `firstprivate`. `I`, `S0` and `S1` are each loop's own.
- **`!A`** is only read by the first loop, so it is shared there and mapped `to`; the second threads it, `tofrom`.

**From a nest to `omp.loop_nest`.** A collapsed nest, followed through each stage; stages 5 and 7 are implemented for one loop without `collapse` (step 5 of the implementation, Translation, "Parallel do", and its reductions, step 7), stage 6 for one loop (steps 6a and 6b, Translation, "Target teams distribute parallel do"), and `collapse` is planned.

```mercury
:- pred add(int::in, array(float)::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
add(N, A, B, !C) :-
    %$omp parallel do collapse(2)
    ( for(I, 0, N - 1) do
        ( for(J, 0, N - 1) do
            !C ^ unsafe_elem(I * N + J) :=
                A ^ unsafe_elem(I * N + J) + B ^ unsafe_elem(I * N + J)
        )
    ).
```

1. **Parse tree.** An `OmpGoal` (`parallel do`, `collapse(2)`) over a `LoopGoal` (`for(I, 0, N - 1)`) whose body is a `LoopGoal` (`for(J, 0, N - 1)`) whose body is a `StateVarFieldUpdate`.
2. **Desugaring and quantification.** Each loop is a `Loop` goal of the clause. An iterator's outer arguments are flattened into goals before its loop: `V1 = N - 1` before the outer loop, and `V2 = N - 1` in the outer loop's body, before the inner one. `!C`'s versions are threaded through both loops, from `C0` before them to `C` after. The inner body is `K = I * N + J, X = unsafe_elem(K, A), Y = unsafe_elem(K, B), Z = X + Y, Cout = 'unsafe_elem :='(K, Cin, Z)`. By the scoping rule, `N`, `A` and `B` are shared by both loops, `I` by the inner one, since it occurs around it, and `K`, `X`, `Y` and `Z` are local to an inner iteration.
3. **Types,** in the clause: `I`, `J`, `N`, `V1`, `V2` and `K` are `int`, the arrays and `C`'s versions `array(float)`, and `X`, `Y` and `Z` `float`.
4. **Translation.** Each loop becomes a `loop` predicate and a `body` predicate, local to the module, with the types fixed and the modes and determinism inferred. `add` calls the outer `loop`, which calls the outer `body` for each `I`. That `body` computes `V2` and calls the inner `loop`, which calls the inner `body`, `add_body_2(I, J, N, A, B, !C)`, for each `J`. These are ordinary predicates, so mode, determinism and uniqueness checking check the loop's sequential meaning, which is also the code generated without the directive.
5. **MLIRGen, with the directive.** `collapse(2)` takes the inner loop into the nest. The outer body holds only the inner loop and the goals of its bounds, and those use only shared variables, so they are computed before the nest. The nest is one `omp.loop_nest` with two induction variables, whose body calls the innermost `body` procedure. The other three predicates are not called and are removed:

   ```mlir
   merk.proc @add(%n: i64, %a: !merk.array<f64>, %b: !merk.array<f64>,
                  %c: !merk.array<f64>) -> !merk.array<f64> {
     %c0 = arith.constant 0 : i64
     %c1 = arith.constant 1 : i64
     %hi1 = arith.subi %n, %c1 : i64          // V1
     %hi2 = arith.subi %n, %c1 : i64          // V2, before the nest
     omp.parallel {
       omp.wsloop {
         omp.loop_nest (%i, %j) : i64 = (%c0, %c0) to (%hi1, %hi2)
             inclusive step (%c1, %c1) collapse(2) {
           %r = merk.call @add_body_2(%i, %j, %n, %a, %b, %c) : ...
           omp.yield
         }
       }
       omp.terminator
     }
     merk.return %c : !merk.array<f64>
   }
   ```

   The induction variables are block arguments, values rather than memory, so `omp.wsloop` needs none of the `private` clauses that Flang gives Fortran's loop variables. The values the region uses from outside, `%n` and the arrays, are captured when `omp.parallel` is outlined. `%r` is not used: by the promise, each iteration gives back the array it was given, so after the nest `C` is `%c`.
6. **On a device,** `target teams distribute parallel do collapse(2)` wraps the same nest as Flang wraps a composite construct: `omp.target`, with `%c0`, `%hi1`, `%hi2` and `%c1` as `host_eval` operands and `omp.map.info` for `%a` and `%b` (`to`) and `%c` (`tofrom`), as in 5b, around `omp.teams`, `omp.parallel`, `omp.distribute` and `omp.wsloop` over the `omp.loop_nest`. The bounds are computed on the host, before `omp.target`, so the kernel is SPMD. `omp-mark-declare-target` finds `add_body_2` through the `merk.call` and compiles it for the device.
7. **Lowering and translation.** Lowering makes the `merk` operations, including those in the `omp` regions, into the LLVM dialect, as it does in a target region, and the `omp` operations stay. MLIR's OpenMP translation then uses the OMPIRBuilder: it outlines `omp.parallel` into a function that `__kmpc_fork_call` runs, builds each loop with `createCanonicalLoop`, signed and inclusive, and joins the two with `collapseLoops` into one loop over `(hi1 + 1) * (hi2 + 1)` iterations, from whose counter it recovers `%i` and `%j`; `omp.wsloop` shares that loop out with the static schedule. LLVM then inlines `add_body_2`.

**Found by following it down,** to be handled in the steps named:

- **Hoisting inner bounds** (with `collapse`, not yet). Computing the inner loop's bounds before the nest is what OpenMP requires of a rectangular nest, and it is legal here because the bounds' goals use only shared variables. But it evaluates them even when the outer loop runs no iterations, which the sequential meaning does not. That matters only for a bound that can throw, such as `N // D` with `D = 0`, so the bounds' goals must be `det`, and a bound that throws is the promise's to keep, as a bound in C is. A bound that uses an outer induction variable, as a triangular nest's does, is not rectangular, which `omp.loop_nest` does not represent, so `collapse` of it is an error.
- **Counter overflow** (step 4, implemented). The translation's `if I > Hi` stops after `I = Hi` only if `I + 1` does not overflow, so with `Hi = max_int` it would never stop. Testing `I = Hi` instead would not do for a step other than 1 or -1, which may step over `Hi`. So the translation computes the trip count before the loop, as the OMPIRBuilder does in `calculateCanonicalLoopTripCount` (with a negative step, the bounds swapped, then the span divided by the step, unsigned, plus one), and its `loop` predicate counts it down, with `I` advancing beside it. Mercury's `int` has no unsigned division, so a stride above 1 divides each bound by `div` and corrects by `mod` (Translation, above). The OMPIRBuilder's own limit is a single loop over the whole `int` range, whose trip count of 2^64 wraps to 0, and a collapsed nest whose product of trip counts overflows; these are the programmer's to avoid, as in C.
- **Values from outside the parallel** (step 6a, implemented). In a generic kernel, a value that the parallel in `teams` uses from outside it made the kernel trap on the GPU, so the bounds' goals and the reads of the region's values are in the parallel (Translation, "Target teams distribute parallel do").
- **Negative steps** (step 5, implemented). `createCanonicalLoop` takes a signed loop with a negative step, `for(I, Hi, Lo, -1)`, by swapping the bounds, so it needs no rewriting; `CodeGen/omp-parallel-do.m` runs one.
- **Regions in Lowering** (step 5, implemented). Lowering flattens the goals of an `omp.target` region; the same must be done for the bodies of `omp.parallel` and `omp.loop_nest`, which end with `omp.terminator` and `omp.yield`, and for the wrappers between them, which hold one operation. It did already: Lowering flattens the goals of each region of any `omp` operation, and the operations in them, nested ones too.
- **Exceptions** (later). A Mercury exception that leaves an iteration would unwind through the outlined region, which OpenMP forbids. Until exceptions are compiled, a parallel loop's body is limited as the rule for every parallel loop says, below; the rule that each iteration catches its own exception and one is rethrown after the join needs the exception support.
- **The removed predicates.** The outer `loop` and `body`, and the inner `loop`, are generated, checked and then not called. They cost only compile time; removing them needs nothing more than not emitting procedures that nothing calls.

**Related work.** Loop syntax for logic programs comes from Schimpf's ECLiPSe loops and Zhou's B-Prolog `foreach`, whose successor, Picat, also compiles loops to tail-recursive predicates and makes variables that occur only in a loop local to each iteration, as ours are; none of them runs loops in parallel. Barklund and Bevemyr's bounded quantification over arrays ("Prolog with arrays and bounded quantifications", LPAR 1993) is the closest earlier parallel construct, run on shared-memory multiprocessors (PLILP 1993) with the abstract machine of Reform Prolog, which runs recursion in parallel (Bevemyr, Lindgren and Millroth). Two other lines have the compiler or the runtime find the parallelism: &-Prolog and Ciao prove independence by abstract interpretation, and Mercury's parallel conjunction `&`, an annotation that, like ours, keeps the sequential meaning, parallelizes recursive loops with Bone, Somogyi and Schachte's loop control ("Controlling Loops in Parallel Mercury Code", DAMP 2012), on CPU threads in Mercury's own runtime. On GPUs, logic programming has so far meant the engine, not the program, as in Martinez-Angeles, Dutra and Santos Costa's Datalog engine ("A Datalog Engine for GPUs", WFLP 2013). The nearest relatives in spirit are single-assignment languages for numerical work: Id's loops name each iteration's new value with `next`, as `fromto` names `Out`; Sisal's `for` loops return reductions and ran in parallel; and SAC's with-loops, above. What is new here is the combination: loops in a typed and moded logic language, with their sequential meaning fully checked, standard OpenMP directives as the parallel promise, maps inferred from modes, and lowering through the `omp` dialect to GPUs.

### Parallel loop annotations

- **Syntax:** a comment line `%$omp parallel do [schedule(…)] [reduction(Op)]` before a goal, mirroring Fortran's `!$omp`. A stock Mercury compiler ignores it, so the sequential meaning is the specification.
- **Targets:** `do` loops controlled by a `for` (Loops, above). Calls to `int.fold_up` and `int.fold_down`, and to the array iterators (`array.map`, `array.generate`), are treated as intrinsics. Self-recursive counted loops are also recognized, with `index(I) upper(N)` naming the induction variable and bound.
- **The promise:** iterations are independent, and writes to a threaded array touch disjoint elements. A reduction's operator is associative. Floating-point reductions may then vary with the schedule.
- **Checked by the compiler:** the body is `det`, pure, and threads no `io`. A violation is an error, not a silent sequential fallback.
- **Exceptions:** each iteration catches its own exception; one is rethrown after the join.

### Target regions

A target directive, such as `%$omp target teams distribute parallel do`, runs its goal on a device. A GPU is only useful on flat data, so only flat values cross into a target region. Implemented so far: `target` with no clauses, on integers, floats and arrays of them (Lowering and code generation, "What is lowered so far"). Our representation (Lowering and code generation, "Type representation") says which values are flat.

- **Scalars:** `int`, `uint` and the sized integers, `float`, `char`, and enums, which are small integers with no cell. A type with one functor of one argument, such as `meters ---> meters(float)`, is represented as its argument, so it is a scalar too.
- **Arrays of scalars:** `array(int)`, `array(float)` and the like are a size and unboxed elements, the contiguous buffer that `omp.map.info` maps. The elements are mapped and the size is passed as a scalar.
- **Tuples, by value:** a tuple is a cell, so `{X, Y}` is unpacked into its scalar fields at the region's boundary. An array of tuples is an array of pointers to cells and is not mapped; packing it into a flat layout can come later.

Lists, strings and other terms are not mapped. The region's goal may not allocate either (Lowering and code generation, "Parallel loops"), so none appear inside it.

**Map types from modes.** The variables that cross the boundary are the goal's nonlocal variables, which quantification computes, so they need not be listed. Each one's mode gives its map type:

| Mode | Map |
| --- | --- |
| An `in` array | `to` |
| An array threaded through the goal, `di` to `uo` | `tofrom`, updated in place, as in experiment E2 |
| An `in` scalar or tuple | `firstprivate` |
| An `out` scalar | `from`, or the variable of a reduction |
| An `out` array | Not allowed: the device would have to allocate it. The caller makes the array and threads it through the goal |

A `map` clause is therefore needed only to override the inferred map type or to map part of an array.

**Array sections.** `map(to: A(Lo .. Hi))` maps elements `Lo` to `Hi` of `A`, inclusive. Both parts are standard term syntax, a higher-order application and the builtin operator `..`, so it reads with no change to the parser, and `..` keeps the meaning it has in the `list` module, where `1 .. 3` is `[1, 2, 3]`. The reading as a section is ours: Mercury has no array sections, and to mmc the term is only part of a comment. A section with no map type takes the type the mode gives. Sections let work be split over several devices:

```mercury
    %$omp parallel do index(D) upper(NumDevices - 1)
    for_each_device(D, ...)

for_each_device(D, ...) :-
    Lo = D * Chunk,
    Hi = int.min(Lo + Chunk - 1, N - 1),
    %$omp target device(D) map(to: A(Lo .. Hi)) map(tofrom: B(Lo .. Hi))
    %$omp&    teams distribute parallel do index(I) upper(Hi)
    loop(Lo, Hi, A, !B)
```

- **Indices are unchanged.** As in OpenMP, a section keeps the array's indices on the device, so the goal still uses `I`, with no offset. Only elements `Lo` to `Hi` are transferred, and for `from` and `tofrom` only those are copied back.
- **Overlap:** sections mapped `to` may overlap, as the halos of a stencil do. Sections that are written must be disjoint between regions that run at the same time. This is the promise a parallel loop already makes, that writes to a threaded array touch disjoint elements; uniqueness is not violated, because the array is still threaded once.
- **Bounds** are checked when the region starts, `0 =< Lo` and `Hi < size(A)`. A section out of bounds is a run-time error that names the array, not a silent write past its end on the device.

**Checked by the compiler,** after type and mode checking: every variable that crosses the boundary has a monomorphic type that can be mapped, or the error names it and its type, as in "`Xs` has type `list(float)`, which cannot be mapped to a device". The goal must also satisfy the rules for a parallel loop: `det`, pure, no `io`, and no allocation.

**No run-time dispatch on the device.** After specialization, a generic `unify` or `compare`, or a closure call whose callee is not known, left in a target region's goal or in a procedure it calls, is an error that names the goal: device code does not depend on run-time type dispatch or function pointers. A polymorphic procedure called at known types is fine, since specialization gives it a clone for them (Optimizations, "What polymorphism costs"). Outside target regions the same goals in a `%$omp` loop are only a remark.

**Open question:** `array2d`, which is flat too, with sections of rows or blocks.

### Sendability

Because every send makes a deep copy, almost every type is sendable, including arrays: the receiver gets its own copy. The exceptions are values whose identity matters: the `io` state, `store` and mutable references, and foreign types without a copy hook. Open terms are sendable with copy semantics (Open terms, below). The compiler infers sendability per type, like Rust's `Send`, and rejects a `send` of anything else. Mailbox handles are sendable; they are identifiers, not pointers. Sending to another node adds one exception, closures, unless both nodes run the same binary (Distribution and wire format).

An optional later optimization lets `send` take a message in a `di` mode and move it without copying, since the sender provably gives it up.

### Generic processes and dispatchers

Polymorphic code is compiled once. Each type variable becomes a hidden type-info argument, and a value of type `T` is one word that the code does not look inside. In E4's `poly.m`, one C function served both `list(int)` and `list(string)`, and a `list(float)` built by polymorphic code used 16 bytes per element. A `send` therefore always has a type info for its message: a static descriptor in monomorphic code, or the hidden argument in polymorphic code. The message copier walks the message with it, so no message size has to be known at compile time.

A dispatcher that forwards payloads of many types uses an existential envelope. The type is hidden from the dispatcher, not lost: the process that builds the envelope knows it, and so does the receiving mailbox.

```mercury
:- type envelope
    --->    some [T] envelope(mailbox(T), T)
    ;       stop.

:- func wrap(mailbox(T), T) = envelope.
wrap(Dest, Payload) = 'new envelope'(Dest, Payload).

router(Self, !IO) :-
    receive(Self, any, infinity, MaybeMsg, !IO),
    (
        MaybeMsg = yes(Msg),
        (
            Msg = envelope(Dest, Payload),
            send(Dest, Payload, !IO),      % same hidden T
            router(Self, !IO)
        ;
            Msg = stop
        )
    ;
        MaybeMsg = no
    ).
```

- **The sender knows the type.** A client calls `wrap(CounterBox, increment(5))` with `T = counter_msg`. `wrap` checks that the payload matches the destination, so `wrap(IntBox, "hello")` is a type error at the client.
- **The router knows only that the halves match.** It sees `Dest : mailbox(T)` and `Payload : T` for one unnamed `T`, which is all `send` needs. Sending `Payload` to a `mailbox(int)` instead is a type error.
- **The receiver knows the type again.** The counter server receives from its own `mailbox(counter_msg)` and sees an ordinary `counter_msg`, with no casts.
- **At run time,** an envelope is a three-word cell holding the type info, the destination and the payload, and the router's `send` passes the stored type info on (E4).
- **Routing on content** needs a key beside the payload, such as `envelope(topic, mailbox(T), T)`. The alternative, carrying a `univ` and testing its type with `univ_to_type`, gives up the compile-time check.

**Sendability of hidden types.** The router cannot see `T`, so sendability needs a rule for polymorphic and existential sends. A `mailbox(M)` can only be created, by `spawn` or `new_mailbox`, where `M` is known to be sendable. This is expressed as a constraint `sendable(M)` that the compiler satisfies automatically for every sendable type, as Rust does for `Send`. Any existing mailbox then proves that its message type is sendable, so `send` needs no constraint of its own and the existential send above is safe. The class has no methods and is erased, so it costs nothing at run time.

In C++ terms, the envelope is type erasure: `wrap` is a templated constructor that runs where `T` is known, the stored type info plays the role of the vtable, and the router's `send` is the virtual call. Across nodes, a relay can forward such payloads without having their types compiled in (Distribution and wire format).

### Open terms (Prolog-style variables)

Mercury never unifies two unbound variables that both stay live, so no term can contain a hole that another term shares. That rules out difference lists and the other Prolog idioms that build a structure through a shared unbound tail; E4 shows mmc rejecting a literal difference list. Open terms lift the restriction for data the programmer marks, and only that data pays for it.

The most common reason Prolog code uses difference lists, building a list in order, needs none of this. Tail-as-input clauses, DCGs, cords and last-call-modulo-constructors already do it in ground code (E4), and they are the faster choice: building through holes costs 2.3× the ground version (E5). Open terms are for the rest: incremental construction where the order is not known in advance, and algorithms built on unification, such as type inference.

**Types.** For any type `T`, `open(T)` has the same constructors as `T`, but any argument at any depth may be an unbound variable; an argument of type `U` has type `open(U)` inside it. Values of an open type have inst `any`, Mercury's existing inst for values that may be partly unbound, and are passed with the existing modes `ia` (`any >> any`) and `oa` (`free >> any`). A free variable of an open type becomes a fresh unbound variable only when mode analysis cannot schedule its conjunction otherwise; the compiler then inserts the code that creates it (Checking rules). In constructor arguments, a ground `U` is accepted where `open(U)` is expected; elsewhere conversion is explicit.

**Operations.**

- `X = Y` on open values is full unification. It may bind variables on either side, including one variable to another, and fails on a constructor clash, so it is `semidet`. Construction into a variable that is still `free` stays `det`.
- `X = f(Y1, ..., Yn)` with `X` an `any` value is the WAM's `get_structure`. If `X` is an unbound variable, it is bound to a new cell, and the arguments that are still free become fresh variables (write mode). If `X` is bound to `f`, the free arguments are assigned its fields, and the others are unified with them (read mode). It is `det` when the type has one functor and every argument is free, and `semidet` otherwise. Only the arguments that are not free need full unification.
- `to_open(X)` turns a ground `T` into an `open(T)`. It is free when the two types share a layout (Lowering) and copies otherwise.
- `from_open(open(T)::ia, T::out) is semidet` fails if any variable is still unbound or the term is cyclic. It returns the term itself when the walk meets no variable cell, bound or unbound, and otherwise copies it into `T`'s layout.
- Plain `=` omits the occurs check, as Prolog does; `unify_oc` performs it. `from_open` rejects the cyclic terms that can result, so ground code never sees one.
- `var`, `nonvar` and term identity are non-logical and therefore `impure`. Open types have no `compare` and no hashing, because the order of terms with variables changes as they are bound.

**Checking rules.**

- **Fresh variables are a last resort.** Mode analysis schedules a conjunction as usual, delaying goals whose inputs are not bound enough. Only when no goal can run does it create a fresh variable for a free open variable that a delayed goal needs, and then it continues. Ground parts of a clause therefore keep the ordinary order: a hole that a later goal fills with `[]` is constructed as `[]`, without a variable cell or a binding. `free` does not match `any`, as in mmc (Overton 2003, 3.2.2), so a merge never turns a variable that is free in one branch into `any`.
- **Negated contexts.** A negation or if-then-else condition that may bind a nonlocal `any` variable needs a `promise_pure` scope, the rule Mercury already applies to solver types. `\+ X = 1` with `X` unbound is the case Prolog answers wrongly.
- **Determinism.** Binding an existing variable is `semidet`, because the compiler cannot know it is still unbound. An inst for "known unbound" would make first bindings `det`; that is a later refinement. Without alias tracking the compiler cannot prove such an inst, since a hole is shared with the term around it, so it would be a promise in the mode declaration that the caller's code relies on, not a checked fact.
- **Sendability.** Open values are sendable, with the semantics of Prolog's `copy_term`. The receiver gets one fresh variable for each distinct unbound variable in the message, so sharing among variables is preserved. Bound variables arrive as their values, and later bindings on either side stay on that side.
- **Parallel loops.** A loop body may not bind an open variable that is nonlocal to the body. The trail belongs to the process, and two threads binding one variable would race.
- **Uniqueness.** Open values are never `unique`, and structure reuse skips them, because the trail may still refer to their cells.

**Example.** The difference list that E4 showed mmc rejecting, written with open terms:

```mercury
:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

    % Fill the hole with Tail and return the ground list.
:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.

close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    from_open(Front, List).

main(!IO) :-
    make_open([1, 2, 3], Front, Hole),
    promise_pure (
        if close(Front, Hole, [4, 5], List) then
            io.write_line(List, !IO)        % [1, 2, 3, 4, 5]
        else
            io.write_string("hole already bound\n", !IO)
    ).
```

`make_open` stays `det`. Its first clause unifies two free variables, which creates one fresh variable and makes the other refer to it, and the mode checker schedules the recursive call before the construction, as it does today. `close` is `semidet` because `Hole` might already be bound. Outside these predicates, `list(int)` keeps its ordinary unboxed layout.

**Example: appending difference lists.** The Prolog idiom `app(A-B, B-C, A-C)`, with a predicate that closes the result into an ordinary list, and a caller in ordinary Mercury:

```mercury
:- type dlist == open(pair(list(int), list(int))).   % Front - Hole

:- pred append_dl(dlist::ia, dlist::ia, dlist::oa) is semidet.
append_dl(A - B, B - C, A - C).

    % Close the hole with [] and return an ordinary list.
:- pred dl_to_list(dlist::ia, list(int)::out) is semidet.
dl_to_list(F - [], L) :-
    from_open(F, L).

:- pred test(list(int)::out) is semidet.
test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).

:- pred first(int::out) is semidet.
first(X) :-
    test(L),
    L = [X | _].
```

`open(pair(...))` makes the pair open as well as the lists, so `A`, `B` and `C` are `open(list(int))`. Below, `<=` is a construction, `=>` a deconstruction, `?=` a `get_structure`, `~=` a full unification and `:=` an assignment, after the notation of Overton (2003, 3.3.1).

`append_dl` desugars to `V1 = A - B, V2 = B - C, V3 = A - C`. For the procedure `(ia, ia, oa)`, `V1` and `V2` start as `any` and `V3` and the locals as `free`.

| Goal | Kind | Determinism | Afterwards |
| --- | --- | --- | --- |
| `V1 = A - B` | `V1 ?= A - B`; `pair` has one functor and both arguments are free | `det` | `A`, `B`: `any` |
| `V2 = B - C` | `V2 ?= B' - C, B ~= B'`; `B` is no longer free | `semidet` | `C`: `any` |
| `V3 = A - C` | `V3 <= A - C` | `det` | `V3`: `bound(any - any)` |

No goal waits, because an `any` input is always bound enough, so the source order stands. `bound(any - any)` matches `any`, so the final insts satisfy the modes, and the procedure is `semidet`, as declared. Used correctly it cannot fail, because the hole `B` is unbound, but proving that needs the "known unbound" inst (Checking rules).

`dl_to_list` desugars to `V1 = F - H, H = [], from_open(F, L)`. `V1 ?= F - H` is `det`. `H` is then `any`, so `H ?= []` binds the hole or tests it, which is `semidet`, since `list` has two functors. `from_open(F, L)` is `semidet`: it still fails if an element, of type `open(int)`, is unbound. Its declared mode `(ia, out)` makes `L` `ground` and its type makes it an ordinary `list(int)`, so the open part ends there without a special rule.

In `test`, every goal waits at first. The constructions need `H1` or `H2`, the call needs `D1` and `D2`, and `dl_to_list(D, L)` needs `D`, because `free` does not match `ia`. With nothing able to run, mode analysis creates fresh variables:

```mercury
test(L) :-
    H1 := new_var, V1 <= [2 | H1], V2 <= [1 | V1], D1 <= V2 - H1,
    H2 := new_var, V3 <= [3 | H2], D2 <= V3 - H2,
    append_dl(D1, D2, D),           % semidet; D: any
    dl_to_list(D, L).               % semidet; L: ground
```

`H1` is in two cells, `V1` and `D1`, which plain Mercury forbids. Here it is a variable cell by then, and both cells refer to it. All the open variables are local to `test`.

In `first`, `L` is `ground` and has type `list(int)`, so `L = [X | _]` is the ordinary deconstruction `L => [X | _]`: a tag test and a field load. There is no dereferencing, no write mode and no trail, and the elements are unboxed again. `first` compiles exactly as it would in a module that never mentions `open`.

At run time, `append_dl` binds `_1` to `[3 | _2]` in constant time and returns `[1, 2, 3 | _2] - _2`. `dl_to_list` binds `_2` to `[]`, and `from_open` meets the two variable cells, so it copies the list into the ordinary layout: `L = [1, 2, 3]` and `X = 1`.

mmc 22.01.9, with `dlist == pair(list(int), list(int))` and modes `in, in, out`, accepts `append_dl`, but only as a test of ground pairs. It rejects building the difference list, at `D1 = [1, 2 | H1] - H1`: "mode error in unification of `D1' and `pair.(V_8 - H1)'. Variable `D1' has instantiatedness `free', term `pair.(V_8 - H1)' has instantiatedness `named inst pair.'-'(free, free)'". That is where the open version creates its fresh variables.

**Why option 1.** Three representations were considered for telling a variable from a value inside an open term. The first test was what each costs code that never uses open terms, and the second what it costs code that does, measured in a C model of the generated code (E5).

| Option | How a variable is recognised | Cost to ground code | Open `list(int)`, time and memory (E5) | Verdict |
| --- | --- | --- | --- | --- |
| 1. Separate open layout | A reserved primary tag in the open layout only; raw `int`, `float`, `char` and enum fields there are boxed or re-encoded | None | 1.18× time, 1.5× memory; 1.03× and 1.0× when small ints use static boxes | Chosen |
| 2. Tagged small integers | Small ints carry their own tag; large ones are boxed | None, if confined to the open layout | 1.02×, 1.0× | Adopt later, inside option 1's open layout |
| 3. Out-of-band bitmap | A header bit per field marks variables; ints stay raw | None, if only open cells get headers | 1.36×, 1.5× | Rejected |

- **Option 1 keeps ground code untouched.** The variable tag and the boxing exist only in the open layout, and a type gets an open layout only if some program uses it under `open(_)`. It is also the simplest to build: unification, the collector and printing all see a word that is either a variable reference or an ordinary tagged value.
- **Option 2 is a refinement, not a rival.** Tagging small ints removes most of option 1's boxing cost, but it adds a second encoding of `int` that every arithmetic operation and conversion in open code must handle, and it does nothing for `float`. Because it changes only the open layout, it can be added later without touching anything else.
- **Option 3 only moves the problem.** A value loaded into a register or passed as an argument has no header, so a loose `open(int)` still needs a tag or a box. Every field read loads the header first, and a variable cell needs a state word as well, since a raw int in it could not be told from a reference. In the model it was slower than boxing and used as much memory.
- **Alias tracking is the other way to accept difference lists.** Definite alias tracking (Overton, 2003) changes the mode checker for every predicate, is still missing from Melbourne Mercury (`LIMITATIONS.md`), and covers only aliasing the compiler can see at compile time. Unification-based algorithms need aliasing that depends on the data. Open terms cover both cases with a runtime mechanism, and the only mode checker change they need is `any` insts, which Mercury already has.

**Precedent.** Mercury's solver types already give values the inst `any`, and its reference manual already requires a promise scope when such a variable is nonlocal to a negated context; open terms reuse both. Undoing bindings on backtracking needs a trailing grade (`.tr`, user's guide). The user's guide also says modules link only if all were compiled in the same grade, so trailing reaches every choice point in the program; E4 saw trail tickets at each one. A solver-type library for Herbrand terms therefore fails the cost rule. HAL ran Herbrand constraints on Mercury with PARMA bindings. This design keeps the solver-type model, gives each open type a generated layout and unify procedure, and trails only where open variables can be bound. The runtime support is in Runtime system, "Support for open terms".

## MLIR dialects

Two new dialects carry Mercury semantics until upstream dialects can take over: `merk` for terms and control, and `mrt` for runtime operations. MLIR has no standard dialect for algebraic data types, determinism or backtracking, so these are ours; everything below them is upstream.

| Level | Dialects | What it represents |
| --- | --- | --- |
| Mercury | `merk` | Procedures with determinism, terms by constructor, switches, disjunction, commits, closures, `type_info` values, parallel loops |
| Runtime | `mrt`, plus `cf`, `scf`, `arith`, `math` | Concrete layout: tagged words, allocation, fields, safepoints, send/receive, exceptions |
| Parallel | `scf.parallel`, `omp` | Formed loops, reductions, OpenMP regions |
| Target | `llvm` | LLVM IR with GC statepoints |

**`merk` types** are Mercury types, not layouts: `!merk.term<"list.list(int)">`, `!merk.closure<…>`, `!merk.typeinfo`, and builtin `i64`, `f64`, `i8` for Mercury's primitive types. Layout is chosen when lowering to `mrt`, so representation work never changes the dialect.

**`merk` operations**

| Operation | Meaning |
| --- | --- |
| `merk.proc` | A procedure: one mode of a predicate or function, with a determinism attribute |
| `merk.construct` | Build a term from a named constructor and its arguments |
| `merk.deconstruct` | Take apart a term whose constructor is known, binding the constructor's arguments |
| `merk.switch` | Branch on a term's constructor; a case of one constructor takes the term apart with `merk.deconstruct` |
| `merk.has_functor` | Test a term's constructor, for `merk.guard` |
| `merk.type_def` | A type's constructors and their arguments' types, from which lowering chooses its layout |
| `merk.call`, `merk.call_closure`, `merk.closure` | First-order calls, higher-order calls, closure construction |
| `merk.ite`, `merk.not`, `merk.disj` | If-then-else, negation, disjunction |
| `merk.succeed`, `merk.fail`, `merk.commit` | Produce a solution, fail, prune remaining solutions |
| `merk.throw`, `merk.try` | Exceptions |
| `merk.par_conj` | Parallel conjunction. OpenMP directives use the `omp` dialect's operations instead of `merk` ones |
| `merk.entry` | Names the program's main procedure, which the runtime calls through `merk_main` |

Defined so far (`Dialect/Merk/IR`): `merk.proc`, `merk.call`, `merk.succeed`, `merk.entry`, `merk.constant` and `merk.foreign_call`, with the type `!merk.string` and the attributes `#merk.determinism<det>` and `#merk.address_spaces<…>` (Lowering and code generation, "Target information" and "What is lowered so far"). For branching, `merk.ite`, `merk.switch`, `merk.not`, `merk.fail` and `merk.yield`, and two that the table does not have:

- `merk.guard %flag` continues if a test succeeded and fails if not; a comparison or a unification of two bound primitive values is an `arith` comparison and a guard, and a test of a term's constructor a `merk.has_functor` and a guard.
- `merk.throw_if %flag, "exception"` throws one of the library's exceptions, such as a division by zero's, until `merk.throw` and terms exist.

For terms (decided 2026-11-04), `!merk.term<"T">`, a value of the type `T` names, which is a type defined by constructors, or a tuple type, written fully qualified with its arguments, as `!merk.term<"list.list(int)">`; `merk.type_def "T" [...]`, one in each module for each such type its procedures use, its own or imported, which lists the type's constructors in their declared order, each a `#merk.constructor<"name"(argument types)>`; and `merk.construct`, `merk.deconstruct`, `merk.has_functor` and `merk.switch`, which name a constructor by `#merk.functor<"name", arity>`, since a type may have two of one name:

```mlir
merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
%l = merk.construct #merk.functor<"[|]", 2>(%h, %t : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
%is = merk.has_functor #merk.functor<"[]", 0> %l : !merk.term<"list.list(int)">
%h2, %t2 = merk.deconstruct #merk.functor<"[|]", 2> %l : !merk.term<"list.list(int)"> -> i64, !merk.term<"list.list(int)">
```

`merk.deconstruct` is not speculatable: another constructor's term has no such arguments. A unification of two terms is a call of the type's unify procedure, `unify.T`, a private `semidet` `merk.proc` that code generation makes the first time a module needs it, in `merk`: it switches on the first term, tests that the second has the same constructor, and compares their arguments, integers and floats by `arith` and terms by their types' unify procedures. These were chosen over three alternatives:

- **Constructors in the type itself**, as parameters of `!merk.term`, would have made each type self-contained, but a recursive type such as `list(int)` names itself among its arguments' types, which an MLIR type can do only as a mutable, identified type, as LLVM's named structs are, and every value's printed type would carry the whole definition. A name and a definition beside the procedures is how LLVM's named structs and CIR's records print too.
- **A layout in the type**, such as tags or a cell size, would make representation work change the dialect, which the rule above forbids: `merk.type_def` lists only what the program says, and lowering chooses the layout.
- **A `merk.unify_test` operation**, expanded by lowering, was the first plan. Lowering would then have to build the recursive comparison in the LLVM dialect itself, with its own recursion, branches and tail calls; as a `merk.proc`, the comparison is lowered by the code that lowers every other procedure, and its recursive call on a list's tail is a `musttail` call. The polymorphic unify of chunk 1's step 4 is then the same procedure taking `type_info`s. The cost is a copy of the procedure in each module that needs it, with internal linkage.

Not yet: types with type variables, with user-defined equality, subtypes, existentially typed constructors, and unifications of terms with `string` or array arguments, which are reported as not yet implemented.

For arrays, the type `!merk.array<T>` and the library's `array` predicates as operations, until the library can be compiled with type_infos: `merk.array_new` (a size and an initial element, or none for an empty array), `merk.array_size`, `merk.array_get`, `merk.array_set`, `merk.array_fill` (from one index to another), `merk.array_copy`, `merk.array_resize` (with an element for the new places, or none to shrink) and `merk.check_index %a[%i], "array.lookup"`, which throws `index_out_of_bounds` with mmc's message, as `index_out_of_bounds("array.lookup: index 4 not in range [0, 3]")`. Those that update an array do it in place and give the same array, since uniqueness checking has made sure its old version is dead; the others allocate. None checks an index itself.

The regions of `merk.ite`, `merk.switch` and `merk.not` have one block each. A region that succeeds ends with `merk.yield`: the condition yields what it binds that the then part uses, as the then part's arguments, and the branches yield the operation's results, the variables that every branch that succeeds binds. A branch can instead end with `merk.fail`, or with `merk.succeed` when the operation is the procedure's last goal. Failure goes to the innermost condition's else part or negation, or else out of the procedure, which then can fail. `merk.switch` is on an integer or a term: its `cases` attribute lists each case's integer constants or functors, and with `can_fail` a value that no case has fails. Arithmetic is the `arith` dialect's, which `merk` depends on. `merk.foreign_call` with `can_fail` is that of a `semidet` `foreign_proc`, whose first result is `SUCCESS_INDICATOR`.

A deterministic procedure, as code generation makes it:

```mlir
merk.proc private @m.double_all_f1_0(%arg0: !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)"> attributes {determinism = #merk.determinism<det>} {
  %0 = merk.switch %arg0 : !merk.term<"list.list(int)"> -> !merk.term<"list.list(int)"> cases [[#merk.functor<"[]", 0>], [#merk.functor<"[|]", 2>]] {
    %1 = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
    merk.yield %1 : !merk.term<"list.list(int)">
  }, {
    %1:2 = merk.deconstruct #merk.functor<"[|]", 2> %arg0 : !merk.term<"list.list(int)"> -> i64, !merk.term<"list.list(int)">
    %2 = merk.constant 2 : i64
    %3 = arith.muli %2, %1#0 : i64
    %4 = merk.call @m.double_all_f1_0(%1#1) : (!merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    %5 = merk.construct #merk.functor<"[|]", 2>(%3, %4 : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    merk.yield %5 : !merk.term<"list.list(int)">
  }
  merk.succeed %0 : !merk.term<"list.list(int)">
}
```

In a `nondet` procedure, a call to another `nondet` procedure has generator semantics. The operations after it run once per solution, and `merk.fail` resumes the most recent choice. Verifiers enforce the determinism rules, so a `det` procedure may not contain an unpruned `nondet` call.

**`mrt` operations:** `mrt.alloc` (bump allocation in the process nursery), `mrt.tag` and `mrt.untag`, `mrt.field_load` and `mrt.field_store`, `mrt.safepoint`, `mrt.yield_check`, `mrt.spawn`, `mrt.send`, `mrt.receive` (a suspension point, over the mailboxes of a selector), `mrt.throw`, `mrt.landing`, and `mrt.rtti` (address of a static type descriptor).

Defined so far (`Dialect/Mrt/IR`, 2026-11-04), for terms' cells: `mrt.alloc N`, a cell of N words on the heap, whose fields are undefined until stored; `mrt.tag %cell, T` and `mrt.untag %term, T`, which add a primary tag to a cell's pointer and take a known one off; `mrt.primary_tag %term`, the tag of a term's word, which the list above did not have, for switches and tests; and `mrt.field_load %cell[W] : T` and `mrt.field_store %v, %cell[W] : T`, a field of a word or less in word W. A store is only into a cell just allocated, since terms are immutable once made. Their values are the LLVM dialect's: a cell or a tagged term is an `!llvm.ptr`, a constant's word an `i64`:

```mlir
%cell = mrt.alloc 2
mrt.field_store %h, %cell[0] : i64
mrt.field_store %t, %cell[1] : !llvm.ptr
%l = mrt.tag %cell, 1
```

Three choices were made against alternatives:

- **The LLVM dialect's types**, rather than `mrt` types of its own such as a word and a cell, since `mrt` is where layouts are concrete: an operation's operand is what LLVM will see, and lowering needs no casts between `mrt` values and the LLVM operations around them, such as the `select` of a secondary tag's address.
- **`merk.switch` on a term stays `merk`'s until lowering flattens it into blocks**, where it becomes `llvm.switch`es on `mrt.primary_tag` and on words that `mrt.field_load` reads, as `Dispatch` makes them. Lowering it to nested switches on integers in the regions of `merk` instead would copy a case's body into each tag that reaches it, as when a case lists two constructors of different tags, where blocks share it.
- **A field is untagged, then loaded at its word**, `mrt.untag` then `mrt.field_load`, rather than one operation of a tagged pointer and a word. Each operation then means one thing, and LLVM folds the two `getelementptr`s into one at `-O1` and above, as the design asks.

**Open terms** need no new `merk` type, because `open(T)` is an ordinary Mercury type. They add `merk.open_unify` (full unification, `semidet`), `merk.to_open` and `merk.from_open` to `merk`, and `mrt.new_var`, `mrt.deref`, `mrt.bind` (which trails when a trail is live), `mrt.trail_mark` and `mrt.trail_undo` to `mrt`. Only procedures with `any` variables contain them.

## Lowering and code generation

Lowering fixes the data layout first, then turns each determinism category into plain control flow, then hands LLVM IR and GC metadata to LLVM. The numbered steps below run in this order.

### Target information

Code generation knows the target through `TargetInfo` (`CodeGen/TargetInfo.h`), made from the triple, CPU and features with LLVM's `TargetMachine`, as Flang's frontend makes it. It gives the LLVM data layout; the address spaces it names for allocas, globals and code; each address space's pointer size and alignment; the word, which `int`, `uint` and `float` fill, a generic pointer wide; and the free tag bits of a cell pointer, 3 for 8-byte words.

- **Address spaces of Mercury code.** Besides the data layout's, `TargetInfo` names the target's numbers for the generic address space, which every pointer converts to, and for global, workgroup-shared, constant and per-thread private memory, as target regions will need: all 0 on a CPU; 0, 1, 3, 4 and 5 for AMDGPU and NVPTX; 4, 1, 3, 2 and 0 for SPIR-V, OpenCL's. ClangIR so far records only the default address space.
- **Recorded on the module.** As in ClangIR, the module records its target, so that a module read back from its text lowers as it would have when written, with no command line: `llvm.target_triple`, `llvm.data_layout` and its DLTI form `dlti.dl_spec`, `merk.target_cpu`, `merk.target_features`, `merk.optimization_level` and `merk.address_spaces`. `-fc1` reads a `.mlir` file written by `-emit-merk` or `-emit-mlir` back, and lowers it for the target it records; a module that records none is an error.
- **Not Clang's `TargetInfo`.** ClangIR rebuilds Clang's `TargetInfo` from the module's triple, for C's type sizes and ABI. Mercury needs few of those facts, and calls into C, such as libc's and libm's, pass only words, floats and pointers. `foreign_proc("C")` and `foreign_type` will need C's `long`, `size_t` and struct ABI exactly as Clang has them; `TargetInfo` can then be built on Clang's without changing what the module records.

### What is lowered so far

Code generation (`CodeGen/MLIRGen.cpp`) makes `merk` for procedures that are `det`, `semidet`, `cc_multi`, `failure` or `erroneous`, have no hidden arguments, and whose values are of primitive types, `io.state`, arrays of values a word wide at most, or types defined by constructors and tuple types without type variables, whose bodies are conjunctions, assignments, constants, tests, constructions and deconstructions of terms, calls and function applications, if-then-elses, switches on integers, characters and constructors, negations and `fail`, or that have a `foreign_proc` for C. A value of type `io.state` is not passed, so `main(!IO)` takes no arguments. Anything else is reported as not yet implemented, at the first such construct of a procedure. Lowering (`CodeGen/Lowering.cpp`) lowers terms to `mrt`, choosing their layouts, and then `merk` and `mrt` to MLIR's LLVM dialect, in one pass over the module, as ClangIR's DirectToLLVM lowers CIR. Then MLIR translates it to LLVM IR, LLVM's pipeline for `-O0` to `-O3` optimizes it, and the `TargetMachine` writes assembly or an object file.

- **Symbols.** A procedure's symbol is its module, then its name, its arity, with an `f` before it for a function, and its mode number, joined by `_`: `hello.main_2_0`, `list.length_f1_0`. Other characters than letters, digits and `_` in a name become `$` and two hex digits, so `int.+` is `int.$2b_f2_0`. A procedure defined and not exported has internal linkage.
- **Entry.** `merk.entry` names the main module's `main/2`; it becomes `merk_main`, a C function that calls it, which the runtime's `main` calls (Runtime system).
- **Constants.** An integer literal, in any base, with `_` and a size suffix, or `0'c`, is a `merk.constant` of its type's width; a float is an `f64`; a character is its code point, an `i32`; a string is a `!merk.string`, with its escapes replaced, and becomes a private constant global of its bytes and a NUL, one for each distinct string in a module. In C a string is `MR_String`, a `char *`, as the reference manual's "C data passing conventions" has it.
- **Foreign code.** A procedure with a `foreign_proc` for C runs its C code instead of its clauses. Mode analysis gives each `foreign_proc` the procedure whose modes its arguments have, and reports one with no such procedure, or two for one procedure, as mmc does. Code generation puts the code in a C function, `merk_foreign_` and the procedure's symbol with `_` doubled, `.` as `_D` and `$` as `_S`, which the procedure calls with `merk.foreign_call`: the function takes the inputs as parameters, as the manual's C types, and returns the only output, or stores several through pointers after the inputs; a variable of a type with no values, such as `io.state`, is an `MR_Word` set to 0. The module records its C code in its `merk.foreign_code` attribute: the C types `MR_Integer`, `MR_Unsigned`, `MR_Float`, `MR_Char`, `MR_String` and `MR_Word`, the module's own `foreign_decl` and `foreign_code` for C, and the functions, with `#line` directives, so that clang reports errors at their lines in the Mercury source. When the module is translated to LLVM IR, `-fc1` writes the code to a temporary file, has `clang`, beside `merkc` or else on the `PATH`, compile it to bitcode for the module's target and optimization level, links it into the module and makes the functions internal, so LLVM's inliner can put the C code into its procedure. A module without foreign code does not need clang. The C function of a `foreign_proc` of a procedure that can fail has `MR_bool SUCCESS_INDICATOR`, set to `MR_TRUE` first, returns it as C's `int`, 32 bits on the targets so far, and stores every output through a pointer; its `merk.foreign_call` has `can_fail`. Not yet: a `foreign_decl` in the interface, which importing modules should see; and foreign types.
- **Branches.** Lowering makes each `merk.ite`, `merk.switch` and `merk.not` blocks of the procedure, outermost first: a guard is a conditional branch to the next goal or to where failure goes, a switch is `llvm.switch`, a yield is a branch to the block after the operation, whose arguments are its results, and a block that only succeeds is copied into the branches that go to it, so that a call before it is in tail position. Failure at the top of a procedure that can fail returns false; in a `det` one it cannot happen, and is `unreachable`.
- **Calling convention.** A procedure that can fail, `semidet` or `failure`, returns its success flag, alone as an `i1` or first in a struct with its outputs; one that fails leaves its outputs undefined. A call of one is a test of the flag. A call whose results are the procedure's own, of a procedure that can fail as the caller can or cannot, is `musttail`: with `tailcc`, LLVM requires only that the return types match, so a procedure's last call does not grow the stack, across modules too. A back end returns a result that needs more registers than its convention has in memory itself, through a hidden pointer, and then cannot make a `musttail` call of the procedure (x86-64 aborts with "failed to perform tail call elimination"). So a procedure whose outputs, after its flag, need more registers than the target returns results in (`TargetInfo::Results`: on x86-64 three words and two floats, as `RetCC_X86_64_C` has them; on AArch64 eight of each; on AMDGPU sixteen words, floats among them; elsewhere two of each) returns them in memory itself: it takes a pointer to a struct of its outputs before its inputs, stores them there, and returns only its flag. A tail call passes on the pointer it was given, since the callee's outputs are the caller's; any other call gives a slot of its own, where a foreign call's slots go, and loads the outputs after the call. Whether a procedure does is a matter of its signature and the target alone, so modules agree on it. `CodeGen/results-in-memory.m` checks it on x86-64 and AArch64 and runs loops of ten million tail calls. LLVM's tail recursion elimination turns a procedure whose recursive calls' results are combined after them, as `2 * f(N - 1)`, into a loop with an accumulator, and so rewrites each of its returns to combine the accumulator, or to select a return value; a return after a `musttail` call must stay next to it, so this tree's `TailRecursionElimination.cpp` keeps such recursive calls, in a function with such a return, instead of rewriting it, which `-O2` failed on with "failed to perform tail call elimination on a call site marked musttail" (2026-11-04; `llvm/test/Transforms/TailCallElim/musttail-other-returns.ll` and `CodeGen/musttail-accumulator.m`). No upstream report of it was found; the fix is one to send upstream.
- **Arithmetic.** The functions and comparisons of `int`, `uint`, the sized integer modules and `float` are built into code generation as `arith` operations, which MLIR's `ArithToLLVM` lowers; `float`'s others are called, and the integer modules' others, such as `int.pow`, are not yet implemented. As in mmc's programs: integers wrap; `//`, `rem`, `div`, `mod` and `float./` throw `domain_error("int.'//': division by zero")` and the like for a zero divisor; `<<` and `>>` throw for an amount outside 0 to the width; `abs` of the least integer throws; `div` and `mod` round toward minus infinity; `-X` of a float is `0.0 - X`. The least integer by -1 gives itself and 0, as mmc folds it, though mmc's programs die of SIGFPE (the register, "The quotient of the least integer by -1"). A conversion between integer types keeps the low bits, and extends by the sign of a signed source. `X = X` succeeds even for a NaN, as with mmc; otherwise float equality is IEEE's.
- **Arrays.** The library's `array` predicates and functions that make, read and update an array, which need no type_info, are built into code generation as `merk`'s array operations: `init`, `make_empty_array`, `size`, `max`, `min`, `bounds`, `in_bounds`, `is_empty`, `lookup`, `elem`, `semidet_lookup`, `set`, `'elem :='`, `semidet_set`, `slow_set`, `semidet_slow_set`, `swap`, `fill`, `fill_range`, `copy`, `resize` and `shrink`, and the `unsafe_` ones, which check nothing; the others, such as `foldl` and `from_list`, are not yet implemented. They check as mmc's library does, with its messages: an index out of range is `index_out_of_bounds("array.set: index 9 not in range [0, 1]")` (`array.lookup` for `elem`, `first argument of array.swap`, `second argument of fill_range`), and a bad size or range is a software error, `Software Error: predicate `array.init'/3: Unexpected: negative size`, `array.fill_range'/5`'s "empty range" when the first index is past the last (checked before either index), `array.resize'/4`'s "cannot resize to a negative size", and `array.shrink'/3`'s "cannot shrink to a larger size" and "cannot shrink to a negative size". An index is checked as an unsigned comparison with the size, so a negative one fails too. An array is a pointer to its size and its elements, a word each (Type representation); `merk.array_get` and `merk.array_set` lower to a load or store after the size word, an element converted to its word (a float's bits, a character extended, a pointer's address), and the others to the runtime's `merk_array_new`, `merk_array_copy`, `merk_array_resize` and `merk_array_fill`, which allocate on the heap, as `mrt.alloc` does (The heap, below). They allocate in the runtime rather than in the generated code because they fill or copy many elements, a loop that is the runtime's either way. An element's load and store are not `mrt` operations yet: `array.set` is a destructive update, which chunk 3's write barrier will need to see. `merk.check_index` calls `merk_index_out_of_bounds`, which formats the message and reports it as `merk_uncaught_exception` does; on a device it is `llvm.trap`. A set updates in place, which is safe because merkc's declarations of `array` are sound (the register, "Arrays' uniqueness"); with `-fc1 -fmmc-uniqueness` a program that keeps an old version, `A1 = A0, array.set(0, 5, A0, A)`, sees the update in `A1`, as mmc's do.
- **Terms.** Lowering first chooses each `merk.type_def`'s layout (Type representation) and erases it, and then lowers the term operations while their operands still have `merk`'s types, which name their layouts: a value is cast to its LLVM type where it is still `merk`'s, and the cast goes once the value is lowered. `merk.construct` of a constant is the constant's word, and of a constructor with arguments an `mrt.alloc` of the cell, the secondary tag's `mrt.field_store` if it has one, an `mrt.field_store` of each argument in its word, as its own LLVM type, and `mrt.tag`. A construction whose arguments are all known at compile time, constants or static cells themselves, is a static cell instead: a private constant global of its words, a field narrower than a word in its word's low bytes and padded, which the term points to with its tag, one global for each cell's contents, whatever its type, since a cell is never updated. `[1, 2, 3]` is three such globals, each pointing to the next. `merk.deconstruct` is `mrt.untag` and an `mrt.field_load` of each argument. `merk.has_functor` compares a constant's word, or the primary tag, and for a shared tag also the secondary tag, which it reads through a `select` of the cell's first word or of `merk_no_secondary_tag`, a static word that is no secondary tag, since a constant has no cell to read: one block, no branch, as a region of `merk` has one block. `merk.switch` on a term switches on the primary tag (`mrt.primary_tag`), and from tag 0 on the constants' words, unless the type has only one, and from the shared tag on the secondary tag; on a type of constants only, on the word, and on a type of one constructor of one argument it branches to its case. A type's layout is chosen alone, from its `merk.type_def`, so modules agree on it.
- **The heap.** `mrt.alloc` loads the thread's heap pointer, `merk_heap.next`, adds the cell's size, and compares it with the heap's limit, `merk_heap.limit`; if the cell fits, it stores the new pointer, and if not, which the branch's weights say is rare, it calls the runtime's `merk_alloc_slow`. The runtime (`runtime/heap.cpp`) defines `merk_heap` in thread-local storage, so that the threads of an OpenMP region each allocate on a heap of their own, with no lock; it is null before a thread's first allocation, which then takes the slow path. The slow path takes a new chunk of 1 MiB from `malloc`, wasting the rest of the old, or gives an allocation of more than 128 KiB, such as a large array, a block of its own, so that the chunk is not wasted. Nothing is freed yet. For testing, `MERK_HEAP_CHUNK_BYTES` sets the chunk's size, and an allocation of more than half of it has a block of its own: with `MERK_HEAP_CHUNK_BYTES=64 llvm-lit`, which lit passes on, every program in the tests runs with nearly every allocation on the slow path. The runtime's own allocations, arrays', bump the same heap through `merkurius::runtime::Allocate` (`runtime/heap.h`), with the same slow path, so that a collector need only hook that. A field is loaded and stored with the word's alignment. On an OpenMP device, which has no heap yet, a construction that allocates is an error at it, in the device compile, while a static cell is not. `CodeGen/heap.m` allocates many chunks' worth, a large array among cells and static cells, with mmc's output; `CodeGen/omp-heap.m` builds a list in each iteration of a parallel loop, on four threads; `CodeGen/omp-target-term.m` checks the device's error.
- **Exceptions, for now.** Until `merk.throw` and `catch` exist, an exception is not caught: `merk.throw_if` calls the runtime's `merk_uncaught_exception`, which flushes the standard output, writes "Uncaught Mercury exception:" and the exception to the standard error as mmc's programs do, and exits with status 1. The library's C code calls it too.
- **Library.** `libmerkstd` compiles `io`, `float` and `math`. `float`'s and `math`'s functions are `foreign_proc`s calling C's math library, with mmc's domain checks, as `domain_error("math.sqrt")`, and its rounding: `round(X)` is `floor(X + 0.5)`, `float.pow` multiplies by squaring, and a float out of `int`'s range converts to `min_int`. Later these should be the `math` dialect's operations and LLVM's intrinsics, which LLVM can fold and vectorize, keeping the checks; the `foreign_proc`s are how the library is written until then.
- **OpenMP target regions.** With `-fopenmp`, a goal after `%$omp target` with no clauses becomes an `omp.target`, a generic kernel, as Flang makes a target region with no loop in it (`target teams distribute parallel do` is SPMD, Language extensions, "Loops", Translation), as Flang and ClangIR lower target regions to the `omp` dialect; the other directives, but `parallel do` and `target teams distribute parallel do` before a loop (Language extensions, "Loops", Translation), are not yet implemented. The goal is generated in the region; the values it uses from outside are mapped `to`, by copy, and the variables it binds that are among its nonlocals are mapped `from`, by reference, each through a stack slot of the procedure (an `llvm.alloca`, in the target's alloca address space and cast to a generic pointer), as Flang maps a scalar: the region loads its inputs and stores its outputs, and the procedure loads the outputs after it. Only integers, floats and arrays of them cross so far (Language extensions, "Target regions"). An array is mapped as ClangIR maps a pointer's section, `p[0:n]`: by its own pointer, as `var_ptr`, with `omp.map.bounds` over its size word and its elements, words 0 to its size, which only the host compile gives, since only the host copies; the region's argument is then its pointer on the device. An array the region only reads is mapped `to`; one it gives, updated in place, `tofrom`, and after the region the variable is the given array, since the device cannot allocate one. When the region's own `merk.array_set` or `merk.array_fill` shows which array it updated, that is all; when it does not, as when the region calls a procedure that updates one, the region compares the array it gives with each given array of its type, by pointer, and gives the number of the one it is, or -1, and the pointer too, each through a slot mapped `from`; the host takes the given array of that number, or else the pointer, which is a new array only when the region ran on the host, where a procedure it calls may allocate one. The given arrays of its type are mapped `tofrom`, then, so a read-only one may be copied back unchanged. Making or filling an array in the region itself is not yet implemented, since the device has no runtime for it, and a procedure the region calls that does is an error at its call in the device compile. Arrays of strings or of arrays do not cross, since their elements are host pointers. `builtin.unrealized_conversion_cast` converts between `!merk.array` and the `!llvm.ptr` that `omp.map.info` takes; lowering removes the casts once arrays are pointers. The `omp` dialect's passes run as ClangIR and Flang run them, the first two on the `merk` module, so that a device compile lowers only what the device runs: `omp-mark-declare-target` marks the procedures the regions call, which it finds through `merk.call`, a `CallOpInterface`, on `merk.proc`, which has the dialect's `DeclareTargetDefaultModel` as `llvm.func` does (a private procedure only regions call is `nohost`, as Flang's are; the host keeps it, for running the region there); `omp-function-filter` removes the others on the device, except those that hold regions. Lowering then flattens the region's goals into blocks of the region, which ends with `omp.terminator`, keeps each procedure's mark on its `llvm.func`, and runs `omp-mark-declare-target` again, for what lowering calls, then `omp-host-op-filter`, which keeps only the regions of the host procedures that hold them, and `omp-stack-to-shared`, which, being the LLVM dialect's, run after lowering, as in ClangIR. The module records the offload attributes (`omp.is_target_device`, `omp.target_triples` and the rest, set with `mlir::omp::setOffloadModuleInterfaceAttributes`), which the `omp` dialect's translation to LLVM IR reads; the host compile, with `--offload-targets=` (or Clang's alias `-fopenmp-targets=`), makes the region an offload entry that it launches with `__tgt_target_kernel`, and without device triples it calls the outlined region directly. The device compile, `-fopenmp-is-target-device` with `-fopenmp-host-ir-file-path`, keeps only the region and the procedures it calls. The directive belongs to the caller: a target region in a procedure that a target region calls would run on the device, as reverse offloading would, so it is an error, found on the `merk` module after `omp-mark-declare-target` has marked what the regions call. On a device:
  - procedures use the C calling convention, since AMDGPU has no `tailcc` (`TargetInfo::HasTailCallConvention`); a call in tail position is still `musttail`, which AMDGPU makes a jump. Neither Flang nor ClangIR sets a convention for OpenMP device functions, which are C's; a kernel's `amdgpu_kernel` is `OpenMPIRBuilder`'s;
  - the module has AMDGPU's metadata from the target info, as ClangIR's `emitAMDGPUMetadata` gives it: the code object version, recorded on the module as `merk.amdgpu_code_object_version` (600, Clang's default and ROCm's device libraries'), which becomes the `amdhsa_code_object_version` module flag, and the PIC level 2 of Flang's device compile;
  - a failed check of `merk.throw_if`, such as a zero divisor, is `llvm.trap`, since the device has no runtime to report it, which stops the kernel and the program;
  - only the module's own procedures are compiled, so a call of another module's, such as the library's, or of foreign code is an error at the call.

  `merkc -fopenmp --offload-arch=gfx1030` runs the jobs that compile and link for offloading (Driver, "Linking and offloading"); its programs run their regions on a gfx1030 GPU, with `OMP_TARGET_OFFLOAD=MANDATORY`.
- **Tests.** `test/CodeGen` checks the `merk`, the LLVM dialect and the LLVM IR of Mercury modules in one file each, and runs the programs that print, checking their output (`hello.m`, `foreign-proc.m`, `foreign-proc-semidet.m`, `count.m`, `factorial.m`, `fib.m`, `switch.m`, and `io-output.m`, `arithmetic.m` and `float-math.m`, whose output is mmc's; `arithmetic-errors.m` checks an uncaught exception's report and exit status; `arrays.m` uses the array predicates, with mmc's output, and `array-errors.m` each of their checks' exceptions, with mmc's messages and exit status; `omp-target.m` checks a target region's host and device compiles, each offload job's output, and runs it on the host; `omp-target-gpu.test` runs it on the machine's GPU, when `amdgpu-arch` finds one and the OpenMP runtime libraries are built; `omp-target-nested.m` checks that a region cannot be in what a region calls, and `omp-target-errors.m`, `omp-target-device-call.m` and `omp-target-array-made.m` what is not implemented yet; `omp-target-arrays.m` checks arrays' maps, host and device, and runs it on the host, with mmc's output, and `omp-target-arrays-gpu.test` on the GPU, checking what each kernel copies); `Driver/compile.test` builds and runs a program, and `Driver/offload.test` checks the offload jobs. `terms.m` uses each layout, switches on constructors, semidet deconstructions and unifications of terms, with mmc's output; `heap.m`, `omp-heap.m` and `omp-target-term.m` the heap (The heap, above). `test/Dialect/Merk` checks the dialect alone: that `.mlir` modules print as they parse and lower for the target they record (`ops.mlir`, `arrays.mlir` for the array operations, and `terms.mlir` for the term operations, each layout and static cells), and its verifiers' errors, at file:line:column (`invalid.mlir`); `test/Dialect/Mrt` the same for `mrt` (`ops.mlir`, with `mrt.alloc`'s fast and slow paths, and `invalid.mlir`).

### 1. Type representation

The starting point is the scheme Melbourne Mercury uses, confirmed by inspecting the C that mmc 22.01.9 generates on x86-64 (experiment E4):

| Kind of type | Representation |
| --- | --- |
| Constants only (an enum) | A small integer, no cell |
| Constant inside a type with functors | A small integer in the word, primary tag 0 |
| Functor with arguments | Pointer with a 3-bit primary tag to a cell of just its arguments; no header word |
| More functors than primary tags | The last tag is shared; a secondary tag sits in the cell's first word |
| One functor, one argument | The argument itself, no cell (notag) |
| Small enum fields | Packed into one word |
| `int`, `float`, `char` | Unboxed, one word |
| `array(T)` | Pointer to a block of the size, a word, then the elements, one word each |
| Ground terms known at compile time | Static data, never allocated |

These cells are not self-describing: the collector and a message copy find their layout from their types (Runtime system, "Memory management", where the alternatives and the reasons for rejecting them are).

**As implemented** (2026-11-04, lowering from `merk.type_def`; Lowering, "What is lowered so far"):

- A type of constants only is a word, each constant its number in declaration order. Packing such fields into one word is not done yet.
- A type of one constructor of one argument is the argument's representation: a `temp ---> temp(celsius)`, `celsius ---> celsius(float)` is a float. A cycle of such types, which has only infinite values, is a pointer.
- In any other type, a constant is its number among the constants, in declaration order, shifted past the 3-bit tag, so its tag is 0; a constructor with arguments is a pointer to a cell of its arguments, a word each, its tag in the pointer's low bits. The constructors with arguments take the tags in declaration order, from 1 if there are constants and from 0 if not; if they need more tags than are left, the last tag is shared by the rest, whose cells' first words are their secondary tags, 0, 1 and so on. A tuple is a cell of its elements with tag 0.
- A field is at its word less the tag, so a field's load is one `getelementptr` and one load of the tagged pointer.

Two choices were made against the alternatives. Tag 0 goes to a constructor when the type has no constants, rather than always being kept for constants, so a type of eight constructors with arguments needs no secondary tag. A test of a shared tag reads the secondary tag through a `select` with a static word, rather than branching around the read, since a constant has no cell; it keeps `merk.has_functor` one operation in one block. Ground terms known at compile time are static cells (2026-11-04; Lowering, "What is lowered so far", Terms), which a collector knows by their addresses, outside every heap (decision D6 of `INPROGRESS.md`).

**Open layouts.** A type used under `open(_)` gets a second layout, used only by open values (Language extensions). One primary tag marks a variable reference, and every field holds a tagged word. `int`, `float` and `char` fields, which hold raw bits in the ordinary layout, are boxed, with static boxes for small values; enum and packed fields are re-encoded as tagged constants. A type that already uses all 8 primary tags moves one more functor under the shared secondary tag. When a type needs none of these changes, as with `tree ---> leaf ; node(tree, tree)` or `list(tree)`, the two layouts coincide and `to_open` is free. The ordinary layout never gives up a tag, so ground code is compiled exactly as before.

### 2. Calling convention

- `det` procedures return their outputs, as a struct when there are several.
- `semidet` procedures return an `i1` success flag plus outputs: the flag alone, or a struct of the flag and the outputs (What is lowered so far).
- `type_info` and `typeclass_info` arguments come first.
- Mercury relies on tail calls between procedures, not only self-recursion. Every call in tail position is emitted `musttail` with LLVM's `tailcc` convention, so the guarantee holds across modules. AMDGPU has no `tailcc`; there the convention is C's, whose `musttail` calls AMDGPU makes jumps (What is lowered so far, "OpenMP target regions"). NVPTX makes no tail calls, so it will need loops instead.

### 3. Deterministic and semidet code

- Switches become `cf.switch` on the primary tag, then on the secondary tag when needed. Those on integers and characters already become `llvm.switch`.
- If-then-else and semidet failure become branches; unifications become loads, stores and compares.
- Last-call-modulo-constructors is on by default. A recursion followed by a construction allocates the cell first and passes the address of its open field (destination passing), so list-building recursions become loops. mmc 22.01.9 does this only with `--optimize-constructor-last-call`.

### 4. Nondeterministic code

Nondet procedures use continuation passing, as mmc's high-level C back end does. A nondet procedure takes a success continuation; producing a solution calls it, and failing returns. One change makes this fit our runtime: each continuation returns an `i1` meaning "stop". A commit is a `true` returned up the chain, so there is no `setjmp`/`longjmp`, and every frame stays visible to the stack maps.

The main alternative is an explicit choice-point stack in the process heap, as mmc's low-level back end has. Continuation passing is chosen because processes have their own growable stacks (Runtime system section), so a deep native stack can be suspended and moved.

### 5. Exceptions

Mercury's `throw` and `catch` use LLVM's zero-cost exception handling: `invoke` and landing pads. Statepoints support `invoke`, so exception edges keep precise roots. An exception that leaves a process body ends the process.

### 6. Parallel loops

A parallel loop is a counted loop: induction variable, inclusive bound, step. A `do` loop under a directive is one already (Language extensions, "Loops"); the other targets (the `fold_up` intrinsics and counted recursive predicates) are rewritten into one. MLIRGen builds it as `omp.wsloop` over an `omp.loop_nest`, with `omp.declare_reduction` for reductions, as Flang lowers `!$omp do`, with no `merk` operation in between. Outlining and the libomp calls come from MLIR's OpenMP-to-LLVM translation. In version 1, a parallel body may not allocate; the compiler checks this after inlining.

This is the transformation that experiment E2 applied by hand to mmc's generated C, where it ran 2× faster on 2 cores with a bit-identical result.

### 7. GC metadata and safepoints

- Calls use LLVM statepoints, so the collector can find and relocate every live heap pointer.
- Safepoints sit at calls and at allocation slow paths. Preemption forces a slow path by lowering the heap limit, so loops that allocate need nothing more; loops that do not allocate get `mrt.yield_check` on their back edge.
- Heap cells are headerless, so every stack map also records where each polymorphic root's `type_info` lives, and the compiler keeps that `type_info` live as long as the root.

### 8. Native code

The LLVM dialect is translated to LLVM IR, optimized by LLVM, and compiled by the normal LLVM back ends. The result is linked with the runtime library into one executable.

## Driver

The compiler is one program with two layers, as Clang and Flang are. The driver reads the command line, works out the jobs and runs them. The frontend, `-fc1`, compiles one module and is what the driver runs for each module. The command line follows Clang and GCC (`-c`, `-o`, `-O2`, `-g`, `-fopenmp`), so the compiler fits Make, Ninja and CMake and behaves like the C, C++ and Fortran compilers next to it; Mercury-specific options follow the same `-f`, `-W` and `-m` conventions. `merkc` and `libmerkrt` are working names for the binary and the runtime library.

Flang's driver cannot be reused as it stands. It is Clang's driver library, `clangDriver`, running in a Flang mode: the Fortran jobs are built in `clang/lib/Driver/ToolChains/Flang.cpp` and the options live in Clang's `Options.td` and `FlangOptions.td`, and a project outside llvm-project cannot add a mode to that library. We write a small driver on LLVM's Option library, imitate Flang's frontend layer, and run Clang's own tools for linking and offloading (Reuse from LLVM and Flang).

### Command line

| Option | Meaning |
| --- | --- |
| `-c`, `-S`, `-o` *file* | Stop after the object file or the assembly; name the output |
| `-fsyntax-only` | Parse and run semantic analysis only |
| `-O0` to `-O3`, `-Os` | Choose the MLIR and LLVM pass pipelines |
| `-g`, `-gline-tables-only` | Debug information (Debugging) |
| `-fopenmp`, `-fopenmp-version=` | Read `%$omp` directives (Front end, Lexer) |
| `-freceive-goal` | Read `receive` goals (Language extensions, "Processes and mailboxes") |
| `--offload-arch=` | Compile target regions for a device (Linking and offloading, below) |
| `--target=`, `-march=`, `-mcpu=` | Target triple and CPU |
| `-I` *dir*, `-module-dir` *dir* | Where interface files are searched for, and where they are written; Flang uses the same two options for `.mod` files |
| `-W`*name*, `-Wno-`*name*, `-Werror=`*name* | Per-message control, by the identifiers in Semantic analysis, Diagnostics |
| `-L`, `-l`, `-fuse-ld=`, `-static` | Passed to the link step |
| `-MD`, `-MF` *file* | Write a Make-style dependency file listing the interface files read |
| `-mllvm`, `-mmlir` | Pass options to LLVM and MLIR |
| `-v`, `-###`, `-save-temps`, `-ftime-report` | Show the jobs as they run, print them without running, keep intermediate files, time each stage |

There are three ways in:

- **`merkc -c foo.m`** compiles one module. It reads the interface files of the modules `foo` imports, which must already exist, and writes `foo.o` and `foo`'s interface file. This is the mode for Make, Ninja and CMake; `-MD` tells them which importers to rebuild when an interface changes.
- **`merkc -o prog main.m`**, without `-c`, builds a program, like `mmc --make`. Starting from the modules named, the driver finds the modules they import, compiles what changed and links (Build mode, below).
- **`merkc -o prog a.o b.o`** only links.

### Build mode

- **Module graph from headers.** The driver reads only each module's `:- module`, `:- include_module`, `:- import_module` and `:- use_module` items, with the term reader (Front end), and finds source files by module name in the directory of the first source and the `-I` directories.
- **Interfaces before code.** Interface files are written in stages before any module is compiled, as the user's guide describes for mmc: a short interface for every module first, from its interface section alone, then the full interfaces, which need only the short interfaces of the modules imported. The reference manual does not forbid modules that import each other, and staging handles them without an order. After that, each compile needs only interface files, so all compiles can run in parallel (`-j`).
- **Rebuild only what changed.** An interface file is rewritten only when its contents change, as mmc does (user's guide), so editing the body of a predicate recompiles that module and relinks.
- **The standard library** ships compiled, with its interface files in the installation directory, which the driver adds to the search path.

### No grades

Melbourne compiles a whole program in one grade, and modules link only if all were compiled in the same grade (user's guide; Language extensions, Open terms). This design has no grades. The cost rule keeps each feature's cost in the code that uses it, so no option changes data representation or the calling convention, and modules compiled with different options link together: `-fopenmp` on one module leaves the others unchanged. Each interface file records the target triple and the compiler version, and the driver rejects a mismatch.

### Frontend (`-fc1`)

`-fc1` runs one action on one module. Its structure follows Flang's `Frontend` and `FrontendTool` libraries:

- **`CompilerInvocation`** turns the `-fc1` command line into option structures. Code generation options are listed in a `.def` file, as in Flang's `CodeGenOptions.def`.
- **`CompilerInstance`** owns the source files and provenance table, the diagnostics, the MLIR context and the output files.
- **Actions** are classes layered by how far they run, as Flang's are (`PrescanAction`, `PrescanAndParseAction`, `PrescanAndSemaAction`, `CodeGenAction`); ours start from reading the source, since there is no prescan, chosen by an `ActionKind` in `ExecuteCompilerInvocation`.
- **Diagnostics** go through our own engine (Semantic analysis, Diagnostics) with Flang's `message.h`. Flang's frontend uses Clang's `DiagnosticsEngine`, which would mean linking Clang's Basic library, so that part is not imitated.
- **Options** for the driver and `-fc1` are in one TableGen file, each marked for the driver, for `-fc1` or for both, as Clang marks its options for the driver and `-cc1`.

| Runs through | Actions |
| --- | --- |
| Source manager | `-E` (the source as the lexer reads it, which is the file unchanged), `-fdebug-dump-provenance` |
| Lexer | `-fdebug-dump-tokens`, `-fdebug-unparse-tokens` (every token's trivia and spelling, which is the file byte for byte) |
| Term reader | `-fdebug-dump-terms` (each item's term in canonical notation, or `error`) |
| Parser | `-fdebug-dump-parse-tree-no-sema` (implemented: the parse tree, one node or field per line), `-fdebug-unparse-no-sema`, `-fdebug-unparse-parenthesized-no-sema`, `-fdebug-unparse-exact` and `-fdebug-unparse-exact-minimal-layout` (implemented: Unparser and round-trip tests) |
| Desugaring | `-fdebug-dump-semantic-form` (implemented: each clause's semantic form, one goal per line; Front end, "Semantic form") |
| Name resolution, type, purity and mode checking | `-fdebug-dump-resolved-names`, `-fdebug-dump-types` (implemented: through purity checking) and `-fdebug-dump-modes` (implemented: through mode checking, with each inst definition and procedure) |
| Semantic analysis | `-fsyntax-only`, `-fdebug-unparse`, `-fdebug-dump-parse-tree`, `-emit-interface` |
| Code generation | `-emit-merk`, `-emit-mlir` (lowered to MLIR's LLVM dialect), `-emit-llvm`, `-S` and `-emit-obj` (implemented: Lowering and code generation, "What is lowered so far"; `-fc1` also reads a `.mlir` file they wrote), `-emit-mrt`, `-emit-llvm-bc` |

The unparser's exact, canonical and fully parenthesized modes (Front end, Unparser and round-trip tests) are chosen with `-funparse-mode=`. The `-fc1` layer with the source and parser actions comes first, in M1, because the round-trip tests run through it; the build mode and linking follow with code generation.

### Linking and offloading

- **Linking** goes through `clang` as the link driver. It is given the objects, the runtime library, `-fopenmp` when any module used it (so Clang links its OpenMP runtime), and the user's `-L`, `-l` and `-fuse-ld=` options. Clang supplies the C start-up files, the C library and the sanitizer runtimes, as `cc` does for rustc and GHC. Implemented so far: without `-E`, `-S` or `-c`, the driver compiles each module to a temporary object file and runs `clang`, beside `merkc` or else on the `PATH`, with the objects, `--target=` if given, and `-L` for the `lib` directory beside `bin`, where `libmerkstd.a` and `libmerkrt.a` are, `-lmerkstd`, `-lmerkrt` and `-lm`, since the library's foreign code may call C's maths library; object files on the command line are linked too. With `-fopenmp`, `clang` is given `-fopenmp` too, so it links `libomp`, as Flang's driver does for any `-fopenmp` link; `-L` names its directory when it is not beside Clang. `libmerkstd` is the standard library's modules that have an implementation so far, `io`'s output predicates, compiled by `merkc` for the host when merkc is built; so a program for another target links only if it calls none of them.
- **Offloading** follows the jobs Flang's driver runs for `-fopenmp --offload-arch=`, which are Clang's for ClangIR too. With `-fopenmp --offload-arch=gfx1030` (a comma-separated list, or repeated), each source's object file comes from:
  1. host `-fc1 -emit-llvm-bc`, with `--offload-targets=amdgpu-amd-amdhsa`, so that its regions are offload entries;
  2. one device `-fc1 -emit-llvm-bc` per GPU, with `-fopenmp-is-target-device` and `-fopenmp-host-ir-file-path` naming the host bitcode, for the GPU's triple, which has its subarchitecture as Clang and Flang give it (`amdgpu10.30-amd-amdhsa`, from `llvm::AMDGPU::getSubArch`), with no `-target-cpu`;
  3. `llvm-offload-binary`, beside `merkc` or on the `PATH`, packages the device bitcode, one `--image=` per GPU;
  4. host `-fc1 -emit-obj -fembed-offload-object=` naming the package, from the host bitcode read back with `-x ir`, as in Flang: the module's triple and data layout are the compile's, the package goes into `.llvm.offloading` as Clang's `EmbedObject` puts it, and the module is optimized and compiled.

  The link is Clang's, with `-fopenmp --offload-arch=`, which runs `clang-linker-wrapper`: it extracts the device bitcode, links it with `libompdevice` and compiles it for each GPU, adds the code that registers it with the offload runtime, and links the host with `libomp` and `libomptarget`. `-L` adds library directories for the link, such as those of the OpenMP runtime when it is not beside Clang. `-v` is passed on to the link, so that Clang shows its own jobs, such as the linker wrapper's for the GPUs' code, as Flang's `-v` does. With `-c` the jobs stop after step 4; `-E` has no offload jobs; `-S` and GPUs other than AMD's are not supported yet. Not yet done, as Flang does: linking the device libraries (`-mlink-builtin-bitcode` with ROCm's `ocml` and the rest), which device code that calls maths will need, and `-flto=full` for the device compile.

### Example

The running example (Front end) built as a program. This is the intended output, not output from a working compiler; `<install>` is the installation directory.

```text
$ merkc -### -fopenmp -O2 -o sumsq sumsq.m
 "merkc" "-fc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-obj" "-O2" "-fopenmp"
   "-I" "<install>/lib/mercury" "-module-dir" "." "-o" "/tmp/sumsq-3f9a1c.o" "sumsq.m"
 "clang" "-fopenmp" "-o" "sumsq" "/tmp/sumsq-3f9a1c.o" "-L<install>/lib" "-lmerkrt"
```

- `sumsq` imports only standard library modules, so there is one compile job; their interface files come from `<install>/lib/mercury`.
- The job writes `sumsq`'s interface file into `.`, though nothing imports it.
- Without `-fopenmp`, both jobs lose the flag, the `%$omp` line stays a comment, and the program links without the OpenMP runtime.

## Debugging

Programs carry standard DWARF debug information, so gdb, lldb, IDE debuggers, `perf` and sanitizers work on them. Melbourne's own debugger and deep profiler are non-goals; the standard tools take their place. The links to the source survive because locations come straight from the parse tree and MLIR carries them through every stage, whereas Melbourne's high-level C grades pass through C. MLIR's LLVM dialect has the DWARF attributes needed: compile units, subprograms, local variables and composite types. Flang's `AddDebugInfo` pass, which attaches them to FIR, is the model.

### What maps to what

| Mercury | DWARF |
| --- | --- |
| Module | Compile unit and source file |
| Procedure, one per mode of a predicate | Subprogram named like `add_square/4 (in, in, in, out)`, with a mangled linkage name |
| Goal | Line-table rows at the location of the parse-tree node it came from |
| Variable | Local variable under its source name. The numbered versions that desugaring makes of a state variable, such as `Acc_0` and `Acc_1` for `!Acc`, are one variable, `Acc`, with a different value at different points |
| `int`, `float`, `char` and the sized integers | Base types |
| Any other type | A word described as an opaque value named by its Mercury type, and printed by a debugger script |
| A suspended process's stack | Unwound by a debugger extension; DWARF alone cannot find it |

### Three tiers

1. **Line tables (M1).** Every `merk` operation already carries the location of its parse-tree node (Semantic analysis, Mapping to `merk`). Lowering keeps locations through rewrites, using fused locations when it merges operations, and gives each procedure a subprogram. MLIR's `ensure-debug-info-scope-on-llvm-func` pass creates minimal subprograms for exactly this. This gives breakpoints by Mercury file and line, stepping, Mercury backtraces, and source lines in `perf` profiles and sanitizer reports.
2. **Variables (M2).** MLIR generation ties each SSA value to its Mercury variable through debug-value records, using the `NameLoc`s it records. Primitive types are described directly.
3. **Terms and processes (M4).** A term is a tagged word whose low bits select the constructor. DWARF has no ready way to describe that, since its tagged unions select on a field's value, and MLIR's debug attributes do not cover them anyway. So, as Go and Rust do, we ship debugger scripts: LLDB data formatters and GDB pretty-printers that call a runtime routine to print a term from its type descriptor, the same descriptors `io.write` uses. A polymorphic variable's script finds its type info in the hidden type-info variable beside it. A second extension lists processes and unwinds the stacks of suspended ones, which no OS thread holds; Go's `runtime-gdb.py` does the same for goroutines.

### Example

The running example (Front end) compiled without `-fopenmp`, with a breakpoint in `add_square`. This is the intended session, not output from a working compiler; line numbers are those of the fragment.

```text
(lldb) breakpoint set --file sumsq.m --line 10
Breakpoint 1: where = sumsq`add_square/4 (in, in, in, out) at sumsq.m:10
(lldb) run
   9   add_square(A, I, !Acc) :-
-> 10      array.unsafe_lookup(A, I, X),
   11      !:Acc = !.Acc + X * X.
(lldb) frame variable
(array(int)) A = array([1, 2, 3, 4])
(int) I = 0
(int) Acc = 0
(lldb) thread backtrace
  * frame #0: add_square/4 (in, in, in, out) at sumsq.m:10
    frame #1: int.fold_up/5 (in(pred(in, in, out) is det), in, in, in, out)
    frame #2: sum_squares/3 (in, in, out) at sumsq.m:5
    frame #3: main/2 (di, uo)
(lldb) next
-> 11      !:Acc = !.Acc + X * X.
(lldb) frame variable X Acc
(int) X = 1
(int) Acc = 0
```

- **`A` prints as a Mercury value** through the term formatter, while `I`, `X` and `Acc` are base types.
- **`Acc` is one variable.** At line 10 it is `Acc_0`; after line 11 it would be `Acc_1`, the procedure's result.
- **Execution order.** Line 11 holds two goals, `#3` (`V_1 = X * X`) and `#2` (`Acc_1 = Acc_0 + V_1`), run in the order the `ProcInfo` gives. `V_1` is a compiler-made variable and is not shown.
- **With `-fopenmp`,** the calls of `add_square` run on OpenMP worker threads, and their backtraces end in the outlined parallel region of `sum_squares`, as in C and Fortran.

### Mercury-specific behaviour

- **Stepping follows execution order.** Mode analysis reorders conjunctions, so stepping can jump between lines even without optimization, much as in optimized C. The order is part of the program's meaning, so it cannot be turned off; the line table makes the jumps explainable.
- **Nondeterminism.** Continuation passing (Lowering, Nondeterminism) puts continuation frames in backtraces. A debugger command that lists live choice points is a later addition.
- **Tail calls.** Loops written as tail recursion leave no frames, as loops in C leave none.
- **The moving collector.** The statepoint rewrite must keep variable locations right for the pointers it relocates. This is a risk to test early, with GC torture mode switched on.
- **Optimized code** degrades as in any LLVM compiler: variables may show as optimized out, and inlined procedures appear as inlined frames.

## Runtime system

The runtime gives each process its own stack, heap and mailboxes, schedules processes M:N over one worker thread per core, and never stops every process at once. It is written in C++, like the compiler, and linked into every executable. So far `libmerkrt` (`runtime/`) is only the program's C `main`, which calls `merk_main` and returns exit status 0, `merk_uncaught_exception`, which reports an exception and exits with status 1, and the array functions, which allocate with `malloc` and never free, and report an index out of range (Lowering and code generation, "What is lowered so far"); output goes through C's stdio, which is flushed when the program exits.

### Processes and scheduler

- **Process record:** identifier, stack, heap, mailboxes, links and monitors, reduction budget, scheduling state.
- **Stacks:** each process has a small native stack that grows by copying. The precise stack maps that the GC needs also let the runtime relocate pointers into a moved stack. Suspension can therefore happen anywhere: in `receive`, at a preemption check, or at a GC.
- **Scheduling:** one worker thread per core, each with a run queue, plus work stealing.
- **Preemption:** rides on the nursery limit check, as GHC does. To preempt a process, the scheduler lowers its heap limit, so the next allocation takes the slow path and yields there. Code that allocates pays nothing extra. Loops that do not allocate get an `mrt.yield_check` on their back edge, which spends a reduction budget; that check is the only preemption cost left.
- **Blocking work:** I/O never blocks a worker. File and socket I/O use the operating system's asynchronous interface where possible, and a separate pool of threads otherwise, as Erlang's dirty schedulers do.
- **Timers:** each worker keeps a timer wheel for receive timeouts.

### Message passing

- **Mailbox handle:** process identifier, mailbox index and generation number. Handles are values, never pointers into another heap.
- **Send:** find the receiver, deep-copy the message into a fresh heap fragment, and push the fragment onto the mailbox's lock-free multi-producer queue. Wake the receiver if it is waiting. The receiver's heap is never locked.
- **Arrival order:** each message is stamped, as it is queued, with the receiving process's arrival count, taken with one atomic increment by the sender. The stamps order messages across all of a process's mailboxes.
- **Receive:** in each mailbox of the selector, find the oldest message its case accepts: the head of the queue for `selecting`, or the first message for which the selector succeeds, scanning in arrival order, for `selecting_matching`. Take the candidate with the lowest stamp. The matched fragment joins the receiver's heap and is merged at its next GC. Unmatched messages keep their order. A process waiting in `receive` is woken by a send to any of its mailboxes.
- **Mailbox lifetime:** a mailbox lives until `close_mailbox` or the owner's exit. Closing it drops its queued messages and makes its handle dead.
- **Dead mailboxes:** a send to a handle whose generation no longer matches, or whose process has exited, is dropped. This is what discards late replies. Generations make handles of closed mailboxes dead forever, even when the mailbox's slot is reused.
- **Cheap mailboxes:** every `request` creates one, so creating and closing a mailbox must cost about as much as a small allocation: a slot in the owner's mailbox table and a new generation number.
- **Observability:** a process can read its own queue lengths, since non-blocking send gives a slow consumer no backpressure.
- **Later:** remember a scan position for repeated selective receives on the same reply, as BEAM does.

### Memory management

- **Per-process heap:** a bump-pointer nursery and an older generation. Minor collection copies survivors out of the nursery; major collection compacts the old generation.
- **Collection runs in the owning process,** at its safepoints, with no global pause.
- **Roots:** the process stack (from stack maps), message fragments received but not yet merged, and the process record.
- **Write barrier:** destructive updates (array `set`, mutables, `store`) record old-to-young pointers in a remembered set. Immutable terms need no barrier.
- **Large objects** (big arrays, long strings) live in a separate space and are never copied by minor collections.
- **Process exit** frees the whole heap at once.
- **Static data** (ground terms, type descriptors, string literals) is outside every heap and never traced.
- **So far** (2026-11-04): a heap for each thread rather than each process, in chunks that are never freed, and large blocks of their own (Lowering, "What is lowered so far", The heap). Where a process's heap pointer and limit live is decision D2 of `INPROGRESS.md`, made with chunk 2's context switch; until then they are thread-local, which keeps the threads of OpenMP regions apart, and only `mrt.alloc`'s lowering and `runtime/heap.h` read them.

**How the collector finds pointers: by type (decided 2026-11-04).** Heap cells have no header, as in the type representation above. The collector and the message copy know a cell's layout from its type: the type of each root comes from the stack maps, which record where each polymorphic root's `type_info` is, and the type of each field from the type descriptors. Forwarding marks during a copy live in a side bitmap, one bit per heap word, and the to-space is scanned through a work stack of typed fields, since it cannot be scanned linearly.

Two alternatives were measured in a C model (experiment E7) and rejected:

- **A header word per cell**, giving its size, which fields are pointers and its constructor, was the first recommendation: a simple Cheney collector and deep copy that need no types, a self-describing heap for debuggers, and room for the secondary tag. It was rejected for its size. It makes a list cell 24 bytes instead of 16, a tree node 32 instead of 24, and a list of pairs 48 instead of 32 per element, and that cost is paid by every program on every traversal once its data leaves a cache: +16% on a large list, +50% on a large tree, +45% on a list of pairs, up to +50% on building, +11% on nrev and +17% on polymorphic nrev, where the header's pointer bits come from a `type_info` at run time. Its advantages are engineering effort, paid once; its cost would be paid on every run.
- **Pages of cells of one shape** (a big bag of pages), with the shape in the page table, keep the headerless size and need no types to trace, and segregating cells by shape even made a list of pairs 22% faster to sum. They were rejected because polymorphic code knows a cell's shape only at run time, so it must choose among bump pointers there (+36% on polymorphic nrev), and because every heap needs a page for each shape it uses, which with a heap per process is the wrong cost for many small processes.

What the choice costs instead: a `type_info` must be findable for every polymorphic root at every safepoint; the tracer is type-directed, with descriptors for every type; and `send`, which must not overwrite the sender's cells, copies by the message's `type_info`, which it receives as a hidden argument anyway. `mrt` hides the layout, so the decision can be revisited: the nrev gate of chunk 1 measures it again on our own compiler.

### Type descriptors

Static type descriptors describe each type's constructors, names, arguments and layout. Generic printing (`io.write`), `compare`, `unify` at polymorphic types, `univ`, the collector and the message copy all use them.

### Exceptions, links and monitors

- `throw` and `catch` work within a process.
- An uncaught exception ends the process, with the exception as its exit reason.
- A link is two-way: the death of one process delivers an exit message to the other. A monitor is one-way and delivers a `down_info` message. Both go to a mailbox the watcher chose, through a function into that mailbox's message type (Language extensions, "Processes and mailboxes").
- Monitoring a process that has already exited delivers a `down_info` message at once, so `request` cannot miss a server's death.
- Supervisors are library code built on monitors.

### Input and output

Each process threads its own `io` state. The standard streams are shared, with internal locking.

### Foreign code

Melbourne Mercury's `foreign_proc` code assumes heap pointers never move. With a moving collector, two kinds of foreign procedure are allowed:

- Those marked `will_not_call_mercury` that do not allocate get raw pointers; no safepoint can occur inside them.
- All others receive handles, which the collector updates.

Foreign code written for Melbourne Mercury therefore needs review before it runs on this runtime.

### OpenMP cooperation

- One scheduler owns the cores. An OpenMP region borrows that process's worker plus idle workers, rather than libomp starting its own pool.
- Version 1: parallel loop bodies may not allocate, so no safepoint occurs inside a region.
- Version 2: per-thread allocation buffers inside the process heap, and the process's collector stops its region at a barrier before collecting.

### Support for open terms

Open terms need new runtime functionality, listed below. None of it runs, or takes space, in a program that never uses `open(_)`: these routines live in their own part of the runtime library, and ground code never calls them.

- **Variable cells.** An unbound variable is a standalone one-word cell in the process heap that refers to itself, with the open layout's variable tag. Fields and registers hold references to it, and binding overwrites it. Because variables are separate cells, every heap pointer still points at the start of a cell, so the moving collector needs no interior pointers. Binding never writes into a constructor cell either, so a cell that `to_open` shares with ground code is never changed.
- **Binding scheme: WAM-style reference chains.** A binding writes one cell and one trail entry; a read follows the chain to its end, and the collector shortens chains as it copies (below). PARMA cycles, as HAL used on Mercury, were the alternative: reads take at most one hop, but a binding writes and trails every cell in the cycle. Chains are simpler and fit the trail and collector described here.
- **Unification.** The compiler generates a unify procedure for each open type. The runtime supplies a generic one that walks type descriptors, for polymorphic code, and the occurs-check variant behind `unify_oc`.
- **Value trail.** Each process gets a trail of (address, old value) pairs, allocated on first use. A binding is trailed only while a choice point or `try` that can bind open terms is live, which costs one compare against a per-process counter. The mode system tells the compiler which those are, because only goals with an `any` variable can bind. Each one saves the trail top; backtracking, and catching an exception as ISO Prolog's `catch/3` does, undo entries down to it; when the last one is gone, the trail is emptied. E5 found an unconditional trail push per binding within noise, so the first runtime skips the WAM's age test and the way it interacts with a moving collector.
- **Collector.** Trail entries are roots and are updated when cells move. The collector recognises variable cells by their tag and short-circuits references to bound variables whose binding is not on the trail, as WAM collectors collapse reference chains. A copying collector handles cyclic terms through its forwarding pointers.
- **Type descriptors.** Each type used under `open(_)` gets a descriptor for its open layout. Generic unify, `from_open`, printing and the collector use it.
- **Conversion.** `from_open` walks the term with cycle detection, fails at the first unbound variable, and copies only if it met a variable cell. `to_open` copies when the two layouts differ.
- **Printing.** `io.write` prints unbound variables as `_1`, `_2` and so on, numbered within one call, and stops on cyclic terms.
- **Message copying.** The copier follows reference chains and copies bound variables as their values. It creates one fresh variable for each unbound source variable, keeping an old-to-new map during the walk, either a forwarding word in the source cell that it restores afterwards or a small hash table. It marks visited cells, so cyclic terms terminate. Message types that cannot contain `open(_)` keep a copier with no variable checks.
- **Later, optional: attributed variables** for `freeze`, `when` and constraint solvers. An unbound variable cell would point to an attribute record, and binding it would wake the attached goals. No second tag is needed, because the cell's contents distinguish a plain variable from an attributed one. This is how open terms would grow into Mercury's solver types.

**Who pays**

| Mechanism | Cost | Paid by |
| --- | --- | --- |
| Variable tag | One primary tag | Open layouts only; a type that uses all 8 tags moves one functor to a secondary tag there |
| Boxed raw fields | One box per `int`, `float` or `char` field; none for small values with static boxes | Open values |
| Dereferencing | A tag test per field read, plus a loop for chains | Open code |
| Trail | Two words per binding | Bindings made while an open choice point or `try` is live |
| Trail top | One load and one store | Choice points and `try` whose goals have `any` variables |
| Variable-aware copying | An old-to-new variable map during a send | Messages whose type can contain `open(_)` |
| Descriptors and routines | Static data and code | Programs that use `open(_)` |
| Nothing | — | Ground code |

## Distribution and wire format

Processes on different machines exchange messages with the same `send` and `receive`. A local send copies the message straight into the receiver's heap. A remote send encodes it into bytes, and the receiving node decodes them into the receiver's heap, so local sends never pay for encoding. Distribution is designed now, because handle and wire formats are hard to change once programs depend on them, and built in milestone M6, after processes.

### Wire format

Messages use a self-describing, pointer-free encoding modelled on Erlang's external term format, which Erlang uses for both `term_to_binary` and distribution. The sender walks the message with its type info and writes it in order, with no addresses; the receiver allocates cells as it parses. The encoding does not depend on word size, endianness or tag layout. The receiver decodes with its own type info for the mailbox's message type, which validates the input as it goes, so a malformed or hostile message is rejected rather than turned into bad pointers. Three additions to Erlang's format:

- **Variables.** The first occurrence of an unbound variable is encoded as "new variable k", and later ones as "variable k". The receiver creates a fresh variable when it first meets k, so sharing among variables survives without patching (Open terms, copy semantics).
- **Back-references.** Erlang's format has no tag for shared subterms, so shared data is expanded. Erlang loses sharing even in local sends: its efficiency guide shows a 22-word structure growing to 4,094 words. A back-reference tag keeps sharing when the term is cyclic, as open terms can be, or when the sender asks for it.
- **Constructors by type.** A constructor is encoded as a type index and a functor number, instead of an atom (below).

### Types instead of atoms

Mercury has no atom table. A Prolog-style atom is a constructor of a declared type, compiled to a small integer that means something only within that type (E4). At connection setup, each side sends a table of the types it will exchange: the name, a fingerprint of the definition, and the functor names in order. The receiver builds a translation array for each type, from the sender's functor numbers to its own; when the fingerprints match, it is the identity. Decoding a constructor is then an array lookup, and a type or functor the receiver does not know rejects the message.

- **Versions.** Matching functors by name tolerates constructors that were added or reordered between versions. Changing a constructor's argument types needs a new version of the type, known to both sides during a rollout. Hot code loading is a non-goal, so there is no in-place upgrade to support.
- **No table to exhaust.** The type tables are fixed at compile time, so a remote peer cannot grow them. Erlang's atom table is never garbage collected, which is why Erlang code must not create atoms from untrusted input.
- **Strings** travel as data and are not interned.

### Opaque payloads at relays

A dispatcher never looks inside an existential payload; it can only forward it (Generic processes and dispatchers). So a node that decodes an envelope whose hidden type is not in its own tables keeps the payload encoded: the bytes as they arrived, with the type's name and fingerprint. Forwarding copies those bytes into the next message, and only the destination decodes them.

- Relays need only the envelope types, so endpoints can add payload types without rebuilding relays.
- Nodes that do know the type can skip the decode and re-encode as well.
- Generic operations on an opaque value, such as `io.write` or equality, either work from the encoded form or report that the type is unknown on this node. With a canonical encoding, equality of ground payloads is byte equality.

### Remote mailboxes, sendability and failure

- **Handles.** A mailbox handle carries a node identifier besides the process, mailbox and generation numbers. Within a node, handles cannot be forged. Across nodes they must be unguessable, or anyone who can reach a node could send to it, so remote handles include a random secret. This is stronger than Erlang, where any connected node that knows the shared cookie can do anything.
- **Closures** contain code addresses, so they can cross only between nodes running the same binary, which the handshake checks, encoded as procedure references. The compiler cannot tell a local mailbox from a remote one, so sending a closure to a node with a different binary raises an exception at the send.
- **Failure.** Links and monitors work across nodes. Losing a connection delivers an exit or `down` message with a "node down" reason to everything watching a process on that node. The delivery guarantee is an open question; Erlang promises ordering between each pair of processes, but not delivery.

### Remote spawn

`spawn/4` takes a closure, and closures cross only between nodes running the same binary, so a process on a node running another program is spawned by its kind's name instead, with `spawn_counter_on` from the `:- process` declaration (Processes and mailboxes).

- **Only the declaration is needed.** A program that imports a module with `:- process counter(int) receives counter_msg` in its interface can spawn counters on another node without containing the code of `counter`. The node that runs it must contain the definition, registered under the kind's module-qualified name.
- **Fingerprint.** A process kind has a fingerprint of its name, its argument and message types, and the fingerprints of the types reachable from them (Processes and mailboxes, "No `sends` clause"). That set is the table of types the two nodes exchange for it. The spawning node sends the name, the fingerprint and the arguments, encoded in the wire format. The other node checks the fingerprint against its own kind of that name before it decodes the arguments.
- **Nodes.** A `node` names a node this one is connected to, from the library's connection API. That API, and how nodes find each other, are still to be designed.
- **Arguments** must be sendable, and closures among them only between nodes running the same binary, as for any message.
- **Result.** `spawn_counter_on` returns `ok(Mailbox)`, with remote handles, or `error(E)`, where `spawn_error` is `node_down`, `unknown_process` or `version_mismatch`. With several mailboxes, the `ok` value is a tuple of them, in the declaration's order.
- **Versions.** Since functors are matched by name, adding a constructor to a message type keeps the kind compatible, as for any message (Types instead of atoms). Changing an argument type, or a constructor's argument types, makes a new version of the kind, which both nodes must know during a rollout.
- **Mercury on both sides.** Processes are written in Merkurius only. A process that needs code in another language calls it through the foreign language interface, inside the process, so the wire format needs no schema for other languages and no foreign peer has to implement it.

### Remote calls

Remote procedure calls need no new mechanism. They are request and reply over `send` and `receive`, exactly as between local processes, and they work across nodes because mailbox handles do.

- **Blocking:** `request` monitors the server, sends the request with a fresh reply mailbox and waits for the reply, returning `reply(R)`, `timeout` or `down(Reason)`.
- **Non-blocking:** `send_request` does only the first half and returns the reply mailbox. The caller carries on, then collects the reply later with `receive`, with `after(0)` to poll or with a real timeout when it needs the answer, or adds the reply mailbox to the selector of its main loop. This is the same pattern as Erlang's `erpc:send_request`.

```mercury
:- pred send_request(mailbox(Req)::in, (func(mailbox(Rep)) = Req)::in,
    mailbox(Rep)::out, io::di, io::uo) is det.

send_request(Server, MakeRequest, ReplyTo, !IO) :-
    new_mailbox(ReplyTo, !IO),
    send(Server, MakeRequest(ReplyTo), !IO).

request(Server, MakeRequest, Timeout, Reply, !IO) :-
    new_mailbox(Downs, !IO),
    monitor(Server, Downs, (func(D) = D), Ref, !IO),
    send_request(Server, MakeRequest, ReplyTo, !IO),
    Sel = selecting(ReplyTo, (func(R) = reply(R)),
        selecting(Downs, (func(D) = down(reason(D))), new_selector)),
    receive(Sel, Timeout, MaybeReply, !IO),
    demonitor(Ref, !IO),
    close_mailbox(ReplyTo, !IO),
    close_mailbox(Downs, !IO),
    (
        MaybeReply = yes(Reply)
    ;
        MaybeReply = no,
        Reply = timeout
    ).
```

- **Replies cannot mix.** Each call has its own reply mailbox, so replies to different calls stay apart, and a reply that arrives after a timeout lands in a mailbox nobody reads.
- **Failure.** The monitor on the server turns a lost node, or a server that dies, into a `down` message, and `request` waits for the reply and the `down` message together, so it returns as soon as either arrives.
- **Later, if wanted:** a `future(T)` library that wraps the reply mailbox and resolves it on first read; syntactic sugar as a front-end rewrite into these calls; and promise pipelining, as in E and Cap'n Proto, which would need support in the wire format. None of these is needed for remote calls to work.

Both predicates pass mmc 22.01.9's type, mode and determinism checks against the API in Language extensions, "Processes and mailboxes".

### Security

Connections between nodes are encrypted and authenticated, as with TLS. For data that must stay secret from relays, a library type `sealed(T)` holds ciphertext tagged with its plaintext type:

```mercury
:- func seal(public_key, T) = sealed(T).
:- pred unseal(secret_key::in, sealed(T)::in, T::out) is semidet.
```

- **Forwarded like any payload.** `sealed(T)` is ordinary sendable data, so dispatchers and relays forward it like any opaque payload. The type parameter stops a `sealed(order)` from being unsealed as a `payment`.
- **Type hidden from relays.** The payload's type name and fingerprint go inside the ciphertext, not in the per-connection tables, so relays learn nothing about the type. After decrypting, the receiver checks them against `T` before decoding.
- **Authenticate, then decode.** `seal` uses authenticated encryption, and `unseal` verifies before the decoder sees a byte. It fails on a wrong key or a tampered message.
- **Keys in handles.** A remote mailbox handle can carry its owner's public key, so sends to it can be sealed automatically.
- **What stays visible:** sizes, timing and routing fields. Padding hides sizes where that matters.
- **No home-made cryptography.** Ciphers come from an established library such as libsodium.

## Optimizations

Most general-purpose optimization comes from LLVM. The Mercury-specific work that matters most removes allocation, polymorphism and closure overhead, and exposes loops for parallelization.

| Optimization | Level | Notes | Priority |
| --- | --- | --- | --- |
| Guaranteed tail calls | LLVM | `musttail` with `tailcc`; Mercury expresses loops as recursion, so this is correctness, not speed | Required |
| Standard pipeline: inlining, CSE, constant propagation, LICM, vectorization | MLIR canonicalization, LLVM | Comes from upstream | Required |
| Last-call-modulo-constructors | `merk` to `mrt` | Destination passing turns list-building recursion into loops; on by default | High |
| Higher-order specialization | `merk` | Closures become direct calls. In mmc this turned `fold_up` into a local loop calling the body directly | High |
| Type specialization | `merk` | Removes `type_info` passing and boxing; automatic at `-O2`, plus the `type_spec` pragma (below) | High |
| Intermodule inlining and specialization | Interface files | Bodies of small exported procedures travel in the optimization section | High |
| Parallel loop formation | `merk` | From `%$omp` on `fold_up`, array iterators and counted recursion | High |
| Switch optimization | `merk`, `cf` | Jump tables, lookup tables, string switches | Medium |
| Escape analysis | `merk` | Cells that never leave a procedure go on the stack, saving nursery traffic | Medium |
| Accumulator introduction | `merk` | Uses associativity `promise` declarations, as mmc has since 1999 | Medium |
| Deforestation, unused-argument and dead-procedure elimination | `merk` | Classic Mercury passes | Later |
| Structure reuse (compile-time GC) | `merk` | Reuse dead cells in place, after Mazur's analysis; off by default in mmc | Later |
| Move instead of copy on send | `merk`, runtime | Messages passed in a `di` mode skip the deep copy | Later |
| Remembered scan position in selective receive | Runtime | Avoids rescanning old messages when waiting for one reply | Later |

### What polymorphism costs

The same membership test was written five ways and run with mmc 22.01.9 (`hlc.gc`): a 1,000-element `list(int)` searched 200,000 times for a missing value, best of five runs. A 1,000,000-element list searched 200 times showed the same pattern, 0.41 s against 1.22 s.

| Version | Time |
| --- | --- |
| Monomorphic, `int` only | 0.32 s |
| Polymorphic with `pragma type_spec(T = int)` | 0.32 s, the same code as the `int` version |
| Type class: a method call through a dictionary | 0.32 s |
| Higher-order: a closure call for each element | 0.32 s |
| Polymorphic, `X = Y` on `T` | 1.00 s |

- **Passing a `type_info` is nearly free:** one more argument, in a register.
- **Indirect calls are cheap in such a loop.** Every call goes to the same target, so branch prediction handles it, and the loop waits on loading the next cell anyway. What they cost is inlining and vectorization, which matter in numeric loops.
- **The 3× comes from generic operations.** In the polymorphic version `X = Y` is a call to the builtin `unify` with the `type_info`, which inspects it to find how to compare. `compare`, used for every `map` and `set` key, hashing and deep copy work the same way, and so do `send`'s deep copy and the wire format, which walk a message by its type.

Two rules follow.

- **Specialization is automatic.** At its default options mmc did not specialize the polymorphic version; it needed the `type_spec` pragma. At `-O2`, when a call passes a `type_info` or `typeclass_info` known at compile time, merkc clones the callee for that type, within a module and across modules through the optimization section of interface files: C++ templates, with the compiler choosing the instances. Polymorphism analysis already finds these calls, those whose `type_info`s and `typeclass_info`s are all constants and cells of constants (Semantic analysis, "Polymorphism analysis"). A user who cares about speed then writes ordinary polymorphic code, and the pragma stays for a type the compiler cannot see, such as one a caller in another program uses.
- **Failures are visible.** If a `%$omp` loop still contains a generic `unify`, a generic `compare` or a closure call that specialization did not resolve, merkc emits a remark naming it, as Clang's `-Rpass-missed` does. Inside a target region it is an error (Language extensions, "Target regions").

Some code stays dynamic: existential types, such as the generic dispatchers, `univ`, and data structures that store closures. That is the same as virtual calls and `std::function` in C++, and under the cost rule only code that uses those features pays for them.

**For later:** packed arrays, such as `array(int8)`, or `array(float32)` if that type is added, would make polymorphic code over `array(T)` read the element size from the `type_info`. Specialization removes that cost in hot code too.

## Standard library, compatibility and testing

The language target is the Mercury 22.01 reference manual. The standard library is the main compatibility cost, because all of it is reimplemented.

**Language version**

- Version 1 implements 22.01 semantics.
- Development snapshots since 22.01 change some things. These are tracked as a separate list and adopted deliberately. The lexical ones, from the current manual's Syntax chapter:
  - **Inverted operator priorities.** A renumbering only (new = 1500 − old), so it is adopted (Front end, Term reader).
  - **The `\e` escape,** for ESC (U+001B), in strings and quoted names. Not adopted: it is an error, so that our test programs still compile with mmc 22.01.9.
  - **Underscores before an exponent,** as in `1.5_e10`. Accepted, because mmc 22.01.9 already accepts it after a fraction (it reads `1.5_e10` as 1.5e10) although the 22.01 manual forbids it. mmc rejects `2_e3`, with no fraction, and so do we.
  - **`<<u` and `>>u`,** unsigned shifts. Not adopted: under 22.01, `A <<u` is `A << u`, as mmc confirms, so the change is not purely additive, and the current manual does not say how `<<uint` splits.
  - Submodule visibility rules also changed; they are not lexical.

**Standard library**

- The whole library is reimplemented from the library reference manual, so the project controls its behaviour, performance and licence. Melbourne's library, LGPL-2 with a linking exception, is neither read nor shipped (Clean-room rules).
- Runtime-coupled modules are reimplemented clean-room from the library reference manual: `builtin`, `int`, `float`, `char`, `string`, `array`, `io`, `exception`, `time`, `benchmarking`, and processes in place of `thread`. Their behaviour is tied to a runtime that is new here, and their sources are not read (Clean-room rules).
- **Unique arrays.** The manual's `array` module declares its unique insts as `ground`, `:- inst uniq_array(I) == array(I).  % XXX work-around`, because, its comment says, "the current Mercury compiler doesn't support `ui` modes", and it takes the array `in` in every predicate that reads one; its header warns that "arrays are currently not unique objects". So mmc checks no array's uniqueness (see the register). merkc's library has two stubs: `array.m`, the manual's declarations, and `sound/array.m`, which `library/make-sound-array.py` makes from it at build time, with the insts the manual's comments give, `uniq_array(I) == unique` and `mostly_uniq_array(I) == mostly_unique`, and each mode that takes arrays `in` preceded by copies that take them `array_ui`, every mix, those with more first, so that reading a unique array keeps it unique, as copied values make possible (Uniqueness checking). The `int` stub has the manual's `array_di`/`array_uo` modes of `fold_up`, `fold_down` and their variants, and so imports `array`. merkc searches `lib/merkurius/sound` first unless `-fmmc-uniqueness` is given (`library/README.md`).
- Pure-Mercury modules (`list`, `map`, `set`, `cord` and others) are reimplemented the same way. Where the manual leaves behaviour unstated, such as which exceptions are thrown, exact output formats, unstated orderings and performance, differential tests against mmc decide.

**Compatibility promise**

- Source compatibility with Melbourne Mercury for the supported language.
- No binary or ABI compatibility.
- Unsupported features (see Non-goals) are rejected with a clear error, never silently miscompiled.

**Testing**

- **Differential testing:** compile each test program with mmc 22.01.9 and with our compiler, and require identical output. Melbourne's test suite is run as an external oracle, not copied into our repository.
- **Pass tests:** unit tests per pass, and FileCheck tests for every dialect and lowering.
- **Diagnostics tests:** expected type, mode and determinism errors, such as the missing-constructor error in Language extensions. To do: a test whose expected errors are `ERR-DAG` lines, checked with `--implicit-check-not=error:`, misses an extra error that falls between two of them, as FileCheck checks the implicit pattern only outside the `DAG` group's matches. Such tests should also check the number of errors, for example with a RUN line that counts them (`grep -c error: %t.err`, compared with the number of `ERR-DAG` lines). Pass 6 found two that passed this way, `uniqueness/backtracking.m` and `scopes.m`; an audit of the error counts then found no others.
- **Runtime stress:** a GC torture mode that collects at every safepoint, and randomized scheduling.
- **Benchmarks:** the four baseline programs (fib, naive reverse, 12-queens, float loop) already exist in C, Mercury, SWI-Prolog and Python. Add the classic Prolog suite (deriv, tak, crypt, poly), parallel-loop scaling runs, and process benchmarks: ring, ping-pong, spawn rate, receive latency. The first parallel-loop benchmark is `benchmarks/bandwidth`, BabelStream's five kernels (copy, mul, add, triad and dot) under `target teams distribute parallel do`, built sequentially, for the host's threads and for a GPU; its GPU numbers measure the maps of each kernel's arrays until there are `target data` regions (benchmarks/bandwidth/README.md).

## Phased plan

Seven milestones take the project from today's experiments to processes and parallel loops in one program, and then across machines. Each ends with a working compiler and a gate measured against the Experiments baseline.

&#91;embedded content: phased plan · 6 milestones, each with its gate\]

M3 and M4 both build only on M2, so their order can be swapped if processes matter more than parallel loops. M6, distribution, builds on M4 and can start before M5. M1 begins with the front end and its unparser, whose round-trip tests then guard every later change to them.

## Open questions and risks

Seven decisions remain open. The two that most constrain the runtime are the nondeterminism strategy and the stack model, so they come first. The designs being worked out for them, and the plans for the pieces under way, are in `INPROGRESS.md`; each moves here when it is decided and implemented, with the options it rejected and why. The ones ticked at the end were decided between 2026-10-27 and 2026-11-04.

**Open questions**

- [ ] Nondeterminism: continuation passing with stop flags, or an explicit choice-point stack.
- [ ] Process stacks: grow by copying, or segmented.
- [ ] When to adopt language changes made after 22.01.
- [ ] Contact with the Mercury team, as a courtesy and for questions the documents leave open. Their `LICENSE` file offers alternative arrangements if the licences do not suit.
- [ ] Sharing in local message copies: keep shared ground subterms, which needs a visited map, or expand them as Erlang does. Variable identity is always kept.
- [ ] Remote delivery guarantees: ordering per pair of processes, as in Erlang, and at-most-once or stronger delivery.
- [ ] Trust between nodes: how nodes authenticate each other and how mailbox keys for `sealed(T)` are published.
- [x] Heap cells: headerless, traced by type, as Melbourne's; a header word and pages of one shape rejected for their measured costs (Runtime system, "Memory management"; experiment E7).
- [x] Compiler implementation language: C++, for direct use of MLIR, the LLVM data structures and TableGen.
- [x] Standard library: reimplemented in full from the library reference manual, for full control.
- [x] Links and monitors: required, and delivered with processes in M4.
- [x] Open-term bindings: WAM-style reference chains (Runtime system, "Support for open terms").
- [x] Parallel annotation syntax: the comment sentinel `%$omp`, attached to the next goal (Front end, "OpenMP directives").
- [x] Prescanner: none. Source files are read unchanged, and the lexer recognizes `%$omp` directives and line number directives, which depend on token context (Front end, "Source manager").
- [x] Lexical rules: 22.01's, with the current manual filling its gaps; where mmc 22.01.9 disagrees with its manual, mmc is followed; of the later changes only the operator renumbering is adopted (Front end, Lexer; Standard library, compatibility and testing).
- [x] Receiving from several mailboxes: selectors, as in Gleam, taking the oldest matching message across mailboxes, as Erlang's receive does; priority by Erlang's two-step receive (Language extensions, "Processes and mailboxes").
- [x] Target regions: only scalars, arrays of scalars and tuples, by value, cross into one; map types follow from modes; array sections, `A(Lo .. Hi)`, are inclusive and keep the array's indices (Language extensions, "Target regions").
- [x] Receive syntax: a `receive` goal whose arms each `select` a message by mailbox and pattern, as sugar over the selector API, behind `-freceive-goal` (Language extensions, "Processes and mailboxes").

**Risks**

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Mode analysis | Definite aliasing is out of scope, because open types cover partly bound data. But full unique modes, which the reference manual says mmc has not fully implemented, the open-term rules and mode inference, which `LIMITATIONS.md` calls "a bit imperfect", have no reference implementation to test against | Implement the 22.01 mode rules first, with differential tests against mmc; build separate test suites for uniqueness and open terms, written from the manual and this design; infer modes only where mmc does without `--infer-modes`, for predicates without a `:- pred` declaration, and compare them with mmc's `--infer-modes`, whose passes are consistent; make mode errors name the waiting goal and variable |
| Precise GC on LLVM statepoints | Subtle correctness bugs, more so in a type-directed tracer, which depends on every root's `type_info`; statepoint overhead | GC torture mode; a heap verifier that walks the heap by type after each collection |
| Allocation-heavy code | Naive reverse is already 3.5× C with Boehm (E1) | Nursery bump allocation, last-call-modulo-constructors, escape analysis; measure every milestone |
| Scope | Many large components | Phased plan in which each milestone is usable on its own |
| Small upstream community | Three developers wrote about 99% of mmc's commits in the last five years | Work from the manual and papers; rely on differential testing, not on help |

## Sources

- [Mercury repository](https://github.com/Mercury-Language/mercury) and the [22.01.9 source distribution](https://github.com/Mercury-Language/mercury-srcdist), used only for the documents and metadata listed under Clean-room rules: `LIMITATIONS.md`, the reference manual, the user's guide, `NEWS.md`, `LICENSE` and commit metadata
- [The Mercury Language Reference Manual, release 22.01](https://www.mercurylang.org/information/doc-release/mercury_ref/index.html) ([PDF](https://mercurylang.org/information/doc-release/reference_manual.pdf)): the language specification this compiler implements
- [The Mercury Language Reference Manual, current version: Syntax](https://www.mercurylang.org/information/doc-latest/mercury_reference_manual/Syntax.html): the comment, name and punctuation rules that the 22.01 manual leaves unstated, and the lexical changes since 22.01.
- [The Mercury User's Guide, release 22.01](https://mercurylang.org/information/doc-release/mercury_user_guide/) ([PDF](https://mercurylang.org/information/doc-release/user_guide.pdf)): compiler options, grades, the debugger and the tools
- [Flang driver design](https://github.com/llvm/llvm-project/blob/main/flang/docs/FlangDriver.md): `flang` and `flang -fc1`, built on Clang's driver library
- [Mercury User's Guide: grade modifiers for parallelism](https://mercurylang.org/information/doc-latest/mercury_user_guide/Grade-modifiers-for-parallelism.html)
- [Mercury benchmarks page](https://mercurylang.org/about/benchmarks.html)
- [Bone, Somogyi and Schachte: Controlling loops in parallel Mercury code](https://paul.bone.id.au/pub/pbone-2012-loop-control)
- [Mazur, Ross, Janssens and Bruynooghe: Practical aspects for a working compile time garbage collection system for Mercury](https://arxiv.org/abs/cs/0110037)
- [Becket on Mercury against C (mercury-users, 2001)](https://lists.mercurylang.org/archives/users/2001-April/000898.html)
- [Nicol: HLC back end speedup for N-queens (mercury-reviews, 2021)](https://lists.mercurylang.org/archives/reviews/2021-January/021741.html)
- [Johansson, Sagonas and Wilhelmsson: Heap architectures for concurrent languages using message passing](https://dspace.lib.ntua.gr/xmlui/handle/123456789/34514?show=full)
- [Clebsch et al.: Orca, GC and type system co-design for actor languages](https://2017.splashcon.org/details/splash-2017-OOPSLA/30/Orca-GC-and-Type-System-Co-Design-for-Actor-Languages)
- [Gleam `gleam/erlang/process`: Subject, receive, Selector](https://gleam-erlang.hexdocs.pm/gleam/erlang/process.html)
- [Erlang reference manual: Expressions, "Receive"](https://www.erlang.org/doc/system/expressions.html): which message a `receive` takes, its cost, and priority messages.
- [GNU GPL v2 FAQ: output of GPL programs](https://www.gnu.org/licenses/old-licenses/gpl-2.0-faq)
- [Erlang external term format](https://www.erlang.org/doc/apps/erts/erl_ext_dist.md)
- [Erlang efficiency guide: Processes](https://docs.w3cub.com/erlang~26/doc/efficiency_guide/processes) (loss of sharing)
- [Erlang efficiency guide: Advanced](https://erlang.org/doc/efficiency_guide/advanced) (the atom table is not garbage-collected)
- [Herbrand constraints in HAL](https://research.monash.edu/en/publications/herbrand-constraints-in-hal/) (Monash research portal)
- [Improving PARMA trailing](https://arxiv.org/abs/cs/0505085) (arXiv cs/0505085)
- [Mercury reference manual: Solver types](https://mercurylang.org/information/doc-latest/mercury_reference_manual/Solver-types.html), and its section [Solver types and negated contexts](https://mercurylang.org/information/doc-release/mercury_ref/Solver-types-and-negated-contexts.html)
- Mercury user's guide: [Grade modifiers for trailing](https://mercurylang.org/information/doc-latest/mercury_user_guide/Grade-modifiers-for-trailing.html) and [The importance of consistency](https://www.mercurylang.org/information/doc-latest/mercury_user_guide/The-importance-of-consistency.html)
- [Jeffery: Expressive type systems for logic programming languages](https://mercurylang.org/documentation/papers/dgj-thesis-final.ps.gz) (PhD thesis, University of Melbourne, 2002): Mercury's type system, with type classes and existential types (Type checking)
- [Overton: Precise and expressive mode systems for typed logic programming languages](https://mercurylang.org/documentation/papers/dmo-thesis.ps.gz) (PhD thesis, University of Melbourne, 2003)
- [Overton, Somogyi and Stuckey: Constraint-based mode analysis of Mercury](https://mercurylang.org/documentation/papers/ppdp02_mode.pdf) (PPDP 2002)
- Measurements in this document were made with Melbourne Mercury 22.01.9 (`hlc.gc`), gcc 13.3, SWI-Prolog 9.0.4 and CPython 3.13 on a 2-vCPU cloud VM.
