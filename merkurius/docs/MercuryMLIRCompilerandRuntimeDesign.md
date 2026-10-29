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
| Receive | Blocking and selective, over one mailbox or several joined by a selector; the oldest matching message wins, across mailboxes; timeout returns `maybe` | Erlang semantics with static types; priority by Erlang's two-step receive |
| Kinds of message | One protocol type per mailbox, a discriminated union; more mailboxes for types the process does not own (replies, `down` messages, other modules' protocols) | Constructors cost nothing; a selector maps each mailbox's type into one result type |
| Unknown messages | Impossible by typing; `det` loops must cover every constructor | No catch-all `_ -> true` needed |
| Request/reply | Fresh reply mailbox per request, and a monitor on the server for the call | Late replies go to a dead mailbox and are dropped; a dead server ends the wait at once |
| Generic dispatch | Existential envelopes; `sendable(M)` required where a mailbox is created | Type-safe forwarding without knowing payload types |
| Process failure | Links and monitors, delivered with processes | Supervision depends on them |
| Heaps | One heap per process, precise moving GC | No global pauses; whole heap freed on exit |
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
| Papers and theses, on mercurylang.org and in the literature | Read | Mode analysis, compilation model, compile-time GC, parallel Mercury |
| Mailing-list posts | Read, skipping any patch or code excerpt from the compiler, runtime or library | Benchmark reports (E6) |
| Commit metadata: dates, authors, messages and changed file names | Counted and classified, with no diffs or file contents | Project activity (E4) |
| mmc as a black box | Built with `configure` and `make`, run on our own programs, its messages read | All experiments; differential tests |
| C that mmc generates from our own programs | Read exactly as mmc writes it | Term representation, nondeterminism, last-call-modulo-constructors (E2, E4) |

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
| MLIR OpenMP dialect: `OpenMPClauseOperands.h` | Clause operand structures used to build `omp.parallel` and `omp.wsloop` when `merk.par_loop` is lowered |
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
| `flang/lib/Lower/OpenMP/Utils.h` | Lowering helpers tied to Fortran's parse tree, symbols and FIR; how it organizes clause processing is still a useful model for lowering `merk.par_loop` |
| `flang/tools/flang-driver/driver.cpp` and `fc1_main.cpp` | Built on Clang's driver library in its Flang mode, which a project outside llvm-project cannot extend. We keep the shape: one binary that hands off to `-fc1` |
| `flang/include/flang/Frontend/` and `flang/lib/FrontendTool/` | Tied to Fortran's options and actions, and uses Clang's `DiagnosticsEngine`. We keep `CompilerInvocation`, `CompilerInstance`, the layered `FrontendAction` classes and the `.def` option lists (Driver) |
| `clang/lib/Driver/ToolChains/Flang.cpp` (`Flang::addOffloadOptions`) | Part of Clang's driver library; it is the recipe for the offload job sequence |

**Rules for copied code**

- Keep the LLVM licence header in every copied file.
- Add a line naming the original path and the llvm-project commit it came from, and stating that the file was modified. Apache 2.0 requires modified files to say so.
- List every copied file with its upstream commit and local path in `THIRD_PARTY.md`, and move its status here from Planned to Copied, then Adapted, so upstream fixes can be diffed in later.
- Prefer depending on an LLVM library to copying from it.

This is an engineering position, not legal advice.

## Experiments

Measurements made on 2026-10-26 and 2026-10-27 set four expectations. Deterministic Mercury already runs at C speed; allocation is the main gap, at 3.5× C on naive reverse; an OpenMP pragma on generated code ran 2× faster on 2 cores; and in a C model, open terms cost about 1.2× in the code that uses them and nothing elsewhere. Nothing was measured beyond 2 cores, and nothing yet on our own compiler.

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
- Consequence: the design bets on bump-pointer nursery allocation. Option B's header word would grow a list cell from 16 to 24 bytes, and must be measured on nrev before it is adopted.

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
- **The OpenMP sentinel.** When OpenMP is enabled (`-fopenmp`), a `%` comment that is the first non-blank text of its line, and that begins `%$omp` followed by whitespace, is a directive. Each following line whose first non-blank text is `%$omp&` continues it. The directive becomes one `directive` token holding its own sub-tokens, lexed with the normal rules. Without `-fopenmp`, these lines stay comments. A `%$omp` inside a string, quoted name or block comment is never seen, because the lexer is already inside that token.
- **Line number directives.** `#` followed by a positive integer and a newline, wherever a token may occur, sets the number of the next line (reference manual, Tokens). The lexer records it in the source file's line table and keeps the directive as trivia, so round trips still reproduce it. A carriage return before the newline is accepted, so files with CRLF line endings work.
- **Reserved:** lines beginning `%$` and a space, for conditional compilation like Fortran's `!$` lines, meaning code compiled only when OpenMP is enabled. They are not in the first version.

### Term reader

- **Output.** The reader reads one item, from its first token to the `end` token, into a term tree. Term nodes are variables, integers, floats, strings, names, compound terms (a functor name, possibly module-qualified, with arguments), list and tuple terms, higher-order applications (`Term(Args)`, read as `''(Term, Args…)` as the manual specifies), and directive-annotated terms. They live in an arena that is reset after each item is converted.
- **Variables** are numbered per item through a table from names to numbers. Each `_` is a fresh variable, and singleton warnings follow the manual's rules for names beginning with `_`.
- **Precedence climbing, hand-written.** `parseTerm(minPriority)` reads a primary term, then extends it with infix and postfix operators whose priority is at least `minPriority`. An `x` argument must have a strictly higher priority than its operator, and a `y` argument a higher or equal one. Arguments of compound terms and list elements are read at priority 501, above `,` at 500. Flang writes Fortran's expression levels as a chain of combinators, one per level; Mercury's table has many more levels, so a table-driven reader is simpler and faster here.
- **Prefix operators.** A name that is a prefix operator is applied as one when the next token can begin a term; otherwise it is an ordinary name, so `-` alone is a name and `- X` is an operator term. The `fxy` operators, such as `some`, `trace`, `try` and `require_complete_switch`, take two arguments, as in `some [X] Goal`. A name immediately followed by `open_ct` is always a compound term, operator or not. Backquoted names are infix operators at priority 1380.
- **The operator table** is data compiled into the front end. Its operators are 22.01's, numbered as in the current manual, where a higher priority binds more tightly. The current numbering is 1500 minus 22.01's, so the two describe the same parses; the current one is used because it is what Mercury programmers now read.
- **Recovery.** On a syntax error the reader reports the token it expected and skips to the next `end` token, so one bad item does not hide errors in later ones. A language server uses the same recovery.

### Converter: combinators over terms

The converter is a small parser-combinator library in the spirit of Flang's, whose primitives match the shapes of terms instead of sequences of tokens:

- `functor("then", A, B)` matches a compound term with that functor and arity, and applies `A` and `B` to its arguments.
- `name`, `variable`, `literal`, `listOf(P)` and `anyOf(P1, P2, …)` match the obvious shapes.
- `construct<T>(…)` builds a parse-tree node of type `T` from the results, and `sourced(P)` records the term's source range on it.

A rule then reads like the manual's definition. In this example, `if C then T else E` is the term `else(if(then(C, T)), E)`:

```cpp
constexpr auto ifThenElse =
    sourced(construct<IfThenElse>(
        functor("else", functor("if", functor("then", goal, goal)), goal)));
```

- **One rule set per category:** items, goals, data-terms, types, insts and modes, each following its chapter of the manual. `anyOf` dispatches on a term's top functor through a table, so choosing an alternative takes constant time.
- **Errors.** A term that matches nothing is reported with the forms expected in its context, such as "expected a goal; found a lambda expression".
- **Kept small.** Rules are `constexpr` objects built from a few templates. Heavy template parsers are known for slow builds and high compiler memory, so the library stays minimal and nodes are allocated in an arena rather than moved around as values.

### Parse tree

- **Types mirror the grammar.** There is one C++ type per construct of the reference manual: each declaration kind, each goal form (`IfThenElse`, `Disjunction`, `SomeGoal`, `TraceGoal`, …), each data-term form (`FunctorTerm`, `FieldAccess`, `Lambda`, `StateVarRef`, …), and the type, inst and mode languages. A category with alternatives is a `std::variant` of its forms, so `std::visit` over it is checked for exhaustiveness, much as a `det` switch is.
- **Immutable.** Nodes are created once by the converter in the `ASTContext` arena and never change. The only fields filled later are annotation slots on names, as Flang's `parser::Name` carries a symbol pointer.
- **Generic walker.** `walk(node, visitor)` visits every node, so a pass handles only the node types it cares about.
- **Unparser.** It prints a parse tree back as Mercury source, either exactly as written or in a canonical layout (Unparser and round-trip tests). Every node records its first and last token, literals keep their spelling beside their value, and parentheses, list notation and operator notation are recorded as written, so nothing needed to print the source back is lost.
- **Debug dumps.** Options print the parse tree and the semantic form, as Flang's do.

### Unparser and round-trip tests

The unparser is the first component to build, together with the source manager, lexer, term reader and converter it tests. Its round-trip tests then guard every later change to them. It has three modes:

- **Exact.** Prints the parse tree using the tokens it recorded, so every spelling, parenthesis, comment and space comes from the tree and the output equals the input, byte for byte. Because it walks the tree instead of copying source ranges, any token or comment that the converter loses or misplaces shows up as a difference.
- **Canonical.** Prints the same tree in a standard layout: four-space indentation, one goal per line, if-then-else and disjunction laid out as in the reference manual, spaces around infix operators, line breaking by a Wadler-style pretty-printer, and at most one blank line between items. Comments are kept, attached to the node they precede or, on the same line, follow, with their text unchanged. Literal spellings such as `0x1F` and quoting are kept as well. This mode is the formatter.
- **Fully parenthesized.** Canonical layout with every operator term in parentheses, used for the mmc comparison below.

Tests, run on every file of every corpus:

1. **Lexer round trip.** Printing every token's trivia and spelling gives back the file, byte for byte.
2. **Exact round trip.** Parsing a file and unparsing it exactly gives back the file, byte for byte.
3. **Canonical fixed point.** Parse the file and unparse it canonically, giving C1; parse C1 and unparse it canonically, giving C2. C1 and C2 must be identical.
4. **Meaning preserved.** The parse trees of the file and of C1 must be equal once locations and trivia are removed. Test 3 alone would pass a printer that consistently dropped something; this test catches it.
5. **Against mmc.** Tests 1 to 4 compare our parser with itself, so a parser that is consistently wrong about an operator's priority would pass them all. mmc's `--convert-to-mercury` option prints mmc's own parse of a module with every operator term parenthesized, as in `(!: Acc) = ((!. Acc) + (X * X))` for the running example. Our parse of the file must equal our parse of that output: the parentheses make it unambiguous, so any difference is a priority or associativity error in our term reader. mmc's printer drops comments and splits a combined `pred`-and-mode declaration into two, so declarations are compared after the same split.
6. **Fuzzing.** Random term trees are printed, parsed and compared. Mutated source files must never crash the front end, and recovery must resume at the next item.

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
| `Scope` | One goal wrapped in a quantifier, `promise_*`, `require_*`, `trace`, `try` or an OpenMP directive |

Each node points back to the parse-tree node it came from, and diagnostics use that pointer, so an error found in the semantic form is reported at the construct the programmer wrote. Together, the parse tree and the semantic form are the AST.

Desugaring builds the semantic form in this order:

1. State variables (`!X`, `!.X`, `!:X`) become numbered variables, following the manual's scoping rules.
2. DCG rules become clauses with two extra arguments.
3. Field access and update become calls to the field functions; function applications and unification expressions become unifications with fresh variables.
4. Lambdas become closure constructions, with their captured variables made explicit.
5. `:- mutable` and `:- process` declarations become ordinary predicates, added as implicit declarations (Language extensions).
6. Bodies are put in superhomogeneous form: each unification holds at most one functor, nested functors become chains of unifications through fresh variables, and head arguments become distinct variables unified in the body.
7. Quantification computes each goal's nonlocal variables by the manual's implicit quantification rules, and renames apart variables that are local to different branches.

Desugaring also numbers the goals and unifications of each semantic form densely, once. Semantic analysis indexes its per-procedure tables by these numbers.

### OpenMP directives

- **Definitions from LLVM.** Directive and clause kinds, and which clauses each directive allows, come from `llvm/Frontend/OpenMP/OMP.td`, which Clang and Flang share. Our directives therefore use the same `llvm::omp` kinds that the `omp` dialect's lowering already understands.
- **Syntax.** `%$omp parallel do` followed by clauses: `schedule(static)` or `schedule(dynamic, Chunk)`; `reduction(Op)`, with `Op` a binary function the programmer promises is associative, such as `+` or `max`; and, for self-recursive counted loops, `index(I)` and `upper(N)` naming the induction variable and its bound (Language extensions). The directive covers the goal that follows it.
- **Parsing.** A small recursive descent parser reads the directive token's sub-tokens into an `OmpDirective` parse-tree node, with one child per clause, each with its own source range.
- **Attachment.** The term reader attaches a directive token to the next term that begins after it. The converter accepts the annotation only where that term is a goal. There it builds an `OmpGoal`, which holds the directive and its associated goal, as Clang's `OMPExecutableDirective` holds its associated statement; desugaring turns it into a directive `Scope`. Anywhere else, such as before a clause head or inside a data-term, the directive draws a warning and is ignored.
- **Checking** is part of semantic analysis: the associated goal must be a loop the compiler can form, and the clauses must make sense for it.
- **Without `-fopenmp`** the lexer leaves directive lines as comments, so the program's meaning is that of the sequential program, which is also what mmc compiles.

### Modules and interface files

- **Modules.** Interface and implementation sections, nested and separate submodules, and the 22.01 visibility rules for `import_module` and `use_module`.
- **Our own interface-file format,** not Melbourne's `.int`/`.opt`. It serializes the exported declarations' parse-tree nodes, with the signatures semantic analysis inferred or checked: types and their representation, insts, modes, predicate and function signatures with modes and determinism, type classes and instances. The module loader reads them back into the same parse-tree types, so imported and local declarations look alike to semantic analysis.
- **Type fingerprints.** Each exported type gets a fingerprint of its definition, stored in the interface file. The same fingerprints identify types in the distribution handshake (Distribution and wire format).
- **Optimization section.** An optional section carries the semantic forms of selected bodies, for intermodule inlining and specialization, and analysis results.
- **Build driver.** It computes module dependencies from imports and rebuilds only what changed, like `mmc --make` (Driver, Build mode).

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
         ├─ OmpDirective parallel do [Reduction +]
         └─ CallGoal int.fold_up [FunctorTerm add_square [VarTerm A], Literal 0,
                                  VarTerm N, Literal 0, VarTerm Sum]
```

**Round trip.** Unparsing this parse tree exactly gives back the source above, including the `%$omp` line and the blank lines. The fragment already follows the canonical layout, so canonical mode should print the same text. mmc's `--convert-to-mercury` prints `add_square` as follows, and parsing that must give the same tree, which checks the priorities of `=`, `+`, `*` and the state-variable prefixes:

```mercury
add_square(A, I, (! Acc)) :-
	array.unsafe_lookup(A, I, X),
	(!: Acc) = ((!. Acc) + (X * X)).
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
- **MLIR.** `#1` becomes `merk.call` with a type-info operand for `int`, `#3` becomes `arith.muli`, `#2` becomes `arith.addi`, and `Acc_1` is the procedure's result. In `sum_squares`, `#2` becomes `merk.closure`, and the directive scope becomes `merk.par_loop` with `reduction(+)`.

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
- **Types** are interned the same way: type constructors applied to arguments, type variables, higher-order types and existentially quantified variables.

### Passes, in order

Analysis takes one strongly connected component (SCC) of the call graph at a time, bottom up, so that every callee outside the SCC is finished first. Inference iterates within an SCC until it reaches a fixpoint, and independent SCCs can be analysed in parallel.

1. **Name resolution:** module-qualify every name, and record on its AST node the candidate declarations of each overloaded predicate, function or constructor.
2. **Type checking and inference:** the manual's type assignments, computed Hindley–Milner style with overloading, type classes (multi-parameter, functional dependencies, superclasses), existential types and equivalence types. Overload resolution picks one candidate per node. Types of local predicates without declarations are inferred, as mmc's `--infer-types` does; exported predicates must be declared.
3. **Purity checking:** `pure`, `semipure` and `impure` goals and `promise_pure` scopes. A predicate's purity is inferred from its body and compared with its declaration.
4. **Mode analysis,** per procedure (below).
5. **Switch detection and determinism analysis:** disjunctions whose arms unify the same input variable with different constructors become switches. Each goal's determinism is then inferred by the manual's rules (`det`, `semidet`, `multi`, `nondet`, `cc_multi`, `cc_nondet`, `erroneous`, `failure`). Local procedures without annotations get the least fixpoint the manual defines, starting from `det`. Incomplete switches in `det` code are errors, which is what makes catch-all receive clauses unnecessary. Commits are inserted where solutions are indistinguishable.
6. **Uniqueness checking:** enforce `di`/`uo` and `mostly_unique` modes, including for arrays, which needs to know where backtracking can happen, so it runs after determinism analysis. The reference manual says unique modes are not yet fully implemented in mmc; we implement them fully, because parallel loops and move-on-send depend on them.
7. **Polymorphism analysis:** for each procedure, decide where every `type_info` and `typeclass_info` comes from: a hidden argument, a field of an existential constructor, a superclass of another dictionary, or a constant built at the call site. This runs after mode analysis, because the source can depend on the mode: a deconstruction yields an existential's type info, while a construction must supply one. MLIR generation materializes the arguments; the AST is not rewritten.
8. **Sendability analysis (new):** decide per type whether values may be sent, and check `sendable(M)` where mailboxes are created (Language extensions).
9. **OpenMP directive checks (new):** the associated goal must be a call to `int.fold_up`, `int.fold_down`, `array.map` or `array.generate`, or a self-recursive counted loop whose `index` and `upper` clauses name its induction variable and bound. Its body must be `det`, pure and free of `io`, and a `reduction` must apply to the loop's accumulator. Violations are errors, never a silent sequential fallback.

### Mode analysis

Mode analysis runs per procedure, starting from the instmap that the argument modes give.

- **Scheduling.** A conjunction's conjuncts are tried in source order. One that cannot run yet, because a variable it needs is not bound enough, is delayed and retried after each conjunct that does run. The result is the conjunction's execution order. If delayed conjuncts remain and none can run, that is a mode error, and the message names each waiting goal and the variable it waits for.
- **Unifications** are classified by the insts of their two sides: construction when the left side is free, deconstruction when it is bound and the right side has free arguments, assignment, a test when both are bound, or a call to the type's unify procedure for tests on structured values. Unifying two free variables that both stay live is an error, except at open types (Language extensions).
- **Calls** choose a procedure of the callee whose initial insts the arguments match. If an argument is bound where the chosen mode outputs it, the call writes to a fresh variable and an implicit test against the original follows; this is an implied mode. A local predicate without mode declarations gets its modes inferred from its calls when mode inference is enabled, as with mmc's `--infer-modes`, iterating within the SCC. A function without mode declarations gets the default mode the manual defines: all arguments `in`, result `out`.
- **Branches.** Each arm of a disjunction, if-then-else or switch is analysed from the same starting instmap, and the results are merged at the join; a nonlocal bound in one arm but not another is an error. Negations and if-then-else conditions may not bind nonlocals, except for open terms inside a promise scope, as the manual's rule for solver types requires.
- **Higher-order code.** Closure insts carry the closure's modes and determinism, so a higher-order call is checked like a first-order one.

### Diagnostics

- **Format.** Modelled on Clang's: a severity, a location, highlighted ranges and attached notes, printed with the source line and a caret. Each message has an identifier, so warnings can be enabled, disabled or made errors one by one.
- **Reported against the parse tree.** A mode error inside a DCG rule, for example, points at the rule as written, not at the clause it expands to.
- **Analysis continues after errors.** A procedure with errors is excluded from MLIR generation, but other procedures and other SCCs are still checked, so one run reports as many problems as possible.
- **Warnings** follow the same passes: singleton variables, unused imports, suspected occurs-check failures, unifications that cannot succeed, and misplaced `%$omp` directives.

### Mapping to `merk`

The output of semantic analysis is the annotated AST, one `ProcInfo` per procedure, and the module's interface file; there is no other intermediate form. MLIR generation reads them directly, one procedure at a time, as ClangIR's code generator reads Clang's AST:

- **Values.** Once modes are known, each variable is produced exactly once on each path. So a map from variables to SSA values, updated at each producer and read at each consumer, is all MLIR generation needs, and the output is in SSA form from the start.
- **Locations.** Each operation carries the location of the parse-tree node behind its semantic node, and variable names are recorded as `NameLoc`s for debug information.
- **Types.** `int`, `float`, `char` and the sized integers become builtin MLIR types such as `i64` and `f64`. Other types become `!merk.term<…>`, closures `!merk.closure<…>`, and type infos `!merk.typeinfo`.

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
| OpenMP directive scope | `merk.par_loop`, with the clauses as attributes of `llvm::omp` kinds |
| `try`, `throw` | `merk.try`, `merk.throw` |
| Quantification, `promise_*` and `require_*` scopes | No operation; they only affect checking |
| `trace` scope | Its goal, if its compile-time condition holds; otherwise nothing |
| Unification of open values | `merk.open_unify` and the other open-term operations |

- **Branch results.** A switch, if-then-else or disjunction produces the variables its arms bind: those the `ProcInfo` records as going from free to bound across it. They become the operation's results, and each arm yields them in the same order, as `scf.if` does. This is why mode analysis keeps instantiation states at branch points.
- **Polymorphism.** Type-info and dictionary operands are materialized where the `ProcInfo` says they come from: a parameter, a field of a deconstructed existential, a superclass of another dictionary, or a constant type-info value.
- **Uniqueness.** Consumed (`di`) operands carry an attribute, so later passes can update in place, or move a message instead of copying it.

## Language extensions

Four extensions are added: processes with typed mailboxes, parallel loop annotations, a sendability rule, and open terms. A program that uses none of them is plain Mercury and pays nothing for them.

### Processes and mailboxes

The API is a library module. This interface, with stub bodies, passes Melbourne Mercury 22.01.9's type, mode and determinism checks, together with a client that uses every part of it: a counter server with a selector over two mailboxes, the two-step priority receive, `request` and `send_request`.

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
:- pred select(selector(R)::in, timeout::in, maybe(R)::out,
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
- **Receive** blocks until a message matches or the timeout expires, and returns `yes(R)` or `no`. It takes the oldest message for which the `semidet` selector succeeds; the rest stay queued in order. Erlang's `receive {reply, Ref, S} -> …` becomes the selector `(pred(M::in, S::out) is semidet :- M = reply(Ref, S))`. `receive` is `select` over a one-mailbox selector.
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
    select(Sel, infinity, MaybeEvent, !IO)
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
        select(Sel, infinity, MaybeEvent, !IO)
    )
```

A selector option for strict priority between its cases can be added later if programs need it. Strict priority can starve the lower cases, which is why it is not the default.

**Request and reply.** `request(Counter, (func(R) = get(R)), after(1000), Reply, !IO)` asks the counter for its value:

1. It monitors the server, delivering any `down` message to a private mailbox. A server that is already dead delivers one at once.
2. It creates a fresh `mailbox(Rep)`, here `mailbox(int)`, builds the request by applying the function to it, here `get(ReplyTo)`, and sends it. These two steps are `send_request`.
3. It selects over the reply mailbox and the `down` mailbox, with the timeout.
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
```

A module defines process kinds; it is not itself a process. Exporting the mailbox as an abstract type and keeping the message constructors private forces other modules through the module's API predicates, as Erlang's gen\_server does by convention.

Failure handling follows Erlang. An uncaught exception ends the process with that exception as its exit reason. A monitor delivers a `down_info` message to a mailbox the watcher names, through a function into that mailbox's message type, so a process with one protocol needs no second mailbox for it. Links deliver exit messages the same way. Both are required and ship with processes in M4; their semantics are in Runtime system, "Exceptions, links and monitors".

### Parallel loop annotations

- **Syntax:** a comment line `%$omp parallel do [schedule(…)] [reduction(Op)]` before a goal, mirroring Fortran's `!$omp`. A stock Mercury compiler ignores it, so the sequential meaning is the specification.
- **Targets:** calls to `int.fold_up` and `int.fold_down`, and to the array iterators (`array.map`, `array.generate`), are treated as intrinsics. Self-recursive counted loops are also recognized, with `index(I) upper(N)` naming the induction variable and bound.
- **The promise:** iterations are independent, and writes to a threaded array touch disjoint elements. A reduction's operator is associative. Floating-point reductions may then vary with the schedule.
- **Checked by the compiler:** the body is `det`, pure, and threads no `io`. A violation is an error, not a silent sequential fallback.
- **Exceptions:** each iteration catches its own exception; one is rethrown after the join.

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

**Types.** For any type `T`, `open(T)` has the same constructors as `T`, but any argument at any depth may be an unbound variable; an argument of type `U` has type `open(U)` inside it. Values of an open type have inst `any`, Mercury's existing inst for values that may be partly unbound, and are passed with the existing modes `ia` (`any >> any`) and `oa` (`free >> any`). A free variable of an open type becomes a fresh unbound variable when it first reaches an `any` position; the compiler inserts the code that creates it. In constructor arguments, a ground `U` is accepted where `open(U)` is expected; elsewhere conversion is explicit.

**Operations.**

- `X = Y` on open values is full unification. It may bind variables on either side, including one variable to another, and fails on a constructor clash, so it is `semidet`. Construction into a variable that is still `free` stays `det`.
- `to_open(X)` turns a ground `T` into an `open(T)`. It is free when the two types share a layout (Lowering) and copies otherwise.
- `from_open(open(T)::ia, T::out) is semidet` fails if any variable is still unbound or the term is cyclic. It returns the term itself when the walk meets no variable cell, bound or unbound, and otherwise copies it into `T`'s layout.
- Plain `=` omits the occurs check, as Prolog does; `unify_oc` performs it. `from_open` rejects the cyclic terms that can result, so ground code never sees one.
- `var`, `nonvar` and term identity are non-logical and therefore `impure`. Open types have no `compare` and no hashing, because the order of terms with variables changes as they are bound.

**Checking rules.**

- **Negated contexts.** A negation or if-then-else condition that may bind a nonlocal `any` variable needs a `promise_pure` scope, the rule Mercury already applies to solver types. `\+ X = 1` with `X` unbound is the case Prolog answers wrongly.
- **Determinism.** Binding an existing variable is `semidet`, because the compiler cannot know it is still unbound. An inst for "known unbound" would make first bindings `det`; that is a later refinement.
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
| `merk.switch` | Branch on a term's constructor; each case binds the constructor's arguments |
| `merk.unify_test` | Compare two ground terms (calls the type's unify procedure when not primitive) |
| `merk.call`, `merk.call_closure`, `merk.closure` | First-order calls, higher-order calls, closure construction |
| `merk.ite`, `merk.not`, `merk.disj` | If-then-else, negation, disjunction |
| `merk.succeed`, `merk.fail`, `merk.commit` | Produce a solution, fail, prune remaining solutions |
| `merk.throw`, `merk.try` | Exceptions |
| `merk.par_conj`, `merk.par_loop` | Parallel conjunction; a loop formed from a `%$omp` annotation |

A deterministic procedure, in proposed assembly syntax:

```mlir
merk.proc @double_all(%xs: !merk.term<"list.list(int)">)
    -> !merk.term<"list.list(int)"> attributes {determinism = #merk.det} {
  merk.switch %xs {
    case @"list.[]" {
      %nil = merk.construct @"list.[]" () : !merk.term<"list.list(int)">
      merk.succeed %nil
    }
    case @"list.[|]" (%h: i64, %t: !merk.term<"list.list(int)">) {
      %two = arith.constant 2 : i64
      %h2  = arith.muli %two, %h : i64
      %t2  = merk.call @double_all(%t) : !merk.term<"list.list(int)">
      %c   = merk.construct @"list.[|]" (%h2, %t2) : !merk.term<"list.list(int)">
      merk.succeed %c
    }
  }
}
```

In a `nondet` procedure, a call to another `nondet` procedure has generator semantics. The operations after it run once per solution, and `merk.fail` resumes the most recent choice. Verifiers enforce the determinism rules, so a `det` procedure may not contain an unpruned `nondet` call.

**`mrt` operations:** `mrt.alloc` (bump allocation in the process nursery), `mrt.tag` and `mrt.untag`, `mrt.field_load` and `mrt.field_store`, `mrt.safepoint`, `mrt.yield_check`, `mrt.spawn`, `mrt.send`, `mrt.receive` (a suspension point, over the mailboxes of a selector), `mrt.throw`, `mrt.landing`, and `mrt.rtti` (address of a static type descriptor).

**Open terms** need no new `merk` type, because `open(T)` is an ordinary Mercury type. They add `merk.open_unify` (full unification, `semidet`), `merk.to_open` and `merk.from_open` to `merk`, and `mrt.new_var`, `mrt.deref`, `mrt.bind` (which trails when a trail is live), `mrt.trail_mark` and `mrt.trail_undo` to `mrt`. Only procedures with `any` variables contain them.

## Lowering and code generation

Lowering fixes the data layout first, then turns each determinism category into plain control flow, then hands LLVM IR and GC metadata to LLVM. The steps below run in this order.

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
| Ground terms known at compile time | Static data, never allocated |

These cells are not self-describing, which matters for the garbage collector. Whether to add a header word to heap cells is the main open representation question (Runtime system section).

**Open layouts.** A type used under `open(_)` gets a second layout, used only by open values (Language extensions). One primary tag marks a variable reference, and every field holds a tagged word. `int`, `float` and `char` fields, which hold raw bits in the ordinary layout, are boxed, with static boxes for small values; enum and packed fields are re-encoded as tagged constants. A type that already uses all 8 primary tags moves one more functor under the shared secondary tag. When a type needs none of these changes, as with `tree ---> leaf ; node(tree, tree)` or `list(tree)`, the two layouts coincide and `to_open` is free. The ordinary layout never gives up a tag, so ground code is compiled exactly as before.

### 2. Calling convention

- `det` procedures return their outputs, as a struct when there are several.
- `semidet` procedures return an `i1` success flag plus outputs.
- `type_info` and `typeclass_info` arguments come first.
- Mercury relies on tail calls between procedures, not only self-recursion. Every call in tail position is emitted `musttail` with LLVM's `tailcc` convention, so the guarantee holds across modules.

### 3. Deterministic and semidet code

- Switches become `cf.switch` on the primary tag, then on the secondary tag when needed.
- If-then-else and semidet failure become branches; unifications become loads, stores and compares.
- Last-call-modulo-constructors is on by default. A recursion followed by a construction allocates the cell first and passes the address of its open field (destination passing), so list-building recursions become loops. mmc 22.01.9 does this only with `--optimize-constructor-last-call`.

### 4. Nondeterministic code

Nondet procedures use continuation passing, as mmc's high-level C back end does. A nondet procedure takes a success continuation; producing a solution calls it, and failing returns. One change makes this fit our runtime: each continuation returns an `i1` meaning "stop". A commit is a `true` returned up the chain, so there is no `setjmp`/`longjmp`, and every frame stays visible to the stack maps.

The main alternative is an explicit choice-point stack in the process heap, as mmc's low-level back end has. Continuation passing is chosen because processes have their own growable stacks (Runtime system section), so a deep native stack can be suspended and moved.

### 5. Exceptions

Mercury's `throw` and `catch` use LLVM's zero-cost exception handling: `invoke` and landing pads. Statepoints support `invoke`, so exception edges keep precise roots. An exception that leaves a process body ends the process.

### 6. Parallel loops

A formed `merk.par_loop` is rewritten into a canonical counted loop: induction variable, inclusive bound, step 1. That loop becomes `scf.parallel` with `scf.reduce` for reductions, and the existing conversion produces `omp.parallel` and `omp.wsloop`. Outlining and the libomp calls come from MLIR's OpenMP-to-LLVM translation. In version 1, a parallel body may not allocate; the compiler checks this after inlining.

This is the transformation that experiment E2 applied by hand to mmc's generated C, where it ran 2× faster on 2 cores with a bit-identical result.

### 7. GC metadata and safepoints

- Calls use LLVM statepoints, so the collector can find and relocate every live heap pointer.
- Safepoints sit at calls and at allocation slow paths. Preemption forces a slow path by lowering the heap limit, so loops that allocate need nothing more; loops that do not allocate get `mrt.yield_check` on their back edge.
- If heap cells stay headerless, every stack map also records where each polymorphic root's `type_info` lives.

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
| Parser | `-fdebug-unparse-no-sema`, `-fdebug-dump-parse-tree-no-sema` |
| Semantic analysis | `-fsyntax-only`, `-fdebug-unparse`, `-fdebug-dump-parse-tree`, `-fdebug-dump-semantic-form`, `-emit-interface` |
| Code generation | `-emit-merk`, `-emit-mrt`, `-emit-llvm`, `-emit-llvm-bc`, `-S`, `-emit-obj` |

The unparser's exact, canonical and fully parenthesized modes (Front end, Unparser and round-trip tests) are chosen with `-funparse-mode=`. The `-fc1` layer with the source and parser actions comes first, in M1, because the round-trip tests run through it; the build mode and linking follow with code generation.

### Linking and offloading

- **Linking** goes through `clang` as the link driver. It is given the objects, the runtime library, `-fopenmp` when any module used it (so Clang links its OpenMP runtime), and the user's `-L`, `-l` and `-fuse-ld=` options. Clang supplies the C start-up files, the C library and the sanitizer runtimes, as `cc` does for rustc and GHC.
- **Offloading**, when device offloading arrives (deferred, like `createMapInfoOp`), follows the job sequence Flang builds in `Flang::addOffloadOptions`:
  1. host `-fc1` to LLVM IR;
  2. one device `-fc1` per `--offload-arch`, with `-fopenmp-is-target-device` and `-fopenmp-host-ir-file-path` naming the host IR;
  3. `llvm-offload-binary` packages the device images;
  4. host `-fc1` to an object, with `-fembed-offload-object=` naming the package;
  5. `clang-linker-wrapper` links the device code, adds the code that registers it with the offload runtime, and runs the host link.

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

The runtime gives each process its own stack, heap and mailboxes, schedules processes M:N over one worker thread per core, and never stops every process at once. It is written in C++, like the compiler, and linked into every executable.

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
- **Receive and select:** in each mailbox of the selector, find the oldest message its case accepts: the head of the queue for `selecting`, or the first message for which the selector succeeds, scanning in arrival order, for `selecting_matching`. Take the candidate with the lowest stamp. The matched fragment joins the receiver's heap and is merged at its next GC. Unmatched messages keep their order. A process waiting in `select` is woken by a send to any of its mailboxes.
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

**Open decision: how the collector finds pointers.**

| Option | Cost | Benefit |
| --- | --- | --- |
| A. Headerless cells, traced using type descriptors and stack maps that record `type_info` locations | Complex tracer; more stack-map metadata | Keeps Melbourne's compact layout (a list cell stays 16 bytes) |
| B. One header word per heap cell (size and pointer layout) | A list cell grows from 16 to 24 bytes | Simple, fast tracer and deep copy |

The recommendation is B for the first runtime, because a correct precise collector matters more than cell size early on. `mrt` hides the choice, so A can replace B later. Allocation already dominates the naive reverse benchmark (3.5× C; experiments E1 and E3), so B's cost must be measured on it.

### Type descriptors

Static type descriptors describe each type's constructors, names, arguments and layout. Generic printing (`io.write`), `compare`, `unify` at polymorphic types, `univ` and, under option A, the collector all use them.

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
- **Type descriptors.** Each type used under `open(_)` gets a descriptor for its open layout. Generic unify, `from_open`, printing and, under option A, the collector use it.
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
    select(Sel, Timeout, MaybeReply, !IO),
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
| Type specialization | `merk` | Removes `type_info` passing and boxing; automatic, plus the `type_spec` pragma | High |
| Intermodule inlining and specialization | Interface files | Bodies of small exported procedures travel in the optimization section | High |
| Parallel loop formation | `merk` | From `%$omp` on `fold_up`, array iterators and counted recursion | High |
| Switch optimization | `merk`, `cf` | Jump tables, lookup tables, string switches | Medium |
| Escape analysis | `merk` | Cells that never leave a procedure go on the stack, saving nursery traffic | Medium |
| Accumulator introduction | `merk` | Uses associativity `promise` declarations, as mmc has since 1999 | Medium |
| Deforestation, unused-argument and dead-procedure elimination | `merk` | Classic Mercury passes | Later |
| Structure reuse (compile-time GC) | `merk` | Reuse dead cells in place, after Mazur's analysis; off by default in mmc | Later |
| Move instead of copy on send | `merk`, runtime | Messages passed in a `di` mode skip the deep copy | Later |
| Remembered scan position in selective receive | Runtime | Avoids rescanning old messages when waiting for one reply | Later |

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
- Pure-Mercury modules (`list`, `map`, `set`, `cord` and others) are reimplemented the same way. Where the manual leaves behaviour unstated, such as which exceptions are thrown, exact output formats, unstated orderings and performance, differential tests against mmc decide.

**Compatibility promise**

- Source compatibility with Melbourne Mercury for the supported language.
- No binary or ABI compatibility.
- Unsupported features (see Non-goals) are rejected with a clear error, never silently miscompiled.

**Testing**

- **Differential testing:** compile each test program with mmc 22.01.9 and with our compiler, and require identical output. Melbourne's test suite is run as an external oracle, not copied into our repository.
- **Pass tests:** unit tests per pass, and FileCheck tests for every dialect and lowering.
- **Diagnostics tests:** expected type, mode and determinism errors, such as the missing-constructor error in Language extensions.
- **Runtime stress:** a GC torture mode that collects at every safepoint, and randomized scheduling.
- **Benchmarks:** the four baseline programs (fib, naive reverse, 12-queens, float loop) already exist in C, Mercury, SWI-Prolog and Python. Add the classic Prolog suite (deriv, tak, crypt, poly), parallel-loop scaling runs, and process benchmarks: ring, ping-pong, spawn rate, receive latency.

## Phased plan

Seven milestones take the project from today's experiments to processes and parallel loops in one program, and then across machines. Each ends with a working compiler and a gate measured against the Experiments baseline.

&#91;embedded content: phased plan · 6 milestones, each with its gate\]

M3 and M4 both build only on M2, so their order can be swapped if processes matter more than parallel loops. M6, distribution, builds on M4 and can start before M5. M1 begins with the front end and its unparser, whose round-trip tests then guard every later change to them.

## Open questions and risks

Eight decisions remain open. The three that most constrain the runtime are the heap cell layout, the nondeterminism strategy and the stack model, so they come first. The eight ticked at the end were decided between 2026-10-27 and 2026-10-29.

**Open questions**

- [ ] Heap cells: header word per cell (option B) or headerless tracing with type descriptors (option A).
- [ ] Nondeterminism: continuation passing with stop flags, or an explicit choice-point stack.
- [ ] Process stacks: grow by copying, or segmented.
- [ ] When to adopt language changes made after 22.01.
- [ ] Contact with the Mercury team, as a courtesy and for questions the documents leave open. Their `LICENSE` file offers alternative arrangements if the licences do not suit.
- [ ] Sharing in local message copies: keep shared ground subterms, which needs a visited map, or expand them as Erlang does. Variable identity is always kept.
- [ ] Remote delivery guarantees: ordering per pair of processes, as in Erlang, and at-most-once or stronger delivery.
- [ ] Trust between nodes: how nodes authenticate each other and how mailbox keys for `sealed(T)` are published.
- [x] Compiler implementation language: C++, for direct use of MLIR, the LLVM data structures and TableGen.
- [x] Standard library: reimplemented in full from the library reference manual, for full control.
- [x] Links and monitors: required, and delivered with processes in M4.
- [x] Open-term bindings: WAM-style reference chains (Runtime system, "Support for open terms").
- [x] Parallel annotation syntax: the comment sentinel `%$omp`, attached to the next goal (Front end, "OpenMP directives").
- [x] Prescanner: none. Source files are read unchanged, and the lexer recognizes `%$omp` directives and line number directives, which depend on token context (Front end, "Source manager").
- [x] Lexical rules: 22.01's, with the current manual filling its gaps; where mmc 22.01.9 disagrees with its manual, mmc is followed; of the later changes only the operator renumbering is adopted (Front end, Lexer; Standard library, compatibility and testing).
- [x] Receiving from several mailboxes: selectors, as in Gleam, taking the oldest matching message across mailboxes, as Erlang's receive does; priority by Erlang's two-step receive (Language extensions, "Processes and mailboxes").

**Risks**

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Mode analysis | Definite aliasing is out of scope, because open types cover partly bound data. But full unique modes, which the reference manual says mmc has not fully implemented, the open-term rules and mode inference, which `LIMITATIONS.md` calls "a bit imperfect", have no reference implementation to test against | Implement the 22.01 mode rules first, with differential tests against mmc; build separate test suites for uniqueness and open terms, written from the manual and this design; keep mode inference opt-in, as mmc does; make mode errors name the waiting goal and variable |
| Precise GC on LLVM statepoints | Subtle correctness bugs; statepoint overhead | GC torture mode; header-word layout first |
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
- [Overton: Precise and expressive mode systems for typed logic programming languages](https://mercurylang.org/documentation/papers/dmo-thesis.ps.gz) (PhD thesis, University of Melbourne, 2003)
- [Overton, Somogyi and Stuckey: Constraint-based mode analysis of Mercury](https://mercurylang.org/documentation/papers/ppdp02_mode.pdf) (PPDP 2002)
- Measurements in this document were made with Melbourne Mercury 22.01.9 (`hlc.gc`), gcc 13.3, SWI-Prolog 9.0.4 and CPython 3.13 on a 2-vCPU cloud VM.
