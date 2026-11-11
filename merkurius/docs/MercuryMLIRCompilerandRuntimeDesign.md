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
| Process failure | Monitors, delivered with processes; links deferred (Runtime system, "Exceptions, links and monitors") | Supervision depends on them |
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

## Coding conventions

Code follows Flang's style, formatted by the repository's `.clang-format`, with these rules of our own.

**Functions, not named local lambdas.** A function's body should read without the implementations of helpers in the middle of it, and a helper should say what it uses. A named local lambda does neither: what it captures is state it imports, much as a global is, and it cannot be used anywhere else, which also hides helpers that do the same job.

- **A static function** if it needs no generator or checker state, and a member function if it does. Names have Flang's leading capital.
- **What it uses is a parameter.** `failed_` and the like stay members, since they are the class's own state already.
- **A lambda stays as an argument** to a higher-order function, such as `llvm::all_of`, written at the call. One that would be named only to be passed on is written at the call too, or is a function if it is long.
- **No short wrapper** where the call it wraps reads as well: a lambda for `IntegerConstant(i64, value, loc)` is that call.
- **A struct for a long list.** A helper that uses many of its caller's locals takes them in a struct the caller fills in, as `ParallelLoop` in MLIRGen, or is a member of a small class that holds them, as `LineWriter` and `TriviaParser` in the canonical unparser, rather than a dozen parameters. Elsewhere, plain parameters.
- **A recursive walk** that remembers what it has visited is an overload that takes the memo table, called by the public function with an empty one, as `ReplaceLeaves` and `ForEachLeaf` in `types.cpp`.

All of `lib` and `runtime` was converted to these rules (finished 2026-11-06), one file at a time and in stages within a file: 267 named lambdas. One is left, in `Parser/message.cpp`: that file is Flang's, whose `Message::Emit` has the same `emitAttachment` lambda, and it stays close to Flang's so that the two can be compared. The unit tests' few are left unless they get in the way. Once they were functions, two pairs of walks over a type turned out to be the same walk for different leaves, and each pair became one.

**A refactoring changes nothing else.** A change that restructures code changes no behaviour and makes no other cleanups, and it is checked so:

- The tests, and the same `merk` and LLVM IR before and after for every program of `test/CodeGen`, with and without OpenMP.
- For a checker, also the errors and the debug dumps of every program of `test/CodeGen` and `test/Semantics`, which the IR does not show: the parse tree, the resolved names, the semantic form, the nonlocals, the types, the polymorphism, the modes, the determinism, and the unparsed source.
- Then the sweeps, generated programs against mmc, and the term fuzzer, for a group of such changes rather than each one.

The only difference the lambda conversion made: a braced list is evaluated in order and function arguments need not be, so `array.bounds` moved a constant ahead of a size it does not depend on.

**Temporary code is marked.** A stand-in that a later step replaces carries a comment `Temporary (chunk N step M, docs/INPROGRESS.md):` saying what replaces it, so that it can be found when that step comes.

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
| Polymorphic predicates, a type class and a `type_spec` pragma (`poly.m`) | One C function per predicate, with a hidden type-info argument per type variable and elements passed as one word each, `MR_Box` being only the word's C type; a `list(float)` costs 16 bytes per element, no more than a `list(int)`, so nothing is allocated for a float; type classes are passed as dictionaries; `type_spec` adds a copy that takes `int` and compares with `X == Y` instead of generic unify | Language extensions: generic processes |
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

   **Running the fuzzer.** `utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]` builds the target with the build tree's clang and runs two fuzzers, one after the other, each for `SECONDS` (default 300) in `JOBS` processes (default 40), after it has loaded its corpus: first one of inputs up to 4 KB, then one of large inputs, up to 64 KB, with a corpus of their own. A run takes about 12 minutes.
   - **Building:** the script writes a ninja file, so the sources compile in parallel and only those that changed are recompiled: about 45 seconds from scratch, a few seconds after a typical change.
   - **Fork mode:** the fuzzer runs with libFuzzer's `-fork`, in which one parent process keeps the corpus and starts short fuzzing jobs. An input is added to the corpus only when it covers a new feature. Earlier runs used `-jobs`, where every process added its inputs to the shared corpus and replayed all of them at startup; the corpus grew to over 250,000 files, and replaying it took most of each run.
   - **Distilling:** features are many more than edges, so even in fork mode a corpus grows with every run, and loading it, which runs every input once in a single process before fuzzing starts, grew with it: by November 2026 the small corpus had 17,716 files and took 12 minutes to load, and the large one, 15,660 files and 195 MB, took over half an hour. Distilling them then kept 12,089 and 11,818 files, 49 MB and 131 MB, so loading is about a third faster: most of a corpus covers features no smaller set does. `utils/fuzz/distill-term-corpus.sh` keeps, with libFuzzer's `-merge=1`, a smallest set of inputs that covers all that a corpus covers. `run-term-fuzzer.sh` runs it before fuzzing, and it distills a corpus that has doubled since it was last distilled (`GROWTH`, default 2), keeping the corpus it replaces as `corpus.prev` until the next time; `--force` distills both at once.
     - *Rejected: distilling on every run.* Distilling runs every input once, as loading does, so it would add a load to every run for little gain; doubling bounds the load to about twice the distilled corpus.
     - *Rejected: capping the corpus size.* Dropping inputs by age or size would lose coverage that distilling keeps.
   - **Loading in parallel:** upstream libFuzzer, in fork mode, runs every input of the corpus once in one process before it starts a job, so that the parent knows every feature the corpus covers; even distilled, that took 8 minutes for the small corpus and 23 for the large one. Our libFuzzer (`compiler-rt/lib/fuzzer`, a local change marked "Merkurius (local, not upstream)") deals the inputs, by size, out to `-fork_load_jobs` processes, by default one per hardware thread, each with a merge control file of its own, and makes one greedy pass over all they recorded: each records the features new to it, so together they record every feature. The small corpus then loads in 46 s and the large in 97 s, held up by its slowest input, of 42 s. `run-term-fuzzer.sh` rebuilds the runtime with the runtimes' ninja and relinks the fuzzer when it changes.
   - **One fuzzer at a time:** measured on 2026-11-06, each configuration once, from the same distilled corpora, with 600 s of fuzzing for each fuzzer on 24 cores of 2 threads:

     | Configuration | Wall time | Load small / large | New features small / large |
     | --- | --- | --- | --- |
     | Upstream, 32 and 8 jobs at once | 33 min | 480 s / 1,363 s | 588 / 360 |
     | Parallel load, 32 and 8 jobs at once | 15 min | 60 s / 299 s | 478 / 226 |
     | Parallel load, 40 jobs, then 40 | 22.5 min | 46 s / 97 s | 366 / 1,155 |

     The large fuzzer gains the most from having every core: with 40 jobs it ran 2.8 times the inputs it ran with 8, and passed the first configuration's 360 new features in about 200 s. The small fuzzer gains nothing from 40 jobs over 32, since 8 more share the cores; its counts vary more from run to run than between these configurations. So the fuzzers run one at a time, for 300 s each.
     - *Rejected: `-keep_seed=1`.* It skips the load, but then the parent knows none of the corpus's features: a job's new inputs are almost all candidates, the parent merges them one job at a time, and it starts no new job meanwhile. With 32 and 8 jobs at once it ran about half the inputs, found 225 and 205 new features, and grew the small corpus from 12,089 to 18,940 inputs, of which 94 covered anything new.
     - *Rejected: two fuzzers at once.* While the large fuzzer loaded, in one process, the small one had the machine; once loading is fast, they share it, and the large one, with a fifth of the jobs, finds the least.
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
  - **Scopes.** The head's `!X`, `!.X` and `!:X` bring it into scope for the clause. Its first version is the head's `!.X`, and its last becomes the head's `!:X`, with a unification only when the body leaves it at a version already in the head. `some [!X]` and a trace goal's `io(!X)` and `state(M, !X)` bring a new one into scope inside the goal, hiding any of the same name, with a warning, as in mmc, though the manual says nothing about it ("the new state variable !X shadows the one outside"); a lambda expression's head does not warn (`desugar/state-var-shadowing.m`). For a trace goal, its first and last versions are the scope's variables.
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
| Type-changing field updates | "The return value is a copy of the first argument with the value of the labelled field replaced by the second argument" ("Field update"), which says nothing of the copy's type | The copy's type may differ in a type parameter that occurs in the field's type and in no other field of the field's constructor, whatever the type's other constructors have: with `box(T) ---> box(contents :: T, count :: int)`, `box("s", 4) ^ contents := 5` is a `box(int)`, and with `u(T) ---> one(x :: T) ; other(y :: int, z :: T)`, `X ^ x := 1` takes a `u(string)` to a `u(int)`. A parameter of two fields, as in `two(T) ---> two(a :: T, b :: T)`, or of none, a phantom one, stays: `X ^ a := 1` on a `two(string)` is a type error | mmc (`Semantics/types/field-updates.m`, `CodeGen/field-retype.m`) |
| Lexical rules | See "Lexer" | | mmc, each case settled by an experiment |
| An `atomic` goal's alternative that does not use its inner state, as in `... or_else Y = 0` | Not described | A mode error: the inner state's final version is left free in that alternative, though it is the alternative's first. No report found among the GitHub issues | Neither side's text decides it; we accept it, since desugaring gives such an alternative an inner state that it leaves as it was. Check again when the STM is implemented |
| `require_switch_arms_<det> [X] Goal` whose goal is a switch on another variable | An error: "If GOAL is not a switch on VAR ... the compiler is required to generate an error message" ("Goals") | No such error; the arms of the switch on the other variable are checked against `<det>`, and their errors name that variable. `require_complete_switch` reports it. No issue on GitHub mentions it | mmc |
| A `promise_equivalent_solution_sets` goal without `arbitrary` | It must have at least one `arbitrary` goal inside it ("Goals": "of which there should be at least one") | No error or warning; the goal is as its own goal, with no cast. No issue on GitHub mentions it | mmc |
| A function mode with more than one solution | Any mode of a function whose arguments are all input can only be `det`, `semidet`, `erroneous` or `failure` ("Determinism categories") | Only the primary mode, `f(in, ...) = out`, is checked, and by its inferred determinism, not the declared one. A body with more than one solution is "Error: invalid determinism for \`f'(in) = out: the primary mode of a function cannot be \`multi'", naming the inferred determinism (`nondet` for a body that can also fail), whether the mode is declared in the `func` declaration or by a `mode` declaration. The declaration is then checked as usual. `:- mode g(in) = out is multi.` with a `det` body has only the warning that the declaration could be tighter. Other modes whose arguments are all input are not checked. No issue on GitHub mentions it | mmc |
| Comparing a NaN | "Implementations must throw an exception when comparing a 'not a number' (NaN) value" ("The standard ordering") | Only for a float reached through a `type_info`, and then the program ends with `Mercury runtime: incomparable floats in compare/3`, status 1, which `try` does not catch. `compare(R, N, 1.0)` gives `(=)`, and so does `compare(R, m2(N), m2(1.0))` for `:- type m ---> m2(float)`, while `compare(R, g2(N), g2(N))` for `:- type g(T) ---> g1 ; g2(T)`, a tuple, a list or an array ends the program. A notag type's `T` is the exception: `compare(R, pp(N), pp(N))` for `:- type pp(T) ---> pp(T)` gives `(=)`, through a `type_info` too. No issue on GitHub mentions it. | mmc, for where it raises: a float declared as one gives `(=)`, and a float a type variable stands for raises. Not for the notag type's `T`, which raises as the manual says, since matching it would take a test of the descriptor in `compare.T`. Merkurius's is an exception, for `try` to catch once it exists (`Uncaught Mercury exception:`, status 1 until then). |
| Overloading in a clause with many overloaded symbols | "There is a type ambiguity, and the compiler must report an error" ("Predicate and function type declarations") | After its warning `highly ambiguous overloading`, an ambiguity it would report in a small clause is not reported, and one type assignment is taken: in one clause with functions `a/0` and `c/0`, `io.write('a')` wrote the character and `io.print('c')` called the function. In a small clause, both are errors. No issue found among GitHub's for "ambiguous overloading". | Not mmc: merkc always reports it, since which is taken has no rule to follow. |
| The type_infos of a `foreign_proc` | Does not mention them | A polymorphic `foreign_proc`'s C code has `TypeInfo_for_T`, an `MR_Word`, for each type variable by the name in the declaration, including one that occurs only inside another type or is not used. | mmc. |
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
| A unification of two unique constants | A unification of two bound values is a test ("Unifications") | With `t ---> c(int) ; kk`, `S = kk, D = kk, ( if S = D then ... else ... )` is "mode error: argument 1 did not get sufficiently instantiated ... Final instantiatedness of `HeadVar__1' was `bound(kk)', expected final instantiatedness was `unique(kk)'" in mmc's own `__Unify__` for `t`, specialized to `in(unique(kk))`; when one side has been passed to an `in` argument first, it is "implicit determinism declaration not satisfied" there instead. A type of constants only is not affected. Found by `utils/codegen-oracle`, 2026-11-04. No GitHub issue | Not followed: the test succeeds (`test/CodeGen/unify-unique-constants.m`). The layout generator builds each value in a function of its own, so that mmc sees only ground values |
| An abstract type as a supertype | The supertype's definition must be "in scope where the subtype definition occurs, by normal module visibility rules" ("Subtypes") | `:- type public =< secret ---> secret.`, where `secret` is abstract in the imported interface, is accepted, and compiles, checked against the definition the module cannot see; an abstractly exported subtype is rejected ("cannot be a supertype because it is an abstract type"). No GitHub issue | The manual: an abstract type's constructors are not visible, so it cannot be a supertype (`test/Semantics/types/subtypes/abstract-supertype.m`) |
| Unresolved predicate overloading after type ambiguities | An ambiguity is an error ("Overloading") | A clause with type ambiguities, such as `show(solutions(g), !IO)` (the function `solutions/1` or the predicate `solutions/2` curried) and `show(list.length(L), !IO)`, that also passes `print_cc` as a closure where both `io.print_cc/3` and the module's own `print_cc/3` fit, is "unresolved predicate overloading ... You need to use an explicit module qualifier", then "Software Error: predicate `hlds.pred_table.find_matching_pred_id'/10: Unexpected: unresolvable predicate overloading", without the type ambiguities. Either alone is reported as an error. Found writing `CodeGen/solutions-library.m`, 2026-11-06; reduced to six goals of its `main`. No GitHub issue | Not compared: merkc's `io` has no `print_cc` yet |
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
| A named unique inst passed to an `in` argument | "A value is not considered `unique` if it might be needed" again; `in` is `ground >> ground` ("Unique modes") | With `:- inst up == unique(p(unique, unique))`, a variable `P0` of inst `up` stays `up` after a call that takes it `in`, so `cp(P0, Q), pdi(P0, P)`, with `cp(P::in, Q::out) :- Q = P` and `pdi(p::di, p::uo)`, is accepted, though `pdi` destroys what `Q` refers to. `upn == unique(p(ut, ut))`, an unnamed `unique(p(unique, unique))` and `w(unique)` with `w(I) == unique(p(I, I))` are shared, as is a named `unique`. The library's `array_di` and `array_uo` are of such an inst, so the hole is visible at run time: after `array.init(2, 1, A0), B = keep(A0), A = put(A0, 5)`, with `keep(A) = A` taking `in` and `put` taking `array_di`, `array.lookup(B, 0)` is 5. No GitHub issue | Not mmc: unsound. The value is shared after the call whatever its inst (`modes/shared-named-inst.m`, not compared with mmc); the program generator avoids the case |
| A lambda expression's mostly clobbered argument | `mostly_clobbered` is "not referenced anywhere, but needed on backtracking" ("Backtrackable destructive update"); a value that `di` clobbered is not available on backtracking | mmc does not check a lambda expression's head variables' final insts when they are clobbered: `P = (pred(X::(unique >> mostly_clobbered), Y::out) is det :- updu(2, X, Y))` is accepted, though `updu` may destroy what the caller keeps for backtracking. Nor does it treat the head variables as used at the end, which is harmless only because one whose modes do not clobber it is shared in the body. No run-time effect yet, as mmc does not reuse such values | Not mmc: unsound, and mmc's rule is not clear (`X::in` clobbered by `ground >> clobbered`, then read, is also accepted). merkc checks the final insts with and without `-fmmc-uniqueness`, and the generator's `--array` lambda expressions take arrays `array_di` (`test/Semantics/uniqueness/lambda-clobbered.m`) |
| Arrays' uniqueness | Arrays are updated destructively ("Destructive update"; library manual, `array`: `set` takes `array_di`) | The library declares `uniq_array(I) == array(I) == ground`, "XXX work-around", so `array_di` takes any array: `A1 = A0, array.set(0, 5, A0, A)` compiles, and the program then prints 5 for `array.lookup(A1, 0)`, which is 0. Documented in the library itself, so not a bug to report | Not mmc: unsound. By default merkc loads a sound stub, with the insts the library's comments intend and `array_ui` modes for the predicates that read arrays (Standard library); mmc's declarations are behind `-fmmc-uniqueness` |
| Unique-input modes and fields of copied values | "It is also not possible to use unique-input modes; only destructive-input and unique-output modes work" ("Limitations of the current implementation"); a unique value has "only one reference" ("Destructive update"), which a copied value always has | A field taken out of a unique term and used after is shared in the term whatever its type, so the term no longer matches `unique`: `T = t(X)`, with `T::ui` and `X::out`, is "argument 1 did not get sufficiently instantiated. Final instantiatedness of `T' was `unique(t(ground))'", and `T0 = t(X), updu(X, T0, T)`, with `T0::di`, is an error at `updu`, for `int`, `float`, `char` and enumeration fields too; so a `ui` argument can only be passed on. No GitHub issue | Not mmc, which only rejects more. By default, a value of a type whose values are copied, a number, a character or an enumeration, matches a unique inst when it is shared or `mostly_unique`, not when clobbered, so reading such fields of a `ui` or `di` term keeps it unique (`uniqueness/copied.m`); a field of another type, as a string, which `string.set_char` may update in place, or a term, still makes the term shared in that part. mmc's behaviour is behind `-fmmc-uniqueness`, which the oracles use |
| `mostly_unique(...)` | Only `unique(...)` is described, "like `bound(...)`" ("Destructive update"), and `mostly_unique` "equivalent to `unique`" for backtracking ("Backtrackable destructive update") | `mostly_unique(p(mostly_unique, unique))` is accepted, and checked as a bound inst whose top level is `mostly_unique`: a field taken out of it as `uo` is an error. No GitHub issue | mmc |
| A `mostly_unique` bound inst with a bound argument | `mostly_unique` is `ground(mostly_unique)` of any value ("Unique modes") | `T6 = t(3), ( T = T6 ; T = T6 )`, with `T::muo`, is an error: "Final instantiatedness of `T` was `mostly_unique(t(mostly_unique(3)))`, expected final instantiatedness was `mostly_unique`". A constant, `T6 = e1`, is accepted, and so is such an inst matched to `ground` (`in`), or of a variable that was an argument of a call before, which mmc has found ground then. No GitHub issue | mmc, which only rejects more |
| A constructed unique value passed `in` after a `ui` call | "A value is not considered `unique` if it might be needed" ("Unique modes"); a callee may keep an `in` argument in an output | A term built in the clause becomes shared when it is passed to an `in` argument, but not after it has been passed to a `unique >> unique` (`ui`) argument: `T0 = t(1), ui(T0), wrap(T0, L), T = T0`, with `wrap(X, [X])`, `T::uo` and `L::out`, is accepted, though `L` holds `T`. Without the `ui` call, or with a `unique(t(ground)) >> unique(t(ground))` or `unique >> ground` one, it is an error. No GitHub issue | Not mmc: unsound, and mmc's rule is not clear. The value is shared after an `in` call either way (`uniqueness/ui-then-in.m`, not compared with mmc); `uisweep/s4/u304_21.m` differs from mmc for this reason |
| A negated call of an `any_pred` closure | "if any non-local solver type variable or higher-order variable with inst `any` is used in a negated context, the goal must be placed inside a ... promise" ("Solver types and negated contexts") | `P = (any_pred is semidet :- r(X)), not P`, with `X::ia`, is accepted with no promise. No GitHub issue | Follow mmc: only a nonlocal whose inst has an `any` part counts, not an `any_pred` closure |
| A type variable whose first constraint has it inside a type | A constraint is `TYPECLASS(TYPE, ...)`, with any type ("Type class constraints on predicates and functions") | For `cons(X, N) <= sh(list(T))`, mmc takes T's type_info from the typeclass_info's place for `list(T)`, and so passes `list(T)`'s as T's: `cons(42, S)` with `S = type_name(type_of(X))` prints `list.list(int)`, and `io.write(X)` for an `int` kills the program: "MR_expand_functor_args: invalid sectag_locn". It does so even when a later constraint has T itself, `<= (sh(list(T)), sh(T))`. No GitHub issue | Sound: the same constraint as mmc, and T's type_info is the argument of `list(T)`'s (`polymorphism/constrained-types.m`); not compared with mmc |
| Several clauses of a predicate with an existential constraint | Each clause gives the existentially quantified variables the same types, and satisfies the existential constraints ("Semantics of type quantifiers", "Existential class constraints"); a predicate may have several clauses | `:- some [T] pred multi(ab::in, T::out) is det => sh(T).` with `multi(a, 1). multi(b, 2).` is a mode error in mmc: "mode mismatch in disjunction. The variable `TypeClassInfo_for_sh' is ground in some branches but not others". mmc makes the typeclass_info in each clause, and gives it back once after their switch, under a variable that only one branch binds. One clause, an if-then-else in one clause, or no constraint (the type_info is made in every branch) is accepted. No GitHub issue | Not followed: merkc accepts it, and each clause gives the typeclass_info for its own proof (`polymorphism/existential-clauses.m`); not compared with mmc. The generator gives existentially typed predicates one clause |
| The quotient of the least integer by -1 | `//` and `rem` are `det` and throw only for a zero divisor; `X rem Y = X - (X // Y) * Y` (library manual, `int`) | With the operands known, mmc folds `min_int // -1` to `min_int` and `min_int rem -1` to 0. With them computed, as from a `no_inline` predicate, the program dies of SIGFPE, the x86-64 division's trap, for `//`, `rem`, `div` and `mod`, and for `int8` and the other signed types alike. No report found among the GitHub issues | The manual: it wraps, to `min_int` and 0, as mmc folds it, wherever the operands come from (`test/CodeGen/arithmetic.m`) |
| Negating a float zero | `- X` is unary minus (library manual, `float`), which in IEEE arithmetic makes 0.0 into -0.0 | `-Z`, with `Z` = 0.0, is 0.0, while `Z * -1.0` is -0.0: mmc's negation is `0.0 - X`. `max` and `min` decide -0.0 and NaN as `X >= Y` and `X =< Y` do, `round(X)` is `floor(X + 0.5)`, and `float.pow` multiplies by squaring, rounding as that does | mmc, as these are what programs print (`test/CodeGen/float-math.m`) |
| Mode inference of a unique array argument | Modes are inferred for a predicate without mode declarations with `--infer-modes` (user's guide); nothing restricts inferring unique modes | `zero(I, !A) :- ( if I < 0 then true else !A ^ unsafe_elem(I) := 0, zero(I - 1, !A) ).`, declared `:- pred zero(int, array(int), array(int)).`, called with an array from `array.init`, is "unique-mode error: the called procedure would clobber its argument, but variable `STATE_VARIABLE_IO_0' is still live", in a clause for `zero(di, (ground >> not_reached), out(not_reached))`: the array's inst at the call is `ground`, and the variable named is not one of that clause's. The same for the translation of `saxpy`'s loop written by hand. No GitHub issue | Not mmc: merkc infers `zero(di, di, uo)`, which is what a loop that updates an array in place translates to (Loops, "Translation"). The message's variable is an mmc bug either way |
| Comparing terms that hold closures | Unifying higher-order terms is an error; "indirect attempts (via polymorphic predicates ...) may result in an error at run-time rather than at compile-time" ("Higher-order") | A comparison or unification that reaches a closure ends the program with "Mercury runtime: called compare/3 for \`pred' type" (or "unify/2"), status 1, for a function's closure too: `compare(R, P, P)`, `compare(R, [F, F], [F])`. But `compare(R, [F], [F])` gives `(=)`: the two lists are one cell, and a term is equal to itself without looking inside. No GitHub issue | mmc's message and status; not its shortcut, which depends on which terms its optimizations merge. merkc always reaches the closure (`test/CodeGen/closure-values.m`) |
| Clauses for one mode in an instance | An instance's clauses "are just like the clauses used to define ordinary predicates or functions" ("Instance declarations"), which may be for one mode ("Different clauses for different modes") | `(step(X::in, Y::out) :- Y = X + 1)` in an instance is an error for each argument: "undefined symbol \`::'/2", and "undefined symbol \`in'/0": mmc reads `X::in` as a term there. No GitHub issue | mmc: an error, once for the clause (`Semantics/modes/clauses-for-one-mode-errors.m`). The manual cannot be followed as written: clauses for one mode make a predicate impure unless a `promise_equivalent_clauses` or `promise_pure` pragma names it, and no pragma can name one instance's method. Nothing is lost: `pred(m/N) is p` in the instance, with `p`'s clauses for each mode and the pragma for `p`, does the same |
| A negative precision from an input in `string.format` | "The implementation uses the sprintf() function in C grades" (library manual, `string`), and a precision may be `.*`; C reads a negative one as if there were none | `format("%.*d|%.*f", [i(-1), i(5), i(-1), f(1.5)])` gives `%.0-1ld|%.0-1f`, the C format that mmc made with the number in it, not a conversion. A negative width from `*` is left-justified, as in C. No GitHub issue | C: the precision is absent, giving `5|1.500000` |
| A range from `min_int` | `Lo .. Hi = [Lo, Lo + 1, ..., Hi] if Lo =< Hi, and [] otherwise` (library manual, `list`) | `min_int .. min_int` and `min_int .. min_int + 1` never return: the program grows until memory runs out (4 GB under `ulimit -v`, 380 GB without). `min_int + 1 .. min_int + 2` and `max_int - 1 .. max_int` are right. No GitHub issue | The manual: `[-9223372036854775808, -9223372036854775807]` (`CodeGen/list-library.m`) |
| An int beyond 32 bits to `char.from_int` | `to_int(out, in)` "fails for integer values outside of the Unicode range" (library manual, `char`) | Only the int's low 32 bits count: `from_int(0x100000041, C)` gives `'A'`, and `from_int(min_int, C)` gives `char.det_from_int(0)`; `-1`, `0x110000`, `max_int` and `0x80000041` fail. `to_int(out, in)` and `det_from_int` alike. No GitHub issue | mmc, as programs could depend on it (`CodeGen/char-library.m`) |

Reports checked and found not to be mmc bugs: GitHub issue 31, "Lambda scoping error", which was the debugger's query translation; the same code compiles in a module. It shows a warning that quantification should give too, "variable `W` has overlapping scopes", when a lambda's argument has the name of a variable outside it.

### mmc bugs to report

The mmc 22.01.9 bugs found so far, to report to the Mercury developers once each has a small reproducer. The register above has the evidence for each; a row's name is given in quotes. Each was searched for among the GitHub issues when it was found. Those marked *test* have a reproducer in merkc's tests already; for the others, a small one is still to be written. Add each new one here when it is recorded in the register.

**Wrong at run time**

- **Comparing a NaN does not throw** ("Comparing a NaN"). `compare(R, N, 1.0)` gives `(=)`, against the manual's "must throw", and one that does stop, through a `type_info`, ends the program where `try` cannot catch it. The same two floats give either, by whether the compiler knew the type. Not reported.

- **An ambiguity goes unreported in a clause with many overloaded symbols** ("Overloading in a clause with many overloaded symbols"). The same goal means either of two things, or is an error, by how much else the clause overloads. Not reported.

- **A negative precision from an input in `string.format` writes the C format** ("A negative precision from an input in `string.format`"). `format("%.*d", [i(-1), i(5)])` gives `%.0-1ld`, where C, which the manual says the implementation uses, gives `5`. Not reported.

- **A range from `min_int` never returns** ("A range from `min_int`"). `min_int .. min_int` runs until memory runs out, where the manual gives `[min_int]`. Not reported. *test*: `CodeGen/list-library.m`.

- **`char.from_int` keeps an int's low 32 bits** ("An int beyond 32 bits to `char.from_int`"). `from_int(0x100000041, C)` succeeds with `'A'`, where the manual says an int outside the Unicode range fails. Not reported. *test*: `CodeGen/char-library.m`.

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
- **Unresolved predicate overloading after type ambiguities** ("Unresolved predicate overloading after type ambiguities"): "find_matching_pred_id ... Unexpected: unresolvable predicate overloading". Not reported.
- **Mode analysis of a coerce through a tuple parameter** ("A coerce through a tuple parameter"): "modecheck_coerce_from_ground_make_bound_functor ... missing constructor for result type". Not reported. *test*: `Semantics/types/coerce/tuple-parameter.m`.
- **Merging branches that leave a variable partly free** ("A unification of two partially instantiated terms after branches"): "merge_instmapping_delta_vars ... error merging var 7". Not reported. *test*: `Semantics/modes/merge-crash.m`.
- **Mode analysis with constrained insts**: "check_hlds.modecheck_util.add_constrained_inst/4: Unexpected: SubInst differs", on `modes_25_24.m` from `generate-mode-programs.py --seed 25`. Not in the register yet, and not reduced. Not reported.
- **Unifying values of a type with existential constraints** ("Unifying values of a type with existential constraints"): a determinism error in mmc's own `__Unify__`, and in a larger module a crash in "modecheck_queued_proc ... found detism error". Not reported.

**Wrong errors or wrong code**

- **A unification of two unique constants** ("A unification of two unique constants"): with `t ---> c(int) ; kk`, `S = kk, D = kk, ( if S = D then ... )` is a mode error, or a determinism error, in mmc's own `__Unify__` for `t`. Not reported. *test*: `CodeGen/unify-unique-constants.m`.
- **Mode inference loses an array's uniqueness** ("Mode inference of a unique array argument"): a recursive predicate that updates an array in place, with its types declared and its modes inferred, is a unique-mode error, which names a variable, `STATE_VARIABLE_IO_0`, that is not in the clause. Not reported.

- **Type inference reports the first pass's errors** ("Errors during type inference"), and **does not reach a fixpoint** ("A fixpoint of type inference") for `f(H1, _, H3) = H4 :- H4 = f(H3, {H1}, H1)`. Not reported.
- **Overlapping instances through a repeated variable** ("Instances that overlap through a repeated variable"): `d(p(A, B))` and `d(p(A, A))` pass mmc's checks, and the C compiler then rejects the generated code. Not reported.
- **An abstract imported type as a supertype** ("An abstract type as a supertype") is accepted. Not reported. *test*: `Semantics/types/subtypes/abstract-supertype.m`.
- **A conditional expression in a term assigned to an unused variable** ("A conditional expression in a term assigned to an unused variable"): a spurious mode error. Reported as GitHub issue 120, closed by its reporter without an answer.
- **A partially instantiated term unified with a bound one** ("A partially instantiated term unified with a bound one"): a mode error in mmc's own `__Unify__`. Reported as GitHub issue 74, open.
- **Several clauses with an existential constraint** ("Several clauses of a predicate with an existential constraint"): `multi(a, 1). multi(b, 2).`, with `=> sh(T)`, is "mode mismatch in disjunction" for mmc's own `TypeClassInfo_for_sh`. Not reported. *test*: `Semantics/polymorphism/existential-clauses.m` (merkc's version); the reproducer is that test's `multi`.
- **The type_info of a type variable constrained inside a type** ("A type variable whose first constraint has it inside a type"): with `<= sh(list(T))`, T's type_info is `list(T)`'s, so `type_of(X)` for an `int` is `list.list(int)`, and `io.write(X)` aborts the program. Not reported. *test*: `Semantics/polymorphism/constrained-types.m` (merkc's sound version); the reproducer is that test with `type_name(type_of(X))`.

- **The quotient of the least integer by -1** ("The quotient of the least integer by -1"): `min_int // -1`, `rem`, `div` and `mod` kill the program with SIGFPE when the operands are computed, though mmc folds them to `min_int` and 0 when they are known, and the manual makes them `det`. Not reported. *test*: `CodeGen/arithmetic.m` has merkc's result.

**The reference manual**

- `:- mode in ^ in = out is semidet` ("`:- mode in ^ in = out is semidet`"): the example is wrong; `in ^ elem(in) = out` is what it means.
- "Limitations of the current implementation" says nested unique modes cannot be used ("Nested unique modes"), but mmc accepts them.
- `mostly_unique(...)` is accepted by mmc, but not described ("`mostly_unique(...)`").
- An instance's clauses are said to be just like a predicate's, but mmc rejects clauses for one mode there ("Clauses for one mode in an instance"). The manual is unclear rather than mmc wrong: its rule for such clauses needs a pragma that cannot name an instance's method. *test*: `Semantics/modes/clauses-for-one-mode-errors.m`.

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
  - **What it exports.** A procedure's body goes in if the procedure is polymorphic, takes higher-order arguments or is small; otherwise only its signature does. Those are the bodies a caller gains from: without them a caller cannot specialize or devirtualize, and mmc without `--intermodule-optimization` calls `array.unsafe_lookup` out of line for each element. The interface section stays the export list: what the optimization section carries gives the importer's optimizer more to work with, and no new names to the importer's source.
  - **Two parts, two rebuild rules.** The signature part is the documented contract, and a change to it rebuilds every importer. The optimization section is the compiler's, and a change to a body in it rebuilds only the importers that inlined or specialized that body. The exception is a body exposed with `pragma inline`, which is part of the contract, as a C++ header is (Optimizations, "`pragma inline`"). This is designed, and depends on the interface-file format below; nothing reads or writes the section yet.
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
- **Duplicate declarations** of the module's own are reported as mmc reports them, and the first is kept (`Duplicate`): a predicate or function declared twice, whatever its types, modes or `with_type`, a method and a predicate of one name and arity among them; a type, inst or mode defined twice, and a type as both a discriminated union type and an equivalence type, reported at the former. A field label used twice in one type, in one constructor or two, is an error, as the manual has it ("Discriminated unions"), reported at the later one, which gets no field access functions; a label may be used again in another type. A type declared abstractly twice is a warning. A type's abstract declaration and a `foreign_type` pragma go with its definition, which becomes the symbol's declaration and source. A duplicate type's constructors are not its type's, and a duplicate type class's methods are reported with the class only. Imported modules' duplicates are not reported.
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
- **Signatures.** Each predicate and function, constructor, field selector (`f/1`) and field update (`'f :='/2`) has a signature, made from its declaration. A field update's result has a type variable of its own for each type parameter that only its field has, which may then change (the register, "Type-changing field updates"). The module's own are made, and its equivalence types expanded, before inference starts, so that their errors are reported; an imported module's on first use. A function's result is its last argument. A `with_type` declaration adds its higher-order type's arguments. A signature has the declaration's class constraints (`<= c(T)`), and each of their variables must occur in the argument types. A method's signature has its type class's parameters as its first type variables, and its type class's constraint first, as if it were declared `<= c(T)`.
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
- **Clauses for one mode** (reference manual, "Different clauses for different modes"). A predicate or function with more than one mode, and a clause whose head arguments have modes, is impure unless it is promised pure or `promise_equivalent_clauses`; the latter does not promise the clauses' own goals pure. With one mode, mode annotations change nothing, as in mmc. Desugaring records the clause's modes, and mode analysis decides which procedures it is for (Mode analysis, "Clauses for one mode").
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
- **Unique modes, the first phase** (Overton, 3.2.1 and 3.2.3; reference manual, "Unique modes"). Liveness decides sharing: a test of two variables both used after makes both shared (definition 3.2.19); an assignment shares the value if its source and its target are both used after (one to a variable not used again mmc deletes); a deconstruction keeps the term's uniqueness, and a part that both the term and its variable use after is shared, in both (3.2.20). A field access or update is matched against the mode `in = out` but leaves the insts of the unifications it stands for: an access the field's part of the term's inst, shared in both if the term is used after, and an update a unique term of the term's other parts, shared if the term is used after, and the value's inst, shared if the value is (`uniqueness/fields.m`). The term's copy of a construction's argument is shared if the argument is used after, or if the term has more than one copy of it, as `p(A, A)` does; if the term is used after too, the argument is shared as well, as in the thesis, where mmc keeps its inst, unsoundly (see the register; `-fmmc-uniqueness`). A unique term's fields may be unique, and these rules apply to them as to any value: a field taken out of a term that is used after is shared in both, one passed to an `in` argument is shared after, and a term rebuilt from a shared field is unique only at its top, `unique(p(ground, unique))`. Passing a unique value to an `in` argument makes it shared after, since the final inst is `ground`, whatever its inst; mmc keeps some named insts unique there, unsoundly (see the register). A value of a type whose values are copied, a number, a character or an enumeration, has no other reference however it is used, so a shared or `mostly_unique` one matches `unique`: taking a number out of a unique term keeps the term unique, which makes `ui` arguments usable; mmc compares them as any other, and rejects that (see the register; `-fmmc-uniqueness`). A named inst is shared as its body is, unique parts below its top level too, and keeps its name if that changes nothing; a recursive one, as `ul == unique([] ; [unique | ul])`, becomes a generated recursive inst, printed unfolded once. A call whose mode clobbers an argument that is used after is "unique-mode error: the called procedure would clobber its argument, but `X` is still live", for a higher-order call too; with one procedure, mmc reports it before an argument's inst that does not match. `mostly_unique` and `mostly_clobbered`, and values needed on backtracking, are the second phase, after determinism analysis (pass 6); mmc reports those as mode errors too, so a test of this phase uses `det` code, or keeps a unique value's construction and its clobbering call free of choice points between them.
- **The `any` inst** (reference manual, "Solver types and negated contexts"). Negation as failure is unsound if a negated goal can constrain a solver variable, so a negation, or an if-then-else's condition, with a nonlocal whose inst has an `any` part, of a solver type or not, is a "purity error" unless it is inside a `promise_pure`, `promise_semipure` or `promise_impure` scope; as in mmc, a `promise_pure` pragma or an `impure` declaration does not count. The rule is checked after the goal inside, so an inner negation is reported first, as in mmc. A `pred` or `func` lambda expression is ground, so a nonlocal whose inst contains `any` is a purity error; an `any_pred` or `any_func` one, which the parser now accepts, may have them. As in mmc, a negated call of an `any_pred` closure needs no promise, though the manual asks for one (see the register).
- **Type conversions** (reference manual, "Type conversions", "Mode checking"). `Y = coerce(X)` waits until `X` is bound, and gives `Y` the inst the manual's traversal builds: a recursive inst node needs its type to be at most the result's; a bound node keeps its functors, each of which the result's type must have; a ground node of the same type stays, and one of a subtype becomes the bound inst of the subtype's constructors, down to a recursive type node, which is ground. A failure is mmc's message, with the path to the subterm ("in the first argument of function symbol `p`") and the reason: a functor the type lacks, or a ground value whose type is not a subtype. The result keeps the argument's uniqueness, and as for an assignment both are shared if `X` is used after. With `Y` bound, as in mmc, the converted value is unified with it, with a warning if that cannot succeed.
- **Mode inference** (Overton, 3.4.2; user's guide, `--infer-modes`; the reference manual does not describe it). A call of a predicate whose modes are inferred uses an inferred procedure whose initial insts its arguments match, as any call does, but not one that would clobber an argument used after, nor a rejected one; otherwise it makes a procedure. Its initial insts are the arguments', normalised: a ground inst becomes ground of the least unique uniqueness of its parts (definition 3.4.1), while a partly free inst or an inst variable stays, since mmc does not normalise below a bound inst either. A unique argument not used after the call is clobbered, `di`; the other final insts are those of the clauses, merged and normalised, so `p(X, Y) :- Y = X + 1` called with a shared `X` is `p(in, uo)`, and `id(X, X)` called with a unique list used after is `id((unique >> ground), out)`, as in mmc. The module is checked in passes, as mmc does with `--infer-modes`: each checks the declared procedures and then the inferred ones, those it made included, and a call of an inferred procedure not yet checked leaves the goals after it not reached in that pass, even when its final insts are known. Passes repeat until none makes a procedure or changes one's final insts, or, as in mmc, one finds a mode error in a procedure not inferred, which ends inference with that pass's errors; the iteration limit is an error, as mmc's "Mode analysis iteration limit exceeded", which a recursive mode with partly free insts can reach, since they grow with each pass, in mmc too (thesis, "Limitations of Mode Inference"). A procedure with a mode error is rejected, mmc's "REJECTED :- mode p(out(...))", and its error is not reported by itself: a call whose insts, normalised, are a rejected mode's waits, as a call that matches no mode does, so `bad(Y), Y = 3` makes `bad((unique >> ground))` instead; if nothing comes, the call is the error, with the rejected mode's error as a note, at the predicate's first clause, or in a lambda expression at it, as in mmc. Without `--infer-modes`, mmc checks a caller's goals after a call that makes a mode in no later pass, and may fail without a message; merkc does what `--infer-modes` does (see the register). Determinism is not inferred yet, so an inferred procedure has none, which pass 5 will give it.
- **Instance methods.** An instance's clauses for a method, and the clause that `pred(m/2) is p` desugars to, `m(HeadVar__1, HeadVar__2) :- p(HeadVar__1, HeadVar__2)`, are checked in each of the method's modes, as a predicate's clauses are, with the instance's types; an error says "in the clause for the method `m(in, out)` of instance `m.c(m.t)`", after mmc's "In clause for type class method implementation for `m(in, out)`".
- **Unifications** are classified by the insts of their two sides: construction when the left side is free, deconstruction when it is bound and the right side has free arguments, assignment, a test when both are bound, or a call to the type's unify procedure for tests on structured values. In a deconstruction or test, a bound argument is tested against a fresh variable for the bound variable's own argument, as mmc splits it, so the variable learns only the functor there: after `M = 1, S = c(M)`, with `S::in`, `S` is `c(ground)`, and `S = c(0)` after it can still succeed (`determinism/case-splits.m`). Without alias tracking, a value with free parts may not be copied while both copies stay live: unifying two free variables that are both live is an error, and so is assigning, or putting into a live term, a live value with free parts, as in mmc; a construction whose term is not used again may have free arguments. A constructed term is `unique`, and a shared copy of a live value is `bound`. A live argument of a construction whose term is live too is shared as well, as an assignment's copies are, which mmc does not do (see the register; `-fmmc-uniqueness` keeps mmc's rule). All of this is except at open types (Language extensions). At open types, a unification of an `any` value with a functor is a `get_structure`, and one of two `any` values is a full unification.
- **Calls** choose a procedure of the callee whose initial insts the arguments match, the most specific if several do, leaving out, as mmc does, one that would clobber an argument used after the call if another does not: `append(["a"], L1, L2)` with `L1` used after it is `append(in, in, out)`, run in place, not `append(di, di, uo)`, delayed until `L1` is no longer used (`modes/clobber-choice.m`). If none does, a bound argument may match a free initial inst: the call writes to a fresh variable and an implicit unification with the original follows; this is an implied mode, which mmc uses for arguments with free parts too. Determinism analysis counts the unification as a test, so the call can fail, and code generation calls the procedure into fresh values and tests each against the bound argument; both once ignored it, and such a call succeeded whatever the argument (2026-11-04, found by `utils/codegen-oracle`; `CodeGen/implied-modes.m`, `determinism/goals.m`). With one procedure, the error is at the first argument that an implied mode cannot fix, as in mmc. An argument that was free where the mode's initial inst is free gets the final inst; another is unified with it. A head variable's final inst must match the declared one; as in mmc the error says it "did not get sufficiently instantiated" if it has free parts, "became too instantiated" if the declared inst has, and otherwise "had the wrong inst", and it is reported at the last goal run, or at the clause for a fact. Only the first mode error of a procedure is reported, as by mmc, though its other clauses are still checked for the modes their calls infer. A function without mode declarations gets the default mode the manual defines: all arguments `in`, result `out`.
- **Branches.** Each branch of a disjunction starts from the instmap before it, as does the else part of an if-then-else, and its then part from the condition's; the branches' insts of the goal's nonlocals are merged after it (Overton, 3.1.4). `fail` is not reached after, and a branch not reached merges as nothing. A variable whose insts do not merge, because it is free in one branch and bound in another, or part of it is, is a "mode mismatch", reported at the goal run last, with a note at each branch, at the branch's first goal, and every other such variable in notes after them, as mmc reports each; with three or more branches, counting those not reached, where the variable is ground in all but one and free in that one, mmc says it is "ground in 2 out of 3 branches" and notes only that one. A variable that is free in some branch and not used after is not merged, as in mmc, but one with a bound inst that does not merge is still an error, even when not used after.
- **Negation and conditions.** While a negation, or the condition of an if-then-else, is checked, the nonlocals of the negation or of the if-then-else are locked: a unification or call that binds one that is free is a "scope error", at that goal, which waits, as a mode error does, for the variable to be bound outside. Quantification makes a variable of the condition and the else part two variables, so the condition may bind what it shares only with the then part. A goal may bind the free parts of a locked variable that is partially instantiated, as in mmc. A negation binds nothing after it. Open terms will be allowed to bind inside a promise scope, as the manual's rule for solver types requires.
- **Trace goals** (reference manual, "Trace goals"). A trace goal may not bind its nonlocals, since it may not be compiled in or not run: they are locked as a negation's are, so a goal outside that binds one runs first, and one that binds it inside is a scope error ("Y is bound inside a trace goal, but is not local to it"). The first version of its `io(!S)` state variable is `unique`, and the last is live at its end and must be `unique` again, as for mmc, which takes the io state with `io.unsafe_get_io_state` and gives it back with `io.unsafe_set_io_state(di)`: clobbering it first is a unique-mode error at the goal that clobbers it, and giving back a shared one is "the trace goal must give back a unique io state". Both versions are of type `io.state`, even if the goal does not use them. The state variable is a new one, and hiding one of the clause's warns, as in mmc ("the new state variable !IO shadows the one outside"). A trace goal is checked whether or not its `compile_time` condition holds, as in mmc; code generation leaves it out. A `state(M, !S)` parameter needs mutables, whose declarations desugaring reports as not yet implemented, so a procedure with one is not checked.
- **Clauses** of a predicate with several are the branches of a disjunction, as in mmc. Each is checked with new head variables, `HeadVar__N`, and head unifications `HeadVar__N = X` at the arguments, before the body; so an argument's variable is free at first, and used only until its head unification runs. A head unification that waits is reported only if no other goal waits, as mmc does, and so is the unification of a single clause's head argument that is not a variable, as `p(A, A)`, which desugaring puts at the start of the body. The head's insts at the end of the clauses are merged, with a mismatch noted at each clause's `:-`, and then compared with the final insts, at the last goal run in the last clause. A `true` that is a branch counts as a goal run, at its position, but one in a conjunction or as a clause's body does not.
- **Clauses for one mode** (reference manual, "Different clauses for different modes"). A procedure's clauses are those without modes and those whose head's modes are the procedure's, compared as duplicate mode declarations are, by what the modes mean, so `R::(ground >> ground)` is for an `in` argument, and a function's are for its default mode too. Name resolution resolves the modes' names, as it does a declaration's. A clause for a mode its predicate does not have is an error, as in mmc, and is then for every mode, as mmc takes it; a predicate whose modes are inferred has no declared mode for such a clause to be for, so it is always this error. A procedure that no clause is for fails: it gets a clause with its predicate's head and `fail` as its body, the empty disjunction, which determinism analysis infers `failure`, as mmc does. An instance's clauses cannot have modes, which mmc reports as a use of an undefined function `::/2` (see "Clauses for one mode in an instance" in the register); desugaring reports it, once for the clause, which is then for every mode. The error is kept, though the manual says an instance's clauses are like a predicate's, because the purity rule for clauses for one mode needs a pragma that names the predicate, which an instance's method does not have, and `pred(m/N) is p` with `p`'s clauses for each mode does the same. Rejected: leaving a procedure without a clause with no body, which its callers would fail to link against, and erroring on it, which mmc does not.
- **No clauses.** A predicate or function of the module, not a type class's method, with neither clauses nor a `foreign_proc` for C is an error, as in mmc ("no clauses for predicate"). Without it, the procedure had no body, and its callers failed to link. It is not reported after an earlier error, since a clause that did not desugar may have been for it and the compilation fails anyway, nor, until the option goes, with `-fstub-unimplemented`, since the library's modules have predicates that code generation builds in, such as `float`'s arithmetic, or that are not written yet (`Semantics/modes/no-clauses.m`).
- **Constrained polymorphic modes** (reference manual, "Constrained polymorphic modes"; Overton, 4.2). The operations on constrained insts are the thesis's: an inst matches one with inst variables if it is known to be compatible with each of them; a unification keeps the inst variables of each side whose inst the result is a sub-inst of, one that matches it with the same uniqueness, so `X = Y` of `(I =< ground)` and `(J =< ground)` is `(I, J =< ground)`, and `X = a` narrows `(I =< ground)` to `(I =< bound(a))`; and a merge keeps those of every branch. A clause starts from the head's insts with the inst parameters in them, and a final inst with an inst variable needs a value that has it; mmc says one without it "became too instantiated". At a call, the callee's inst parameters are bound by the arguments' insts at their occurrences in the initial insts, to the greatest lower bound of the argument's inst and the constraint, if it matches the constraint; as in mmc, an earlier argument's binding is kept, and only occurrences in one argument are merged, so `app([A], [B], L)` of `bound(a)` and `bound(b)` is a mode error, though the thesis merges them all. An argument whose inst there is an inst variable narrowed by a unification binds nothing, as in mmc (see the register). The modes, with the bindings, are then checked and passed as without inst variables; a callee's inst parameter left unbound stays in the outputs. A lambda expression's inst variables are its own. A higher-order term of a name has the procedure's own modes, since mmc makes it a lambda expression that calls it: its remaining arguments are passed from their initial insts, with what the curried ones bound, and must get their final insts, so `P = id(A)` is a mode error. A higher-order call binds the term's inst variables as a call does, but, as in mmc, those numbered as the procedure's own are its own (see the register). As in mmc, two modes that differ only in inst variables are the same mode.
- **Higher-order code** (reference manual, "Higher-order"; Overton, 3.2.5). A higher-order inst carries the modes and determinism of the term, and a higher-order call is checked against them as a call is against a procedure's, implied modes included; the closure must have a higher-order inst of its kind and arity, or for a function no higher-order inst, which means the default `func(in, ..., in) = out is det`. So a function's higher-order inst matches `ground` only if it is the default, and an error that expects `ground` there prints the default, as mmc's does. One higher-order inst matches another of the same kind, arity and determinism if its modes accept the other's, as in mmc (the manual does not say): for each argument the expected initial inst matches the closure's, and the closure's final inst matches the expected one. So `pred(di(uniq_array), out(uniq_array))`, a named inst that expands to `unique`, is passed for `pred(di, uo)`, and so is `pred(ground >> clobbered, uo)`, which asks for less; `pred(in, out)`, which gives less, is not, nor is `pred(di, uo)` for `pred(in, out)` (`test/Semantics/modes/closure-variance.m`). Unifying two higher-order terms, or a lambda expression or higher-order term with a bound variable, is an error ("attempt at higher-order unification").
  - A lambda expression waits until its nonlocals are ground, as mmc requires. Its body is then checked as a clause is, from its arguments' modes and the nonlocals' insts made shared, since the closure may be called more than once, with the nonlocals not live, as in mmc, so that clobbering one in it is an error of its inst ("T0 has inst ground, but unique is expected"), and its arguments' final insts must match; an error is reported at the goal in it, or at the lambda expression for a final inst, naming the argument `LambdaHeadVar__N`. Its term gets the higher-order inst of its modes, unique, and the nonlocals are shared. A function's lambda expression without modes has the default ones.
  - A predicate's or function's name, with its first arguments, makes a higher-order term of one of its procedures. With several procedures, the one is chosen whose initial insts the arguments of higher-order types match, as the manual says; mmc does not look at the others, so `P = m(1)` of a predicate `m(in, out)`, `m(out, in)` is an error, which says that the modes match more than one. The arguments are passed as a call's are, implied modes included, and must then be ground, like a lambda expression's nonlocals; the term gets the higher-order inst of the remaining modes.

### Determinism analysis

Determinism analysis (`check-determinism.cpp`) works on each procedure's body in the form mmc gives it after mode analysis, which `-fdebug-dump-determinism` prints. The procedure has one set of variables; a procedure of several clauses is a disjunction of them over head variables `HeadVar__N`, with an argument's variable the head variable itself when desugaring made it for a pattern; conjunctions are flattened and in the order mode analysis ran them; and a unification of a bound `X` with `f(Y1, ..., Yn)` deconstructs `X` into new variables, each followed by a test against its `Yi` if that was bound, as mmc writes `U = u(V, A), Z = V`. As mmc's mode analysis does, the form leaves out a construction or assignment of a variable not used after it, replaces a unification that cannot succeed by `fail` and leaves out the goals written after it (the head unifications, which come first, are written before the body), leaves out a branch of a disjunction, or a clause, that is `fail`, makes a disjunction in a disjunction its branches, and a disjunction of one branch that branch. So far the pass detects switches, eliminates common subexpressions, infers the determinism of goals, checks declared determinisms, checks committed choice, determinism casts and determinism requirements, and infers the determinisms of procedures declared without one. It runs only if mode analysis found no errors, as mmc stops after them.

- **Switches** (reference manual, "Switches"; Henderson, Somogyi and Conway, "Determinism analysis in the Mercury compiler", 1996, section 3). The candidates of a disjunction are its nonlocals that are bound before it, in variable order. A disjunct tests a candidate if, after only unifications other than calls, a deconstruction tests it or a variable unified with it, either way round; a `some` scope is looked into, and a nested disjunction whose every branch, followed by the goals after it, tests the candidate is split into one disjunct for each, with those goals copied; mmc does not split a disjunction inside such a branch. A candidate that every disjunct tests, against at least two functors, makes a switch; if none does, one that some disjuncts test against at least two functors makes a partial switch, the others following it as a disjunction, in which switches are detected again. The disjuncts that test a functor are its case, a disjunction if there are several, those that test it directly first, in order, then the parts of split ones, the last split first; cases are in order of functor name and arity, integers by value. A disjunct that starts with a disjunction whose branches each only deconstruct the candidate itself, with arguments not used after, is in mmc the goals after it in the case of each of those functors, without the tests, and the cases that have only it are one case for all of their functors (`case b/0 or c/1`); one with goals before it, a `some` scope, an alias or any other goal in a branch is split as above. A disjunction is split on the candidate and the variables unified with it outside `some` scopes; a test after a scope may be of a variable unified with the candidate in it. A switch can fail if the variable's inst before it allows a functor of its type that has no case, as `bound(a ; b)` does not; one on a type without constructors, such as `int`, can always fail. Inside a case, a test of its variable against a constant is `true` in mmc, so it does not count for another switch, nor is a disjunction after it split, though a test after it counts (`determinism/case-splits.m`), and a switch on the same variable has only that case, which cannot fail, or is `fail`; a test of a variable unified with it is kept.
- **Choosing a candidate**, as mmc does, which the manual leaves open. Of full switches, the first that cannot fail, else the first on a type without constructors, else the first; of partial ones, the first that cannot fail, else the one with the most functors, else the first. If every disjunct tests some candidate against the same functor, there is no switch but a full one that cannot fail or is on a type without constructors, and if there is none the disjuncts are left as they are, for common subexpression elimination to take the test out (see the register).
- **Common subexpression elimination** (ibid., section 4). In a disjunction, or the cases of a switch, the first candidate that every branch deconstructs with the same functor, after only unifications, through aliases and passing over the tests of a switch's variable against constants (but not those of a variable unified with it, which mmc keeps: in a case of the switch on `X`, `Z = X, Z = 0, ...` and `V = X, V = 0, ...` both test `X` against 0, which moves out), is deconstructed before it instead, into new variables, which each branch assigns to its own; a variable whose inst has free parts in the functor's arguments is left alone. If anything moved, the body is simplified as mmc's mode analysis does when it runs again: knowing the functor of each switch's variable in its cases, and of variables deconstructed and assigned, it makes a deconstruction against another functor `fail`, with the rest of its conjunction, and deletes the goals of a case that cannot be reached; it deletes constructions and assignments of variables not used after them, where a compound goal after one uses the variables it had before its own goals were deleted, and a `fail` replacing a unification uses its variables, and it deletes the tests of a switch's variable against constants, and branches that are `fail`, and a conjunction left in a conjunction, as a disjunction with one branch, is part of it; then switches are detected again in the whole body, until nothing moves. A single-functor type, `u ---> u(t, int)`, is so switched on its arguments.
- **Goal determinism** (reference manual, "Determinism checking and inference"; Henderson, Somogyi and Conway, section 5.1). A determinism is whether a goal can fail and whether it has zero, one or more solutions. A call has its callee's declared or inferred determinism, a higher-order call its term's inst's, and a function application its function's. A field access or update is the unification the manual defines it by, which fails when the term's constructor is not the field's: det when the type has only that constructor, and semidet when it has others (`Semantics/determinism/field-access.m`). An assignment and a construction are det, and so is a lambda expression, whose body is inferred with the arguments its modes make inputs bound and checked against its declaration; a test of two variables is semidet, even of a type with one value. A deconstruction's argument bound before it is a test of the argument, unless the variable's argument is free, which the unification binds, as when mode analysis built the term first from free variables: `P = p(T3, mk(2)), P = p(T4, _)` with `T3` still free is det, as in mmc. A deconstruction is det if its variable's inst allows no other functor (an inst `bound(...)` of an `int` counts), if it is the test of a switch's case, or if a deconstruction before it, or a switch around it, gives that functor, passed on by assignments; that last only when common subexpression elimination changed the body, after which mmc checks its modes again, and so knows more than the first mode analysis did: without that, `Z = S` before `S`'s case test does not tell mmc `Z`'s functor. A construction gives none. A conjunction can fail if a conjunct can before one with no solutions; a disjunction can fail if all its disjuncts can, and has more than one solution if one disjunct has, or two have any; a switch takes the most of its cases and can fail if a case or the switch itself can; an if-then-else whose condition cannot fail is its condition and then part, else as both parts; a negation is det if its goal is failure, failure if det or multi, erroneous if erroneous, else semidet. A call of an erroneous or failure procedure makes the rest unreachable for mode analysis, as in mmc, so `( X > 0, Y = X ; error("no") )` binds `Y` in every branch that can succeed; mode analysis then leaves out the goals after it, and the then part of an if-then-else whose condition it is, so their errors are not reported, though their variables count as used until mmc quantifies the body again, as for a unification replaced with `fail`. An if-then-else whose condition has no solutions is its else part, as in mmc, where the manual counts the then part too (register). The negation of `true`, or of a conjunction of `true`s, is `fail` for mode analysis too, as in mmc, so the goals after it are not reached; a negation whose goal fails only by inference is not. A goal that fails only by inference cuts nothing: the goals after it are checked, and still give the reasons that a goal can fail, but not that it can succeed more than once, unless no solution is allowed.
- **After mode errors.** As in mmc, determinism analysis runs on the procedures without mode errors when other procedures have them; only errors before mode analysis, such as type errors, stop it (`CheckDeterminism`'s `analyse`). A procedure without a declared determinism that has a mode error is `erroneous` for its callers, since it stays at the fixpoint's start, as in mmc.
- **No outputs.** A goal with more than one solution, but no variable that it binds and that is used outside it, has at most one, unless it is impure (reference manual, "Interfacing nondeterministic code with the real world"). Used outside goes by quantification after switch detection: the head, the other conjuncts and the parent's, the condition and then part of an if-then-else for each other, but not the other branches of a disjunction or switch, so a goal that switch detection copied into several cases is counted once in each. The innermost such goal is pruned, as mmc does for disjunctions. mmc infers a conjunct other than a disjunction that has no outputs, such as a switch, in a context where only its first solution is wanted, and so marks it and the disjunctions inside it `cc_multi` or `cc_nondet` instead of pruning them; the procedure's determinism, and its errors, are the same.
- **Committed choice.** A committed choice call is allowed only where just its first solution is wanted (reference manual, "Committed choice nondeterminism"). That is the body of a `cc_multi` or `cc_nondet` procedure or lambda expression, and any goal with no outputs (as above), including a negated goal. Where a conjunction is in such a context, so is its last conjunct, and so is each conjunct before conjuncts that cannot fail. Where an if-then-else is, so are its then and else parts, and so is its condition if the then part cannot fail. Before a goal that mode analysis finds `erroneous`, by declared determinisms, only the first solution is wanted, as nothing after it asks for another; the manual says only that a goal after a committed choice call may not fail, and mmc does not count a goal `erroneous` only by inference. The parts of disjunctions, switches and scopes inherit the context. A committed choice call, higher-order call or function application anywhere else is an error at the call, as in mmc ("call to predicate \`p'/1 with determinism \`cc_multi' occurs in a context which requires all solutions"). Its notes are the goals after it that can fail, in the conjunctions around it and in an if-then-else's then part, found as for a declaration's notes; of one after it that cannot fail, those in its disjuncts, a switch's cases, an if-then-else's then and else parts (not its condition) and a scope's goal, as mmc records them. A goal with no outputs, where only the first solution is wanted, keeps the goals after it as notes, for a call in it that a goal in it makes need all its solutions, and so does a determinism cast's goal, but not a negated goal (`determinism/no-outputs.m`). The call then counts as having more than one solution. Where only the first solution is wanted, a goal with more than one is committed choice, as mmc marks it: a call if its callee is, and a compound goal if each of its parts with more than one solution is in such a context too. A condition outside such a context, whose then part can fail, makes its if-then-else `nondet` inside a `cc_nondet` procedure, for example. The procedure's own determinism is the committed choice one if its body has more than one solution. A parallel conjunct must be `det` or `cc_multi` (ibid., "Goals"), but mode analysis does not check parallel conjunctions yet, so neither does this pass.
- **Determinism casts** (reference manual, "Goals"). `promise_equivalent_solutions Vars Goal` and `arbitrary Vars Goal` succeed at most once where their goal can succeed more than once, so `multi` and `cc_multi` become `det`, and `nondet` and `cc_nondet` become `semidet`; only the first solution of their goal is wanted, so it may make committed choice calls. `promise_equivalent_solution_sets Vars Goal` is as its goal; the `arbitrary` scopes in it are the casts. Each lists the variables its goal binds: those whose insts have free parts before it and that are used outside it, as mmc's quantification finds them after mode analysis, none if mode analysis finds that the goal's end cannot be reached (after a call or a unification that cannot succeed, in every branch). Mode analysis deletes a construction or an assignment of a variable used nowhere after it, so a variable used only there is not used outside (`R1 = M` in `( arbitrary [M] m(M), R1 = M ; R1 = 3 ), R = 0`). Which ones it deletes goes by its liveness, from before the deletions, as in mmc: an assignment of a variable that only a deleted one uses is kept, so in `( promise_equivalent_solutions [C] ( 3 > 1, C = A ), D = C ; D = A ), promise_equivalent_solutions [E] ( E = D )`, with `E` not used after, only `E` is extra (`determinism/before-deletion.m`). It turns a unification that cannot succeed into `fail`, but quantification still finds its variables, for the cast and for whether a goal has outputs, until mmc quantifies the body again: after it deletes an assignment (not a construction), or checks the modes again after common subexpression elimination. As in mmc, one that binds variables it does not list is an error ("the \`promise_equivalent_solutions' goal binds a variable that is not listed: Y"), and so is one that lists others, bound before, local to the goal, or not in it ("lists an extra variable: X"), each with the variables in their order. An `arbitrary` scope outside a `promise_equivalent_solution_sets` scope is an error, where a lambda expression starts afresh and `promise_equivalent_solutions` does not count, as is one that lists a variable of the scope it is in, and a `promise_equivalent_solution_sets` scope in another, with a note at the outer scope; the casts apply all the same. The errors do not keep the pass from checking the procedure's declared determinism, as in mmc. Where a cast's goal can succeed more than once, the reasons given inside it for a declaration not satisfied are only the goals that make it fail, as in mmc; where it cannot, they are as elsewhere. Switch detection and common subexpression elimination look into scopes of every kind, not only `some`, as mmc does, and stop at a call in one. Where no solution is allowed, a negation that can succeed is a reason too ("the negated goal can fail", in mmc's words). A scope that switch detection copies is checked once. `generate-switch-programs.py --casts` adds casts to the generated modules (`random-casts-*.test`, `casts.m`). Of 2,400 generated modules (1,600 with `--committed --casts`, 800 with `--casts`), every procedure's determinism and every error line agree with mmc; 58 differ only in the labels of their goals, as do 51 of the 2,400 modules without casts. (Two procedures whose determinism differed, c27_11 p1 and c3_19 p2, came from a test's bound argument and a split after a case's constant test, now fixed.) (A committed choice call in an if-then-else condition that mmc reported and merkc did not, c28_21, came from deleting assignments transitively, now fixed.) Not handled: mmc's common subexpression elimination of a case's own functor tests in a disjunction inside the case, which merkc splits instead, and so does not quantify the body again.
- **Trace goals** (reference manual, "Trace goals"). A trace goal must be `det` or `cc_multi`: only the first solution of its goal is wanted, so it may make committed choice calls, and as it binds nothing used outside, the goals around it see at most one. One that can fail is an error ("the trace goal has determinism semidet, but must be det or cc_multi"), with the committed choice determinism, `cc_nondet` for a goal that can also succeed more than once, and the procedure's determinism is then one that can fail, with the goals inside that make it so as the reasons, as in mmc. Whether its `compile_time` condition holds makes no difference. Its impurity does not leak (Purity checking).
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
- **After an if-then-else,** what is used is used after its condition, then part and else part, since backtracking into one of them runs it again. mmc counts only what the then part uses as used after the condition, and nothing as used after the then and else parts, so nothing used after an if-then-else is `mostly_unique` because of backtracking into it (see the register); `-fmmc-uniqueness` does the same. Where only the first solution is wanted, nothing after the condition is run again unless the then part, or something after the if-then-else, can fail, and then nothing counts as used after the condition, in both modes, as in mmc: `( if m(X), X > 1 then updu(X, T0, T) else T = T0 )` in a `cc_multi` procedure is accepted. It once counted the then part's uses, and rejected such a condition in `main`, which mmc accepts (2026-11-06, found by the first nondeterministic code; `Semantics/uniqueness/sound.m`).
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
  A constant's fields are filled in as a cell's are, and only the dump tells them apart, so that code generation has the instance, its type_infos and its premises for either. The predicates made from do loops take the clause's constraints, the same array and so the same numbers, where the loop's body needs them. Only a loop that needs an existential type's constraint is not translated.
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
- **As implemented,** only `unselected-message` (Processes and mailboxes, "The `receive` goal") has an identifier, and the options `-Wunselected-message` and `-Wno-unselected-message`; it is the only warning mmc does not give. Its message ends with `[-Wunselected-message]`, as Clang's do.

### Mapping to `merk`

The output of semantic analysis is the annotated AST, one `ProcInfo` per procedure, and the module's interface file; there is no other intermediate form. MLIR generation reads them directly, one procedure at a time, as ClangIR's code generator reads Clang's AST:

- **Values.** Once modes are known, each variable is produced exactly once on each path. So a map from variables to SSA values, updated at each producer and read at each consumer, is all MLIR generation needs, and the output is in SSA form from the start.
- **Locations.** Each operation carries the location of the parse-tree node behind its semantic node, and variable names are recorded as `NameLoc`s for debug information.
- **Types.** `int`, `float`, `char` and the sized integers become builtin MLIR types such as `i64` and `f64`. Other types become `!merk.term<…>`, closures `!merk.closure`, and type infos `!merk.typeinfo`. `io.state` has no values to pass, so its arguments are left out.
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
:- pred spawn_with_stack(int::in,           % stack size in bytes
    pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
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
:- type received(R) ---> message(R) ; timeout.
:- pred receive(selector(R)::in, timeout::in, received(R)::out,
    io::di, io::uo) is det.

    % One mailbox.
:- pred receive(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    timeout::in, received(R)::out, io::di, io::uo) is det.
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
- **Receive** blocks until a message matches or the timeout expires, and returns `message(R)` or `timeout`, which only ever means that the time ran out. `no` would say less, and would also have covered what follows. A receive that can never get a message, from a closed mailbox, from another node's, or over a selector with no open mailbox, is a mistake in the program, as only a mailbox's owner receives from it and closes it, so it is a run-time error at once; on a device, which cannot see that a mailbox is closed, it is a timeout (Rejected: a third result, `closed`, which every receive would have to handle for a case that should never happen). It takes the oldest message for which the `semidet` selector succeeds; the rest stay queued in order. Erlang's `receive {reply, Ref, S} -> …` becomes the selector `(pred(M::in, S::out) is semidet :- M = reply(Ref, S))`. `receive/6`, on one mailbox, is `receive/5` over a one-mailbox selector.
- **Order is Erlang's.** Erlang's `receive` takes the first message in the queue that matches any clause; the order of the clauses only decides which clause handles a message that matches several (Erlang reference manual, Expressions, "Receive"; checked with OTP 24). Typed mailboxes never hold an unknown message, so a mailbox is a plain FIFO, and a selector keeps this rule across all its mailboxes: it takes the matching message that arrived first.
- **Exhaustiveness:** a receive loop declared `det` must handle every message constructor. Removing one case is a compile error: "the switch on Msg does not cover \`stop'/0".
- **No backtracking over messages:** send and receive thread the unique `io` state, so they cannot appear in nondeterministic code.
- **`any` is `det`** in fact, and mmc warns that its `semidet` declaration could be tighter. It is declared `semidet` only to fit the selector's mode; a `det` `receive_any` would avoid the warning.
- **As implemented** (2026-11-07, `library/process.m`): the mailboxes of one process, `new_mailbox`, `close_mailbox`, `send`, the one-mailbox `receive` and `any`, with this interface, which mmc accepts. A handle is a slot of the runtime's table and its generation (Message passing, "So far"). `receive` is written in Mercury over `foreign_proc`s: it waits for the message at a position, tries the selector on it, and removes it if the selector succeeds, or goes on to the next. Selectors too, with `new_selector`, `selecting`, `selecting_matching` and `receive/5`: a selector is a list of cases, each a mailbox and a function from its message type, existentially typed, so that each case keeps its own. Each case looks in its mailbox from where it last stopped, past the messages it rejected; the runtime stamps each message in the order of arrival over all mailboxes, and the receive takes the case's message with the lowest stamp. Of cases that accept the same message, which can happen only when they share a mailbox, the one added first takes it. If no case finds a message, the receive waits for any arrival or close after the count it read before looking, so it needs no registration with each mailbox; it finds nothing at once when every mailbox is closed or another node's. `spawn`, monitors and request and reply come with processes (`CodeGen/process-local.m`, `CodeGen/process-messages.m`, `CodeGen/process-selectors.m`). As mmc has no process module to compare with, `CodeGen/process-checks.m` checks itself: round trips of generated terms of every kind a message holds, checks that a message shares nothing with the sender's term, which `foreign_proc`s overwrite in place, and a run of random sends, receives, closes and stale handles against a model, a list of queues. `CodeGen/process-selectors.m` checks receives over random selectors in the same way, against queues whose messages carry their order of arrival. The runtime's unit tests (`unittests/Runtime/MailboxTest.cpp`) test what one thread cannot reach: receives woken by a send or a close, timeouts, many senders, a selective scan while messages arrive, the table of mailboxes growing, and a receive over two mailboxes woken by either, by a close, or by a message to neither, which it ignores, and taking messages in order while they arrive.

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

**Which receive to write.** There are three forms, and `select` belongs only to the last:

- **Every message, in order:** `receive(Box, any, Timeout, Got, !IO)`, with a match on `Got` after it. This is the form to use when one mailbox's messages are all handled as they arrive, since it says that no message is skipped. A message that fits no later pattern has been taken all the same, so the program must handle it. It is the only form a target region has today (Send and receive in a target region).
- **Selected messages of one mailbox:** `receive(Box, Selector, Timeout, Got, !IO)`, with a `semidet` predicate, such as `is_reply(Ref)`, to take a reply out of order. Messages it does not accept stay queued, in order, for a later receive.
- **Several mailboxes, or arms by pattern:** a selector built with `selecting` and `selecting_matching`, or the `receive` goal below, with `-freceive-goal`. Each arm of the goal is `select(Mailbox, Pattern) -> Goal`, even if there is only one mailbox; what its arms leave out stays queued.

**The `receive` goal.** Building a selector, a result type to map every mailbox into, and a switch on `message(...)` is boilerplate. The `receive` goal writes the same loop directly. The goal receives one message; each arm selects the messages it wants, by mailbox and pattern:

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

- **Syntax.** `receive Params Arms`, shaped like `trace` and `try`. The parameters are `timeout(T)`, with a `timeout` from the library, and `io(!IO)`, the state variable that the wait and the arms thread. Each arm is `select(Mailbox, Pattern) -> Goal`, or `timeout -> Goal`, and the arms are joined by `;`. `select` is not an operator, so the arms are ordinary terms; the term reader needs only `receive` as a binary prefix operator, with `trace`'s priority. `receive(...)` written as a call, with no layout before the parenthesis, stays an ordinary call, qualified or not, so the library predicates keep their name. As with `trace`, `receive` as a name on its own is then an operator, so it needs parentheses, `(receive)`, and a module cannot be named `receive`.
- **The option.** The operator departs from mmc's operator table, so it exists only with `-freceive-goal`. Without it, a file reads exactly as mmc reads it, and test 5 (Unparser and round-trip tests) stays exact, as `-fopenmp` keeps plain files plain.
- **Conversion.** The converter reads the arms into a `ReceiveGoal` node with one child per arm, instead of an if-then-else chain. An arm that is neither form, a missing `io(!IO)`, a parameter given twice, more than one `timeout` arm, or a `timeout` arm without a `timeout` parameter or the other way round, is an error. With neither, the timeout is `infinity`. The canonical layout is a disjunction's, with each arm's goal on the lines below its head.
- **Meaning.** The goal is the selector code above:
  - Arms for different mailboxes may be written in any order, mixed freely. The compiler groups them by mailbox, keeping the order of each mailbox's arms, which is the only order that matters: a message is in one mailbox, so only that mailbox's arms compete for it. Each mailbox, by the text of its term, becomes one `selecting_matching` case, the first written added first, whose `semidet` predicate tries that mailbox's patterns in order and returns the arm and the pattern's variables. A predicate whose first arm takes anything, as `select(Downs, _)` does, accepts the first message, so its scan stops there, as a `selecting` case's would.
  - `receive/5` takes the oldest message, across the mailboxes, that some arm selects. A message that several arms select goes to the first of them, as in Erlang's clauses.
  - A switch on the result runs the arm's goal, with the pattern's variables bound. The goal is `det` if every arm's goal is.
  - **A pattern's variables.** One that occurs outside the goal, as `Ref` in `select(Replies, reply(Ref, S))`, is tested in the predicate, so it must be ground before the goal; mode analysis says so otherwise. The others are the arm's own: fresh copies in the predicate, which come back through the result. So an arm's pattern cannot bind a variable for the goals after the receive goal; its goal can. The rule goes by where a variable occurs because desugaring, before mode analysis, builds the selector and cannot know insts. Deciding by inst instead, Erlang's rule, comes later (`INPROGRESS.md`, The receive goal: later), and changes no program that compiles now: under this rule a variable of a pattern that is not ground before the goal, such as a hole of an open term, is a mode error, as for any lambda expression's nonlocal. One case gives a puzzling error under this rule: a tested variable bound only after the goal, as in `receive [io(!IO)] ( select(B, N) -> true ), N = 1`, lets mode analysis move the pure goals of the switch on the result before the call, whose implied mode then makes the goal `semidet`. The code is still right, since only pure goals move; the inst-based rule removes the error, since the variable is then the arm's own.
  - **The result** needs no type of the program's: `library/process.m` exports, for the expansion only, `receive_arms(A, B)`, whose `receive_rest` applied I times to `receive_arm` of the variables is arm I's result, the last arm's without `receive_arm`, with `receive_none` for no variables and a tuple for several, and `receive_goal`, which is `receive/5` with a result of process's own type, since the module need not import `maybe`. So the expansion names only process, which the module must import, or a receive goal is an error.
  - **Every mailbox closed.** A receive whose mailboxes are all closed would wait for a message nobody can send; with or without a `timeout` arm, it is the run-time error "receive: every mailbox is closed or another node's" instead.
- **Selective arms.** The arms need not cover a mailbox's type. A message that no arm selects stays queued for a later `receive`, as in Erlang. Since that can leave messages piling up, a mailbox whose arms leave a constructor out draws the warning `unselected-message`, unless some other `receive` in the same predicate selects it. `-Wno-unselected-message` turns it off.
  - **What counts as selected.** A mailbox is its term's text, as for grouping the arms, over all the clauses of the predicate, so `Self` in two clauses is one mailbox. An arm selects the constructor of its pattern's functor, whatever the arguments, so `select(Box, data(1, _))` counts for every `data/2`: the warning is for messages that nothing in the predicate is written to take, not a check that the patterns cover every value, which would warn about every receive that waits for a particular reply. An arm whose pattern is a variable of its own selects every message; a tested variable, one that occurs outside the goal, selects no constructor. A type without a known set of constructors, such as `int`, a type variable or an abstract type, is selected only by such a variable.
  - **Where.** Once per predicate and mailbox, at the mailbox of the first arm for it, listing the constructors left out in the order of the type's definition. A predicate with a type error is left out, since its arms' constructors are not known.
  - **When.** It needs the mailboxes' types, so it is a pass of its own after type checking (`lib/Semantics/check-receives.cpp`); it runs after mode analysis, whose table of the types' constructors it uses. Desugaring records each selector case's mailbox and which of its arms take every message on the case's lambda expression (`ReceiveCase`), and the pass reads the patterns' constructors from the conditions of the case's chain of if-then-elses, where type checking chose them.
- **Tests.** `Parser/convert/receive.m` and `receive-errors.m` read the goal and its errors, with the canonical layout and round trips in the unparser's tests; `Semantics/desugar/receive.m` checks the expansion, and `Semantics/receive/` the errors of each pass and the warning; `CodeGen/receive-goal.m` runs the counter loop above and the other cases at -O0 and -O2, its output written by hand, since mmc has no receive goal.
- **Priority** is still the two-step receive: a `receive` with `timeout(after(0))` and only the urgent arms, whose `timeout` arm is the ordinary `receive`.
- **Guards, a later extension.** An arm may add a guard after its pattern, `select(Box, reply(R)), R > 0 -> Goal`, as Erlang's `when` does. The guard runs inside the mailbox's `selecting_matching` predicate, after the pattern matches; if it fails, the arm does not select the message, which the mailbox's later arms may still select, or which stays queued. A guard is any pure `semidet` goal. Erlang restricts guards to a fixed set of built-ins that have no side effects, so that choosing a message cannot change the world; purity checking gives the same guarantee with ordinary predicates. A guard sees the pattern's variables and those bound before the goal, and the variables it binds go to the arm's goal with the pattern's. It needs no new operator, since `,` binds more tightly than `->`.

**Request and reply.** `request(Counter, (func(R) = get(R)), after(1000), Reply, !IO)` asks the counter for its value:

1. It monitors the server, delivering any `down` message to a private mailbox. A server that is already dead delivers one at once.
2. It creates a fresh `mailbox(Rep)`, here `mailbox(int)`, builds the request by applying the function to it, here `get(ReplyTo)`, and sends it. These two steps are `send_request`.
3. It receives from the reply mailbox and the `down` mailbox, with the timeout.
4. It removes the monitor and closes both mailboxes, then returns `reply(N)`, `timeout` or `down(Reason)`.

As implemented (2026-11-08, `library/process.m`), the server's mailbox must be on this node, as a monitor's must, and a remote one reports that it is not supported yet; `send_request` works across nodes. A server whose process lives on but has closed the mailbox is not noticed, as a monitor watches the process: the request then waits for its timeout.

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

**Process declarations as implemented** (2026-11-10, step 4a of GPU processes without `omp target`, docs/INPROGRESS.md):

- **Syntax and the option.** `:- process Kind(Types) receives Types`, with `on gpu` after it for a kind that runs on a GPU (decision D1). `process` is a prefix operator with the other declarations' priority, and `receives` and `on` are infix operators that bind less tightly than `,`, which separates several message types: the declaration reads as `process((Kind receives Types) on gpu)`. mmc has none of them, so they exist only with `-fprocess-declarations`, as `receive` exists only with `-freceive-goal`. Even then they are operators only in an item that begins with `:- process`, and `process` only there. **Rejected: operators everywhere**, as `do` and `receive` are. Every program with processes imports the module `process`, and `:- import_module int, process, string` would then be an error, an operator used as an operand; a constructor `on`, as in `turn ---> on ; off`, would be one too.
- **Conversion.** The converter reads a `ProcessDecl`: the kind's name, which is not module-qualified, its argument types, its message types in order, and whether it runs on a GPU. While the declaration's terms exist, it makes the expansion as the terms it would be written as, and converts them as items: the kind's `pred` declaration, and, for one message type, `spawn_Kind`'s declaration and its clause. The names in them are qualified, `io.io`, `process.mailbox`, `process.spawn` and the module's own kind, so that they mean the same whatever else the module declares. The node keeps the expansion out of its fields, so the parse tree prints and unparses as written. The canonical layout puts message types that do not fit on the line below it, indented.
- **Expansion.** `ExpandProcesses` (`lib/Semantics/expand-processes.cpp`) puts the predicates into the items, after the declaration, which stays: the declarations where the declaration is, so that a declaration in the interface exports them, and `spawn_Kind`'s clause in the implementation section, or at the end of the module. Every later pass sees ordinary declarations and clauses. The module loader expands an imported module too, without the clause. The module must import `io` and `process` where the declaration is, as for any declaration that names them; name resolution says so otherwise. A kind declared twice, or with its predicate declared as well, is a predicate declared more than once.
- **Spawning at a place** (step 4c): the declaration also declares `spawn_Kind_at(Place, Args, Mailbox, !IO)`, with a clause that spawns at `process.host` through `spawn_Kind`, and at `process.gpu(N)` through the kind's kernel, or with a software error for a kind that runs on the host (GPU processes, "Spawning on a GPU as implemented").
- **Not yet implemented,** each reported: several message types, and a call of `spawn_Kind_on`, which needs remote spawn. Name resolution reports the last instead of an undefined predicate when a `spawn_Kind` that a process declaration made, of one less arity, is visible.
- **Tests.** `Parser/convert/processes.m` and `process-errors.m` read declarations and their errors, with the canonical layout and round trips in the unparser's tests; `Semantics/desugar/processes.m` checks the expansion, and `Semantics/processes/` the errors after conversion; `CodeGen/process-declarations.m` spawns kinds of its own module and of another's interface, one polymorphic, at -O0 and -O2, its output written by hand, since mmc has no process declarations.

Failure handling follows Erlang. An uncaught exception ends the process with that exception as its exit reason. A monitor delivers a `down_info` message to a mailbox the watcher names, through a function into that mailbox's message type, so a process with one protocol needs no second mailbox for it. Links, deferred for now, would deliver exit messages the same way. The semantics are in Runtime system, "Exceptions, links and monitors".

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
  - *A barrier after each round* (shared with Flang, in `OpenMPIRBuilder` and the device runtime). On the device, the loop is a call to the runtime's `__kmpc_distribute_for_static_loop`, which deals the iterations out in rounds: a team's threads take the next `threads` iterations, all the teams' threads `teams × threads` apart, as clang's lowering deals them out. Clang's has the implicit barrier of its inner worksharing loop after each round, and the runtime's had none, so a team's warps drifted apart. In a stencil, the warps of a team read the same rows of the input, and warps far apart read them from memory again instead of from the cache. The 4096² Game of Life read 24% more from memory than the same loop compiled by clang and was 26% slower, the reacting front 18% slower (Measurements, "Dense stencils"); Flang's build of the same Life was slower still. The runtime now goes round in lockstep, with a barrier across the team after each round, when the call's flags ask for it, which `OpenMPIRBuilder` does for a combined loop on the device unless it is `nowait` or `spmd_no_loop`, as clang has its barrier. Every thread goes round as often as the team's first, so that all reach every barrier, and only in SPMD mode with all the block's threads in the parallel region, which a barrier across the block needs. `LIBOMPTARGET_LOOP_BARRIER=on` or `off` overrides the flags for the whole program, to measure a loop both ways; any other value is reported and ignored.
    - *Why not the round loop in the kernel,* as clang's, with `OpenMPIRBuilder`'s own barrier in it: it would take the body out of the runtime's call and change much more of what Flang's tests check, for the same barrier. The flag keeps the decision per loop with the compiler, as clang's is, in one constant.
    - *Why not only the environment variable:* it would set every loop of a program alike, those that gain, such as stencils, and those that may lose, such as irregular loops whose warps would wait for the slowest, or pure streaming. *Why not in Merkurius only,* with its own loop around `omp.parallel`: it would give up what `omp.distribute` and `omp.wsloop` give, the trip count that sizes the launch, SPMD detection and the reductions, and leave Flang as slow.
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

A target directive, such as `%$omp target teams distribute parallel do`, runs its goal on a device. A GPU is only useful on flat data, so only flat values cross into a target region. Implemented so far: `target` with no clauses, on integers, floats, arrays of them, and terms whose cells hold only scalars, among them mailbox handles (Lowering and code generation, "What is lowered so far"). Our representation (Lowering and code generation, "Type representation") says which values are flat.

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


**Data regions** (implemented for arrays, 2026-11-04). `%$omp target data map(tofrom: A, B)` before a goal keeps arrays on the device for the goal, so that the target regions in it, even in procedures it calls, find them there: their own maps then copy nothing. A map names a variable, or a state variable `!A`, whose version at the goal is mapped. The map types are `tofrom`, the default, `to`, `from` and `alloc`.

- **Lowering.** As Flang lowers `!$omp target data`: `omp.target_data` with the arrays' `omp.map.info` entries around the goal, in the procedure, each array mapped as a target region maps it, with bounds over its size word and elements. What the goal binds and is used after it leaves through a word's slot, since a value in the region does not dominate what follows.
- **Checked by the compiler.** The goal is `det`, since failing would leave the region without unmapping its arrays. A state variable that the goal updates has a map type that copies it back. The goal may do I/O and allocate on the host, unlike a loop's body.
- **Checked at run time.** The end of the region copies the device's elements to the array that came in, so a state variable leaving as another array, which `array.copy` or `resize` would give, throws a software error, rather than leaving the program with stale elements. It does so without a device too, where nothing is mapped, so that a program does not work only on the host.
- **The promise.** The host does not use a mapped array's elements in the goal, which the compiler cannot see through calls. Without a device or OpenMP the elements are simply the host's, as the sequential meaning says. For a plain variable, as `map(tofrom: A0)` for a goal that binds `A`, the compiler does not know which output is which, so the run-time check is the state variable's alone.
- **Not yet.** Scalars, sections, map modifiers, the `if`, `device` and `use_device_*` clauses, and regions in target regions are "not yet implemented".

Rejected:

- **`target enter data` and `target exit data`** as BabelStream's OpenMP model uses. They have no enclosing goal, so they would have to thread through `io`, could be left unbalanced, and give no place to check that an array stays the same.
- **The output array taken to be the input one**, with no slots: true of a correct program, but it would hide a replaced array, the one error here that gives wrong answers silently.

**Group-private arrays** (open design, 2026-11-06). OpenMP 6.1 adds the `dyn_groupprivate` clause, "to support the use of small, fast memory on devices for data with dynamic lifetimes": AMD's LDS, CUDA's shared memory (Technical Report 15, the 6.1 public comment draft of 9 July 2026, sections 13.9 and 34.13; the release is expected in November 2026). On `target` or `teams`, `dyn_groupprivate(Size)` gives each contention group, a team, one uninitialized block of `Size` bytes, computed at run time, which lives exactly as long as the team's tasks. A size over the limit, which `omp_get_gprivate_limit` gives, does what the `fallback` modifier says: `abort` ends the program, `null` makes no block, and `default_mem`, the default, takes the block from an implementation-defined memory space instead. The team reaches its block through `omp_get_dyn_gprivate_size` and the routines beside it. mmc has nothing of the kind: one heap, Boehm's collector or none (`--gc`), and no way to direct an allocation to a kind of memory; only a `foreign_type` holding a pointer that foreign code manages.

- **Arrays only.** A group-private value is an array of scalars, flat and with no pointers in it, as everything else on a device is, so neither the collector nor a message copy ever looks inside one.
- **No collection and no reference counts.** OpenMP fixes the lifetime, the team's, so the array is allocated with its region and gone with it, as a stack frame is. What must be checked is that it does not escape: an array made group-private in a target region reaches no output of the region, no map back to the host and nothing that outlives the team. That is a scoped check of uniqueness, close to what `array_di` and `array_uo` already have, and its error names the array.
- **Generic pointers.** Under `default_mem` an array may be in group memory or in global memory, decided at run time, so code that uses it takes a generic pointer, to which `TargetInfo`'s workgroup-shared address space (3 on AMDGPU and NVPTX) converts. Using address space 3 directly, when the size is known to fit, is an optimization for later.
- **Lowering** follows Flang's, as the other directives do, once Flang lowers `dyn_groupprivate`: the omp dialect's operation for it, if it has one by then, and our lowering compared with Flang's output for the Fortran equivalent.
- **Not this:** arrays that outlive a region and move between kinds of memory, host, device and group, which reference counting might suit. They are large objects, and belong to the runtime's large-object space (Runtime system, "Memory management").
- **Open:** the syntax, whether a clause on the directive naming the array, `dyn_groupprivate(A)` with its size from the array's, or a library predicate in the goal that makes the array; and how `null`'s missing block appears in Mercury, where an array cannot be absent, which may mean allowing only `abort` and `default_mem`.

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

**`merk` types** are Mercury types, not layouts: `!merk.term<"list.list(int)">`, `!merk.closure`, `!merk.typeinfo`, and builtin `i64`, `f64`, `i8` for Mercury's primitive types. Layout is chosen when lowering to `mrt`, so representation work never changes the dialect.

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

Not yet: types with type variables, with user-defined equality, subtypes, and unifications of terms with `string` or array arguments, which are reported as not yet implemented. Existentially typed constructors are under Lowering, "Existentially typed constructors".

For polymorphic code, the types `!merk.word`, a value of a type variable, and `!merk.type_info`, with `merk.type_info` to make one from a type constructor and its arguments, `merk.type_info_word` to take one out of a cell, `merk.to_word` and `merk.from_word` to convert a value at a call where its callee has a type variable, `merk.type_ctor` for a constructor's descriptor, `merk.unify_word` for a unification by a `type_info`, and `merk.compare_word` for a comparison by one.

For type classes, `!merk.typeclass_info`, a dictionary, and:

- **`merk.instance`,** a symbol at the module's level, with the table of an instance's dispatch procedures, or with none for another module's instance or a class without methods.
- **`merk.typeclass_info`,** a dictionary made from an instance's symbol and its type_infos and premises.
- **`merk.typeclass_info_word`,** a word taken out of one, typed as what is there, a type_info or a superclass's dictionary.
- **`merk.call_method`,** a call of a slot of a dictionary's table.
- **`merk.nondet_call_method`,** such a call of a nondet or multi method, whose region is the rest of the code, as for `merk.nondet_call`.

These stay in the dialect, and do not become loads and indirect calls until lowering, so that a `merk`-level specializer can see which instance a dictionary is.

For arrays, the type `!merk.array<T>` and the library's `array` predicates as operations, until the library can be compiled with type_infos: `merk.array_new` (a size and an initial element, or none for an empty array), `merk.array_size`, `merk.array_get`, `merk.array_set`, `merk.array_fill` (from one index to another), `merk.array_copy`, `merk.array_resize` (with an element for the new places, or none to shrink), `merk.array_from_list` (of a `list.list(T)`'s elements, in order) and `merk.check_index %a[%i], "array.lookup"`, which throws `index_out_of_bounds` with mmc's message, as `index_out_of_bounds("array.lookup: index 4 not in range [0, 3]")`. Those that update an array do it in place and give the same array, since uniqueness checking has made sure its old version is dead; the others allocate. None checks an index itself.

The regions of `merk.ite`, `merk.switch` and `merk.not` have one block each. A region that succeeds ends with `merk.yield`: the condition yields what it binds that the then part uses, as the then part's arguments, and the branches yield the operation's results, the variables that every branch that succeeds binds. A branch can instead end with `merk.fail`, or with `merk.succeed` when the operation is the procedure's last goal. The then part of a condition that cannot succeed, such as a unification mode analysis replaced with `fail`, is only `merk.fail`, so that the variables after the operation are the else part's; it once counted as a branch that succeeds and binds nothing, and they had no values (2026-11-04, found by `utils/codegen-oracle`; `CodeGen/condition-cannot-succeed.m`). Failure goes to the innermost condition's else part or negation, or else out of the procedure, which then can fail. `merk.switch` is on an integer or a term: its `cases` attribute lists each case's integer constants or functors, and with `can_fail` a value that no case has fails. A switch on a string or a float is instead a chain of `merk.ite`s, one for each case, whose condition compares the value with the case's constants, by `merk.string_equal` or by `arith.cmpf oeq` as `==` does, so that `0.0` is `-0.0`'s case; after the last case it fails, or, for a complete switch, the last case is what is left (`CodeGen/switch-strings.m`). Hashing or binary search over the constants is left to switch optimization (Optimizations). Arithmetic is the `arith` dialect's, which `merk` depends on. `merk.foreign_call` with `can_fail` is that of a `semidet` `foreign_proc`, whose first result is `SUCCESS_INDICATOR`.

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

Code generation (`CodeGen/MLIRGen.cpp`, with the `MLIRGen*.cpp` files beside it for type information, calls, unification, control flow, built-in calls and OpenMP) makes `merk` for procedures of every determinism that have no hidden arguments, and whose values are of primitive types, `io.state`, arrays of values a word wide at most, or types defined by constructors and tuple types without type variables, whose bodies are conjunctions, assignments, constants, tests, constructions and deconstructions of terms, calls and function applications, if-then-elses, disjunctions, switches on integers, characters and constructors, negations and `fail`, or that have a `foreign_proc` for C. A value of type `io.state` is not passed, so `main(!IO)` takes no arguments, and neither is an argument of mode `free >> free`, which carries nothing either way, as with mmc; a closure's entry takes a word for it, as for every argument, and does not pass it on. Such a disjunction, whose determinism determinism analysis records as `det`, `semidet`, `cc_multi` or `cc_nondet`, is an if-then-else whose condition is its first branch and whose else part is the rest, so the last branch's call is a tail call; one with more solutions, which a goal after it may backtrack into, is nondeterministic code (Lowering, "Nondeterministic code"), as are nondet and multi procedures and their calls, and if-then-elses whose condition has more than one solution. Anything else is reported as not yet implemented, at the first such construct of a procedure, and so is a procedure whose clauses mode analysis did not check, such as one with a `try` goal or a trace goal's `state` parameter, which has no body to generate; none is left out silently, for its callers to fail to link. Lowering (`CodeGen/Lowering.cpp`) lowers terms to `mrt`, choosing their layouts, and then `merk` and `mrt` to MLIR's LLVM dialect, in one pass over the module, as ClangIR's DirectToLLVM lowers CIR. Then MLIR translates it to LLVM IR, LLVM's pipeline for `-O0` to `-O3` optimizes it, and the `TargetMachine` writes assembly or an object file.

- **Symbols.** A procedure's symbol is its module, then its name, its arity, with an `f` before it for a function, and its mode number, joined by `_`: `hello.main_2_0`, `list.length_f1_0`. Other characters than letters, digits and `_` in a name become `$` and two hex digits, so `int.+` is `int.$2b_f2_0`. A procedure defined and not exported has internal linkage.
- **Entry.** `merk.entry` names the main module's `main/2`; it becomes `merk_main`, a C function that calls it, which the runtime's `main` calls (Runtime system).
- **Constants.** An integer literal, in any base, with `_` and a size suffix, or `0'c`, is a `merk.constant` of its type's width; a float is an `f64`; a character is its code point, an `i32`; a string is a `!merk.string`, with its escapes replaced, and becomes a private constant global of its bytes and a NUL, one for each distinct string in a module. In C a string is `MR_String`, a `char *`, as the reference manual's "C data passing conventions" has it.
- **Foreign code.** A procedure with a `foreign_proc` for C runs its C code instead of its clauses. Mode analysis gives each `foreign_proc` the procedure whose modes its arguments have, and reports one with no such procedure, or two for one procedure, as mmc does. Code generation puts the code in a C function, `merk_foreign_` and the procedure's symbol with `_` doubled, `.` as `_D` and `$` as `_S`, which the procedure calls with `merk.foreign_call`: the function takes the inputs as parameters, as the manual's C types, and returns the only output, or stores several through pointers after the inputs; a variable of a type with no values, such as `io.state`, is an `MR_Word` set to 0. On a polymorphic procedure, a value of a type variable is an `MR_Word` holding its canonical word, and the function's first parameters are its type_infos, `MR_Word TypeInfo_for_T` by the declaration's names, as in mmc; an existential type or a class constraint is still "not yet implemented". The prelude defines the manual's `MR_word_to_float`, `MR_float_to_word` and the same for `int64` and `uint64`, but not yet its other macros on terms, such as `MR_list_head`. The module records its C code in its `merk.foreign_code` attribute: the C types `MR_Integer`, `MR_Unsigned`, `MR_Float`, `MR_Char`, `MR_String` and `MR_Word`, the module's own `foreign_decl` and `foreign_code` for C, and the functions, with `#line` directives, so that clang reports errors at their lines in the Mercury source. When the module is translated to LLVM IR, `-fc1` writes the code to a temporary file, has `clang`, beside `merkc` or else on the `PATH`, compile it to bitcode for the module's target and optimization level, links it into the module and makes the functions internal, so LLVM's inliner can put the C code into its procedure. A module without foreign code does not need clang. The C function of a `foreign_proc` of a procedure that can fail has `MR_bool SUCCESS_INDICATOR`, set to `MR_TRUE` first, returns it as C's `int`, 32 bits on the targets so far, and stores every output through a pointer; its `merk.foreign_call` has `can_fail`. Not yet: a `foreign_decl` in the interface, which importing modules should see; and foreign types.
- **Branches.** Lowering makes each `merk.ite`, `merk.switch` and `merk.not` blocks of the procedure, outermost first: a guard is a conditional branch to the next goal or to where failure goes, a switch is `llvm.switch`, a yield is a branch to the block after the operation, whose arguments are its results, and a block that only succeeds is copied into the branches that go to it, so that a call before it is in tail position. Failure at the top of a procedure that can fail returns false; in a `det` one it cannot happen, and is `unreachable`. Blocks that no branch reaches, such as the else part of a condition that cannot fail, are then erased, since the conversion of the `arith` dialect to LLVM skips them and would leave their operations behind (`CodeGen/unreachable-branch.m`).
- **Calling convention.** A procedure that can fail, `semidet` or `failure`, returns its success flag, alone as an `i1` or first in a struct with its outputs; one that fails leaves its outputs undefined. A call of one is a test of the flag. A call whose results are the procedure's own, of a procedure that can fail as the caller can or cannot, is `musttail`: with `tailcc`, LLVM requires only that the return types match, so a procedure's last call does not grow the stack, across modules too. A back end returns a result that needs more registers than its convention has in memory itself, through a hidden pointer, and then cannot make a `musttail` call of the procedure (x86-64 aborts with "failed to perform tail call elimination"). So a procedure whose outputs, after its flag, need more registers than the target returns results in (`TargetInfo::Results`: on x86-64 three words and two floats, as `RetCC_X86_64_C` has them; on AArch64 eight of each; on AMDGPU sixteen words, floats among them; elsewhere two of each) returns them in memory itself: it takes a pointer to a struct of its outputs before its inputs, stores them there, and returns only its flag. A tail call passes on the pointer it was given, since the callee's outputs are the caller's; any other call gives a slot of its own, where a foreign call's slots go, and loads the outputs after the call. Whether a procedure does is a matter of its signature and the target alone, so modules agree on it. `CodeGen/results-in-memory.m` checks it on x86-64 and AArch64 and runs loops of ten million tail calls. LLVM's tail recursion elimination turns a procedure whose recursive calls' results are combined after them, as `2 * f(N - 1)`, into a loop with an accumulator, and so rewrites each of its returns to combine the accumulator, or to select a return value; a return after a `musttail` call must stay next to it, so this tree's `TailRecursionElimination.cpp` keeps such recursive calls, in a function with such a return, instead of rewriting it, which `-O2` failed on with "failed to perform tail call elimination on a call site marked musttail" (2026-11-04; `llvm/test/Transforms/TailCallElim/musttail-other-returns.ll` and `CodeGen/musttail-accumulator.m`). No upstream report of it was found; the fix is one to send upstream.
- **Arithmetic.** The functions and comparisons of `int`, `uint`, the sized integer modules and `float` are built into code generation as `arith` operations, which MLIR's `ArithToLLVM` lowers; `float`'s others are called, and the integer modules' others, such as `int.pow`, are not yet implemented. The reverse modes of the integer modules' `+`, `-` and `xor`, `X + Y = R` with `X` or `Y` the output, are built in too, wrapping as the others, since a unification of two applications, `A + B = C + D`, calls the second with its result bound when an argument of it is computed after it, as in mmc (`CodeGen/reverse-modes.m`). As in mmc's programs: integers wrap; `//`, `rem`, `div`, `mod` and `float./` throw `domain_error("int.'//': division by zero")` and the like for a zero divisor; `<<` and `>>` throw for an amount outside 0 to the width; `abs` of the least integer throws; `div` and `mod` round toward minus infinity; `-X` of a float is `0.0 - X`. The least integer by -1 gives itself and 0, as mmc folds it, though mmc's programs die of SIGFPE (the register, "The quotient of the least integer by -1"). A conversion between integer types keeps the low bits, and extends by the sign of a signed source. `X = X` succeeds even for a NaN, as with mmc; otherwise float equality is IEEE's.
- **Polymorphism.** A procedure takes a `!merk.type_info` first for each type variable of its signature, and a value of a type variable is a `!merk.word`. A call passes what polymorphism analysis says: the caller's own, or `merk.type_info`, which lowers to a static value when its arguments are and otherwise to a new cell. An argument at a callee's type variable goes through `merk.to_word`, and an output through `merk.from_word`: a narrow integer is zero-extended and a float keeps its bits, as in a field. A value with no bits, such as `io.state`, is the word 0, and an output of it is dropped. A term has one layout for all its type's instances, and an array too, so `list(int)` is passed as `list(T)` through its word: `merk.to_word`, then `merk.from_word`. `list(T)` has its own `merk.type_def`, whose field of a type variable is a `!merk.word`, which is why a field must be a whole word however it was stored. A unification at a type variable is `merk.unify_word`: it loads the descriptor from the `type_info`, by its tag, and calls the descriptor's unify entry, a generated procedure taking the `type_info` and two words, which takes the arguments' `type_info`s out of it and calls `unify.T` for the most general instance, whose `type_info`s come first. A unification at a known type calls the monomorphic `unify.T` as before. An array's elements are compared the same way, by `unify.array`, whose loop is a tail call, and strings by `merk.string_equal`. A descriptor's table goes with the type wherever the descriptor does. A comparison has the same shape, with `merk.compare_word`, `compare.T`, `compare.array` and `merk.string_compare` ("Type descriptors"). A descriptor's entries are not compiled for a device, so `merk.unify_word` and `merk.compare_word` are errors there, though a comparison at a known type is not.
- **Existentially typed procedures.** A procedure's first results are the type_infos of its quantified variables that no existential constraint has, then a `!merk.typeclass_info` for each existential constraint, as polymorphism analysis lists them; its other outputs follow. A clause has the type it chose, where the declaration and so every caller have a type variable, so its procedure takes its types from the declaration, and at the end of the body its outputs become words, and a dummy one a zero word.
  - **What goals give** are kept in `values_` under variable numbers beyond the body's. An if-then-else or a switch then treats them as it does a variable: a branch's own stay its own, as the types they describe do, and if the other branches cannot succeed they leave with the branch. Keeping them in a map of their own would have needed the same branch logic again.
  - **At the end of a body,** the clauses give the same types, but one that cannot succeed there may have had its goals in a branch left behind, so the first clause whose values all exist gives them.
  - **Tests and comparisons** of a type containing a given type have no procedure of their own, as `list(E)` has no variable for one to take: they use `merk.unify_word` and `merk.compare_word` with the type_info the goal passes.
  - **A call that never succeeds,** erroneous or failure, ends its branch with `merk.fail`, which is never reached after an erroneous one, so that the branch need not bind what others bind.
  - **Not yet:** `foreign_proc`s and instance methods with existential types, the second being what polymorphism analysis leaves out; and a do loop using a value whose type a goal gave, as its predicate's signature would need a type variable for that type ("this procedure's type_infos"). A do loop inside an existentially typed procedure, using only other values, works.
  - **Rejected.** Hidden outputs last: nothing depends on the order, but polymorphism analysis's list, and mmc, have them first. Choosing one clause's results by position: a clause whose arm fails has none to use at the end.
- **Existentially typed constructors.** A constructor's cell has, before its arguments, a `!merk.type_info` for each of its existentially quantified type variables that no constraint has, then a `!merk.typeclass_info` for each of its constraints, in polymorphism analysis's order, which is mmc's: mmc's generated C for `box ---> some [T] box(T) => sh(T)` has the typeclass_info in field 0 and the value in field 1. `merk.type_def`'s `#merk.constructor<"box"(!merk.typeclass_info, !merk.word) hidden 1>` says how many fields are hidden, and the functor keeps its Mercury arity: `merk.construct` and `merk.deconstruct` take or give the hidden fields' values first, and lowering finds the functor by its arity without them. An argument of an existential type is a `!merk.word`; one whose type has one, such as `list(T)`, is the term of that type with the variable, and a construction or deconstruction coerces its values to and from those types through their words. A construction stores what polymorphism analysis passes, and a deconstruction gives what it gets to the goals after it, as a call's outputs are given, unless nothing uses its arguments. In code generation's types, a constructor's existential variables are numbered past every variable of the type it is in, from `kCellExistential`, so that a term of an existentially typed constructor inside another's, as `nest ---> some [B] nest(tagged(B))`, keeps their variables apart; the cost is type_def names such as `list.list(T536870913)`.
  - **Unify and compare.** `unify.T` fails if the two terms' existential types differ, by `merk.compare_type_info`, and then unifies the arguments with the first term's type_infos. `compare.T` compares, before each argument's value, the types of the existential variables it has that no argument before it has, so `tagged(1, 2)` is less than `tagged(2, 'c')`, though `int` is greater than `character`, as mmc's `typed_compare` does for an argument (`CodeGen/existential-constructors.m`). `merk.compare_type_info` calls the runtime's `merk_compare_type_infos`, which orders types as mmc does, found by experiment: by their constructors' modules, then names, then arities, then their arguments in order, a primitive type's module being `builtin` and a tuple's name `tuple`, so `'x' < 1.0 < 1 < 1i8 < "a" < {1, 2} < 1u < [1]`, `aa(1) < bb`, and `{"z", "z"} < {1, 2, 3}`. The runtime is the host's, so `merk.compare_type_info` is an error on a device, as dispatch on a type class is.
  - **A constrained variable's type_info** is in the typeclass_info of the first constraint that has it as an argument, after its method table and its superclasses' typeclass_infos; one only inside a constraint's argument's type is "not yet implemented".
  - **Rejected.** Hidden fields after the arguments: the order is free, but mmc's cells and polymorphism analysis's lists have them first. A functor arity counting the hidden fields: the switches, `merk.has_functor` and the descriptor would each need the Mercury arity back. The constructor's own numbers for its existential variables, as mode analysis's constructor table keeps them: they clash with the variables of a type such as `t(V1)`, where `some [E] f(A, E)` would have `V1` for both. Comparing all the existential types before the first argument, which was the first version: mmc compares an argument before it first.
- **Type classes.** A procedure's hidden arguments are its type_infos, then a `!merk.typeclass_info` for each constraint; an instance's method takes the instance's first, then the method's own. The first dictionary of a method call is its class's, and chooses the procedure.
  - **A dictionary's layout.** Word 0 is the instance's method table; then the superclasses' dictionaries; then the type_infos of the class's types; then the type_infos of the instance's type variables its constraints do not have; then its constraints' dictionaries. The places of superclasses and types do not depend on the instance, which is what lets polymorphic code find them. A dictionary whose arguments are all static is a static global, like a type_info; any other is a new cell.
  - **A table's slots.** Word 0 is the shape of the instance's dictionaries, how many superclasses, type_infos and constraints they hold, for the message copy and the collector. Then one slot for each mode of each method of the class, in declaration order. Each slot holds a dispatch procedure for that instance, which takes the dictionary, the method's own hidden arguments and the arguments as the class declares them, takes the instance's hidden arguments out of the dictionary, converts words to the instance's representations, and calls the instance's method procedure. The call through a table keeps tail calls: 3,000,000 iterations through a method run in constant stack, while mmc's `hlc.gc` overflows its stack on the same program. A nondet or multi method's dispatch procedure is nondet or multi, and passes its continuation on to the instance's procedure; a call through the table is `merk.nondet_call_method`, whose region is the rest of the code, which outlining makes a semidet `merk.call_method` whose last inputs are a continuation and its environment, as for a closure (`CodeGen/nondet-methods.m`).
  - **Known instances.** A method call at an instance the caller knows, whose body this module has, calls the method's procedure directly, with the instance's hidden arguments as values, and makes no dictionary for it. An imported instance still goes through its table, since its method procedures are private to its module.
  - **OpenMP.** A parallel loop's body procedure is given the procedure's own hidden arguments. On a device, a known instance's method without premises needs no dictionary, and runs (`omp-typeclass-gpu.test`); making a dictionary or dispatching through one in a procedure the device compiles is an error, as there are no tables there, and so is a target region in a procedure with hidden arguments, or passing a dictionary into a region: nothing maps one yet. On the device the tables are empty, so that the filter that removes host procedures is not stopped by their symbols.
  - **Rejected.** Having the table hold instance procedures themselves, each taking the dictionary and finding its hidden arguments there: a direct call of a known instance would then have to make a dictionary, which a device cannot, and the method's arguments would be words in every call, so a float would leave its register even when the instance is known. A dispatch procedure gives the indirect path that cost alone. Leaving the direct calls to LLVM: gcc did not devirtualize even a constant closure in the side measurement below, and our compiler knows dictionaries as values, without having to recover them from loads.
- **Closures.** A closure is a `!merk.closure`, which has no type parameters and lowers to a pointer to a cell: its entry's code address, the address of a static layout that says what it captured (null on a device), and those values, a word each. `merk.closure @entry(...)` makes one, `merk.closure_word` reads a captured word, and `merk.call_closure` calls one: it passes the closure and then a word for each input, and gets back a word for each output, with a semidet call's success flag first as a semidet procedure returns it. `test/CodeGen/closures.m`, `closure-tail-calls.m` and `closure-values.m` check them against mmc's output.
  - **A word for every argument,** a type with no values too: an `io.state` is word 0. A polymorphic caller of `pred(T)` does not know whether `T` is `io`, so a dummy argument cannot be left out of the call as a first-order call leaves it out. `!merk.closure` has no type parameters, as the call's argument types say what the closure takes and the words do not depend on them.
  - **Entries.** One entry for each procedure and number of captured values, `<symbol>$closure<N>`, private, shared by every closure of that procedure with that many captured. It takes the closure and the other inputs as words, reads the captured words, converts each word to the procedure's representation, calls the procedure, and gives back words.
  - **Lambda expressions** are procedures of their own, `<symbol>$lambda<k>`. They take the values they capture, in their own representations: the type_infos of the enclosing body, by type variable, then its typeclass_infos, then the values goals gave, then the variables bound before the expression that its body uses, each in order of its number. Then they take their inputs. They give a word for each output, so that a closure call as their last goal is a tail call. The entry calls the procedure like any other. A word copy's body has the same lambda expressions as the procedure's, and uses the procedure's procedures for them.
  - **Closures of named procedures** capture what a call would pass, the type_infos and typeclass_infos polymorphism analysis gives the call, and then the curried arguments. A closure of a procedure built into code generation, such as `int.+`, has an entry `merk.builtin.<symbol>$closure<N>` whose body is code generation's for a call.
  - **Word copies.** `musttail` requires the caller's and the callee's return types to be the same, but an entry and a lambda's procedure return words, and the procedure they call returns its outputs typed: an `f64`, a `ptr`, nothing for `io`. So a procedure of this module that an entry or a lambda calls last, or that a word copy calls last, also gets a copy, `<symbol>$words`, with the same inputs and each output a word, `io`'s as word 0; its body is the procedure's, and calls those last from there go to copies as well. Every loop through closures within a module then runs in constant stack, whatever its types. The words of outputs of types with no values go through the branches of a word body with the others, as a zero word in a branch that has none. A nondet or multi procedure's copy is nondet or multi too, and in a body that gives words, `merk.each`'s continuation takes the body's outputs as words, so that a call in a disjunction's branch or a soft cut's then or else part that gives them on unchanged passes its continuation on, through the continuation, which only gives them as the body's solution (`CodeGen/nondet-word-copies.m`). A call in a commit's generator or a soft cut's condition calls no copy, as its solutions go to the code after it, not to the body's continuation.
  - **Lowering's tail calls.** A call whose results reach the procedure's `merk.succeed` only through `merk.to_word` and `merk.from_word` of them, each to the type the procedure returns at that position, with no other use, is still a call in tail position: the conversions lower to nothing or cancel, and are erased with the call. A block that only converts and succeeds is copied into the branches that go to it, as one that only succeeds is. Code generation folds a word converted to a value and straight back.
  - **Higher-order type_infos.** The type constructor is `pred/N` or `func/N`, where `N` counts a function's result too, of variable arity as tuples are. Its descriptor has the kind `closure`, no module and the name `pred` or `func`. Its unification and comparison procedures call the runtime's `merk_higher_order_called`, which writes mmc's message, "Mercury runtime: called unify/2 for \`pred' type" or "called compare/3 ...", \`pred' for a function's closure too, and exits with status 1. A comparison at a closure type goes through the same procedure, with the type's type_info (register, "Comparing terms that hold closures"). `io.write` writes `'<<predicate>>'` or `'<<function>>'`, as mmc does.
  - **OpenMP.** Making or calling a closure in a procedure the device compiles is an error, as a device does no run-time dispatch. It is checked before `omp-function-filtering`, which removes an entry, as a host procedure, and leaves its closure a poison value. Higher-order specialization will make such calls direct, and needs a typed procedure for each lambda expression to call.
  - **Nondet and multi closures.** A nondet or multi lambda expression's procedure is nondet or multi, as is the entry of a closure of a nondet or multi procedure, which calls it with `merk.nondet_call` and gives each solution in words. A higher-order call of such a closure is `merk.nondet_call_closure`, whose region is the rest of the code, as for `merk.nondet_call`; outlining makes it a semidet `merk.call_closure` whose last inputs are a continuation and its environment, and makes the entry, as every nondet procedure, take them. An entry passes its continuation on as a tail call, as its procedure gives words: a lambda expression's procedure, or the word copy of a procedure of this module; one of an imported procedure, which gives typed outputs, converts them in a continuation of its own.
  - **Not yet:** closures of methods; a closure of an imported procedure calls its typed procedure, so a loop through another module's closures grows the stack; `io.write_list` to a stream; `int.fold_up` and the library's other higher-order predicates (step 6).
  - **Rejected.** *An entry for each `merk.closure` operation*: simplest to make, but the same entry again for each closure of a procedure. *A lambda expression's procedure as its own entry*, taking the closure and words and reading what it captured from the cell: no wrapper, and its last closure call a tail call for free, but a second way of making procedures, with its own reading of hidden arguments and results in memory, and no typed procedure for higher-order specialization or a device to call directly; with entries shared, LLVM inlines the lambda's procedure into its one entry at -O2 anyway. *Tail calls only where the representations agree*, ints, enums and words: the smallest change, but a loop through a float, a string, a cell or `io` would grow the stack a frame per round. *Every procedure returning words*: uniform, but every float result would go through an integer register, `io` would take a register in every call, and results in memory would change for all code. *`!merk.closure<…>` with the closure's argument types*: the cell and the call do not depend on them, and polymorphic code would have to convert between closure types.
- **Field accesses.** A call of a field's function or its update function, which no module defines, is the unifications the manual defines it by ("Field access functions are implemented directly as unifications", "Field selection"): `merk.deconstruct` of the term by the field's constructor, after `merk.has_functor` and `merk.guard` when the type has other constructors, and for an update `merk.construct` of the constructor again with the new value in the field's place (`CodeGen/field-access.m`, of another module's types `CodeGen/field-imports.m`). An update may change the type of a type parameter only its field has (the register, "Type-changing field updates"), and the new value goes in the cell as the other fields stay, since none of them has that parameter (`CodeGen/field-retype.m`).
- **Arrays.** The library's `array` predicates and functions that make, read and update an array, which need no type_info, are built into code generation as `merk`'s array operations: `init`, `make_empty_array`, `size`, `max`, `min`, `bounds`, `in_bounds`, `is_empty`, `lookup`, `elem`, `semidet_lookup`, `set`, `'elem :='`, `semidet_set`, `slow_set`, `semidet_slow_set`, `swap`, `fill`, `fill_range`, `copy`, `resize`, `shrink`, `array` and `from_list`, and the `unsafe_` ones, which check nothing. `generate`, `generate_foldl`, `generate_foldl2` and `foldl` to `foldl5` are `array.m`'s, in Mercury over these, compiled into `libmerkstd`, which compiles the sound stub, as calls are numbered by its modes; a module checked with `-fmmc-uniqueness`, whose numbers are `array.m`'s, reports their calls as not yet implemented (`CodeGen/array-folds.m`). The others, such as `foldr` and `to_list`, are not yet implemented. They check as mmc's library does, with its messages: an index out of range is `index_out_of_bounds("array.set: index 9 not in range [0, 1]")` (`array.lookup` for `elem`, `first argument of array.swap`, `second argument of fill_range`), and a bad size or range is a software error, `Software Error: predicate `array.init'/3: Unexpected: negative size`, `array.fill_range'/5`'s "empty range" when the first index is past the last (checked before either index), `array.resize'/4`'s "cannot resize to a negative size", and `array.shrink'/3`'s "cannot shrink to a larger size" and "cannot shrink to a negative size". An index is checked as an unsigned comparison with the size, so a negative one fails too. An array is a pointer to its size and its elements, a word each (Type representation); `merk.array_get` and `merk.array_set` lower to a load or store after the size word, an element converted to its word (a float's bits, a character extended, a pointer's address), and the others to the runtime's `merk_array_new`, `merk_array_copy`, `merk_array_resize`, `merk_array_fill` and `merk_array_from_list`, which allocate on the heap, as `mrt.alloc` does (The heap, below). They allocate in the runtime rather than in the generated code because they fill or copy many elements, a loop that is the runtime's either way. `merk_array_from_list` counts the list's cells, then copies their heads' words, following the tails while a word's primary tag is that of `[|]`, which lowering passes from the list's layout (`CodeGen/array-from-list.m`). An element's load and store are not `mrt` operations yet: `array.set` is a destructive update, which chunk 3's write barrier will need to see. `merk.check_index` calls `merk_index_out_of_bounds`, which formats the message and reports it as `merk_uncaught_exception` does; on a device it is `llvm.trap`. A set updates in place, which is safe because merkc's declarations of `array` are sound (the register, "Arrays' uniqueness"); with `-fc1 -fmmc-uniqueness` a program that keeps an old version, `A1 = A0, array.set(0, 5, A0, A)`, sees the update in `A1`, as mmc's do.
- **Terms.** Lowering first chooses each `merk.type_def`'s layout (Type representation) and erases it, and then lowers the term operations while their operands still have `merk`'s types, which name their layouts: a value is cast to its LLVM type where it is still `merk`'s, and the cast goes once the value is lowered. `merk.construct` of a constant is the constant's word, and of a constructor with arguments an `mrt.alloc` of the cell, the secondary tag's `mrt.field_store` if it has one, an `mrt.field_store` of each argument in its word, as its own LLVM type, and `mrt.tag`. A construction whose arguments are all known at compile time, constants or static cells themselves, is a static cell instead: a private constant global of its words, which the term points to with its tag, one global for each cell's contents, whatever its type, since a cell is never updated. `[1, 2, 3]` is three such globals, each pointing to the next. `merk.deconstruct` is `mrt.untag` and an `mrt.field_load` of each argument. `merk.has_functor` compares a constant's word, or the primary tag, and for a shared tag also the secondary tag, which it reads through a `select` of the cell's first word or of `merk_no_secondary_tag`, a static word that is no secondary tag, since a constant has no cell to read: one block, no branch, as a region of `merk` has one block. `merk.switch` on a term switches on the primary tag (`mrt.primary_tag`), and from tag 0 on the constants' words, unless the type has only one, and from the shared tag on the secondary tag; on a type of constants only, on the word, and on a type of one constructor of one argument it branches to its case. A type's layout is chosen alone, from its `merk.type_def`, so modules agree on it.
- **The heap.** `mrt.alloc` loads the thread's heap pointer, `merk_heap.next`, adds the cell's size, and compares it with the heap's limit, `merk_heap.limit`; if the cell fits, it stores the new pointer, and if not, which the branch's weights say is rare, it calls the runtime's `merk_alloc_slow`. The runtime (`runtime/heap.cpp`) defines `merk_heap` in thread-local storage, so that the threads of an OpenMP region each allocate on a heap of their own, with no lock; it is null before a thread's first allocation, which then takes the slow path. The slow path takes a new chunk of 1 MiB from `malloc`, wasting the rest of the old, or gives an allocation of more than 128 KiB, such as a large array, a block of its own, so that the chunk is not wasted. Nothing is freed yet. For testing, `MERK_HEAP_CHUNK_BYTES` sets the chunk's size, and an allocation of more than half of it has a block of its own: with `MERK_HEAP_CHUNK_BYTES=64 llvm-lit`, which lit passes on, every program in the tests runs with nearly every allocation on the slow path. The runtime's own allocations, arrays', bump the same heap through `merkurius::runtime::Allocate` (`runtime/heap.h`), with the same slow path, so that a collector need only hook that. A field is loaded and stored with the word's alignment. On an OpenMP device, which has no heap yet, a construction that allocates is an error at it, in the device compile, while a static cell is not. `CodeGen/heap.m` allocates many chunks' worth, a large array among cells and static cells, with mmc's output; `CodeGen/omp-heap.m` builds a list in each iteration of a parallel loop, on four threads; `CodeGen/omp-target-term.m` checks the device's error.
- **Last call modulo constructor.** Before the term operations lower, the pass of "Deterministic and semidet code" below rewrites the calls whose results go into cells. `merk.hole` lowers to null, `merk.field_ref` to the field's address in the cell, from the term's untagged pointer, and `merk.ref_store` to a store of the word there, a narrow integer zero-extended as a field is (`Dialect/Merk/holes.mlir`, `CodeGen/last-call-modulo-constructor.m`).
- **Exceptions, for now.** Until `merk.throw` and `catch` exist, an exception is not caught: `merk.throw_if` calls the runtime's `merk_uncaught_exception`, which flushes the standard output, writes "Uncaught Mercury exception:" and the exception to the standard error as mmc's programs do, and ends the process, or the program with status 1 (Runtime system, "Exceptions, links and monitors"). The library's C code calls it too.
- **Library.** `libmerkstd` compiles `io`, `float`, `math`, `require`, `pair`, `maybe`, `list`, `solutions`, `process`, `string`, `char` and part of `array` (Lowering, "Nondeterministic code", "Collecting solutions"), with `-fstub-unimplemented`, which gives a procedure code generation cannot make yet, such as a nondet mode, a body that reports it at run time (`docs/INPROGRESS.md`, step 6). `require`'s errors throw `software_error` with mmc's messages. `list` is written in Mercury, every mode of it, with clauses of their own for modes such as `reverse(out, in)`; its behaviour where the manual leaves it open is mmc's, found by running mmc's programs: `sort` and `merge` are stable, `merge` takes the first list's element on a tie, `remove_dups` and `sort_and_remove_dups` keep the first of equal elements, a negative or too large count fails in `split_list`, `take` and `drop`, `duplicate` of a count that is not positive is `[]`, `zip` ends with the rest of the longer list, and the `det_` predicates and `take_upto` throw mmc's messages, `det_index1` as `det_index0` and the functions `det_index0` and `det_last` as their predicates. Its constructing recursions, such as `append` and `map`, are tail calls by last call modulo constructor (Lowering, "Deterministic and semidet code", "Last call modulo constructor"), so it runs on a million elements in an 8 MB stack, as mmc's library, compiled with `--optimize-constructor-last-call`, does; `foldr` and `filter/4` overflow there in both. `float`'s and `math`'s functions are `foreign_proc`s calling C's math library, with mmc's domain checks, as `domain_error("math.sqrt")`, and its rounding: `round(X)` is `floor(X + 0.5)`, `float.pow` multiplies by squaring, and a float out of `int`'s range converts to `min_int`. Later these should be the `math` dialect's operations and LLVM's intrinsics, which LLVM can fold and vectorize, keeping the checks; the `foreign_proc`s are how the library is written until then.
- **OpenMP target regions.** With `-fopenmp`, a goal after `%$omp target` with no clauses becomes an `omp.target`, a generic kernel, as Flang makes a target region with no loop in it (`target teams distribute parallel do` is SPMD, Language extensions, "Loops", Translation), as Flang and ClangIR lower target regions to the `omp` dialect; the other directives, but `parallel do` and `target teams distribute parallel do` before a loop (Language extensions, "Loops", Translation), are not yet implemented. The goal is generated in the region; the values it uses from outside are mapped `to`, by copy, and the variables it binds that are among its nonlocals are mapped `from`, by reference, each through a stack slot of the procedure (an `llvm.alloca`, in the target's alloca address space and cast to a generic pointer), as Flang maps a scalar: the region loads its inputs and stores its outputs, and the procedure loads the outputs after it. Only integers, floats, arrays of them and terms whose cells hold only scalars cross so far (Language extensions, "Target regions"). Such a term is of a type of one constructor, of two or more integers or floats, so its pointer has no tag, and is mapped `to` as an array is, by its pointer, with its cell's words as the section; a ground one known at compile time, a static cell, too. A mailbox handle is one, `process.mailbox`'s cell of five words, whose layout code generation knows, as it knows an array's, since the type is abstract outside the library. A region gives only scalars and arrays: a cell the device makes, with the device runtime's `merk_device_alloc` (Send and receive in a target region, As implemented), is in device memory, which the host cannot read (`CodeGen/omp-target-cells.m` and `omp-target-mailbox.m`, with their GPU runs; 2026-11-07). An array is mapped as ClangIR maps a pointer's section, `p[0:n]`: by its own pointer, as `var_ptr`, with `omp.map.bounds` over its size word and its elements, words 0 to its size, which only the host compile gives, since only the host copies; the region's argument is then its pointer on the device. An array the region only reads is mapped `to`; one it gives, updated in place, `tofrom`, and after the region the variable is the given array. An array the device made in the region is copied to the host as the region ends, since the device's arena ends with it (`merk_device_give_array`). When the region's own `merk.array_set` or `merk.array_fill` shows which array it updated, that is all; when it does not, as when the region calls a procedure that updates one, the region compares the array it gives with each given array of its type, by pointer, and gives the number of the one it is, or -1, and the pointer too, each through a slot mapped `from`; the host takes the given array of that number, or else the pointer, which is a new array only when the region ran on the host, where a procedure it calls may allocate one. The given arrays of its type are mapped `tofrom`, then, so a read-only one may be copied back unchanged. Making or filling an array in the region itself is not yet implemented, since the device has no runtime for it, and a procedure the region calls that does is an error at its call in the device compile. Arrays of strings or of arrays do not cross, since their elements are host pointers. `builtin.unrealized_conversion_cast` converts between `!merk.array` and the `!llvm.ptr` that `omp.map.info` takes; lowering removes the casts once arrays are pointers. The `omp` dialect's passes run as ClangIR and Flang run them, the first two on the `merk` module, so that a device compile lowers only what the device runs: `omp-mark-declare-target` marks the procedures the regions call, which it finds through `merk.call`, a `CallOpInterface`, on `merk.proc`, which has the dialect's `DeclareTargetDefaultModel` as `llvm.func` does (a private procedure only regions call is `nohost`, as Flang's are; the host keeps it, for running the region there); `omp-function-filter` removes the others on the device, except those that hold regions. It replaces what a call of a removed procedure gave, outside the regions, by a poison value: `ub.poison` for a type that is not the LLVM dialect's, such as an array's (this tree's change to the filter, which made `llvm.mlir.poison`, of LLVM types only); lowering gives it the type it lowers to and keeps it `ub.poison`, so that `omp-host-op-filter` makes it an argument of the function, as the `var_ptr` of a map entry must be: an LLVM poison constant has no uses of its own, so translation could neither tell two entries' apart nor replace it by the kernel's argument (`CodeGen/omp-target-host-made.m`; found 2026-11-07, with no upstream report found). The filter kept the arguments of every block of the host procedure as they were, but only the entry block's go with the procedure it rebuilds, so a region in a branch, given a value that the branch's block took as an argument, stopped the compile on an assertion; it now replaces those of other blocks by placeholders too, as its documentation says (this tree's change; `CodeGen/omp-target-branch.m`; found 2026-11-07). Lowering then flattens the region's goals into blocks of the region, which ends with `omp.terminator`, keeps each procedure's mark on its `llvm.func`, and runs `omp-mark-declare-target` again, for what lowering calls, then `omp-host-op-filter`, which keeps only the regions of the host procedures that hold them, and `omp-stack-to-shared`, which, being the LLVM dialect's, run after lowering, as in ClangIR. The module records the offload attributes (`omp.is_target_device`, `omp.target_triples` and the rest, set with `mlir::omp::setOffloadModuleInterfaceAttributes`), which the `omp` dialect's translation to LLVM IR reads; the host compile, with `--offload-targets=` (or Clang's alias `-fopenmp-targets=`), makes the region an offload entry that it launches with `__tgt_target_kernel`, and without device triples it calls the outlined region directly. The device compile, `-fopenmp-is-target-device` with `-fopenmp-host-ir-file-path`, keeps only the region and the procedures it calls. The directive belongs to the caller: a target region in a procedure that a target region calls would run on the device, as reverse offloading would, so it is an error, found on the `merk` module after `omp-mark-declare-target` has marked what the regions call. On a device:
  - procedures use the C calling convention, since AMDGPU has no `tailcc` (`TargetInfo::HasTailCallConvention`); a call in tail position is still `musttail`, which AMDGPU makes a jump. Neither Flang nor ClangIR sets a convention for OpenMP device functions, which are C's; a kernel's `amdgpu_kernel` is `OpenMPIRBuilder`'s;
  - the module has AMDGPU's metadata from the target info, as ClangIR's `emitAMDGPUMetadata` gives it: the code object version, recorded on the module as `merk.amdgpu_code_object_version` (600, Clang's default and ROCm's device libraries'), which becomes the `amdhsa_code_object_version` module flag, and the PIC level 2 of Flang's device compile;
  - a failed check of `merk.throw_if`, such as a zero divisor, is `llvm.trap`, since the device has no runtime to report it, which stops the kernel and the program;
  - only the module's own procedures are compiled, so a call of another module's, such as the library's, or of foreign code is an error at the call.

  `merkc -fopenmp --offload-arch=gfx1030` runs the jobs that compile and link for offloading (Driver, "Linking and offloading"); its programs run their regions on a gfx1030 GPU, with `OMP_TARGET_OFFLOAD=MANDATORY`.
- **Tests.** `test/CodeGen` checks the `merk`, the LLVM dialect and the LLVM IR of Mercury modules in one file each, and runs the programs that print, checking their output (`hello.m`, `foreign-proc.m`, `foreign-proc-semidet.m`, `count.m`, `factorial.m`, `fib.m`, `switch.m`, and `io-output.m`, `arithmetic.m` and `float-math.m`, whose output is mmc's; `arithmetic-errors.m` checks an uncaught exception's report and exit status; `arrays.m` uses the array predicates, with mmc's output, and `array-errors.m` each of their checks' exceptions, with mmc's messages and exit status; `omp-target.m` checks a target region's host and device compiles, each offload job's output, and runs it on the host; `omp-target-gpu.test` runs it on the machine's GPU, when `amdgpu-arch` finds one and the OpenMP runtime libraries are built; `omp-target-nested.m` checks that a region cannot be in what a region calls, and `omp-target-errors.m`, `omp-target-device-call.m` and `omp-target-array-made.m` what is not implemented yet; `omp-target-arrays.m` checks arrays' maps, host and device, and runs it on the host, with mmc's output, and `omp-target-arrays-gpu.test` on the GPU, checking what each kernel copies); `Driver/compile.test` builds and runs a program, and `Driver/offload.test` checks the offload jobs. `terms.m` uses each layout, switches on constructors, semidet deconstructions and unifications of terms, with mmc's output; `heap.m`, `omp-heap.m` and `omp-target-term.m` the heap (The heap, above). `utils/codegen-oracle/compare-runs.py` compiles programs with mmc and with merkc, at each optimization level asked for, runs merkc's with each heap chunk size asked for, and checks that each prints what mmc's prints, with its exit status; `generate-layout-programs.py` writes random modules for it, of types of every layout, with constructors with arguments around the primary tags' boundaries, 7 to 9 of them with and without constants, so that some share the last tag, and fields of ints, floats, chars, other types, the type itself, lists and tuples, which print every constructor by a switch, code each one by clauses, and print, unify and take apart values built both static and, with some integers known only at run time, allocated, each in a function of its own, so that mmc's mode analysis sees only ground values. `CodeGen/oracle/layouts-1.test` runs twenty of them at `-O0` and `-O2` and with chunks of 64 bytes; 200 more, seeds 2 to 11, agreed on 2026-11-04. `generate-run-programs.py` writes modules of int functions whose bodies are arithmetic, division by constants, bit operations, shifts, `min` and `max`, if-then-elses and calls of the earlier procedures, det predicates of two outputs, semidet predicates, semidet switches on ints, tail recursive and other recursive functions, float functions, and functions that build values of the layout generator's types from an int and fold them back, by clauses; values are kept small, results modulo a prime and products by small constants, since the language leaves overflow to the implementation. `main` calls each procedure on constants and in a loop on values computed at run time, and prints the results. It found the implied-mode, condition and unreachable-block bugs, and the reverse modes that code generation lacked (2026-11-04); `CodeGen/oracle/runs-1.test` runs twenty, and 200 more, seeds 2 to 11, agreed after the fixes. It leaves out what code generation does not do yet: disjunctions other than switches, including a negated conjunction of negations, which becomes one; strings other than literals, chars other than literals and tests, and the library modules without an implementation, such as `bool`, whose calls fail only when the program is linked. `test/Dialect/Merk` checks the dialect alone: that `.mlir` modules print as they parse and lower for the target they record (`ops.mlir`, `arrays.mlir` for the array operations, and `terms.mlir` for the term operations, each layout and static cells), and its verifiers' errors, at file:line:column (`invalid.mlir`); `test/Dialect/Mrt` the same for `mrt` (`ops.mlir`, with `mrt.alloc`'s fast and slow paths, and `invalid.mlir`).

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
- An abstract type of another module, whose constructors a module cannot see, has no `merk.type_def` there, and its values are pointers: they are passed and stored as they are, and unified, compared, written and copied through the type's descriptor, which the module that defines it makes (Type descriptors). That module's `merk.type_def` of a type its interface declares abstractly is marked `abstract`, and lowering reports as not yet implemented one whose layout does not pass as a pointer does: a pointer, or a word, as an enum is, which goes in the same registers. A notag of a float, which goes in a float register, or of a narrower integer or a character is reported (`CodeGen/abstract-type-errors.m`). Giving such a type a word, the float's bits in it, would lift that (`CodeGen/abstract-types.m`).
- A field narrower than a word, a `char` or a sized integer, is its whole word, zero-extended, in a store and in a static cell alike, and a load truncates it. Polymorphic code passes a value as one word it does not look inside, so every field must be a whole word that does not depend on how it was made. Zero extension needs no signedness, which `merk` and `mrt` types do not have, and only the value's bits have to be determined: a descriptor says how to compare them. A narrow store leaving the other bytes unset, as at first, was the alternative: monomorphic code cannot tell, but a polymorphic read of the word would see them.

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
- **Trace goals** (reference manual, "Trace goals"). One whose `compile_time` condition does not hold is left out. `flag(Name)` holds with `--trace-flag=Name`, as with mmc's option; `tracelevel(shallow)` and `tracelevel(deep)` never hold, as there is no execution tracing; and of the grade tests only `c` and `mlds` do, as in mmc's default grade on Linux, `hlc.gc`, since the foreign language is C and the code is high level, as mmc's MLDS back end's is: the others name what Merkurius does not have, though `trail` should hold once there is a trail (`docs/INPROGRESS.md`, item 3). One with a `run_time` condition is in `merk.ite`, whose condition is `and`, `or` and `not` as `arith` operations on the runtime's answer for each `env(Name)`, `merk_env_at_start`, whether the variable existed when the program started, whatever its value, even empty, as the manual says; main saves a copy of the environment before the program runs, so that the program's own changes do not count (Runtime system). mmc reads each variable into a global when the program starts; asking each time the goal runs costs a search of the saved environment, which a trace goal that does I/O hardly notices. The io state has no value to pass, so the goal is its own code. One whose goal has more than one solution is in `merk.commit`, as any goal that has at most one solution to the goals around it is (`CodeGen/trace-goals.m`).
- **Last call modulo constructor** is on at every optimization level (decided 2026-11-06). A recursion followed by a construction allocates the cell first and passes the address of its open field (destination passing), so list-building recursions become loops. mmc 22.01.9 does this only with `--optimize-constructor-last-call`, which its library is compiled with; without it a user's `append` of a million elements overflows mmc's stack.
  - **How.** A `merk` pass in lowering (`CodeGen/LastCall.cpp`), run once the types' layouts are known, so that it knows which constructors are cells, rewrites a call whose results go into cells the procedure returns. The procedure makes each cell before the call, with `merk.hole`, null, where the result goes, takes the field's address with `merk.field_ref`, a `!merk.ref`, and calls the callee's variant, `<proc>.lcmc`, which takes the procedure's inputs and then a reference for each output it stores; it returns the others, such as an accumulator's final value. In the variant, each output it stores is stored through its reference with `merk.ref_store` where it is made, in every branch: a cell made the same way is stored there first and then its own hole is passed on, so the recursive call is last and `musttail`; a call whose result is the output itself passes the variant's own reference on, also as a `musttail` call.
  - **Which calls.** A det or semidet procedure's output, or outputs, made through if-then-elses and switches whose results only the procedure's success uses; in a branch, a call each of whose results either goes into one cell of a constructor stored on the heap (not a constant or a no-tag type), which the branch returns, or is an output itself; with nothing after the call but operations free of effects, so that the call can go last. A callee gets a variant when such a call of it is in its own strongly connected component of the call graph; which of its outputs the variant stores are those that go into cells at such calls, and, found to a fixpoint, those that a call in another variant passes on, so that in mutual recursion such as `odd_pos` and `even_pos` (`CodeGen/last-call-modulo-constructor.m`) both variants call each other last. Of two calls whose results go into one cell, as a tree's `node(double(L), X, double(R))`, the last one is.
  - **What it covers, against mmc.** Every list operation of the library and of the test runs on a million elements in an 8 MB stack. mmc, with the option, needs a bigger stack for the mutual recursion, for a constructor with an existentially typed argument and for the tree, and its library's `zip` overflows; `foldr` and `filter/4`, whose recursive call comes before the test of the element, overflow in both. A call before an if-then-else whose branches construct is not last in either, so the library's `filter/3` and `filter_map` test first and then call (mmc's need the same, as `filter/4` shows).
  - **Rejected: only with an option, as mmc,** or only at `-O2`: a program that runs at one level and overflows its stack at another is a difference in what a program does, not in how fast it is, and processes have fixed stacks (Runtime system, "Processes and scheduler"). **Rejected: the library written with accumulators and `reverse`,** which would leave users' own recursions overflowing, allocate every list twice and be a workaround for a missing compiler feature. **Rejected: in MLIRGen or on the semantic form,** before the layouts are known; and in LLVM, which cannot see that a field of a fresh cell is the only use of a call's result.

### 4. Nondeterministic code

Nondet procedures use continuation passing, as mmc's high-level C back end does. A nondet procedure takes a success continuation; producing a solution calls it, and failing returns. One change makes this fit our runtime: each continuation returns an `i1` meaning "stop". A commit is a `true` returned up the chain, so there is no `setjmp`/`longjmp`, and every frame stays visible to the stack maps.

The main alternative is an explicit choice-point stack in the process heap, as mmc's low-level back end has. Continuation passing is chosen because each process has its own stack (Runtime system section), so a deep native stack can be suspended. That stack has a fixed size, so a deep search needs a process spawned with enough of it.

**In `merk`: regions, then outlined** (decided 2026-11-06). MLIRGen emits nondeterminism structured, as it does if-then-else and switch: a nondet or multi procedure keeps its outputs as results, `merk.succeed` in it gives a solution and failure backtracks; `merk.nondet_call` has a region, the rest of the computation, run once per solution with the callee's outputs as arguments; `merk.disj` runs its branches in turn, each ending as the code around it does; `merk.each` runs a goal with more than one solution, such as a disjunction or a switch or if-then-else whose branches have more than one, whose solutions, given by `merk.solution`, go to one continuation region the branches share, so the goals after it are not copied into each; `merk.soft_ite` is an if-then-else whose condition has more than one solution, whose then part runs for each of them, given by `merk.solution`, and whose else part runs only if there is none, even when the then part fails; and `merk.commit` runs a goal for its first solution only, failing if there is none. Code generation emits each goal after a nondet one into its continuation, so the rest of the procedure ends up in the innermost region. A `merk` pass (`CodeGen/Outlining.cpp`) then outlines each continuation into a procedure, named after the one it came from with `.cont.` and a number, that returns `i1` "stop", with the values it uses from outside in an environment, and lowering sees only that explicit form: a nondet or multi procedure becomes a semidet one with no outputs that takes a `!merk.cont` and a `!merk.env` after its inputs; a solution is `merk.call_cont` of the continuation, then `merk.succeed`, which returns true if the continuation asked to stop; and failure, as in a semidet procedure, returns false or goes to the next branch of `merk.disj`. A region that only gives its arguments as its own solution is not outlined: the call passes on its caller's continuation, as a `musttail` call, so `member(out, in)` runs in constant stack (`CodeGen/nondet-tail-calls.m`).

```mlir
merk.proc @pairs(%l: !merk.term<"list.list(int)">, %k: i64) -> (i64, i64)
    attributes {determinism = #merk.determinism<nondet>} {
  merk.nondet_call @m.member_2_1(%l : !merk.term<"list.list(int)">) {
  ^bb0(%x: i64):
    merk.nondet_call @m.member_2_1(%l : !merk.term<"list.list(int)">) {
    ^bb0(%y: i64):
      %s = arith.addi %x, %y : i64
      %eq = arith.cmpi eq, %s, %k : i64
      merk.guard %eq
      merk.succeed %x, %y : i64, i64
    }
  }
}
```

- **Where a commit goes.** Determinism analysis records each goal's determinism as the goals around it see it, which has at most one solution for a goal with no outputs and for one where only the first solution is wanted. A goal that has at most one solution so seen but a part with more, such as a call of a nondet procedure, a conjunction with such a call, or a switch or if-then-else with such a branch or condition, is in `merk.commit`, which gives the values of the variables it binds; so is the body of a committed choice procedure or lambda expression. A disjunction with at most one solution so seen is an if-then-else chain already. Only the values that the code after `merk.each` or `merk.commit` uses stay in their solutions.
- **A soft cut's then part** is outlined as a continuation whose environment has one slot more, after the values it uses, which it sets on entry: that the condition has a solution. In place of `merk.soft_ite`, the slot is cleared and a `merk.disj` runs the condition, which gives its solutions to the then part, and then, once the condition has no more, the else part if the slot is still clear. mmc's high-level C back end does the same: in the C it generates for our test program, a flag in the environment struct in the procedure's frame is cleared before the condition, set first thing in the then part's continuation, and tested after the condition's call. The rejected alternative, a then part that returns whether it was entered alongside "stop", would change every continuation's type for one construct.
- **A commit's generator** is outlined too, into a procedure that takes an environment of the values it uses and slots for the solution: its first solution is stored there, and it stops. The code around the commit calls it as a semidet procedure and reads the slots.
- **Collecting solutions** (decided 2026-11-06). `merk.aggregate` folds a step over the solutions of a goal: its operands are the accumulators' first values; its generator gives solutions with `merk.solution`; its step takes a solution's values and the accumulators and ends with `merk.yield` of an `i1`, whether to go on, and the new accumulators; its results are the accumulators after the last solution or after a step said not to go on. The step runs on each solution as the generator gives it, in the generator's order, and backtracking into the generator does not undo it, which is the one thing Mercury cannot express; the operation succeeds exactly once. Outlining gives the generator and the step one environment, in the caller's frame: the values either uses, then a slot for each accumulator. The step becomes a continuation that reads the accumulators from their slots, writes the new ones back, and fails, so that the generator goes on, unless the step said not to, when it succeeds, which stops the generator. The generator becomes a procedure, as a commit's does, since a solution ends the procedure it is in; it is called in the condition of a `merk.ite`, since it fails when it has no more solutions and succeeds when stopped, and either way the code after it reads the results from the slots (`Dialect/Merk/aggregate.mlir`). Each accumulator is threaded through the steps and never copied, so `di`/`uo` accumulators such as `io` are sound, and a `muo` solution is used by its step before the generator backtracks over it.
  - **The library.** `solutions.unsorted_aggregate/4`, `unsorted_aggregate2/6` and `do_while/4` have no clauses: code generation makes each of their procedures, one per declared mode, a `merk.aggregate` whose generator calls the generator closure and whose step calls the step closure, with the closure's determinism, `det` or `cc_multi`; `do_while`'s step goes on if the closure gives `yes`, the others' always. Every mode has the same body, as every value but the closures is a word, of a type variable. The rest of `solutions` is Mercury on them: `unsorted_solutions` folds `list.cons` from `[]`, so its list is in the reverse of the order the solutions came in, as mmc's is; `solutions` sorts it with `list.sort_and_remove_dups` inside `promise_equivalent_solutions`, as the sorted list does not depend on that order; `aggregate` and `aggregate2` fold over `solutions`' list with `list.foldl` and `foldl2`. The `multi` modes of `solutions` and `unsorted_solutions`, which promise `non_empty_list`, have clauses of their own. mmc calls `unsorted_aggregate`'s and `do_while`'s steps in the order the solutions come, and `do_while` stops at the first `no` (`CodeGen/solutions-library.m`, run against mmc's output).
  - **Rejected.** A private builtin, `builtin_aggregate`, that the public predicates call: the three predicates are already folds over a generator's solutions, so they are the operation's own users, without hidden names or forwarding clauses, and the interface stays the manual's. `solutions.m` in plain Mercury on a mutable heap cell updated by impure `foreign_proc`s in a failure-driven loop: it needs no compiler construct, but allocates a cell for each collection, updates the heap destructively, which the collector's write barrier must then see, and needs `unsafe_promise_unique` for `di`/`uo` accumulators. The fold as C `foreign_proc`s: C cannot call the `tailcc` continuations, and it would tie the C code to the calling convention. An attribute that makes the step's flag optional: the steps that never stop yield a constant `true`, which folds away, and one form keeps the verifier and outlining simple.
- **Environments are in the frame that makes the call,** not on the heap: a continuation is called only while that frame is live, since nothing stores one, so nondeterminism allocates nothing. Its roots are in frames, but in memory, not in SSA values, so statepoints do not find them on their own; how they are recorded is under GC metadata and safepoints, below. A call that passes an environment of its own frame is not a tail call; one that passes on its caller's continuation is.
- **The structured form is kept for optimizations that need the whole procedure:** inlining a nondet callee into its continuation, so that `member(X, L), X > 3` becomes a loop over `L`; a commit over a search becoming a search that returns at its first match; and removing a continuation once a goal is known to have one solution. None is done yet.
- **Rejected: MLIRGen emitting the explicit form,** continuation procedures and environment operations, directly. It allows no optimization the structured form does not, since outlining produces it, but the reverse needs structure that would be gone; and one Mercury procedure would be spread over several `merk.proc`s, as would its tests.

### 5. Exceptions

Mercury's `throw` and `catch` use LLVM's zero-cost exception handling: `invoke` and landing pads. Statepoints support `invoke`, so exception edges keep precise roots. An exception that leaves a process body ends the process.

### 6. Parallel loops

A parallel loop is a counted loop: induction variable, inclusive bound, step. A `do` loop under a directive is one already (Language extensions, "Loops"); the other targets (the `fold_up` intrinsics and counted recursive predicates) are rewritten into one. MLIRGen builds it as `omp.wsloop` over an `omp.loop_nest`, with `omp.declare_reduction` for reductions, as Flang lowers `!$omp do`, with no `merk` operation in between. Outlining and the libomp calls come from MLIR's OpenMP-to-LLVM translation. In version 1, a parallel body may not allocate; the compiler checks this after inlining.

This is the transformation that experiment E2 applied by hand to mmc's generated C, where it ran 2× faster on 2 cores with a bit-identical result.

### 7. GC metadata and safepoints

- Calls use LLVM statepoints, so the collector can find and relocate every live heap pointer.
- Safepoints sit at calls and at allocation slow paths. Preemption forces a slow path by lowering the heap limit, so loops that allocate need nothing more; loops that do not allocate get `mrt.yield_check` on their back edge.
- Heap cells are headerless, so every stack map also records where each polymorphic root's `type_info` lives, and the compiler keeps that `type_info` live as long as the root.

Six points follow from how LLVM's statepoints work, found on 2026-11-06 and to be checked in chunk 3. Statepoints came from Azul's Falcon, an LLVM JIT tier for a JVM with a moving collector (2014–2015). Azul and GraalVM's Native Image, which compiles ahead of time as we do, use them in production, and the code is maintained: fixes in 2026 include relocations of constants and `alloca`s across blocks. Go's LLVM compiler, gollvm, uses a fork of the rewrite (`GoStatepoints.cpp`, BSD licence, last changed in 2024) for a collector that does not move, so it never relocates. It still shows two of the answers below working.

- **Which pointers are roots.** `RewriteStatepointsForGC` asks the function's `GCStrategy` which pointer types the collector manages. LLVM's built-in strategies say `ptr addrspace(1)`; gollvm's says every pointer. Ours must tell heap pointers from the other pointers a frame holds (continuations, environments, string literals, type descriptors), and the type is all it sees, so heap pointers move to `ptr addrspace(1)` and everything else stays in address space 0. Today every one lowers to plain `ptr`. Three consequences:
  - Static ground terms are globals in address space 0, so a static term used as a heap pointer needs an `addrspacecast`, which the rewrite does not track since it is a constant, as D6 needs. Code generation cannot emit an `addrspacecast` inside a static initializer, such as a static cell that points to another; gollvm's `RemoveAddrSpace` pass replaces those with `ptrtoint` and `inttoptr`, and we would do the same.
  - `merk.word`, the boxed value of polymorphic code, lowers to `i64`. A heap pointer held as an integer is invisible to the rewrite, so a word that may point into the heap becomes `ptr addrspace(1)` too, with unboxed integers and floats put in it by `inttoptr`, and the collector telling them apart by `type_info` as it does fields.
  - A tagged term is a pointer derived from its cell, which statepoints relocate correctly, but only if the tag is added on the pointer itself, by a GEP or `llvm.ptrmask`. A round trip through `ptrtoint` and `inttoptr`, as lowering does for constant words today, hides the base.
- **Roots in frame slots.** Statepoints track SSA values only. A heap pointer *stored* in a frame slot is not a root to them, and we keep some there: the environments of nondeterministic code (`merk.env_new`, a struct `alloca` in the caller's frame) and the slots callees write their outputs into. The fix is to give each live slot's address to the statepoint as a deopt operand: the stack map records it as a "Direct" location, a frame register plus an offset, which needs no relocation, since the slot itself does not move. The collector reads and updates the slot in memory, by the environment's `layout` attribute; since its address is passed to the callee, LLVM reloads it after the call. This is how gollvm records its `alloca`s, with a liveness analysis of slots and a pointer bitmap for each aggregate. A polymorphic value in a slot needs its `type_info` recorded as well.
- **`musttail` calls.** The rewrite turns `musttail call @f` into `musttail call @llvm.experimental.gc.statepoint(...)`, which the verifier rejects (tested 2026-11-06 with `opt -passes=rewrite-statepoints-for-gc`). A real tail call needs no statepoint, because it replaces the caller's frame, leaving nothing in it to scan, and the callee's own safepoints cover the arguments. So lowering marks every `musttail` call `"gc-leaf-function"`, which the rewrite skips; it still compiles to a `jmp`. A plain `tail call`, though, becomes a statepoint and loses its tail call (`callq` and `ret`), so with the collector on, LLVM no longer turns calls into tail calls on its own. Every tail call we need must therefore be a `musttail` from lowering, as last calls of nondeterministic code and tail recursion are now.
- **Holes of last call modulo constructor.** A `!merk.ref` lowers to the address of the hole, a pointer into the middle of a cell. Within one function, statepoints relocate it as a pointer derived from the cell, with `gc.relocate`'s base and derived pair. But a variant gets it as an argument, without the cell, and a cell has no header from which the collector could find its start, so the variant's own safepoints cannot relocate it. So a `!merk.ref` is two arguments, the cell's tagged pointer, which is a root like any other, and the hole's offset in bytes from that pointer, and `merk.ref_store` adds them; the variant's signature has a pair for each reference. Built 2026-11-08. Rejected: one LLVM struct of the two, as the statepoint rewrite does not take a struct that holds a heap pointer. The hole is null from the cell's allocation until it is filled, so a collection in between, as the variant's own allocations can start, finds no pointer there.
- **The hole's store and the remembered set.** Filling a hole is the one store into a cell after it is made. A collection between the cell's allocation and the store can promote the cell to the old generation, and the value stored, a cell the variant made after the collection, is young, so the store needs the write barrier that destructive updates and bindings have: if the hole's cell is old, the hole is recorded in the remembered set. Usually no collection has happened in between and the cell is still in the nursery, which a comparison with the nursery's bounds tells, so the barrier costs a compare and a branch.
- **Walking the stack.** `.llvm_stackmaps` holds one record per safepoint, found by the call's return address. The runtime builds a hash table from them at startup and walks a process's stack frame by frame, scanning what each frame's record lists. Mercury code is compiled with frame pointers (`"frame-pointer"="all"`) and the walk follows their chain. Frames without stack maps (C frames of the runtime, of `foreign_proc` code and of libomp) hold no roots, because the runtime's C++ holds no raw heap pointer across a safepoint, but the walk must get past them, and a chain breaks at a C frame built without frame pointers. So every call from Mercury into C records the last Mercury frame's stack pointer and return address in the process record, as OCaml does, and the walk jumps from there to the next Mercury frame.

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
| `-O0` to `-O3`, `-Os` | Choose the MLIR and LLVM pass pipelines. Without one, code generation for an AMD GPU is at `-O2`, as Clang's linker wrapper compiles OpenMP's device code, since AMDGPU's code generation is not well tested at `-O0`; the passes before it are at `-O0` (`CodeGen/target.m`) |
| `-g`, `-gline-tables-only` | Debug information (Debugging) |
| `-fopenmp`, `-fopenmp-version=` | Read `%$omp` directives (Front end, Lexer) |
| `--trace-flag=`*name* | Compile in the trace goals whose `compile_time` condition needs `flag("`*name*`")`, as mmc's option of the same name (Lowering, "Deterministic and semidet code") |
| `-freceive-goal` | Read `receive` goals (Language extensions, "Processes and mailboxes") |
| `-fprocess-declarations` | Read `:- process` declarations (Language extensions, "Processes and mailboxes") |
| `--offload-arch=` | Compile target regions, with `-fopenmp`, and process kinds that run on a GPU, with `-fprocess-declarations`, for the GPUs named (Linking and offloading, below) |
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

- **Linking** goes through `clang` as the link driver. It is given the objects, the runtime library, `-fopenmp` when any module used it (so Clang links its OpenMP runtime), and the user's `-L`, `-l` and `-fuse-ld=` options. Clang supplies the C start-up files, the C library and the sanitizer runtimes, as `cc` does for rustc and GHC. Implemented so far: without `-E`, `-S` or `-c`, the driver compiles each module to a temporary object file and runs `clang`, beside `merkc` or else on the `PATH`, with the objects, `--target=` if given, and `-L` for the `lib` directory beside `bin`, where `libmerkstd.a` and `libmerkrt.a` are, `-lmerkstd`, `-lmerkrt` and `-lm`, since the library's foreign code may call C's maths library; object files on the command line are linked too. With `-fopenmp`, `clang` is given `-fopenmp` too, so it links `libomp`, as Flang's driver does for any `-fopenmp` link; `-L` names its directory when it is not beside Clang. `libmerkstd` is the standard library's modules that have an implementation so far, such as `io`'s output predicates, its reading of lines from the standard input and its exit status, which the runtime's `main` returns, compiled by `merkc` for the host when merkc is built; so a program for another target links only if it calls none of them.
- **Offloading** follows the jobs Flang's driver runs for `-fopenmp --offload-arch=`, which are Clang's for ClangIR too. With `-fopenmp --offload-arch=gfx1030` (a comma-separated list, or repeated), each source's object file comes from:
  1. host `-fc1 -emit-llvm-bc`, with `--offload-targets=amdgpu-amd-amdhsa`, so that its regions are offload entries;
  2. one device `-fc1 -emit-llvm-bc` per GPU, with `-fopenmp-is-target-device` and `-fopenmp-host-ir-file-path` naming the host bitcode, for the GPU's triple, which has its subarchitecture as Clang and Flang give it (`amdgpu10.30-amd-amdhsa`, from `llvm::AMDGPU::getSubArch`), with no `-target-cpu`;
  3. `llvm-offload-binary`, beside `merkc` or on the `PATH`, packages the device bitcode, one `--image=` per GPU;
  4. host `-fc1 -emit-obj -fembed-offload-object=` naming the package, from the host bitcode read back with `-x ir`, as in Flang: the module's triple and data layout are the compile's, the package goes into `.llvm.offloading` as Clang's `EmbedObject` puts it, and the module is optimized and compiled.

  The link is Clang's, with `-fopenmp --offload-arch=` and the `-O` given, which runs `clang-linker-wrapper`, which compiles the GPUs' code at that `-O`, or at `-O2` without one, as Clang's and Flang's links do: it extracts the device bitcode, links it with `libompdevice` and compiles it for each GPU, adds the code that registers it with the offload runtime, and links the host with `libomp` and `libomptarget`. `-L` adds library directories for the link, such as those of the OpenMP runtime when it is not beside Clang. `-v` is passed on to the link, so that Clang shows its own jobs, such as the linker wrapper's for the GPUs' code, as Flang's `-v` does. With `-c` the jobs stop after step 4; `-E` has no offload jobs; `-S` and GPUs other than AMD's are not supported yet. The link adds Merkurius's device runtime, `libmerkdevice-amdgcn.a` beside `libmerkrt.a`, bitcode for any AMD GPU that this build's Clang compiles from `runtime/device`, with `-Xoffload-linker-amdgcn-amd-amdhsa`, so that the linker wrapper links it into the device image beside `libompdevice` (Send and receive in a target region, As implemented). Not yet done, as Flang does: linking the device libraries (`-mlink-builtin-bitcode` with ROCm's `ocml` and the rest), which device code that calls maths will need, and `-flto=full` for the device compile.
- **Images of GPU process kinds** (docs/INPROGRESS.md, GPU processes without `omp target`, step 4b). With `-fprocess-declarations --offload-arch=`, each source first gets, for each GPU:
  1. device `-fc1 -emit-llvm-bc -fprocess-kernels`, for the GPU's triple and without OpenMP, which compiles a kernel `merk_process_<kind>` for each kind marked `on gpu` and keeps only what the kernels reach; its bitcode is named `.o`, so that Clang hands it to the linker as it is;
  2. `clang --target=amdgcn-amd-amdhsa -mcpu=<gpu> -flto -O2`, which links the image, a shared object, from it, `merk-process-start-amdgcn.o` (the start object, which defines libc's RPC client and `merk_device_number` so that LTO exports them), `libmerkdevice-amdgcn.a`, and libc and libm for the GPU, which Clang finds beside itself. It optimizes at the `-O` given, or else at `-O2`, as the linker wrapper does for OpenMP's device code: AMDGPU's code generation is not well tested at `-O0`, and fails at `-O0` and `-O1` on a `musttail` call in divergent control flow (LLVM bugs to report, below).

  The host's compile, the object's or with `-fopenmp` the host bitcode's, gets `-fembed-process-image=<gpu>:<image>` for each, and embeds the image of a module that has such kinds as a constant, which a constructor registers with the runtime (`merk_register_process_image`). The link is the host's alone without `-fopenmp`, and adds `-lLLVMOffload`, through which the runtime launches the kernels (step 4c). A module without such kinds gets an image without kernels, which its host compile leaves out.

#### LLVM bugs to report

Bugs found in LLVM, to report upstream once checked against its main branch. Each was searched for among the GitHub issues when it was found.

- **AMDGPU: "failed to annotate CFG" on a `musttail` call in divergent control flow** (found 2026-11-10). `llc -O0` or `-O1` for `amdgpu10.30-amd-amdhsa` stops in `SI annotate control flow` on this, which `llvm-reduce` made from a GPU process's kind; `-O2` compiles it. Issue 170830 is the same error from a `callbr`, not this. Not reported. An explicit `-O0` or `-O1` meets it, as LLVM's bug, not worked around: `CodeGen/process-spawn-at-O0.m` is expected to fail until it is fixed, and of the GPU tests' programs, `omp-target-arrays.m`, `omp-target-receive-ring.m`, `omp-target-two-nodes.m` and `omp-target-team-arrays.m` do not link at `-O0`.

  ```llvm
  define void @f(i64 %0, double %1, ptr %2, ptr %3) {
    %5 = icmp eq i64 %0, 0
    br i1 %5, label %6, label %9
  6:
    %7 = icmp eq i64 %0, 0
    br i1 %7, label %9, label %8
  8:
    musttail call void @f(i64 0, double 0.0, ptr null, ptr null)
    ret void
  9:
    ret void
  }
  ```

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

The runtime gives each process its own stack, heap and mailboxes, schedules processes M:N over one worker thread per core, and never stops every process at once. It is written in C++, like the compiler, and linked into every executable. So far `libmerkrt` (`runtime/`) is only the program's C `main`, which saves a copy of the environment, for the `run_time` conditions of trace goals (`merk_env_at_start`), calls `merk_main` and returns exit status 0, `merk_uncaught_exception`, which reports an exception and exits with status 1, and the array functions, which allocate with `malloc` and never free, and report an index out of range (Lowering and code generation, "What is lowered so far"); output goes through C's stdio, which is flushed when the program exits.

### Processes and scheduler

- **Process record:** identifier, stack, heap, mailboxes, links and monitors, reduction budget, scheduling state.
- **Stacks (decided 2026-11-04):** each process has a native stack of a fixed size that never moves, a slot of an `mmap` with `MAP_NORESERVE`, a growing run of slots for each size, committed by the kernel a page at a time as it is touched. Its size is given when the process is spawned (Language extensions, "Processes and mailboxes"), with a default of 64 MB, and is rounded up to a size class, each with a free list, so that spawning a process reuses a slot without a system call. Suspension can happen anywhere: in `receive`, at a preemption check, or at a GC. Since a stack never moves, C frames, `alloca`s and `sret` slots on it are safe, and foreign code and libomp run on it directly. A slot that recursed deeply keeps its pages until they are released with `MADV_DONTNEED`.
- **Stack overflow:** a procedure on a cycle of non-tail calls compares the stack pointer with the process's limit on entry, and overflow ends that process with an error, which its monitors see. A tail call does not grow the stack, so for the stack alone loops and tail recursion would need no check (they have one for preemption). Separate compilation cannot see a cycle through another module or through a closure, so a procedure that makes a non-tail call out of its module or to an unknown procedure is checked as well; a leaf, or a procedure that calls only unchecked procedures of its module, is not. The runtime's walkers (`io.write`, generic `unify` and `compare`, the message copy) check the same limit, since they recurse on a term's depth. The limit lies a margin above the bottom of the slot, which costs only address space until touched, for the frames that are not checked and foreign code. Foreign code that goes past the margin is not caught.

Measured on x86-64 in a C model, 2026-11-04: a stack touched once costs 8.25 KB in a 64 MB slot (a page and 4.25 KB of page tables, which stacks this far apart do not share), 6 KB at 1 MB and 4.1 KB at 64 KB, and about 2 µs for the first touch. A million slots of 64 MB took 61 TB of address space, in one `mmap`. Switching stacks and callee-saved registers took about 6 ns each way. Erlang's process is about 2.7 KB, so we expect about 10 KB with a heap and a record, which is three to four times as much, in exchange for stacks that need no copying.

The check is added to LLVM IR after the optimizer (`AddStackChecks`), since that is where the calls that remain can be seen: a recursion LLVM turned into a loop needs none, and a procedure inlined into its caller cannot take its check away. A cycle may go through tail calls, so the components of the call graph count them, but only a caller that makes a call that is not a tail call in its component is checked, as every cycle that grows the stack has one. A procedure that may leave the module through tail calls counts as unknown to its callers. A thread that runs no process has a null limit, and passes. The first process has the limit of its thread's stack until it has a process stack.

Rejected:

- **The check at a procedure's start in the MLIR lowering,** before the optimizer, as first built. It stopped LLVM from turning `fib(N - 2)` into a loop, and naive Fibonacci took 2.3 times as long, against 1.37 afterwards.
- **Growing a small stack by copying**, as first planned here and as Go does. Relocating a stack needs every pointer into it, and LLVM's stack maps list only GC roots, not the addresses of `alloca`s, `sret` slots and saved frame pointers, and foreign frames have no maps at all. Go can do it because it owns its compiler.
- **Guard pages** as the way to find an overflow. They cost nothing at run time, but each splits the mapping, and at Linux's default `vm.max_map_count` of 65530 `mprotect` failed at 32,754 processes. Raising the limit needs administrator rights, and users must be able to run these programs without them.
- **Segmented stacks** (LLVM's `split-stack`), which grow without copying and start smaller. Untried with `musttail`, and a hot call on a segment boundary is slow, the reason Go and Rust left them.
- **Scheduling:** one worker thread per core, each with a run queue, plus work stealing.
- **As built, 2026-11-08** (`runtime/process.cpp`): a worker for each core, or `MERK_WORKERS`, runs each process until it blocks, with a queue for each worker and stealing, and a lock for each mailbox (How the scheduler was built). `main` is a process too, on a 64 MB stack. A process that ends closes its mailboxes, and the program ends when `main` does, whatever others are doing: its meaning is what `main` does, and a forgotten server should not keep it alive. Only a mailbox's owner receives from it. Rejected: *any process may receive*, since a send would have to find waiters for each mailbox, and two receivers on one mailbox have no order. *Waiting for every process at the end of `main`*: one process waiting for ever would keep the program alive. The runtime calls a closure through a small assembly stub, which restores the stack pointer as the tail calling convention's callee may pop from the stack.
- **Preemption:** rides on the check of the stack's limit, as Go's does. A ticker thread, started with the first spawn, wakes every 10 ms; if a process waits for a worker, it stores a value above any stack in each worker's `merk_stack_limit`, so that the next check fails and its slow path, `merk_stack_limit_passed`, yields. The same check finds a stack overflow. It is added after LLVM's optimizer to the procedures that need it for the stack, to any on a cycle through tail calls or that tail calls unknown code, and to the header of each loop LLVM made, where a heap address moved out of the loop is taken again after the check. The ticker also fires timers when every worker is busy. A process in a target or parallel region does not yield, as libomp ties those to the thread. **Rejected: lowering the heap limit, with `mrt.yield_check` on loops that do not allocate,** as first planned. That is two mechanisms, a process would also have to be able to move at every allocation, including the runtime's own in C++, which keep thread-local addresses across calls, and a tail recursion needs a check for the sake of the other anyway.
- **Blocking work:** I/O should not block a worker. Built on a pool of threads, as Erlang's dirty schedulers do: the process moves to one of them for the call (OpenMP cooperation). The operating system's asynchronous interface is not used so far.
- **Timers:** planned as a timer wheel for each worker; built as one binary heap under a small lock, as nothing has needed more.

#### How the scheduler was built

Done 2026-11-08, as route step 3, chunk 2. The order, and what was decided on the way.

**Where it starts.** A program is one process. Each OS thread has a heap in
thread-local storage (`merk_heap`). Mailboxes have one global lock and a
condition variable, and a receive blocks its OS thread. An uncaught exception
ends the program. `spawn`, links and monitors do not exist.

**Steps,** each its own commit, tested on the CPU first, in the order built:

1. **A process's context.** A record for each process, with its heap's pointer
   and limit (`runtime/process.h`). `merk_heap` holds the running process's
   heap, saved to its record when another runs. `main` is the first process.
   Nothing else changes. Done.
2. **Stacks and switching.** Slots with `MAP_NORESERVE`, size classes with free
   lists, and a switch of stack and callee-saved registers, x86-64 only so far
   (`runtime/stack.h`), are done. Each size class maps a growing run of slots
   rather than one mapping for all, since a limit on the address space can
   refuse a large one, and it then asks for less. The limit check is done too,
   added after LLVM's optimizer. Done.
3. **`spawn` on one worker.** Done. A process runs a closure on its own stack,
   and `main` is one too, on 64 MB. The program's first thread runs them all,
   each until it blocks: `receive` switches to the next runnable process, and
   the thread spins and then sleeps when there is none. A send wakes the
   mailbox's owner. Only the owner receives from a mailbox, as with Gleam's
   subjects, since a send must know whom to wake and two receivers would have
   no order; it closes when its owner ends. The program ends when `main` does.
   A heap starts at 4 KB and doubles. Node connections and device bridges stay
   OS threads, which wake processes. Benchmark: `benchmarks/processes`.
4. **Workers.** One for each core or `MERK_WORKERS`, started as there is work
   for them, so a program that spawns nothing has only its first thread. Done
   so far: reading the heap's address again after a call (D1), and several
   workers under the one lock, with one run queue. A worker with nothing to run
   does so on its thread's own stack, since another may resume the process it
   last ran. A process in a target region stays on its thread and blocks it, as
   libomp ties the region to the thread. Each worker now has a run queue with
   its own lock, and steals half of another's when its own is empty. Timers are
   a binary heap under one small lock, rather than a wheel for each worker, which
   has not been needed. Each mailbox has a lock, each process a lock for whether
   it is asleep, and names and free slots one more. A process's messages are
   stamped from its own count, as the design has it. Slots are in chunks that
   never move, so that finding one takes no lock. Measured 2026-11-08 before the
   locks were split: at 24 workers over half the 3.84 million acquisitions of the
   one lock were contended, with about 10 threads waiting on it at any time. Why 16 independent pairs still did not scale
   was found on 2026-11-08: a fragment took a 4 KB chunk for every message, so
   each message cost a page fault, 85% of a round on one worker, and with many
   workers the faults contended in the kernel. Chunks now start at 256 bytes and
   double. Lock-free queues are not needed for now: to be measured again if many
   senders to one mailbox show the lock.
5. **Preemption.** Done, through the check of the stack's limit (design
   document, Processes and scheduler). A ticker thread, started with the first
   spawn, makes the running processes yield every 10 ms if another waits. Still
   to do: a compile option to leave out what only preemption needs, for programs
   that do not spawn.
6. **Ending a process.** Each piece its own commit. Done: an uncaught exception
   or a stack overflow is reported and ends only its process; in `main`, on a
   thread with no process, or in a target or parallel region it ends the
   program. `throw` and `catch` are not needed first (decided 2026-11-08): a
   process has nothing to unwind, and until they exist the exit reason is the
   exception's text, which becomes a `univ` with them; `down_reason` is abstract
   so that programs do not change. Monitors are done (design document, Exceptions,
   links and monitors), on one node. Links are deferred (decided 2026-11-08): two monitors give the same two-way message, and they can be added later if something needs them. `request` is done, so this step is. Open: a
   monitor across nodes or places reports that it is not supported yet.
7. **What blocks.** Each piece its own commit. Done: target regions and parallel
   regions, on a blocking pool, the latter with the claim of a team's size
   (design document, OpenMP cooperation). Blocking I/O is done too, by the same
   move: `listen`, `connect`, `lookup`, and reading the standard input when it
   would wait. Not moved: a send to another node when its socket is full,
   whose write blocks its worker for up to a tick time. Node
   connections stay OS threads, which wake processes.
8. **The GPU with processes.** Done: a region's launch does not hold a worker
   (step 7); regions of several processes at once on one device, with
   `device(N)` (`process-target-many.m`); and benchmark modes `cpu-process` and
   `gpu-process`, `main` against a process of its own node. The numbers are
   in the design document. Not a worker's job: the send ring, if it is built
   (The send ring), since workers may all be busy. Regions on
   several devices at once are the same code, but need a machine with two GPUs
   to run.

**Decisions:**

- **D1, how code reaches the current process's heap.** A process moves between
  OS threads, and LLVM may keep the address of a thread-local variable across a
  call within a function, so after a switch to another thread it would use the
  old thread's. Options: (a) a register reserved for the context, as GHC's
  `BaseReg`, which is the fastest and which foreign code keeps if it is
  callee-saved, but needs a calling convention and changes to the back end for
  each target; (b) a hidden argument to every procedure, which needs no back-end
  change but costs a register and every signature; (c) keep the thread-local
  variable, and have lowering read its address afresh after each call, through
  an operation LLVM cannot merge, which is the smallest change and costs a read
  after each call. Chosen (the user, 2026-11-08): (c) first, measured at the
  nrev gate, leaving (a) for later. Built: compiled code calls a stand-in,
  `merk.heap_address`, which LLVM knows reads only inaccessible memory, so that
  its calls merge unless something between them may write that memory, as a call
  that can switch the process may. `llvm.threadlocal.address` would be merged
  across such a call. The stand-in becomes that intrinsic after the optimizer,
  and nothing after the optimizer merges the reads across calls. Inline assembly
  would have needed each target's thread-pointer sequence. nrev did not change.
- **D2, spinning under the scheduler.** Decided: a worker with an empty queue
  spins for the same 50 µs, with a yield, before it sleeps.

### Message passing

- **Mailbox handle:** process identifier, mailbox index and generation number. Handles are values, never pointers into another heap.
- **Send:** find the receiver, deep-copy the message into a fresh heap fragment, and push the fragment onto the mailbox's lock-free multi-producer queue. Wake the receiver if it is waiting. The receiver's heap is never locked.
- **Arrival order:** each message is stamped, as it is queued, with the receiving process's arrival count, taken with one atomic increment by the sender. The stamps order messages across all of a process's mailboxes.
- **Receive:** in each mailbox of the selector, find the oldest message its case accepts: the head of the queue for `selecting`, or the first message for which the selector succeeds, scanning in arrival order, for `selecting_matching`. Take the candidate with the lowest stamp, if it was queued before the mailboxes were looked at; one queued while they were looked at may be younger than a message that arrived in a mailbox already looked at, so they are looked at again. The matched fragment joins the receiver's heap and is merged at its next GC. Unmatched messages keep their order. A process waiting in `receive` is woken by a send to any of its mailboxes.
- **Mailbox lifetime:** a mailbox lives until `close_mailbox` or the owner's exit. Closing it drops its queued messages and makes its handle dead.
- **Dead mailboxes:** a send to a handle whose generation no longer matches, or whose process has exited, is dropped. This is what discards late replies. Generations make handles of closed mailboxes dead forever, even when the mailbox's slot is reused.
- **Cheap mailboxes:** every `request` creates one, so creating and closing a mailbox must cost about as much as a small allocation: a slot in the owner's mailbox table and a new generation number.
- **Observability:** a process can read its own queue lengths, since non-blocking send gives a slow consumer no backpressure.
- **Later:** remember a scan position for repeated selective receives on the same reply, as BEAM does.
- **So far** (2026-11-07, `runtime/mailbox.cpp`): a program is one process, so one table of mailboxes, one lock and one condition variable on the monotonic clock serve every mailbox, and a receive that waits blocks its thread on the condition variable. A mailbox is a queue of messages, each in a fragment of its own, which is never freed until the collector merges fragments. `send` copies the message by its `type_info` before taking the lock, as the collector will trace (Type descriptors): a constant, number or character is its word, a string and an array are copied, an array's elements by its element type, and a cell field by field with its tags, the secondary tag too, the last field of the term's own type in a loop rather than a call, so that a long list uses no stack. A closure or a value of an existentially typed constructor cannot be sent to another node yet, and the runtime reports them. A closure is copied by its layout. An existentially typed constructor's hidden fields are copied too: a type_info cell by its descriptor, and a typeclass_info by its method table's shape. Each queued message has a stamp, its place in the order of arrival over all mailboxes, and a count of arrivals and closes lets a receive over a selector wait for any of its mailboxes on the one condition variable. A receive over a selector reads the last stamp given before its cases look, and takes the oldest message they find only if its stamp is not later; otherwise they look again, from where they stopped, and the next look takes that message or an older one, so it ends even while another thread sends. Taking it at once lost the order under a concurrent sender: 1 arrived in the first mailbox after it was looked at, 2 in the second before it was, and 2 was taken first (`CodeGen/process-selectors-sending.m`, found by `MailboxTest.SelectorTakesTheOldestWhileSending`, which failed about 3 runs in 100; 2026-11-07).

### Memory management

- **Per-process heap:** a bump-pointer nursery and an older generation. Minor collection copies survivors out of the nursery; major collection compacts the old generation.
- **Collection runs in the owning process,** at its safepoints, with no global pause.
- **Roots:** the process stack (from stack maps), message fragments received but not yet merged, and the process record.
- **Write barrier:** destructive updates (array `set`, mutables, `store`) record old-to-young pointers in a remembered set. Immutable terms need no barrier.
- **Large objects** (big arrays, long strings) live in a separate space and are never copied by minor collections.
- **Process exit** frees the whole heap at once. Built 2026-11-08: a process has one list of chunks, its heap's and those of the messages it has read. A message's chunks become its reader's at the first read, even if the message stays in the mailbox, as a selective receive looks at messages without taking them. Rejected: freeing a message's chunks when it is removed, since the value it was read for may still be in use.
- **Static data** (ground terms, type descriptors, string literals) is outside every heap and never traced.
- **So far** (2026-11-04): a heap for each thread rather than each process, in chunks that are never freed, and large blocks of their own (Lowering, "What is lowered so far", The heap). Where a process's heap pointer and limit live is decision D2 of `INPROGRESS.md`, made with chunk 2's context switch; until then they are thread-local, which keeps the threads of OpenMP regions apart, and only `mrt.alloc`'s lowering and `runtime/heap.h` read them.

**How the collector finds pointers: by type (decided 2026-11-04).** Heap cells have no header, as in the type representation above. The collector and the message copy know a cell's layout from its type: the type of each root comes from the stack maps, which record where each polymorphic root's `type_info` is, and the type of each field from the type descriptors. Forwarding marks during a copy live in a side bitmap, one bit per heap word, and the to-space is scanned through a work stack of typed fields, since it cannot be scanned linearly.

Two alternatives were measured in a C model (experiment E7) and rejected:

- **A header word per cell**, giving its size, which fields are pointers and its constructor, was the first recommendation: a simple Cheney collector and deep copy that need no types, a self-describing heap for debuggers, and room for the secondary tag. It was rejected for its size. It makes a list cell 24 bytes instead of 16, a tree node 32 instead of 24, and a list of pairs 48 instead of 32 per element, and that cost is paid by every program on every traversal once its data leaves a cache: +16% on a large list, +50% on a large tree, +45% on a list of pairs, up to +50% on building, +11% on nrev and +17% on polymorphic nrev, where the header's pointer bits come from a `type_info` at run time. Its advantages are engineering effort, paid once; its cost would be paid on every run.
- **Pages of cells of one shape** (a big bag of pages), with the shape in the page table, keep the headerless size and need no types to trace, and segregating cells by shape even made a list of pairs 22% faster to sum. They were rejected because polymorphic code knows a cell's shape only at run time, so it must choose among bump pointers there (+36% on polymorphic nrev), and because every heap needs a page for each shape it uses, which with a heap per process is the wrong cost for many small processes.

What the choice costs instead: a `type_info` must be findable for every polymorphic root at every safepoint; the tracer is type-directed, with descriptors for every type; and `send`, which must not overwrite the sender's cells, copies by the message's `type_info`, which it receives as a hidden argument anyway. `mrt` hides the layout, so the decision can be revisited: the nrev gate of chunk 1 measures it again on our own compiler.

How the collector finds roots in the stack, and the four points it rests on (which pointers are roots, roots in frame slots, `musttail` calls, walking the stack), is under Lowering, "GC metadata and safepoints". Two alternatives were considered and not chosen:

- **Walking the stack with the DWARF unwinder**, as gollvm does: every call becomes an `invoke`, its stack map is attached to the landing pad as its type info, and the runtime finds it through the unwinder's tables. It gets past C frames for free, since they have unwind tables too, but each frame costs a table lookup and an unwinder step, and every call must be an `invoke`. Frame pointers cost a register in each frame but make the walk a chain of loads. If the record kept on calls into C proves fragile, this is the fallback.
- **Environments on the heap** would make their roots ordinary heap fields, but every choice point would allocate, which the frame keeps free, and a continuation never outlives the frame that made it.

#### Closures that say what they captured

Done 2026-11-08. The first thing the collector needs: `spawn` and `monitor` do not share cells with another process, and closures can be sent between the processes of a node. A monitor's function is copied only if the watcher's mailbox is another process's, so `request` costs no copy.

**Before it.** A closure's cell is the entry's code, the number of
captured values, and the values, a word each. The captures are, in order, the
type_infos and typeclass_infos, and then the values. Nothing says what a value's
type is, and entries cannot say: `<symbol>$closure<N>` is shared by every
closure of a procedure with N captures.

**As built:**

- **Word 1 of the cell becomes the address of a static layout,** in place of the
  count, which nothing reads. A layout is the count and a type for each capture.
  The cell stays its size, a closure is as cheap to make, and a call is the
  same. One layout for each distinct content, as `linkonce_odr` or private
  globals, made from an attribute on `merk.closure`.
- **A capture's type is a field's type word, as in descriptors,** with a type
  variable as the place of its type_info among the captures. That is what an
  existentially typed constructor's hidden fields do already (`kHiddenTag`: a
  place, and a word within a typeclass_info if the variable is constrained), so
  `Resolve` needs nothing new. With the closure's captures as the cell.
  - A lambda expression's variables are places in its captured type_infos, or
    in its captured dictionaries.
  - A closure of a named procedure uses the callee's declared types, whose
    variables are its hidden arguments in order. A specialised callee has
    concrete types and no hidden arguments.
- **Marks in a layout, for the captures that are not values:** a type_info, a
  typeclass_info, and "unknown", which cannot be copied. The hidden fields of existentially typed constructors
  are the same two, so that a value of such a type can be sent, which is refused
  now.
  - A type_info: tagged 0, static, kept as it is; tagged 1, a cell of a
    descriptor and arguments, copied.
  - A typeclass_info: its shape depends on the instance. The method table gets
    a prefix, before its slots, with the counts of superclasses, class types,
    other type_infos and constraints. Tables already exist per instance, so
    this costs nothing at run time.
- **The message copy learns `Closure`,** by reading the layout: a new cell with
  the same code and layout, and each capture copied by its type, type_infos and
  dictionaries first as the values' types use the original cell's. The collector
  will trace the same way.
- **`spawn` copies** the closure and the handle's cell into a fragment that the
  child owns, and `monitor` the function, which removes both Temporary comments.
- **On a device the layout is null.** A device never copies a closure (design
  document, GPU processes: closures do not cross), and it has no descriptors for
  most types.
- **To another node stays "not supported yet".** The design has closures cross
  between nodes running the same binary, as procedure references. That is a
  separate piece, and not on the way to the collector.

**Rejected:**

- **A layout for each entry,** as prefix data before its code, which would leave
  the count alone. The entry is shared by closures of different sites. It would
  need an entry for each site, and prefix data on every target.
- **The layout in word 0, with the code in the layout.** Every closure call pays
  one more load.
- **A third word.** 8 bytes on every closure, when the count is unused.
- **Keeping shared closures, and having the collector not move them.** A
  closure's captures are in the parent's heap, which its collector moves.

**Decision:**

- **D1, what `spawn` does with a capture it cannot copy,** such as a foreign type
  with no descriptor. Decided 2026-11-08: it reports an error at the spawn, as a
  send does. Rejected: copying such a word as it is, which is convenient but
  assumes that a foreign value never points into a heap.

### Type descriptors

Static type descriptors describe each type's constructors, names, arguments and layout. Generic printing (`io.write`), `compare`, `unify` at polymorphic types, `univ`, the collector and the message copy all use them. A descriptor has its header, its kind, arity, module and name, the entries that unify and compare two words of the type, and a table of its functors, in the order declared. Each functor has its name, arity, tag, number of hidden fields, secondary tag, constant's word and fields' types, and `byTag`, `constants` and `shared` map a word to its functor. An existentially typed constructor's cell has its hidden fields, the type_infos and typeclass_infos of its existential variables, before its arguments, and the fields' types are its arguments'. A field's type is a `type_info`'s word, with a tag of 2 for a type parameter of the term's type, which is its number in the high bits, so `[|]`'s fields are `var 0` and a static cell of `list` and `var 0`, and a tag of 3 for an existential variable, whose type_info is in the cell: in the hidden field of the low 16 bits of the number, or, if its high bits are not 0, in that word of the typeclass_info there. So `box`'s field is hidden field 0, word 1, and a walker, which keeps the cell with the environment of its fields, finds the type there (`runtime/descriptor.h`). A walker does not allocate, so it does not make a `type_info` for each field: it keeps each type's arguments on its own stack, with the type they were made in. The `flags` byte, for the collector's `PointerFree`, is still 0, until something reads it.

A module defines the descriptors of the types it exports, whether or not it uses them, for the modules that import them. Primitive types', arrays' and tuples' are `linkonce_odr` where they are used, since their entries are `tailcc` procedures, which the runtime's C++ cannot define.

Comparison is the standard ordering (reference manual, "The standard ordering"). `compare.T`, like `unify.T`, is made for each instance a module compares, taking a `type_info` for each type variable left in it. Its constructors compare by their position in the type's declaration, which a `merk.switch` gives whatever their layout, and then their arguments from the first, with the last one's comparison a tail call, so that comparing long lists does not grow the stack. Integers compare with their sign, or without it for `uint`s and characters, strings by C's `strcmp`, and arrays by size and then elements, as mmc's do. `builtin.compare`, `ordering`, `array_compare` and the operators `@<`, `@=<`, `@>` and `@>=` are built in on it. A float that a type variable stands for raises on a NaN, even in a monomorphic instance, as `g(float)`'s `compare.T`, where one declared as a float gives `(=)` (see "Comparing a NaN" in "Where mmc and the manual disagree").

`io.write`, `io.print` and their `_line` forms are the first walker, `merk_write` in `runtime/write.cpp`, which `io.m` declares as polymorphic `foreign_proc`s. It writes terms as mmc's programs do, with its operator table being the parser's `operators.def`. An operand's context allows it a priority, of 999 for an argument, and any for the whole term, a list's tail and the argument of a tuple of one, written `{ X }`; an operator atom in any other context is written in parentheses without quotes, as `f((+))`. A list's elements are written in a loop while its tail has the list's own type, so its length uses no stack, though the depth of other terms does, as with mmc: a term nested 100,000 times overflows an 8 MB stack with both. The walker is the host's, so calling these on a device is an error, as for any procedure not compiled for the device.

### Exceptions, links and monitors

- `throw` and `catch` work within a process.
- An uncaught exception ends the process, with the exception as its exit reason. So does a stack overflow. It is reported on the standard error, and the program goes on. Nothing is unwound, as a process's stack and heap go as a whole. In `main`, on a thread that runs no process, or in a target or parallel region, it ends the program with status 1, as libomp's team and a device's region cannot be abandoned on their own.
- **Links are deferred** (decided 2026-11-08): two monitors, one each way, give the two-way message, and supervisors are built on monitors. They can be added if something needs them. The rest of this item is what they would be. A link is two-way: the death of one process delivers an exit message to the other. A monitor is one-way and delivers a `down_info` message. Both go to a mailbox the watcher chose, through a function into that mailbox's message type (Language extensions, "Processes and mailboxes").
- Monitoring a process that has already exited delivers a `down_info` message at once, so `request` cannot miss a server's death.
- **Monitors as implemented** (2026-11-08, `runtime/mailbox.cpp`, `library/process.m`). A monitor watches the process that owns the target mailbox, and a closed mailbox counts as an ended process, with the reason `noproc`. A process closes its mailboxes first and tells its monitors after, so that a watcher told of its end finds its mailboxes closed. The function is applied by the receiver, when it reads the message: the ending process queues what happened, and the first read replaces that with the function's result, on the receiver's heap. **Rejected: applying the function in the ending process,** which would run a watcher's code on a stack that may just have overflowed, and read a closure in another process's heap; **and a helper process for each monitor,** which costs a spawn for each `request`. A monitor is in a table by id, in shards with a lock each, and on its target's list, so that `demonitor` is safe against the target ending. As with Erlang's `demonitor` without `flush`, a message already queued stays. The reason's text is `normal`, `noproc`, or the report of the failure, until exceptions can be caught and it can carry the exception. Monitors across nodes or places, and a monitor's message to a mailbox a device receives from, report that they are not supported yet.
- Supervisors are library code built on monitors.

### Input and output

Each process threads its own `io` state. The standard streams are shared, with internal locking.

### Foreign code

Melbourne Mercury's `foreign_proc` code assumes heap pointers never move. With a moving collector, two kinds of foreign procedure are allowed:

- Those marked `will_not_call_mercury` that do not allocate get raw pointers; no safepoint can occur inside them.
- All others receive handles, which the collector updates.

Foreign code written for Melbourne Mercury therefore needs review before it runs on this runtime.

### OpenMP cooperation

Decided 2026-11-04, for chunk 2; none of it is implemented, and today's parallel loops simply use libomp's pool. Processes run on the scheduler's workers and loop teams on libomp's threads, and the only bridge between them is the process that enters a region. Two facts keep this simple: a parallel body is `det`, pure and threads no `io` (Language extensions, "Parallel loop annotations"), so a team thread never sends, receives, spawns or blocks in the scheduler and need not be a process; and the sequential meaning is the specification, so a team of one is always correct.

**Entering a region:**

1. *The process is pinned* to its OS thread until the join, since libomp ties a team's primary thread to that thread. It is not preempted, and a request to end it waits for the join. Between regions it may move: each OS thread that forks is its own libomp root.
2. *The worker hands off its scheduling role and run queue*, as Go does on entering a system call, and a spare thread takes them, so processes queued there are not held up by the region. Blocking foreign calls can share this mechanism.
3. *The team's size is claimed atomically*: 1, plus as many idle workers as a shared count allows, capped by a `num_threads` clause, and passed as `num_threads`. The claim is released at the join. The claimed workers stay parked while libomp's threads use their cores. Two regions at once get smaller teams, and one that claims nothing runs alone on its thread.

A process that wakes while a team holds the cores unparks its worker, and the operating system shares the cores for a while. Latency stays bounded without preempting a team thread. libomp's waiting at barriers suits that badly, so the blocktime is to be short or 0, which makes starting a loop slower.

The region runs on the process's own stack, which never moves (Processes and scheduler). A team thread has a stack of libomp's, which must be large enough for what the body calls, set with `kmp_set_stacksize_s` or `OMP_STACKSIZE`, and has its own stack limit for the overflow check.

**Heap.** Each version of a parallel body's allocation is a step from the one before:

- Version 1: bodies do not allocate, so no safepoint occurs inside a region.
- Version 2: each team thread allocates from chunks of its own in the process's heap, there is no collection during the region, and the chunks join the heap at the join, followed by a collection if one is due. A team thread starts with a null `merk_heap`, so its first allocation takes the slow path, which attaches it to the region's process. Array stores record their old-to-young pointers in per-thread buffers, merged at the join. This needs no roots from inside the region, where heap values in the structs of shared variables and in reductions' private copies are hidden from stack maps. A long loop that allocates a lot can use a lot of memory.
- Version 3, only if version 2's memory use proves a real problem: the process's collector stops its region at a barrier, which needs safepoints and root maps through outlined code.

**Target regions.** A `target` region can block for seconds, so it does not hold a worker: it runs on a thread of the blocking pool with the process suspended. Mapped arrays cannot move meanwhile, since only the process itself collects its heap.

As implemented (2026-11-08, `runtime/process.cpp`): a process can change OS threads at any call, so at `merk_region_enter` it moves, on its own stack, to a thread of the pool, and the worker goes on with its queue; at `merk_region_exit` it goes back on a run queue. The pool's threads are created as needed and reused, so each is a long-lived libomp root. A program that has never spawned does not move, so a program of only OpenMP pays for none of this. The move costs about 7 µs a region (Measurements). A process in a parallel region does not move either, as libomp ties that to its thread.

A parallel region on the host moves too, once the program has spawned, which is the hand-off above: rather than a spare thread taking the worker's queue, the process goes to the spare thread and the worker keeps its queue. Its team's size is what `merk_parallel_enter` returns, passed as `num_threads`: 1 and the cores that no worker is busy on and no other region has claimed, at most `omp_get_max_threads`, claimed atomically and released at the join. A claimed core is not kept from the workers: as above, a process that wakes shares it for a while. With `MERK_WORKERS=1` a team is one thread. A program that has not spawned gets libomp's team, and a parallel region within a target region or on a device neither moves nor claims.

Blocking calls move the same way, once the program has spawned: `listen`, `connect` and `lookup`, and reading the standard input if a refill of its buffer would wait, so that input already there costs nothing. A send to another node does not move, as most do not wait and the move would add about 7 µs to each, so when the socket is full its write holds its worker for up to a tick time.

The program ends when `main` does, which can be while another process is in a region. `exit` would then tear down libomptarget under it. So once the program is ending, a region neither starts nor cleans up, and if one is running, standard output is flushed and the library's destructors do not run. A failure that ends the program goes the same way; the runtime's own fatal errors (`Fail`) do not yet.

**Rules that follow.** `threadprivate` cannot be offered as it stands, since a process can change OS threads between regions. Clauses are per region, rather than set with `omp_set_num_threads`, whose state belongs to the thread. A program with processes and no directives pays for none of this.

**Measured** on the in-tree libomp, in C with `pthread`s standing in for workers, on 48 cores, 2026-11-04. Each concurrent root gets its full team with correct reductions, so libomp needs nothing more for two processes in regions at once. But it does not share the cores: two roots asking for 48 got 48 each, 94 workers in all. `OMP_THREAD_LIMIT` does not cap the total, two roots with a limit of 4 each getting 4, so it cannot be our cap. A root's team threads are kept while the root lives, and are not lent to another root: 4 roots of 8 left 28 idle threads, and a further root of 8 started 7 more. They stay in the pool after a root exits. So the idle threads can reach workers × (team size - 1), 2,256 on 48 cores if every worker thread once ran a full team, which costs memory rather than CPU. Long-lived workers and spare threads keep the roots few and reused. Still to measure, on Merkurius programs: the blocktime's cost to starting a loop, the idle threads' memory, and how far oversubscription hurts.

Rejected:

- **One pool, with a region borrowing the scheduler's workers**, as first planned here. It would avoid oversubscription, but libomp creates its workers itself, with `pthread_create` in `__kmp_create_worker`, and offers no way to supply them, so we would carry a patch to libomp or write our own `__kmpc_*` runtime. `target` regions depend on libomp too. To be revisited only if oversubscription is measured to hurt.
- **Sizing a team from the idle count alone**, without a claim. Two processes can read the same count and take the same workers.
- **Running a region on the worker's system stack**, as Go does for foreign calls, proposed while process stacks moved. Stacks no longer move, so it has no use.

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

- **Handles.** A mailbox handle carries a node identifier and a place within the node, its host or one of its GPUs (GPU processes, below), besides the process, mailbox and generation numbers. Within a node, handles cannot be forged. Across nodes they must be unguessable, or anyone who can reach a node could send to it, so remote handles include a random secret. This is stronger than Erlang, where any connected node that knows the shared cookie can do anything.
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

### GPU processes

A process may run on a GPU and exchange messages with processes on the CPU
over PCIe, starting with a restricted version (decided 2026-11-05). It
builds on OpenMP offload, and comes before processes on the CPU are
complete: its first form is a target region that sends and receives, whose
partner is on another node (decided 2026-11-07; The route there, below).
This is the direction; how it works in detail is still to be designed.
Decided 2026-11-10: a GPU process is a kernel that the runtime launches through
liboffload, not a target region, while OpenMP stays the model for parallelism
(docs/INPROGRESS.md, GPU processes without `omp target`).

**A GPU is a place of its host's node.** The design already has remote
mailboxes and remote spawn ("Remote mailboxes, sendability and failure" and
"Remote spawn" above), and a GPU fits them as a place within a node: the
node's host is one place and each of its GPUs another. A GPU shares its
host's connections, handshake and security, and only the node's place table
names it (decided 2026-11-06). Rejected: a GPU as a node of its own, which
would give it connections and a handshake of its own though it reaches the
rest of the world only through its host.

- **Handles** carry the node and the place, so a mailbox on the GPU is named
  by a value like any other, as the invariant "a mailbox handle is a value"
  allows (Handles, routing and places, below).
- **Closures do not cross.** The device image is another binary, so closures
  are not sent, and a GPU process is spawned by its kind's name from its
  `:- process` declaration, as on a node running another program. `spawn`'s
  closure form is not used for the GPU. Later, since the device image is
  compiled from the same source in the same compile, the procedures compiled
  for both could share a table of procedure references, so that closures over
  them cross between the places of one node, though not to a node running
  another binary; the handle and wire formats should not rule this out.
- **Messages are copied**, as `send` on the host deep-copies into a
  fragment by the message's `type_info` (Memory management, decided
  2026-11-04). One-word messages and arrays of one-word elements go as a
  word or a block. Any other message is copied by its `type_info` with the
  host's own walk (`runtime/copy.h`), which the GPU's runtime runs too: the
  GPU copies a message it sends into host memory, from which the host
  queues it, and the host copies a message for the GPU into host memory, from
  which the GPU copies it into its arena (Spawning on a GPU as implemented,
  "Messages"). Decided by the user 2026-11-10, for the functionality: "If
  this is an inefficient way of doing things we will have to revisit and
  come up with something more optimal". **Rejected for now: a copy that
  code generation specializes for each message type**, as first planned,
  which has no dispatch on the GPU but repeats the terms' layout, which
  Lowering decides, in generated code, and would have covered flat types
  first; it may come back for the types where the walk is too slow. The
  sendability rule applies, and the GPU adds to what cannot cross: foreign
  handles, `io`, `store` and mutable references are already excluded, and
  closures and existentially typed values are too, as they name procedures
  and descriptors of the other image.
- **The protocol is the message type.** `mailbox(M)` says which messages may
  be sent, and a message's constructor stands for the kind of communication.
  The order of messages is not checked, as between CPU processes; session
  types could be considered later.

**The transport is LLVM libc's GPU RPC** (`libc/shared/rpc.h`): a `Client` on
the device and a `Server` on the host exchange fixed-size buffers through
ports in pinned host memory, with an opcode header, and the offload plugins
already run its server (`offload/plugins-nextgen/common/src/RPC.cpp`). It
builds for amdgpu, so it covers gfx1030. A host thread bridges its server to
the CPU scheduler's mailboxes. Rejected: a transport of our own, which would
redo libc's handling of lanes, ports and memory ordering; and one heap shared
through unified memory, which needs fine-grained coherence that not every
platform has (PCIe atomics), and breaks the invariant that a message lives in
a fragment of its own.

**The restricted first version:**

1. `send` and `receive` on the device for flat message types only (no
   pointers), over libc's RPC, with the host bridge.
2. A target region that threads `io`, which sends and receives in its
   sequential part (Send and receive in a target region, below). Decided
   2026-11-07: this form comes first, and a GPU process spawned by its
   kind's name, as a persistent kernel, from the CPU only, comes later, as
   it needs `:- process` declarations through semantic analysis.
3. Deep-copied messages, the copy specialized at the message type, with
   sendability checked at compile time.

Not in it: spawning from the GPU, monitors and links, selectors over several
mailboxes, timeouts other than polling, and garbage collection on the device.

**Questions for the design:**

- **Waiting.** A GPU process cannot block; it polls its mailbox, sleeping
  between polls (`s_sleep` on AMD), and occupies a compute unit while it lives.
  The host's server thread must always run, or a host waiting for the kernel
  and a kernel waiting for the host deadlock.
- **What a process is on a GPU:** a wavefront or a work-group, not a lane,
  with one lane using the mailbox for all of them or one port slot per lane as
  libc's RPC does. Between messages it may run data-parallel loops. The
  first version: a kernel of one team, whose initial thread communicates
  (The device runtime, below).
- **Its heap:** chunks of device memory reserved at launch, bump-allocated.
  Without a collector on the device, a process is short-lived or resets its
  heap between messages (The device runtime, below). Decided 2026-11-10:
  a GPU process makes no cells but arrays, so it needs neither (Messages
  and memory on the device, below).
- **Memory coherence:** fine-grained host memory with system-scope atomics
  where the platform supports PCIe atomics, and coarse-grained memory with
  explicit fences and polling otherwise.
- **Latency:** a round trip over PCIe takes microseconds, so messages should
  carry work items and results, not fine-grained traffic.

**Later: GPU to GPU.** Two GPU processes on different devices exchange messages
the same way, as two places. The first version relays them through the host;
direct transfers over peer-to-peer PCIe, or AMD's Infinity Fabric links, need
mailboxes in device memory that the other device maps, and are designed when
that is added. Routing (below) chooses them where they exist, so programs do
not change when they are added.
Collectives across GPUs are an open proposal (`docs/INPROGRESS.md`,
Collectives and direct GPU transfers).

**Spawning on a GPU as implemented** (2026-11-10, steps 4c and 4d of GPU processes without `omp target`, docs/INPROGRESS.md):

- **The library.** `process` has `:- type place ---> host ; gpu(int)` and `gpus(Count, !IO)`, how many GPUs the node has for processes (decision D2). A process declaration declares `spawn_Kind_at(Place, Args, Mailbox, !IO)` beside `spawn_Kind` (Process declarations as implemented). At `host` it is `spawn_Kind`. At `gpu(N)`, a kind declared `on gpu` calls `process.spawn_kernel` with the module's and the kind's names and its arguments' words, which `process.gpu_argument`'s instances give: one for an `int`, a `float` (its bits) or a `character`, five for a mailbox (its handle). An argument of another type is a type class error at compile time. A kind that runs on the host is a software error at `gpu(N)`, as is a GPU that does not exist, or a module without an image for that GPU's architecture.
- **The GPUs** (`runtime/devices.cpp`). The first use calls liboffload's `olInit` and takes its devices other than the host, each with a context and its name and UID. A GPU's number is OpenMP's for it, found by UID with `omp_get_device_from_uid`, so that a region and a process name a device alike; one that libomptarget does not number, or every GPU without libomptarget, is numbered after OpenMP's devices in liboffload's order. liboffload itself has no numbers, only handles in `olIterateDevices`' order, and libomptarget numbers devices in the order that compatible images are first registered. The runtime's RPC callback is registered with each platform. Memory and copies use libomptarget for OpenMP's devices and liboffload for the others. liboffload's functions are declared weakly in `runtime/liboffload.h`, from its `.td` files, since its generated header is built after merkrt; a program with GPU kinds links `-lLLVMOffload`.
- **A spawn** (`runtime/gpu-processes.cpp`) loads the module's image on the GPU the first time (`olCreateProgram`), and sets its `merk_device_number`. It makes the process's mailbox on the host, owned by no host process, and launches `merk_process_<kind>`, one team of one thread, on a queue of its own, with the words and then the mailbox's handle, each an 8-byte argument, as AMDGPU's plugin copies them by the kernel's metadata. The GPU receives from the mailbox through its ring, as a region does. A host function queued after the kernel closes the mailbox when it ends; the queue is destroyed at the next spawn. The handle's place is still the host's, 0, until handles are routed by place. Messages between GPU processes, regions and host processes all go through the host's mailboxes, so a device's number only chooses memory.
- **Memory.** Each GPU process has an arena of its own: the host allocates its words on the device and passes them as the kernel's first argument, which the device's runtime reads from the kernel's dispatch packet in a process's image (`merk_device_process_image`). A region's cells and its table of rings go when it ends (The arena, below). When a GPU process's kernel ends, a host function on liboffload's thread closes its mailbox and takes the mailbox's ring and the copy of a message there (`merk_mailbox_close_ended`); liboffload's thread must not call liboffload, so a thread of the runtime's, the reaper, frees them, the arena and its chunks and the queue. The process counts as a user of its mailbox's ring until then (The mailbox's ring, below). The program ends at once when main does, even while GPU processes run. With `MERK_CHECK_DEVICE_MEMORY` set, the program's normal end reports how many chunks of the devices' arenas and how many rings are still allocated, for tests. Rejected: freeing the shared arena when the last GPU process ends, as a server whose GPU processes overlap would never free it.
- **Messages** (step 4e). A message that is neither one word nor an array of one-word elements, of a ground type with no higher-order argument, is copied by its `type_info` (`runtime/copy.h`, the host's copy, templated on where the cells go), between GPU processes or target regions and the host's mailboxes. The device's compile passes the message type's `type_info` to `merk_device_send_term` and `merk_device_receive_term`, and keeps its descriptors' functors, by which the copy walks, though not their unify and compare entries. A send copies the message into chunks of host memory that the host gives (`rpcPinnedChunk`), and the host queues it from there (`rpcSendTerm`) and frees them. A mailbox that a device receives such messages from has a ring of terms, whose slots only say that a message is there: the device asks the host to copy the oldest into host memory (`rpcTakeTerm`), copies that into its arena, and moves the ring's head, after which the host frees its copy (`merk_mailbox_copy_for_ring`). A mailbox that closes frees the copy if the head has passed its message, as it has when a GPU process ends after its last receive; a copy the device may still be reading goes with the ring, when the last region or GPU process that attached it ends. A message that a device has taken from its ring is freed when the host next trims the ring, and those still in it when its mailbox closes are freed then: a device reads a word in the ring, and the host copies any other message, an array as a term, with the mailbox's lock held, so nothing reads a message after its lock is released. With `MERK_CHECK_PINNED` set, the program's normal end waits up to ten seconds for running regions and GPU processes and reports how many chunks of such copies are still allocated, and with `MERK_CHECK_MESSAGES` how many queued messages neither a process read nor a mailbox freed, leaving out those still in open mailboxes, whose processes may still run, for tests. A copy for the other image refuses a closure and an existentially typed constructor's `type_info`s and `typeclass_info`s, which name the other image's procedures and descriptors ("cannot copy a closure between the host and a GPU"); since step 5a a message type that is not flat is reported at compile time, which leaves the refusal as the runtime's own check. A team mailbox still holds only one-word messages and arrays.
- **Tests.** `CodeGen/process-gpu-messages.m` and `-gpu.test`: a counter that takes `add(N)` and `get(Reply)`, a ping-pong of a hundred rounds between two GPU processes through the host, a list and a string to a GPU process and back, and a closure that cannot cross; `CodeGen/omp-target-term-messages.m` and `-gpu.test`: a target region that receives and sends a point, a string, a list and an array of strings. `CodeGen/process-spawn-at.m` spawns at the host and checks the software errors; `CodeGen/process-spawn-at-gpu.test` spawns on the GPU with an `int`, a `float`, a `character` and a mailbox, ends a program while its GPU process waits, and runs a program compiled for another GPU; at `-O0` the image does not link (LLVM bugs to report).

#### Handles, routing and places

A mailbox handle says where the mailbox's queue is, so that `send` can choose
how to reach it, and a message can travel from a GPU to its host, to another
node and on to a GPU there, without the program naming the path. Systems that
let GPUs communicate address a flat rank instead: NVSHMEM and rocSHMEM put and
get on a symmetric heap from device code, GPU-aware MPI is driven from the
host, and NCCL's device API is newer and addresses ranks too (What the
hardware offers, below). Realm, Legion's runtime, is the closest: its ids
for processors and memories encode the node and the kind of place, but it
runs tasks rather than passing messages.

**The handle** is small and fixed in size: the node id, the place, and the
process, mailbox and generation numbers and the secret (Remote mailboxes,
sendability and failure). What else is known about a node or a place is in
tables that nodes exchange when they connect, as they exchange their type
tables (Types instead of atoms).

- **Node id.** A random 128-bit id chosen at start-up, with an incarnation
  number, as Erlang's `creation`. Discovery maps it to the node's transports,
  such as TCP, a Unix socket or later RDMA, so a node that reconnects or
  changes transport keeps its handles valid. Rejected: an IP address and a
  process number, since an address is not an identity: machines have several
  interfaces, NAT and containers rewrite addresses, and process numbers are
  reused.
- **Place.** A small number into the node's place table, 0 for the host.
  Each entry gives the place's kind, host or GPU, its architecture, such as
  `gfx1030`, its OpenMP device number, and a stable physical identity, the
  device's UUID or PCI address. Rejected: the OpenMP device number alone,
  which depends on the process and on `ROCR_VISIBLE_DEVICES` or
  `CUDA_VISIBLE_DEVICES`, so another node cannot interpret it. The physical
  identity lets two nodes on one machine see that they share a GPU, or that
  two GPUs are peers, which is what makes a direct path possible. OpenMP 6.0's
  `omp_get_uid_from_device` and `omp_get_device_from_uid` map a device number
  to a unique id string and back, and libomptarget implements them
  (`offload/libomptarget/OpenMP/API.cpp`; on the device they return nothing).
  Its plugins form the id from the vendor's UUID, `cuDeviceGetUuid` or HSA's
  `HSA_AMD_AGENT_INFO_UUID`, prefixed with the plugin's name. An AMD agent
  without a UUID reports `GPU-XX`, and the plugin then falls back to the device
  number, which is not stable; the Radeon RX 6950 XT (gfx1030) of our test
  machine is one. For such a device the place table uses its PCI address from
  HSA instead. The host's id is not unique, so the host place is named by the
  node id alone.

**Routing.** `send` compares the handle's node and place with its own:

- **The same place:** the message is queued locally, as now.
- **Another place of the same node:** from a device to the host over libc's
  RPC; from the host to a device by writing into the mailbox's ring in
  host-pinned memory, which the device reads directly (memory of the device
  itself would need the plugin to map it for the host); from a device to another device peer to peer where
  `hipDeviceCanAccessPeer` allows it, and through the host otherwise.
  `omp_target_memcpy` between two devices is a portable first version.
- **Another node:** to the next hop of the node's route table, encoded once at
  the source. Each hop forwards the bytes without decoding them (Opaque
  payloads at relays), and only the destination decodes them; a GPU there gets
  them from its host.

Each node computes its route table at the handshake from the place tables.
It is the only thing that changes when direct paths, such as GPUDirect RDMA,
are added, and programs never see it.

**What the hardware offers** (checked 2026-11-06). A GPU runs no network
stack, so it never speaks IP itself, and connections, addresses and the setup
of every transport stay with its host. The data, though, often bypasses the
host's CPU and memory, and on some systems the GPU starts the transfer
itself. From the most common to the most direct:

1. **The host sends from GPU memory.** With GPUDirect RDMA, or AMD's
   ROCmRDMA, the NIC reads and writes GPU memory directly over PCIe, but the
   CPU posts each operation. This is what GPU-aware MPI does, such as Cray
   MPICH on Slingshot. On Frontier each of a node's four MI250X modules has a
   Slingshot NIC attached to it. On Aurora the eight NICs of a node hang off
   PCIe switches at the CPUs, not at its six GPUs, yet data still passes
   between the GPUs and the NICs without going through CPU memory. On El
   Capitan each MI300A has a x16 link to its NIC. AMD documents ROCmRDMA for
   Instinct accelerators, with a large BAR and the IOMMU off or in
   pass-through; nothing found says that a consumer Radeon supports it.
2. **The GPU starts the operation.** The host creates the NIC's queues once,
   in GPU memory or mapped into it, and kernels then post work and ring the
   NIC's doorbell themselves, with no CPU proxy. NVSHMEM's IBGDA transport does
   this on InfiniBand, with Mellanox NICs; NVIDIA's guide notes that a GPU
   thread fills in a work request more slowly than a CPU thread, so each
   message takes longer, though throughput at scale is better. rocSHMEM's GDA
   backend does the same on AMD GPUs with Mellanox, Broadcom Thor and Pensando
   Pollara NICs, and falls back to a host proxy when it cannot; AMD provides
   it with limited support. NCCL has a device-side API for GPU-initiated
   networking since 2.28.7, still marked experimental, over GDAKI or a CPU
   proxy. HPE's stream-triggered operations on Slingshot 11 are a step short of
   this: the CPU queues operations on the NIC as triggered operations, in
   advance, and the GPU's stream triggers them, so the CPU is out of the
   control path when they run. A 2026 implementation for MPI's
   point-to-point operations over the CXI libfabric provider ran on up to
   8,192 of Frontier's GPUs. All of these address registered memory at a
   rank, not a mailbox.
3. **Loads and stores between GPUs.** Within a node, GPUs joined by NVLink or
   Infinity Fabric read and write each other's memory directly, as Frontier's
   and El Capitan's do. NVIDIA's GB200 NVL72 makes one NVLink domain of the 72
   GPUs of a rack.
4. **Memory shared by the CPU and the GPU.** The MI300A's CPU and GPU cores
   share the same HBM, with one virtual address space, though memory from
   `hipMalloc` is faster for the GPU than memory from `malloc`. On Grace
   Hopper (GH200) each has its own memory, and either reads the other's over
   NVLink-C2C, coherently, including memory from `malloc`. There, a message
   between the host and the device is a write to memory, and the round trip
   over PCIe and the coherence questions above largely go away.

**Places say what they can do.** Routing needs to know which of these a pair
of places can use, so each entry of the place table also gives:

- whether the place's memory can be registered with which of the node's NICs
  (level 1), and whether the GPU can post to their queues itself (level 2);
- which of the node's other GPUs it can reach by peer loads and stores, and
  over which link (level 3);
- whether it shares memory with the host, coherently or not (level 4).

The route table then has five kinds of hop: through the host, which works
everywhere; a send from GPU memory posted by the host; a send posted by the
GPU into a mailbox's queue in the remote GPU's registered memory; a peer
store; and a write to shared memory. In every case the host keeps the
control path: it makes the connections, registers the memory, creates the
NIC's queues and does the handshake. A GPU is a place of its host's node even
on the most direct hardware. The transports under the node id are libfabric,
as Slingshot's CXI provider, UCX or verbs endpoints, with TCP as the fallback.
Only the relay through the host can be tested on our machine, whose gfx1030
is a consumer GPU; the direct paths need Instinct or NVIDIA datacenter GPUs
with RDMA NICs.

**What choosing among links adds** (discussed 2026-11-08; not built). With two
GPUs joined by Infinity Fabric, or by NVLink and NVSwitch, the program still
names only a mailbox, and the route depends on the pair of places. Going
faster than through the host needs four more things than the table above:

- **Where the mailbox's queue lives.** Every queue is on the host today. To
  gain from a fast link, the queue must be in memory that the sending GPU can
  write directly, normally the receiving GPU's. So the queue's place is part of
  the choice, made when a device first uses the mailbox.
- **Several senders.** The host's queue is protected by a lock, and the ring it
  fills has one producer. GPUs writing straight into a peer's ring have no such
  lock, so that ring needs its slots reserved atomically.
- **The order of messages.** Messages from one sender to one mailbox stay in
  order, and mixing routes can break that: small messages through a ring and
  large ones through a copy can overtake each other. Either a sender and a
  mailbox keep one route, or messages carry sequence numbers.
- **The size of the message.** A small message suits a direct store into the
  ring. A large one suits a DMA copy, with only its address and size going
  through the ring. So the lookup is by source, destination and size class.

#### Send and receive in a target region

A target region whose goal threads `io` may send and receive. It is then, in
effect, a short-lived process on the device, beside the persistent GPU
processes above. It is the restricted first version's form, and has its
limits until they are lifted. Handles are flat values, so they are mapped into the region
`to` as any other value.

- **Before there is a scheduler,** the thread of the process that launched
  the region waits for it, and the host's bridge thread forwards the
  region's messages. The region's partner is then a process elsewhere: in
  the first slice, the main process of another node (The route there).
- **The launch does not block,** once there are processes. A region that waits for a message from a CPU
  process of its own node would deadlock if the thread that launched it
  waited for it and was the one that runs that process. A region that
  communicates is launched asynchronously, as with `nowait`, and the Mercury
  process that launched it is suspended until it completes; its thread goes
  back to the scheduler.
- **Only the sequential part sends and receives.** `io` threads through the
  sequential part of the region, its initial thread in a generic kernel, and
  there `send` and `receive` work as on the CPU. The iterations of a
  `distribute parallel do` have no `io`, so they cannot. This is wanted: a
  receive in an iteration depends on forward progress between teams, which
  GPUs do not guarantee beyond the teams that fit at once, so a team could
  wait for a message that only a team not yet running would send.
- **Loops produce messages as values.** An iteration's messages are its
  results, in an array or a reduction into an outbox, which the sequential
  part then sends. Messages are batched, as their latency asks (Latency,
  above), and nothing sends from each lane. Rejected: `send` from each lane,
  which needs `io` in the iterations and contends for the RPC's ports.
- **Mailboxes made in a region live as long as it does.** A message that
  arrives after the region has ended goes to a mailbox nobody reads, as for
  a closed mailbox; the generation number tells them apart.
- **A kernel that faults** sends a `down` message to whatever monitors the
  region's process, with a reason saying that its place failed.

- **As implemented** (2026-11-07, docs/INPROGRESS.md, The device's side
  of the first slice): a target region sends one-word messages, integers,
  floats, characters and constants of a type of constants only, with
  `process.send` in the region or in a procedure it calls: in the device's
  compile every procedure runs on the device. A handle crosses as a cell
  of five words (Lowering, "OpenMP target regions"). The send is a call of
  `merk_device_send` with the handle's cell, the message type's key and
  the message as a word. The key is a 64-bit hash (xxh3) of the type's
  name, computed by code generation, since the host encodes the message by
  the host's `type_info` if the mailbox is another node's, and a procedure
  that a region calls has no host value to take it from: the host's
  compile of a module with target regions registers each key of its
  sends' ground one-word message types with the type's `type_info`, in a
  procedure that runs at start-up (`merk.constructor`, a C constructor in
  `.init_array`), and the callback looks the key up; two types with the
  same key stop the program at registration. Rejected: the device sending
  its own `type_info`'s address, which the host maps back through
  libomptarget's table of declare target globals, which has no reverse
  lookup, so the host needs a registry anyway; and giving procedures that
  regions call the `type_info` as an argument, which changes their
  signatures in both compiles and every caller's. Outside a region, a send
  that the device does not have, such as a polymorphic one, stays the call
  of `process.send`, an error only if a region calls its procedure. On a
  GPU, `merk_device_send` is the device's
  runtime's (`runtime/device/device.cpp`), which sends the opcode
  `'m' << 24 | 1` (`runtime/rpc-opcodes.h`) over the offload plugin's RPC
  server; `merkrt` registers a callback for our opcodes with
  libomptarget's `__tgt_register_rpc_callback` at start-up, weakly, so
  that programs without offloading link without it
  (`runtime/offload.cpp`). The callback queues the message with
  `merk_mailbox_send`, which sends it on if the mailbox is another node's,
  and answers, so that the device's send returns once the message is
  queued: without the answer, a region ended before the host had queued
  its messages, and a later send could take another port and be served
  first. On the host, `merk_device_send` is `merkrt`'s, which queues it
  directly. Arrays of one-word elements are sent too (The device's
  communication, below); a message of another type is not supported yet.
  `CodeGen/omp-target-send.m`, `omp-target-send-call.m`, each with its GPU
  run, `omp-target-send-errors.m` and `omp-target-send-polymorphic.m`.

  The first slice (`CodeGen/omp-target-two-nodes.m`, with its GPU run):
  node 1's region exchanges integers with node 2's `main` over TCP, node 2
  having been given node 1's mailbox by node 1's host. A send to another
  node's mailbox is encoded and written to the connection by the callback,
  on the plugin's server thread, which TCP's flow control can then stall
  for every device of the process. The benchmarks did not show this as a
  limit (The device's communication, below); ping-pong has one message in
  flight, and large arrays are limited by the encoding and TCP.

  A region receives with `receive(Mailbox, any, Timeout, Result, !IO)`,
  one-word messages, in the region or in a procedure it calls: in the
  device's compile every procedure runs on the device, and a receive needs
  nothing of the host's, as a send needs its `type_info`; outside a
  region, a receive the device does not have stays the call of
  `process.receive`, an error only if a region calls its procedure, which
  may run only on the host. It is a call of
  `merk_device_receive` with the handle's cell and the timeout's
  milliseconds, a `merk.switch` on the `process.timeout`, which succeeds
  with the oldest message as a word; the result is `yes` of it, or `no`.
  `any`'s closure is not made. On a GPU, the device's runtime reads the
  mailbox's ring in host-pinned memory (The device's communication, below);
  between looks the device sleeps (`s_sleep`, about 8 microseconds), and it
  stops when the timeout passes by `omp_get_wtime`, the OpenMP device
  runtime's steady clock. On the host, `merk_device_receive` waits on the
  mailbox as `receive/5` does. A cell made on the device, such as `message(N)`,
  comes from `merk_device_alloc`, the device's arena. Another selector, or a
  message of another type, is an error at the receive; a selector over
  several mailboxes is an error at its closures in the device's compile.
  `CodeGen/omp-target-receive.m`, with its GPU run,
  `omp-target-receive-errors.m`, `omp-target-receive-selector.m`, and
  `omp-target-term.m`, with its GPU run.

#### The device's communication, as built

What runs today (2026-11-08) for a region's messages, within a node and
between nodes, and why. The numbers are in Measurements.

**The path of a message.** Between two nodes' regions, a message goes from
the device to its host over RPC, is encoded there, crosses TCP, is decoded by
the other node's reader thread, and is shown to the other device in a ring in
host memory. Nothing was added for GPU to GPU beyond the two halves: a
region's send to another node's mailbox and a region's receive of what arrives
are each the first slice's. Both hosts stay on the path; a copy from the device
straight to a network card is not built.

**What LLVM's offload runtime gives.** An RPC server for every device, on a
host thread while a kernel runs, speaking libc's protocol
(`libc/shared/rpc.h`); it tries registered callbacks first, then its own
opcodes, then libc's. A callback is registered with
`__tgt_register_rpc_callback`, and on the device
`rpc::Client __llvm_rpc_client`, a weak symbol of `libompdevice`, opens a port
for any opcode. `omp_target_memcpy` works from a callback, after a fix below.
One server thread serves every device under a lock, so **a callback must never
wait for a message**: it would stop every device's RPC, `printf` included.

**`parallel do` in a target region,** directly or in a procedure the region calls,
as Flang has it: a generic kernel with an `omp.parallel` in it. On a device the
thread that starts the loop has a stack the rest of the team cannot read, so the
loop's reduction variables and the scalars it uses from outside are in
`__kmpc_alloc_shared` memory, as Clang's globalization has for a shared variable
(`SharedSlots`, `ShareCapturedScalars`); pointers pass as they are. The host's slot
for a reduction is inside the region, or it would be an argument of the kernel that
the device's compile does not have. The driver turns off OpenMPOpt's SPMD conversion
at the device's link (`-openmp-opt-disable-spmdization`): its guarded form of such a
kernel faulted at address 0 when a value read from memory before the region reached
the region through a call. Not reduced further or reported; a kernel we mark SPMD,
such as `target teams distribute parallel do`, is not affected, though a generic
kernel does not get the speed of the conversion. `teams` cannot be used this way:
OpenMP does not allow statements beside it in a `target`.

**Output from a region.** On a device, `io.write_string`, `io.write_int` and `io.nl`
are io's procedures, copied into the region's compile with their foreign code
(Library code on the device, below). Their C is the host's,
`fputs(S, stdout)` and so on, which on a GPU is LLVM's libc. Its `stdio` sends each
call to the host over libc's own RPC opcodes. The offload plugin's server writes the
text into the host's `stdout` and answers once it has. So a region's output is in
order with what the host prints, through the same buffer, and is flushed as the
host's is: at the end of a line on a terminal, in blocks into a pipe or file, and
when the program exits, also after a device's error. A region on the host calls the
same C.

Each write is a round trip on the plugin's one server thread, so it is for messages,
not for a loop that is being timed. Several teams' writes may interleave within a
line, as several threads' do on the host.

Rejected:
- *Output opcodes of our own* (`rpcWriteText`, `rpcWriteInt`, until 2026-11-10),
  whose callback flushed at the end of each line. The flush was not needed for order,
  since both go into the host's one `stdout` buffer. It cost a system call per line
  into a pipe. libc's `stdio` is developed and improved upstream, and ours would not
  be.
- *Flang's I/O over RPC* (`flang-rt/lib/runtime/io-api-gpu.cpp`), which buffers each
  statement on the server, which Mercury's separate write calls do not map to.

**The transport is opcodes of our own,** `'m' << 24 | n`
(`runtime/rpc-opcodes.h`), in `runtime/device/device.cpp` and the callback in
`runtime/offload.cpp`:

| n | Name | What |
| ---: | --- | --- |
| 1 | `rpcSend` | a one-word message, by the type's key; answered once queued |
| 2 | `rpcAttach` | the mailbox's ring: its address and head |
| 3 | `rpcChunk` | a chunk for the device's arena |
| 4 | `rpcGiveArray` | an array the device made and copied into host memory, copied again for a region's output |
| 5 | | unused: was a send of an array that the host copied from the device |
| 6 | `rpcDrain` | move messages waiting on the host into the ring |
| 7 | `rpcTakeArray` | the oldest array message, copied into host memory that the device reads |

Rejected: libompdevice's host call (`__llvm_omp_host_call`), which needs a host
function's address passed into each region and allocates a copy of the
arguments for each call; and a server thread of our own, which would redo the
plugin's.

**Encoding happens on the host,** by the host's `type_info`, found by the
message type's key (As implemented, above). The device does no run-time
dispatch and never reads a `type_info`.

**A send waits for the host.** Without the answer, a region could end before
its messages were queued, and the next send could take another port, which
the server, scanning from port 0, may serve first, out of order. It costs an
RPC round trip per send, and it is what the GPU has left over the CPU between
nodes, with the wait on the ring. Removing it needs another way to keep sends
in order, such as a send ring in host memory drained by a host thread (not
designed).

**The arena.** The device bump-allocates cells and arrays with an atomic add,
in chunks of at least 1 MiB that the host allocates on the device when asked
(`rpcChunk`, `omp_target_alloc` in the callback). One thread asks at a time,
since a GPU has thousands, and the others sleep. Each region is bracketed on
the host by `merk_region_enter` and `merk_region_exit`, which count the
running regions and give each a slot, one of 256 (`kRegionSlots` in
`runtime/device-ring.h`); a region that starts while all are taken waits for
one. The region passes its slot to the device as its first input, and each of
its teams gives it to `merk_device_region_enter` before it makes a cell,
which keeps the slot's words in the team's local memory: in `omp.teams`, and
in a parallel region nested in it, where an SPMD kernel's operations are. A
region's image has, by slot, the current chunk, whether a wave is asking for
one, a target teams region's table of its teams' rings, and the table of the
mailboxes' rings the region has attached (The mailbox's ring, below). A
chunk request carries the slot, and `merk_region_exit` takes back the slot's
chunks on each device, so a region's cells go when it ends, though other
regions or GPU processes still run (2026-11-10;
`CodeGen/omp-target-region-arena-gpu.test` runs 40 regions beside a waiting
GPU process within a 32 MiB limit).

A short region pays for its arena only once (2026-11-11). The host keeps up
to 16 chunks of 1 MiB a device that regions have finished with, counted
towards the limit and freed if an allocation would pass it, and gives them to
the next to ask, without `omp_target_alloc` or `omp_target_free`. The slot's
words are cleared by the next region of the slot on the device, not by the
host: the host gives the slot with a generation, new each time, in the high 32
bits of the region's word, and the first lane of each wave compares the slot's
generation with it as the region starts; the first to find it older swaps in a
mark that it is clearing, clears the words, and stores the new generation,
while others that see the mark wait. It is the first lane of each wave, so
that no lane spins waiting for a lane of its own wave, which is masked off
beside it. `CodeGen/omp-target-region-slot-reuse-gpu.test` reuses one slot
600 times, with team tables and rings in it; without the clear it hangs. Each
region that allocated cost about 45 µs more than one that did not, the
allocation, the free and the host's copy to clear the slot; now it costs about
the chunk's RPC (Measurements, Target regions). Rejected: the host clearing
the slot as before, a synchronous copy to the device of about 12 µs per
region; clearing it at `merk_device_region_enter` without a generation, since
a team that starts late would clear what earlier teams of the region had
made; and the device keeping freed chunks itself, which needs a free list on
the device shared by regions and processes, for the same saving. Rejected: one arena for the image, freed
when the last running region ends, as it was, since regions that overlap, or
any region beside a GPU process, which lives as long as it likes, never let
it end; a table on the device keyed by a launch id, its entries in the
image's arena, since the entries then pile up while anything runs; an arena
that the host allocates on the device at each launch, which costs a device
allocation and copy per region and needs the device before the region has
chosen it; and keying by the launch's kernel-argument address, which the host
cannot match to the region's end. The chunks are limited to 8 GiB in all, or `MERK_DEVICE_ARENA_BYTES`, past which
the program stops with an error: a runaway allocation otherwise fills the VRAM
and can make the driver reset a GPU that also drives the display, as one did.
`Fail` there uses `_Exit`, since `exit` on the server thread hangs.

**Arrays of scalars,** `array(T)` for a one-word `T`. `runtime/array.cpp` is
also compiled for the device, allocating in the arena. The wire format is
8-byte little-endian items, the length and then the elements.

- **A send** copies the array into host memory (`rpcPinnedChunk`), and
  `rpcSendTerm` gives the host the copy, which the send copies into the
  message's fragment or encodes.
- **A receive** is the ring giving the size, `rpcTakeArray` copying the message
  into host memory that the device reads, and the device copying that into an
  array in its arena. The device decides the place, so the arena's accounting
  stays on one side.
- **Rejected: copying into or out of the device's memory while it runs**
  (`omp_target_memcpy`, `olMemcpy`), as both did until 2026-11-10. The
  arena is coarse-grained memory, whose lines the device's L2 cache keeps
  without regard to a DMA copy; a system-scope fence on gfx10 invalidates
  the L0 and L1 caches but not L2. A GPU process that took an array, the
  first thing in an arena whose memory another process had used, read old
  cells, and a GPU page fault followed (`process-gpu-arrays-gpu.test`).
  Regions met it less as they start as kernels, when the caches are
  invalidated. The copy was faster: a send from a GPU within a node reached
  about 1.2 GB/s. One lane copying word by word is slower, until messages
  travel as their bytes and a wave's lanes copy them together
  (docs/INPROGRESS.md, GPU processes without omp target, step 5).
- **Copied and encoded as blocks.** An array of scalars is copied into a
  message with one `memcpy`. An array of `int`, `uint` or `float` (64-bit
  ones) is encoded and decoded as one, its bytes on the wire being its bytes in
  memory; other scalars are not, as `char`s and enumerations are checked when
  decoded, and narrower integers masked. An element-at-a-time encode by
  `type_info` had capped sends between nodes at 300 to 700 MB/s.

**The mailbox's ring** (`runtime/device-ring.h`). The device looks for a message
by reading host-pinned memory (`llvm_omp_target_alloc_host`) directly, with no
RPC and no host thread involved.

- **One ring for each mailbox a device receives from,** 64 slots. The first
  receive attaches it (`rpcAttach`); the device keeps its address in a table
  in the arena, by the mailbox's slot, chaining entries that collide, as
  replacing one would call the host at every look.
- **The host's list stays the mailbox's queue;** the ring shows the device its
  oldest messages. A slot is a word message itself, or an array's size. The
  device moves `head` when it has taken a message, and the host drops it from
  its list.
- **What does not fit waits on the list,** in order, and a flag tells the
  device to ask (`rpcDrain`), after which it looks again, so that a receive
  with no time to wait still finds it.
- **A ring lives while something running has it attached:** the regions
  that attached it, each counted once by its slot, which the attach request
  carries, and the GPU process whose mailbox it is. As the last of them
  ends, the ring is detached and freed, and the mailbox is its list alone,
  holding what the device did not take, so a mailbox can pass between
  regions and the host. A region keeps its table of rings in its slot, so a
  later region attaches again, with one RPC, and never finds a ring that has
  gone. A mailbox that closes while a region still reads its ring hands the
  ring, and the copy of a message the device may still be taking, to an
  orphan, by the mailbox's slot and generation, which the last user frees
  (`merk_mailbox_region_ended`, `merk_mailbox_close_ended`). Nothing waits
  for a moment when nothing runs, which beside a GPU process may never come
  (2026-11-10; `CodeGen/omp-target-region-rings-gpu.test` runs 40 regions,
  each receiving from a mailbox of its own, beside a waiting GPU process).
  Rejected: freeing every ring when the last running region ends, as it
  was; keeping the regions' tables of rings in the image, whose entries
  would outlive the rings they point to; and a pool of rings for reuse,
  until a measurement asks for it, as a region that receives allocated a
  ring before too.
- **One device for each mailbox:** a second device attaching to a mailbox
  while its ring is in use stops the program with an error. A ring has one
  consumer.

Why host-pinned memory (chosen by the user, 2026-11-08). The first design
had a queue in device memory that the host writes. A probe measured a one-word
round trip between a host thread and a kernel through host-pinned memory at
2.0 µs, against 7.5 µs for an RPC round trip and about 10 µs for
`omp_target_memcpy`, so a queue the host could reach only by copying would
be slower than the poll it replaced. Rejected: device memory mapped for the
host (a large BAR), which needs the plugin to offer it, a change to take
upstream, and depends on the machine; its gain would be the device's looks not
crossing PCIe. **On an APU** there is one memory, so this choice is the better
one there too.

**Unified shared memory** (not built, not measured; no APU here) would go
further for arrays: the device could read a received array where the host has
it, with no copy into the arena, and a send could give a pointer in place of
the host's copies. The message's memory must then live as long as the device
holds it.

**The host spins before it blocks.** A round between nodes had three blocking
wakes, each node's reader thread and the receiver's wait, and a wake costs more
than the round's other steps (this machine's deepest idle state alone takes
18 µs to leave). A thread about to block for a message spins first, for 50 µs,
or `MERK_SPIN_MICROSECONDS`, 0 for none: on the socket with `MSG_DONTWAIT`,
and on the count of arrivals. **It yields as it spins:** without that, a round
with both nodes on one core took 237 µs in place of 39, as the spinner kept the
core from the thread it waited for; with it, 28. A receive with no time left
does not spin. The cost is up to 50 µs of a core for each wait.

**Choosing a device.** `device(N)` on `target`, on a combined target loop and
on `target data`, on the target leaf, as Flang's: the operand `device` of the
operation, evaluated on the host. N is a non-negative integer, or an integer
variable bound before the directive; a stock compiler reads the clause as a
comment, so such a variable should have another use, or it warns of a singleton.
Other terms and `ancestor` are reported as not yet implemented. A number the
machine does not have stops the program in libomptarget, with or without
`OMP_TARGET_OFFLOAD=MANDATORY`. Only device 0 has been run, on this machine's
one GPU.

**Using several devices at once.** A target region blocks the thread that
launches it, so `main` alone runs one at a time. Two ways are open with
`device(N)`: regions launched from the threads of `omp.parallel`, or from
Merkurius processes, once the scheduler exists (`target nowait` belongs with
it). The rest of a node with several GPUs is still to do: each GPU as a place
of its own, with a place table and a stable identity, and a direct path between
two GPUs, which goes through the host's mailboxes today. A mailbox's ring has
one device at a time (above).

**Fixes this needed in LLVM,** candidates for upstream:

- The AMD plugin queued a small copy behind a running kernel's own copy back
  on the same copy engine, so `omp_target_memcpy` from a callback waited for
  the kernel, which waited for it. A copy with a pending dependency is now
  submitted once the dependency completes.
- MLIR's `omp-stack-to-shared` freed shared memory between a `musttail` call
  and its return.
- MLIR's `omp-host-op-filter` kept the arguments of every block of the host
  function, which broke regions in branches.

**What the plugins lack, for later.** A stable identity for a GPU (the gfx1030
has no UUID, and liboffload no PCI address), which the place table needs once a
node has two GPUs or nodes compare places. RPC on the host plugin, so the
device's transport does not run on the CPU, and a region on the host calls the
host's runtime directly (Testing on the CPU, below).

**Tested.** `omp-target-receive-ring.m` (more messages than the ring holds, a
mailbox passing between regions and the host, colliding table entries),
`omp-target-two-nodes.m` (one region, or both nodes in regions, each on the CPU
over TCP too), and their GPU runs. Two processes on one machine stand for two
machines, over loopback, sharing its one GPU: two GPUs, two machines and a real
network are still to be tried.

#### Testing on the CPU

With the CPU as the offload target, everything above runs on one machine,
for tests and for debugging. libomptarget has a host plugin
(`offload/plugins-nextgen/host`) that runs target regions on the CPU as on a
device, which Clang and Flang use with
`-fopenmp-targets=x86_64-unknown-linux-gnu`:

- **Four devices** (`NUM_DEVICES`), and so four places like GPUs on one
  machine, enough to route between places.
- **Device memory of its own,** buffers from `malloc`, so maps copy: a map
  that is missing or wrong gives a wrong result, as on a GPU.
- **Kernels are host code,** so gdb, rr, ASan and TSan work in device code,
  where none of them work on a GPU. Mailbox queues, the host bridge and
  routing are debugged so.

What it takes:

- **The driver.** `addOffloadJobs` in `tools/merkc/driver.cpp` offloads to
  AMDGPU only; it is to accept the host's triple as an offload target too,
  and make a device image for it.
- **A device keeps a GPU's restrictions.** No allocation in a region, no
  run-time dispatch and only flat values crossing hold in the device compile
  for any device triple, not only `amdgcn`, so that a test that passes on the
  CPU does not fail on a GPU.
- **The same transport.** A CPU place sends over libc's RPC protocol between
  threads, not through the host's own mailboxes, so that tests run the code
  that the GPU runs. `rpc.h` is generic over the number of lanes, so a client
  of one lane on the CPU should work; that is to be checked.
- **Capabilities can be faked.** A CPU place can be declared in the place
  table with any of the capabilities above, peer access, NIC registration or
  memory shared with the host, carried out by plain copies, so that each kind
  of hop of the route table is tested on one machine, including those our
  gfx1030 lacks. Several nodes are several processes on the machine. A
  message from a GPU through two hosts to another GPU then runs on the CPU,
  under a debugger.

What it cannot show:

- **Forward progress.** CPU threads always progress, so a receive that would
  deadlock between teams on a GPU completes on the CPU. A test mode runs a
  region's teams one at a time, so that waiting on a team that is not running
  hangs instead of passing.
- **The memory model.** x86 orders memory more strongly than a GPU, so a
  missing fence in the queues goes unnoticed; TSan finds races but not an
  order too weak for a GPU. Tests on a GPU are still needed for that.
- **Asynchronous launch.** The host plugin runs a kernel in the thread that
  launches it (`launchImpl`), and a `nowait` region runs as a task on libomp's
  hidden helper threads. That tests that the launching Mercury process is
  suspended, though not with a GPU's timing.

#### The device runtime

Discussed 2026-11-07. Code on a device that uses only scalars and arrays of
scalars, with lifetimes known at compile time, needs almost none of the
runtime a CPU process has, so the device's own runtime is a few hundred
lines beside the OpenMP device runtime, which a target region needs anyway.

**What the device does without:**

- **A collector.** An array of scalars holds no pointers, so nothing is
  traced, and with known lifetimes nothing has to be found: no stack maps,
  safepoints or write barriers on the device.
- **Type information at run time.** `type_info`s and `typeclass_info`s serve
  polymorphic code and generic unification, comparison and writing. Device
  code is monomorphic, so code generation specializes all of it.
- **General allocation.** A device's `malloc` is slow and fragments memory.

**What it keeps:**

1. **The arrays a region uses,** which the host allocates and maps in with
   `map` clauses, as target regions do now.
2. **An arena for arrays made on the device,** one for each launch or each
   process: allocation bumps a pointer, and everything is freed at once when
   the kernel ends or the process resets it. This is region-based memory
   management in its simplest form; mmc had a region grade (`rbmm`, after
   Phan and Janssens), which is why `grade(rbmm)` is a trace goal's test. If
   the sizes are known when the kernel starts, the host sizes the arena
   exactly, and it cannot overflow.
3. **Uniqueness for the rest.** An `array_di`/`array_uo` update is in place,
   and the last use of a unique array is known at compile time, so the
   compiler can reuse or free its space (compile-time garbage collection, as
   Mazur's thesis on structure reuse for Mercury has it). A `send` of a unique
   array hands it over: the sender cannot use it again, so nothing waits for
   the send to finish before the array is reused, and nothing is copied
   defensively.
4. **Mailboxes:** a small ring in host-pinned memory that the host writes into,
   and `send` and `receive` over libc's RPC. Flat messages are copied with
   `memcpy`, so the host and the relays forward bytes without reading them.
   libc's RPC moves a small buffer for each lane on each round trip, so a
   large message is written by the host straight into device memory, and
   only its address and length cross by RPC.
5. **Errors.** An index out of range, or an exception nothing catches,
   reports through the RPC, as `printf` does, and traps; the first version
   does not unwind on the device. A process that traps is a place's failure
   for its monitors (Send and receive in a target region, above).
6. **A small stack** for each thread. Tail calls and last call modulo
   constructor keep loops flat; recursion that is not a tail call is kept
   shallow, as a device thread's stack does not grow.

**Messages and memory on the device.** Decided by the user on 2026-11-10.
The GPU is for fast parallel execution of work that splits easily, not for
pointer structures such as lists and trees; where two places share no
memory, copying a message and sending it is the better option. Step 4e
(`INPROGRESS.md`, GPU processes without `omp target`) had let GPU processes
and target regions send and receive any ground term, copied by its
`type_info`, which moved away from the flat messages and known lifetimes
above. This brings them back.

- **Messages to and from GPU code are flat,** in GPU processes and target
  regions alike: a fixed-size value of scalars, enumerations and mailboxes,
  such as `add(int) ; get(mailbox(int))`; or one block of variable size, an
  array of scalars, a multi-dimensional array or a view of one, or a
  string; or a shared array's handle (below). No message holds a pointer,
  so any sender writes it where the receiver reads it, as bytes, and the
  receiver reads it in place. A message type of another kind in a GPU
  kind's or a region's `send` or `receive` is an error at compile time.
  Messages between the host's processes keep any type.
- **Flat records** (added by the user on 2026-11-10), as Fortran's derived
  types and C's structs are used in GPU code, such as the element of an
  array of particles, or the accumulator of a reduction that OpenMP's
  `declare reduction` combines: a type of one constructor, not recursive,
  whose fields are flat, that is scalars, enumerations, mailboxes, other
  flat records, which are inline, and arrays of a size fixed by the type
  (INPROGRESS, step 5, "Flat records"). A tuple of flat types is one too.
  It has a layout of its own, C's for the target, so that its bytes are
  the same on the host and the GPU and in C or HIP code. In GPU code it is
  a value, never a cell: it is kept in registers or on the stack, passed
  and returned by value, and an array of them holds them inline, so
  nothing made in one iteration of a loop, or one call of a recursion,
  outlives it but by being copied into the next by value, or stored into an
  array. A flat record is a fixed-size message, and an array of them a
  one-block message.
- **A GPU process's code makes only such values,** checked by type: a goal
  that makes a value of another type, by a construction or as a call's
  result, is an error at compile time that names the type. GPU code is
  monomorphic, so every type is known. A GPU process then allocates only
  arrays, which are freed at their last use where uniqueness shows it, as
  compile-time garbage collection does, or when the process ends. It needs
  no collector, no turns and no check of what lives across a `receive`.
- **As built (step 5a, 2026-11-10),** `lib/Semantics/check-gpu-code.cpp`,
  after uniqueness analysis, on the bodies that determinism analysis builds.
  A value is flat when it is a scalar (the integers, `float`, `char`), an
  enumeration, a mailbox, a value of a type with constructors that is not
  recursive, has no existentially typed constructor and whose fields are
  such values, or a tuple of them; or, as a whole value only, a string or
  an array of scalars or enumerations. `received(T)` is flat when `T` is. A
  type variable passes, as GPU kinds are monomorphic. An abstract or
  foreign type is not flat, as its representation is unknown here.
  - The messages: of a GPU kind's own mailbox, at its declaration; of each
    `send` and `receive` in a GPU kind's code or a target region, and in
    the module's procedures that they call.
  - The values: every variable that a GPU kind's code, or a procedure of
    the module that it calls, makes, but for `io.state`, `any`, a receive's
    selector, which code generation removes, and what a send or receive
    passes, which is checked as a message. A procedure of another module is
    checked by what it gives back.
  - Each problem is reported once, at the goal, naming the type and why:
    "recursive", "a higher-order type", "has no fixed size, which only a
    whole message may have", and so on. Tests:
    `test/Semantics/processes/gpu-flat.m` and the region cases in
    `omp-target-send-errors.m`.
  - Fields that are arrays of a fixed size wait for flat records' step 3,
    and arrays of flat records for step 2. Until flat records are values on
    the device, a message of several words still travels by its `type_info`
    copy (step 4e, above), marked Temporary.
- **A target region's code may make terms,** as code compiled from ordinary
  procedures does, in an arena that lives as long as the region. A region
  is a computation and ends; a GPU process is an actor and may live for
  ever. Messages handled for long belong in a GPU process.
- **Portions of arrays travel by value:** `send_slice` sends a range of a
  one-dimensional array and `send_tile` a tile of a multi-dimensional one,
  each as a dense array of its own, copied straight from the sender's array
  into the receiver's queue, by a rectangular DMA or a strided copy kernel,
  with no packing in the program.
- **Multi-dimensional arrays** have their rank in their type, and a view of
  one is a strided descriptor, as MLIR's `memref` has: a handle, an offset,
  and an extent and a stride for each dimension, so a tile is a view, not a
  list of ranges, and views of views compose. Code generation follows
  `memref`'s semantics, so that MLIR's tiling, vectorization and mapping to
  GPUs apply. A layout stored tile by tile, which makes each tile one
  block, may come later, if measurements ask for it.
- **Shared arrays,** `shared_array(T)`, made from a unique array, are
  read-only and counted: the sender increments the count before the
  message leaves, each holder decrements it when done, and the array is
  freed at zero. An array of scalars holds no reference, so no cycle can
  keep one alive. It lives in a heap of its own on the device, not in a
  process's memory, from LLVM's GPU libc's `malloc`. Within a GPU,
  processes read it in place, as workers of a reduction that each send
  back a scalar do; another place copies it unless peer access lets it
  read it. A view of a shared array holds a reference to the whole array.

Rejected:
- **Terms in messages to GPU code,** step 4e's, which needs a walk of
  `type_info` on the device, pointers fixed up for each place, and memory
  whose lifetime the compiler does not know.
- **Turns,** decided earlier the same day: a GPU process's memory reset at
  each `receive`, with a check at compile time that nothing that has cells
  lives across one. Without cells in GPU processes there is nothing to
  reset, and the check by type is simpler than one by liveness.
- **A collector on the device,** copying with stack maps, or collecting at
  each `receive`: time on every turn, which the GPU's performance does not
  allow, and known lifetimes need none.
- **Pointers to parts of arrays in messages:** the receiver may have another
  address space, and the sender may free or update the array.
- **Counting every array once it is sent:** a cost that the program does not
  see. Only shared arrays are counted.

**Who communicates.** In a generic kernel, the code outside every
`parallel` runs on each team's initial thread only, while the other threads
wait in the runtime's state machine, so `send` and `receive` in the
sequential part (Send and receive in a target region) already run on one
thread for each team, with no directive. The first version runs a GPU
process as a kernel of one team, `num_teams(1)`, so that a process is one
team and one thread talks to the host. Later, a process for each team in an
SPMD kernel would use `masked` (OpenMP 5.1's replacement for the deprecated
`master`, `omp.masked` in MLIR's dialect, as Flang lowers it), with a team
barrier after a `receive` whose message the other threads read. Either way,
the threads that wait do so at a barrier or in the state machine, never by
spinning on a flag the talking thread sets: lanes of one wavefront spinning
while another lane must progress can deadlock on AMD.

**The subset this needs,** checked in the device compile and reported as
not supported on the device rather than run slowly: scalars, arrays of
scalars and tuples of them; no strings, no terms built from constructors,
which need a heap and a copy that knows their shape, no closures and no
polymorphism. The checks, and proving the sizes an arena is made for, are
the work; the runtime itself stays small. Futhark and Accelerate keep about
as little on the GPU: neither has a collector there, and both plan memory
statically, Futhark reusing memory blocks by its size analysis.

#### Library code on the device

Built 2026-11-10 (`lib/CodeGen/DeviceLibrary.cpp`). A target region calls the
library's procedures as it calls the module's own: the device's compile copies
in what its regions reach, from the library's `merk` bytecode, with their
foreign code compiled for the GPU. Before, a region could call only its own
module's procedures and the runtime's `merk_device_` functions, and MLIR
generation turned the few library predicates a region needed into special
cases. On the host, programs link `libmerkstd` as before.

**The library as bytecode.** The library's build writes each compiled module's
`merk` dialect as MLIR bytecode, `lib/merkurius/device/<module>.merkbc`
(`-emit-merk-bc`), compiled as for a device. That is the form before lowering:
each procedure's body with its types, `type_info`s and closures, and the
module's foreign C as text (`merk.foreign_code`). The 12 modules give 4 to 61
KB each, about a quarter of their text. A module compiled from its bytecode gives
the same LLVM IR as from its source (`emit-merk-bc.m`).

**Copying what regions reach.** After MLIR generation, the device's compile
follows the calls from its target regions:
- **Procedures:** a declaration that the regions reach is replaced by the
  library's procedure, found by the longest module name its symbol begins
  with. What the copy refers to comes from the same module: procedures,
  transitively and across modules, closures' entries, and instances without
  their methods. Copies are private.
- **Types:** the descriptor of each type whose `type_info` the module makes
  and only declares comes from its module, as `linkonce` without its unify and
  compare entries, since no library is linked into a device's image. The
  copies' constructors (`merk.type_def`) come with them.
- **After it,** everything works on the copies as on the module's own code:
  `MarkRegionsForTeamMailboxes`, omp-mark-declare-target, and the checks for
  closures and type classes. Polymorphic first-order code works as it is:
  a `type_info` is a constant in device memory, so `list.length` runs with
  its `type_info` argument.
- **Not copied:** the module's own procedures, and `array`'s under
  `-fmmc-uniqueness`, whose procedures are numbered by `array.m` rather than
  the sound stub. A procedure of another module of the program is not compiled
  for a device yet (`omp-target-other-module.m`); its bytecode could come from
  the build driver.

**Errors come at compile time, at the call.** Each copy records the procedure
that first reached it and where. What cannot run on a device is reported with
a note for each call back to the module's own code, also from the check after
lowering, as in "list.filter's closure" (`omp-target-library-errors.m`). The
frontend prints MLIR's notes, without MLIR's own note of the operation.

**Foreign code on the device** (`CompileDeviceForeignCode`):
- **Which C:** each `foreign_proc`'s C function in a module's foreign code is
  guarded by `#if !defined(MERK_DEVICE) || defined(MERK_WANT_<name>)`. Lowering
  compiles the module's own foreign code, and that of each library module a
  copied `foreign_proc` came from, as separate translation units, defining
  `MERK_WANT_` for the functions the device's module calls. The bitcode is
  linked after translation with `LinkOnlyNeeded`, so other functions of a
  `foreign_code` block, which may be for the host alone, stay out.
- **How:** as clang compiles a C file's declare target functions: `-fopenmp
  --offload-arch=<gpu> --offload-device-only -nogpulib -nogpuinc`, with the code
  inside `#pragma omp begin declare target`. The GPU comes from `-target-cpu` or
  the triple's sub-architecture (`amdgpu10.30-amd-amdhsa`, as the driver gives
  it). Without either, it is an error, as clang would look for a GPU on the
  machine. Library C that differs on a device chooses with `__AMDGPU__`.
- **What the device has:** what a compiled function needs, following the
  functions it calls, must be an intrinsic, the device runtime's
  (`merk_device_`, `merk_array_`), OpenMP's, or defined in the GPU's `libc.bc`
  or `libm.bc` beside clang, which the device's image links. Anything else is
  an error at the `foreign_proc`, "its foreign code calls X, which is not on the
  device", with the chain of calls: `math.sqrt` needs `merk_uncaught_exception`,
  `io.write` needs `merk_write`. C that does not compile for the device gets
  clang's errors and "<module>'s foreign code does not compile for the device".

**What moved out of the compiler.** `io.write_string`, `write_int` and `nl` need
no special case: their C, `fputs(S, stdout)` and so on, is LLVM's GPU libc on a
device (Output from a region, above). `process.team_number` and
`number_of_teams` are `foreign_proc`s that call `omp_get_team_num` and
`omp_get_num_teams`, on the host through weak references, so a program without
OpenMP's runtime gets team 0 of 1. `send`, `receive` and `team_mailbox` stay
builtins, as they need the message type's key, and `send` and `receive` choose
by the type between a word and an array.

**`process.team_barrier`** waits until every team of the region has called it as
often. It is a counter, not messages: both runtimes keep, in the region's team
table, the teams that have arrived in this round and the rounds that have
ended. A team reads the rounds before it arrives, and the last to arrive clears
the count and ends the round. The others wait for the rounds to change, on a
device with `s_sleep`, on the host spinning for `MERK_SPIN_MICROSECONDS` and then
on a condition variable. A region whose code reaches it waits for its teams to
start, as one with team mailboxes does, so teams that cannot all run at once are
reported and do not hang the GPU. The teams wait while others still start, and
the region is reported once none has started for ten seconds
(`kTeamsStallMilliseconds`). Rejected: a second from each team's own start,
the first version, which a full test run broke now and then, when other
programs' kernels, such as hundreds of GPU processes, filled the GPU and the
region's last teams started only as those ended. Waiting longer did not help:
with one GPU test per lit worker, the GPU had more programs' queues than its
scheduler holds, and a region's later teams did not start for more than ten
seconds, though the first had (the kernel logs "Runlist is getting
oversubscribed"). Once, too, a queue could not be taken off the GPU in time and
the driver reset it. So lit runs at most four `-gpu.test` files at once
(`MERK_LIT_GPU_JOBS` changes it, `test/lit.cfg.py`). A program that shares a
GPU with others can still be reported, which is the error's purpose. On the GPU it costs 1 µs or less at up to 64
teams, against 29 µs for a tree of messages (Measurements, "Barriers between
teams"). libc has nothing for teams to wait for each other.

**Tests:** `omp-target-library.m` (`list.length`, `reverse` and `append` on
`list(maybe(int))`), `omp-target-foreign.m` (the module's C with `printf`,
`sin` and `hypot`, `math.ceiling`, `string.length`, and a host-only function in
`foreign_code` that calls `getpid`), `omp-target-team-barrier.m` (team 0 polls
for every team's message after the barrier: without it, 150 of 200 polls fail
on the host and 189 on the GPU), each with a `-gpu.test`; and the error tests
`omp-target-library-errors.m`, `omp-target-foreign-errors.m` and
`omp-target-foreign-c-error.m`.

**Rejected alternatives.**
- **A device archive of LLVM bitcode,** compiled once from the whole library,
  as the OpenMP device runtime and libc for GPUs are. Every procedure of every
  module would have to compile for a device, or be a stub that fails when it
  runs; an error would come at link time, as a missing symbol, not at the call;
  and polymorphic and higher-order bodies would reach no specializer. It can
  come back for large monomorphic procedures, if compile time needs it.
- **Reading the library's source in each device compile,** as the module loader
  reads interfaces. Each compile would type-check and mode-check the library's
  bodies again; bytecode costs one write when the library is built.
- **The type-checked semantic form,** which this document has for the interface
  file's optimization section and `pragma inline`. Nothing serializes it yet,
  while MLIR bytecode exists, and the specializer is a `merk` pass
  (Optimizations, "`pragma inline`"), which this form feeds directly.
- **Indirect calls for closures on a device.** AMDGPU has them, but a call to an
  unknown function keeps OpenMPOpt from making a kernel SPMD, and its register
  use follows the calling convention. A specializer removes the call instead.
- **Compiling foreign C as plain `--target=amdgcn-amd-amdhsa` C.** The GPU libc
  has no `unistd.h` and others, so the host's are found and do not parse.
  Foreign code is the host's C, which OpenMP compiles for the device with the
  host's headers.
- **clang's GPU wrapper headers** (left out by `-nogpuinc`). Their maths
  functions call ROCm's `__ocml_` functions, which no device image here links.
  The GPU's libm defines the C names.
- **Special cases in MLIR generation** for each library predicate a region
  needs, and copies of library code in programs, as before: each new predicate
  would have added another.

#### Data-parallel arrays: an OpenACC-like module

Proposed 2026-11-07; not designed in detail yet. OpenACC's model has two
parts, levels of parallelism and data regions, and both have a simple
counterpart: the levels map onto OpenMP, and the data regions become a
type.

| OpenACC | OpenMP on a GPU | OpenMP on a CPU |
| --- | --- | --- |
| `gang` | `teams distribute` | `parallel do` |
| `worker` | `parallel do` | none |
| `vector` | the team's threads (`simd` on a GPU is mostly lanes) | `simd` |

Translating OpenACC into OpenMP has been done before: Clacc, from Oak Ridge
National Laboratory, implements OpenACC in Clang that way. Flang instead has
an `acc` dialect of its own. A translation suits Merkurius, which already
lowers `%$omp target teams distribute parallel do` loops.

**Data regions as a type.** OpenACC's `copyin`, `copyout`, `create` and
`present` say where an array lives across kernels. Here that is the type:

```mercury
:- type device_array(T).          % an array that lives on the device

:- pred to_device(array(T)::array_di, device_array(T)::uo) is det.
:- pred from_device(device_array(T)::di, array(T)::array_uo) is det.
```

While an array is on the device the host holds no handle to it, so it
cannot read stale data, and no `present` clause is needed, since the type
says where the data is. Uniqueness does what OpenACC leaves to the
programmer.

**Operations** on `device_array`: `map`, `zip_with`, `generate` (an array
from its indices), and `reduce` and `scan` with an operator promised to be
associative, as reduction clauses are. Each is pure, with no order between
elements, so each is one `target teams distribute parallel do` loop. Hints
for the levels, as gang and vector sizes, can become `num_teams` and
`thread_limit`, but the defaults should serve most code: OpenACC's own
experience is that the levels are mostly a tuning knob.

**How.** The module can be written in Mercury, each operation a `do` loop
with an OpenMP directive (`-fopenmp -fdo-loops`), so that the same source
runs on the CPU too, with or without `-fopenmp`, as the OpenMP tests do.
Rejected for now: building the operations into code generation, as the
array operations are, which needs more code for the same loops. Either way
the operations take closures, which the device does not have, so the
prerequisite is **specializing higher-order calls**: a call such as
`map(func(X) = X * 2.0, A0, A)` becomes a loop whose body is the lambda's
code, with no closure left, as mmc's `--higher-order` does for calls whose
closure is known. That is useful well beyond the device. The same
specialization lets a GPU process use selectors over several mailboxes,
whose `selecting` takes a closure; until then a GPU process has the
one-mailbox `receive`, specialized at its type.

**Order:** higher-order specialization; then `to_device`, `from_device`,
`map`, `zip_with`, `generate` and `reduce`; then `scan` and the level hints,
once there are programs to tune.

#### GPU to GPU across hosts: why, and how it is measured

Decided 2026-11-07: messages between processes on GPUs of different hosts,
GPU0 to CPU0 to CPU1 to GPU1, are the use case the implementation works
towards. Routing needs no table for it: a device sends everything not for
itself to its host, hosts are connected to each other directly, and a host
delivers to its own places, so each hop finds the next from the handle's
node and place (Handles, routing and places). Messages between a pair of
processes take one path, so they stay in order.

**What it replaces.** On a cluster today this takes MPI between hosts, CUDA
or HIP, or OpenMP offload, on the devices, and often OpenMP threads on the
hosts. GPU-aware MPI and NCCL or RCCL move the data, but ranks, tags,
streams, device pointers and synchronization are the programmer's, and the
common bugs are:

- a send and a receive that do not match in tag, size or order, which hangs;
- a buffer reused before a non-blocking send has finished, or read before a
  receive has;
- host and device pointers confused, or a copy synchronized on the wrong
  stream;
- a failed rank aborting the whole job, MPI's default; ULFM, the proposal for
  surviving failures, is not part of the standard.

**What the design gives instead:**

- **One operation:** `send` to a handle, wherever its mailbox is, on this
  GPU, another GPU of the host, or a GPU of another node.
- **Typed messages and selective receive** instead of tags, so a mismatch is
  a type error at compile time, not a hang.
- **Uniqueness instead of completion tracking:** a sent unique array belongs
  to the receiver, and a received one is complete when it arrives (The
  device runtime).
- **Robustness:** a GPU process that dies, a lost device or a lost host is an
  exit or `down` message to whatever monitors it, and a supervisor restarts
  the work, moves it to another GPU or reports it, in ordinary code.

**Performance decides whether it matters,** since programmers accept MPI's
difficulty for its speed. The acceptance criteria, each measured against an
MPI+HIP program doing the same:

1. **Ping-pong between GPUs of two hosts:** latency and bandwidth over
   message sizes. The hop from the device to its host and the relay are the
   extra costs, so large messages go into device memory directly with no
   copy at each hop; GPUDirect RDMA, later, skips the hosts' memory.
2. **A halo exchange:** a stencil on a grid split across GPUs, the common
   pattern of HPC communication, which shows whether communication overlaps
   computation.
3. **Recovery:** a GPU process killed during the halo exchange, and a
   supervisor that recovers, which the MPI program cannot do.

#### The route there

Decided 2026-11-07, and ahead of the rest of chunk 1 in `INPROGRESS.md`.
Deferred until the use case works or something on the way needs them: `try`
and `atomic` goals, open types, closures of methods, and the library's
array predicates (chunk 1, step 6 (d)); `io.command_line_arguments`, step
6 (e), came first, as node 1 connects to an address on its command line. Calls with hidden arguments are not needed first either: mailboxes of
concrete types have specialized `send` and `receive`, and only generic
processes need them.

The order takes the part with the most risk first, the GPU's side, where
performance is decided, and builds it together with as much distribution as
it needs, so that the first thing to work is the use case itself, GPU to
node 1 to node 2. Processes on the CPU come after.

1. **The first slice: GPU to node 1 to node 2.**
   - **On the device:** `send` and `receive` in the sequential part of a
     target region (Send and receive in a target region), the device
     runtime, and flat messages.
   - **On node 1:** the region's launching process, its main, waits for it,
     while the host forwards the region's messages to node 2 and shows node
     2's replies to the region in the mailbox's ring in host memory (The
     device's communication).
   - **On node 2:** main is the region's partner, with `send` and `receive`
     on a mailbox of its own. With one process on each node, a `receive`
     that blocks waits on a condition variable, so no scheduler is needed.
     Done: mailboxes, `send` and `receive` within one process (Processes
     and mailboxes, "As implemented"), remote handles over TCP, with the
     cookie (`INPROGRESS.md`, Remote handles and TCP), selectors and the
     `receive` goal.
   - **Between them:** TCP, with handles carrying the node and the place.
     The handshake and the handle's format are designed properly now, since
     they are hard to change once programs depend on them (Handles, routing
     and places). How node 1 gets node 2's mailbox is the smallest part of the
     connection API that is still to be designed (Remote spawn, "Nodes"):
     node 1 connects to an address given on its command line, and asks for a
     mailbox that node 2's main registered under a name.
   - **Then, before the device's side:** selectors over several mailboxes
     and the `receive` goal (Processes and mailboxes). The slice does not
     need them, as node 2's main receives from one mailbox, but the device's
     `receive` is designed once what a full `receive` lowers to is known.
     They do not depend on TCP, since a process receives only from its own
     mailboxes.
   - **First benchmark:** ping-pong between the GPU and node 2, over loopback
     on one machine, then between two machines. Done on one machine, and
     also between a GPU region of each of two nodes, processes sharing the
     machine's GPU (Measurements); two machines are still to do.
2. **Exceptions, enough to end a process:** `throw`, and an exception nothing
   catches ends the process with it as the exit reason. `catch` comes later;
   ending a process needs no unwinding.
3. **Processes on one host** (chunk 2, leaking): `spawn`, mailboxes, `send`
   and `receive` with the copy into a fragment, the scheduler, stacks of a
   fixed size, and links and monitors, which robustness needs. A region's
   launch becomes asynchronous, its process suspended until it completes,
   and ping-pong between main and a GPU of its own node works.
4. **The other benchmarks:** the halo exchange, and recovery, with
   node-down and a place's failure delivered to monitors.
5. **As they are needed:** GPU processes spawned by kind name, as persistent
   kernels; higher-order specialization, which gives the device selectors
   over several mailboxes and the data-parallel array module.

**Rejected orders:**

- **Bottom-up,** processes complete with the scheduler first, then the GPU,
  then distribution: nothing temporary, but the part with the most risk
  comes last, and the scheduler, a large piece, is not on the use case's
  critical path.
- **A thin slice with a process for each OS thread,** then each layer done
  properly: the earliest end-to-end test, but the stand-in for the scheduler
  is thrown away, and threads share a heap where processes need their own.
- **Distribution first,** between two mains: the least novel part first, and
  for flat arrays the wire format proves little.
- **GPU processes spawned by kind name first,** which would let main and a
  GPU of its node talk without a second process, but needs `:- process`
  declarations through semantic analysis first.
- **A minimal `spawn` first,** so that main and a GPU of its node talk with a
  target region: a stand-in for processes before the scheduler.

**The collector stays deferred** (chunk 3). A long ping-pong that receives a
new array with every message leaks, though; if that stops the benchmarks,
freeing a unique array at its last use, which the compiler can find, comes
first, as it needs no collector.

**Open:**

- **Hardware.** One machine with one GPU tests the whole route: GPU to
  node 1 to node 2, with the nodes two processes on the machine over
  loopback, and the reply back. That has every kind of hop: device to host,
  host to host with a relay, and host to device; a GPU on node 2 would only
  repeat the last, and the GPU cannot tell whether node 2 has one. Other
  machines, which are available, are for the performance figures, which
  loopback cannot give, and for a last check with a GPU on each host.
- **The transport between hosts.** TCP first, which keeps failure handling
  ours, and libfabric, UCX or verbs later (What the hardware offers). MPI
  underneath, hidden from programs, would bring RDMA, cluster interconnects
  and GPU-aware transfers on HPC machines, but adds a dependency, and by
  default MPI aborts the job when a rank fails.

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
| Intermodule inlining and specialization | Interface files | Bodies of polymorphic, higher-order, small and `pragma inline` procedures travel in the optimization section | High |
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

That loop hides the cost of a call, because it waits on the chain of cell loads anyway. The same operations over an array, where the load is cheap, show it (nanoseconds per element, measured with mmc):

| Version | List | Array |
| --- | --- | --- |
| Monomorphic | 1.6 | 0.77 |
| Type class | 1.6 | 1.64 |
| Closure | 1.6 | 2.25 |
| Polymorphic `unify` | 5.0 | 5.30 |

So a method or closure call costs two to three times the monomorphic loop over an array, and also blocks vectorization, whose cost this scalar table does not show. In the C that mmc generates, gcc did not devirtualize even a constant closure, so specialization has to happen in our compiler, where closures and dictionaries are known values.

Two rules follow.

- **Specialization is automatic.** At its default options mmc did not specialize the polymorphic version; it needed the `type_spec` pragma. At `-O2`, when a call passes a `type_info` or `typeclass_info` known at compile time, merkc clones the callee for that type, within a module and across modules through the optimization section of interface files: C++ templates, with the compiler choosing the instances. Polymorphism analysis already finds these calls, those whose `type_info`s and `typeclass_info`s are all constants and cells of constants (Semantic analysis, "Polymorphism analysis"). A user who cares about speed then writes ordinary polymorphic code, and the pragma stays for a type the compiler cannot see, such as one a caller in another program uses.
- **What exists so far.** Only the simplest case: a method called at an instance known in the caller, whose body is in the same module, is a direct call (Lowering, "Type classes"). A polymorphic callee given a known dictionary is not cloned yet, so a loop in it still dispatches. No chunk's steps schedule the clones, `pragma inline` (below) or the optimization section they need across modules ("Modules and interface files"); they are one piece of work.
- **Failures are visible.** If a `%$omp` loop still contains a generic `unify`, a generic `compare` or a closure call that specialization did not resolve, merkc emits a remark naming it, as Clang's `-Rpass-missed` does. Inside a target region it is an error (Language extensions, "Target regions").

Some code stays dynamic: existential types, such as the generic dispatchers, `univ`, and data structures that store closures. That is the same as virtual calls and `std::function` in C++, and under the cost rule only code that uses those features pays for them.

**For later:** packed arrays, such as `array(int8)`, or `array(float32)` if that type is added, would make polymorphic code over `array(T)` read the element size from the `type_info`. Specialization removes that cost in hot code too.

### `pragma inline`

Automatic specialization decides for itself, so a programmer cannot rely on it to remove polymorphism. `pragma inline` is how to ask for that and be told if it did not happen. This is designed; the pragma parses, and nothing acts on it yet.

- **The promise.** At a call whose types are known, a procedure marked `pragma inline` is inlined or cloned for those types, and the call passes no `type_info` or `typeclass_info` for them. Which of the two is the compiler's choice: a recursive procedure cannot be expanded completely, so it is cloned, and the clone calls itself directly.
- **When it cannot be kept.** A call's types may be unknown even after its caller has been specialized, a type may have polymorphic recursion, whose instances have no end, or a `foreign_proc` may not be duplicated (below). merkc then compiles the ordinary call and gives a remark naming it, as with "Failures are visible" above. Inside a target region it is an error, since a device has no dictionaries (Language extensions, "Target regions"). A polymorphic caller's own call is not reported as long as the caller's clones resolve it; only what is left after specialization is.
- **At every optimization level.** The promise does not depend on `-O`, so a target region that compiles at `-O2` also compiles at `-O0`, and a program does not change which of its calls dispatch when it is debugged. Automatic specialization of unmarked procedures stays at `-O2`.
- **Duplication.** The manual tells a `foreign_proc` with labels or static variables to have `pragma no_inline` or `may_not_duplicate`, because inlining would duplicate its C code. merkc therefore never inlines or clones such a procedure, whatever else is marked. `pragma no_inline` on a procedure without foreign code stops inlining but not a clone, since a clone is what removes the dictionary without growing each call site.
- **Across modules.** A marked exported procedure's body goes in the interface file whatever its size, and is part of its contract, so a change to it rebuilds every importer ("Modules and interface files"). What the body calls becomes a visible symbol for the linker, but not a name in importers' source, as with Swift's `@usableFromInline`; the interface section stays the export list. A marked procedure in the implementation section is inlined within its module.
- **The form in the interface file.** The type-checked semantic form, with its names resolved. Inlining substitutes the known types into it, which gives the same monomorphic code that re-inferring it would, and it means the same in every importer.
- **How close to one file.** A marked procedure's body reaches the same specializer in an importer as in its own module, so at a call with known types the expected code is the same as if it were in one file. That holds as far as bodies travel. An unmarked procedure that a marked body calls, if it is not polymorphic, higher-order or small, has only its signature in the interface file, so it stays a real call, where in one file LLVM could inline it. That is deliberate: the promise reaches inner calls only if they are marked too, and merkc does not mark them on its own, so what crosses a module boundary stays the programmer's choice. C in a `foreign_proc` is compiled by Clang and stays a call in either case. Link-time optimization could also close those gaps, but this design does not need it. None of this has been measured, as nothing is implemented; once it is, the comparison to run is the same program in one module and in two.
- **The pass.** Inlining and cloning happen in the `merk` specializer, the same pass as automatic specialization, because that is where instances are still visible (MLIR dialects). `pragma inline` makes the pass act on a procedure and report the outcome; it is not a second mechanism. `pragma type_spec` stays for a type no call in the program shows.
- **Compatibility.** The manual calls `pragma inline` "a hint to the compiler", so doing more than mmc does is within it, and a program means the same under both compilers: only its speed differs. No new syntax is needed, so the same source compiles with mmc.

**Rejected alternatives**

- **A hint, as in mmc.** Nothing tells a programmer that a hot loop still dispatches, which is the problem this solves. The compiler's own inlining of small procedures remains the unmarked case.
- **Source in the interface file,** so that type inference works on the body in each importer, as Swift does for `@inlinable`. Re-checking repeats work in every importer, and a body's overloads could resolve differently in the importer's scope, so the check has to be made against the defining module's declarations anyway. It would gain only what substituting types into a typed body already gives. Swift needs textual interfaces for binary compatibility across compiler versions, and that does not apply here.
- **A new keyword or syntax.** Programs would stop compiling with mmc, which our tests compare against, and the pragma already exists with a compatible meaning.
- **Separate pragmas for inlining and cloning,** as GHC has `INLINE`, `INLINABLE` and `SPECIALIZE`. What the programmer wants is that the polymorphism goes; the choice between the two depends on recursion and size, which the compiler knows.
- **LLVM's `alwaysinline`.** By then a dictionary is loads and an indirect call, which LLVM would have to recover, and it knows nothing of a recursive procedure's clones. gcc did not devirtualize even a constant closure in the measurement above.

**What other languages do,** from their documentation rather than from measurements here:

| Language | Construct | What crosses the module boundary |
| --- | --- | --- |
| Swift | `@inlinable`, `@usableFromInline` | Source text in `.swiftinterface`, or serialized SIL in `.swiftmodule` |
| GHC Haskell | `INLINE`, `INLINABLE`, `SPECIALIZE` | Typed Core in `.hi` files, which is what this design most resembles |
| Rust | Generics, `#[inline]` | MIR in crate metadata |
| C++ | Templates, `inline` | Source, in headers |
| Kotlin | `inline`, `reified` | Bytecode of the inline function |
| Mercury (mmc) | `pragma inline`, `pragma type_spec` | `.opt` files, with `--intermodule-optimization` |

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

- **What is declared but not written yet is an error at compile time,** at
  the call: "not yet implemented: code generation for calls of
  io.format/5, which its module declares but does not define yet". The
  module loader records what each imported module's file defines: a
  clause, counting a state variable `!X` as two arguments; a DCG rule; a C
  `foreign_proc`; a process kind's `spawn_Kind` and `spawn_Kind_at`; and
  `solutions`' aggregates, which code generation makes. Code generation
  checks a call or closure of another module's predicate or function
  against it, after the calls it generates itself. A sound stub
  (`library/sound/`) declares what another file defines, so its module is
  not checked. Before this, such a call compiled and failed when the program
  linked, as `io.format` did (test `undefined-calls.m`).

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

## Measurements

The speed tests' numbers, so that a change can be held against them. How each
is run, and what is learned from it, is in its README in `benchmarks/`
(`pingpong`, `array-messages`, `bandwidth`, `team-barrier`).

All were taken on one machine: an AMD Ryzen Threadripper PRO 5965WX (24 cores,
48 threads) and an AMD Radeon RX 6950 XT (gfx1030) on PCIe, with the programs at
-O2. "Between nodes" means two processes over TCP on loopback, so no real
network is in these numbers, and "GPU / GPU" means two processes sharing that
one GPU. Runs between nodes vary from one to the next, by up to a third before
the spin, so only differences of several microseconds mean anything. Each table
gives the date and the commit of the tree it was measured on; a later change
may have moved its numbers.

### Target regions

2026-11-11, gfx1030, the build of this tree at -O2: the time of a target
region by what it does, from a loop of 500 regions on the host, after 20 to
warm up. The probe was written for this and is not kept; `benchmarks/life`'s
two-node Life is what pays these each step.

| The region | Before | After the chunk pool and the device's clear |
| --- | --- | --- |
| does nothing | 17.9 µs | 17.6 µs |
| sends an int | 25.8 µs | 26.5 µs |
| makes an array of one int and sends an int | 54.0 µs | 38.5 µs |
| returns an int to the host | not measured | 44.8 µs |

- **A region that makes a cell** costs about 12 µs more than one that does
  not, from 28 µs: the chunk's RPC, about 8 µs, and the slot's clear on the
  device.
- **A region's scalar output** is mapped `from`, which costs a copy from the
  device and a wait, about 27 µs. It is what is left of the two-node Life's
  exchange beyond its send and receive, and is not addressed yet.

### Dense stencils

2026-11-11, gfx1030, 4096 × 4096 cells, 100 steps, at -O2: the average step,
the median of five rounds, of `benchmarks/life` and `benchmarks/front` against
their C programs, built by the same tree's clang. The first column is the tree
of 6eb3f50c92be with the barrier after each round of a combined loop
(Translation, "Target teams distribute parallel do"), on by default;
`LIBOMPTARGET_LOOP_BARRIER=off` gives the runtime's loop as it was. Life in
Fortran is a program written for this and not kept, the same step as
`life.c`'s, built by the same tree's flang. Every build gives the same cells
and sums.

| Program | With the barrier | Against C | Without | Against C | C |
| --- | --- | --- | --- | --- | --- |
| Game of Life, Mercury | 0.671 ms | 6% slower | 0.822 ms | 29% slower | 0.635 ms |
| Game of Life, Fortran | 0.699 ms | 10% slower | 0.933 ms | 47% slower | 0.635 ms |
| Reacting front, Mercury | 2.402 ms | 5% slower | 2.700 ms | 17% slower | 2.298 ms |

C's own loops have the barrier as clang builds them, so its time is the same
either way.

- **Without the barrier,** a profile of Life's step (rocprofv3) gave the
  same launch as C's, 320 teams of 256 threads, and the same 4.2 million
  memory instructions, but 24% more
  64-byte reads from memory, 2.66 million against 2.15 million, and 27% more
  requests to L2. The kernels' loops are nearly instruction for instruction
  the same. A C loop written as the runtime's, with no barrier, was slower
  than Mercury's, 0.96 ms, and with a barrier after each round 0.67 ms.
- **Not the cause:** clang's fused multiplies and adds (C built with
  `-ffp-contract=off` is as fast, 2.34 ms for the front), the arrays starting
  8 bytes past a cache line after their size word (C's shifted so is as
  fast, with the same reads), the claim of the region's slot, and the
  arrays' places on the device.
- **What is left** of the gap, 6% for Life and 5% for the front, is not
  diagnosed. Life's C program with 1-byte cells takes 0.258 ms, which narrow
  fields in flat records would give Mercury.

`benchmarks/bandwidth`'s kernels, which reuse nothing, gained too. Against
`bandwidth.c`, built by the same tree's clang, at 5192433284b7, Mercury and C
taking turns, three rounds, the median of each kernel's best of 100, in MB/s,
2^25 elements:

| Kernel | With the barrier | Without | C | With the barrier, against C |
| --- | --- | --- | --- | --- |
| Copy | 332087 | 303421 | 530183 | 37% slower |
| Mul | 317131 | 274011 | 456577 | 31% slower |
| Add | 371603 | 317882 | 63243 | 5.9 times as fast |
| Triad | 368700 | 323441 | 62671 | 5.9 times as fast |
| Dot | 476855 | 424203 | 521129 | 8% slower |

- **C's Add and Triad** were as slow on 2026-11-04 (`benchmarks/bandwidth`'s
  README), so it is clang's code, not the barrier, that is slow there; why
  is not known.
- **Copy and Mul** are what is left to close: a kernel that reads one array
  and writes another, where C is near the card's peak.

**Where an array's elements start.** The two programs launch the same teams
and threads, deal out the same iterations, and their loops are the same load
and store; the kernels themselves differ (rocprofv3: Copy 1.57 to 1.70 ms,
C's 1.00 ms). A Mercury array's size word comes first, so on the device its
elements start 8 bytes past the 2 MB boundary that libomptarget gives the
mapped block. `bandwidth.c` built with its arrays moved, by mapping elements
before them under `target data`, two rounds, in GB/s, 2026-11-11 at
5192433284b7:

| The elements' offset from the block | Copy | Mul | Dot |
| --- | --- | --- | --- |
| 0, as `bandwidth.c` is | 530 | 457 | 523 |
| 8 bytes, as Mercury's are | 405 | 352 | 437 |
| 16 to 128 bytes | 386 to 455 | 345 to 392 | 419 to 441 |
| 256 bytes, 512 bytes, 4 KB | 530 | 458 | 523 |
| Mercury | 336 | 320 to 335 | 476 |

- **Streaming wants elements on 256 bytes,** not on a cache line: 128 bytes
  off still loses 15% of Copy. Off by 8, C loses a quarter of Copy and Mul.
- **Not for stencils:** Life's C program with its arrays 8 bytes off took the
  same time and fetched the same bytes (Dense stencils, above), since its
  rows are read from the cache again.
- **Off by 8, Mercury is 17% below C** on Copy, but aligned it is level
  with C (below), so the misalignment costs Mercury's loop more than C's;
  why is not known. Its Dot is above C's off by 8, 476 against 437.
- **Not changed yet.** Aligning a large array's elements would pad it before
  its size word and map it from the padding, and every allocation of an
  array on the host would have to pad, the heap's, a message's and a
  collection's copies, or a map would copy another object's bytes. In-place
  slices (`docs/INPROGRESS.md`, Split and join) bear on how arrays are laid
  out, so it waits for them. A program can align a kernel's elements itself
  meanwhile: element 31 of an array is at 256 bytes on the device, as the
  size word is at the block's start, so an array 31 elements longer, indexed
  from 31, has its elements aligned.

`bandwidth.m` with its arrays 31 elements longer and its loops from 31, a
copy written for this and not kept, against `bandwidth.m` as it is and
`bandwidth.c`, three rounds taking turns, the median of each kernel's best of
100, in GB/s, 2026-11-11 at 5192433284b7; its check passed:

| Kernel | Mercury | Indexed from 31 | C | From 31, against C |
| --- | --- | --- | --- | --- |
| Copy | 340 | 542 | 531 | 2% faster |
| Mul | 326 | 470 | 456 | 3% faster |
| Add | 372 | 474 | 89 | 5.3 times as fast |
| Triad | 370 | 486 | 89 | 5.5 times as fast |
| Dot | 475 | 486 | 521 | 7% slower |

With its elements aligned, Mercury's streaming kernels are as fast as C's,
and Add and Triad gain about 30% too. Only Dot, a reduction, stays behind.

### Message latency

The round trip of a one-word message, in microseconds: 20000 rounds, after 100
to warm up. Five runs each, 2026-11-08, at a0a56a9a0a10, without the spin
(`MERK_SPIN_MICROSECONDS=0`) and with its default of 50 µs:

| Between | No spin | Spin |
| --- | ---: | ---: |
| GPU and node 2 | 47.5 to 67.7 | 33.9 to 36.0 |
| CPU and node 2 | 38.5 to 64.4 | 21.9 to 25.7 |
| GPU and a thread of its node | 23.3 to 25.8 | 19.2 to 19.4 |
| CPU and a thread of its node | 11.8 to 16.7 | 6.8 to 6.9 |
| CPU and node 2, both on one core | about 39 | about 28 |

Between regions of two nodes, with the spin, five runs, 2026-11-08, at the tree
of 2dbae6c632a3:

| Node 1 / node 2 | Microseconds a round |
| --- | --- |
| GPU / GPU | 45.2, 45.7, 46.4, 44.9, 44.4 |
| GPU / CPU | 35.8, 34.3, 33.0, 36.5, 35.8 |
| CPU / GPU | 33.7, 35.9, 37.7, 38.0, 37.4 |
| CPU / CPU | 25.3, 25.6, 24.5, 25.3, 21.6 |

The floors, plain round trips with nothing of ours, three runs, 2026-11-07,
except the last line:

| Floor | Microseconds a round |
| --- | --- |
| TCP on loopback, two processes (`tcp-floor`) | 15.4, 15.1, 15.3 |
| Two threads through condition variables (`condvar-floor`) | 8.8, 9.1, 9.0 |
| A region's RPC with the host (`rpc-floor`) | 7.5, 7.6, 7.5 |
| A word through host-pinned memory, a host thread and a kernel, 2026-11-08 | 2.0 |

What the mailbox's ring for the device did, the committed runtime before it
(dc7a2eb76c55) and the ring (a3601dcc7e7d) run in turn, four runs each, before
the spin:

| Between | Before | With the ring |
| --- | ---: | ---: |
| GPU and a thread of its node | 28 to 32 | 24 to 26 |
| GPU and node 2 | 45 to 53 | 49 to 65 |

### Barriers between teams

A barrier between the teams of a target teams region, from team mailboxes
(`benchmarks/team-barrier`), in microseconds a barrier: 10000 rounds, after 100
to warm up. `central` has every team report to team 0, which releases each;
`tree` is a binary tree. Medians of three runs, 2026-11-10, at 801205c01704 with
the benchmark added. The runs differed by under 3% on the GPU:

| Teams | GPU, central | GPU, tree | Host's teams, central | Host's teams, tree |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 3.79 | 3.74 | 2.03 | 2.05 |
| 4 | 6.72 | 7.99 | 8.15 | 8.61 |
| 16 | 33.4 | 17.1 | 90.1 | 53.7 |
| 64 | 152.9 | 28.7 | 2251 | 1567 |

- **On the GPU, a hop of a real message costs about 1.2 µs,** from central's 126
  hops in sequence at 64 teams. The estimates in docs/INPROGRESS.md assumed the
  240 ns of a word written to memory and polled, so both shapes are 5 to 10
  times those estimates: 153 µs against 30, and 29 µs against 3. Where the
  rest goes is not measured yet. Each receive also checks the handle, reads the
  clock, and makes its result's cell in the arena by an atomic add that all the
  teams share.
- **The tree pays off from 16 teams.** At 64 it is 5 times faster than central.
- **The host's teams are threads with condition variables,** so a hop costs a
  thread's wake-up. At 64 teams there are more threads than the machine's 48
  hardware threads, and a barrier takes milliseconds.

With `process.team_barrier` (`library`, Library code on the device, above)
beside them, medians of three runs, 2026-11-10, at 6a90cadd2fac. The runs
differed by under 4% on the GPU, but 9% for `library` at 2 teams, and by up to
half on the host's teams below 64:

| Teams | GPU, central | GPU, tree | GPU, library | Host's teams, central | Host's teams, tree | Host's teams, library |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.76 | 3.74 | 0.68 | 2.02 | 2.02 | 0.27 |
| 4 | 6.85 | 7.93 | 0.76 | 9.11 | 8.40 | 1.09 |
| 16 | 34.0 | 17.2 | 0.79 | 105.7 | 48.2 | 10.1 |
| 64 | 151.4 | 28.8 | 0.97 | 2264 | 1541 | 35.4 |

- **On the GPU, the library's barrier hardly grows with the teams:** 0.68 µs at
  2 and 0.97 at 64, 5 times faster than the tree at 2 and 30 times at 64. The
  design expected about one atomic round trip per team. The teams' adds to one
  counter overlap rather than queue, so the cost stays near one round trip and
  the wait's poll.
- **On the host's teams it is 5 to 8 times faster than the tree up to 16
  teams, and 43 times at 64,** and still grows with the teams: past the
  machine's 48 hardware threads, the teams that wait block, and the last to
  arrive wakes them all.
- **The shapes from messages are unchanged** from the first table, within the
  runs' spread, so the change from our output opcodes to libc's and the library
  copies did not slow messages.

### Array bandwidth

Arrays of ints as messages, in MB/s, the median of three runs, 64 MiB a
measurement, every message checked. 2026-11-08, at dc7a2eb76c55: before the
ring and the spin, which change the cost of a message, not of its bytes.

From node 1 (`to`), where `memcpy` is `omp_target_memcpy` alone, from the
device to the host:

| Words | `memcpy` | GPU, its node | CPU, its node | GPU, node 2 | CPU, node 2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8 | 0.4 | 4.9 | 0.3 | 1.4 |
| 16 | 13 | 5.9 | 79 | 5.0 | 23 |
| 256 | 203 | 93 | 1240 | 68 | 319 |
| 4096 | 2550 | 753 | 2151 | 917 | 1242 |
| 65536 | 8565 | 1471 | 2008 | 1256 | 1422 |
| 262144 | 8360 | 1174 | 2199 | 1504 | 1662 |
| 1048576 | 7338 | 1184 | 2234 | 1127 | 1416 |

To node 1 (`from`), where `memcpy` is from the host to the device:

| Words | `memcpy` | GPU, its node | GPU, node 2 | CPU, node 2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 0.3 | 0.3 | 1.4 |
| 16 | 14 | 5.0 | 4.2 | 23 |
| 256 | 226 | 79 | 68 | 310 |
| 4096 | 3397 | 1098 | 677 | 1271 |
| 65536 | 10539 | 4181 | 1179 | 1252 |
| 262144 | 20404 | 6572 | 1242 | 1363 |
| 1048576 | 23715 | 10384 | 821 | 680 |

Between regions of two nodes, in the same way, 2026-11-08, at the tree of
2dbae6c632a3:

| Words | GPU / GPU, `to` | GPU / GPU, `from` |
| ---: | ---: | ---: |
| 1 | 0.3 | 0.3 |
| 16 | 4.7 | 4.7 |
| 256 | 69 | 72 |
| 4096 | 776 | 843 |
| 65536 | 1315 | 1347 |
| 262144 | 1488 | 1283 |
| 1048576 | 1179 | 835 |

What copying and encoding arrays of scalars as blocks did at 2^20 words (MB/s,
b0a0770703e9 and dc7a2eb76c55; at 2^18 words, GPU to node 2 went from 335 to
1504):

| Between | Element by element | As a block |
| --- | ---: | ---: |
| GPU to its node | 1024 | 1184 |
| CPU within its node | 1301 | 2234 |
| GPU to node 2 | 380 | 1127 |
| CPU to node 2 | 590 | 1416 |
| GPU from node 2 | 640 | 821 |
| CPU from node 2 | 517 | 680 |

### Memory bandwidth

Five array kernels modeled on BabelStream, in MB/s, one run of `run.sh` with the
default sizes, 2026-11-04. The card's memory is rated at 512 GB/s.

| Kernel | Sequential | OpenMP, host (48 threads) | OpenMP, GPU |
| --- | ---: | ---: | ---: |
| Copy | 27,876 | 99,676 | 297,358 |
| Mul | 27,728 | 96,123 | 264,506 |
| Add | 31,355 | 105,249 | 309,747 |
| Triad | 30,239 | 106,687 | 324,757 |
| Dot | 20,316 | 127,659 | 417,102 |

Against the same program in C with OpenMP on the GPU, built by the same clang,
the median of 7 rounds, each a kernel's fastest of 100 runs, 2026-11-04. C's Add
and Triad vary the most over the rounds, from 70,000 to 141,000 and from 69,000
to 97,000:

| Kernel | Mercury | C |
| --- | ---: | ---: |
| Copy | 311,982 | 531,733 |
| Mul | 271,219 | 456,067 |
| Add | 308,900 | 78,069 |
| Triad | 324,071 | 78,711 |
| Dot | 413,621 | 515,025 |

The stack's overflow check, 2026-11-08, on 6d58c6434fe0 and its change, -O2, the same compiler with the check on and off, 7 alternating runs, median seconds. nrev is 30000 rounds of a 400-element list; Fibonacci is `fib(42)`, the worst case, a small procedure that is nearly all calls.

| Program | No check | Check |
| --- | ---: | ---: |
| nrev | 18.19 | 18.20 |
| Fibonacci | 0.82 | 1.12 |

Processes on one worker, 2026-11-08, on e742eaaa8b15 and its change, `-O2`, `benchmarks/processes`.

| Measurement | Result |
| --- | ---: |
| Ping-pong between two processes, a round | 3.5 µs |
| The same between `main` and a thread, as before processes (`pingpong cpu-local`) | 7 µs |
| Spawning a process, 64 MB stack, one million | 2.8 µs |
| Spawning a process, 64 KB stack, one million | 2.2 µs |
| A process's first run | 1.8 to 2.1 µs |
| A message to it and its end | 3.7 to 3.9 µs |
| Resident memory of a waiting process | 8 KiB |

Ping-pong with the GPU did not change: `gpu-local` 19.5, CPU and CPU between nodes 22.7, GPU and CPU 35.0, GPU and GPU 44.8 µs a round, the median of 5.

Reading the heap's address again after each call (How the scheduler was built, D1), 2026-11-08, on f67539332be0 and its change, `-O2`, nrev as above, 5 alternating runs: 18.14 s before and 18.14 s after, the median.

Several workers under one lock, 2026-11-08, on f42910dd8074 and its change, `-O2`, `benchmarks/processes`, on 24 cores with 48 threads. `work` is 48 processes each computing `fib(34)`; `pairs` is 16 pairs playing ping-pong, the time for a round over all of them.

| Workers | Ping-pong, µs a round | `work`, ms | `pairs`, µs a round in all |
| ---: | ---: | ---: | ---: |
| 1 | 3.6 | 1088 | 3.6 |
| 2 | 5.6 | 545 | 3.7 |
| 8 | 5.3 | 138 | 3.8 |
| 24 | 5.1 | 61 | 7.1 |
| 48 | 5.3 | 37 | 9.7 |

Work that does not communicate scales, 29 times on 48 threads. Messages do not: the one lock makes 16 pairs slower with more workers. With all workers, spawning a process took 9 µs against 2.8 µs on one worker's earlier numbers, and a message to it and its end 13 µs against 3.8.

A run queue for each worker, with stealing, 2026-11-08, on db41894a8ce6 and its change, the same conditions as the table above.

| Workers | Ping-pong, µs a round | `work`, ms | `pairs`, µs a round in all |
| ---: | ---: | ---: | ---: |
| 1 | 3.6 | 1108 | 3.7 |
| 2 | 5.3 | 544 | 3.1 |
| 8 | 6.2 | 139 | 2.5 |
| 24 | 5.9 | 56 | 7.3 |
| 48 | 5.8 | 39 | 8.2 |

With 48 workers, spawning a million processes took 7.0 µs each, and a message to one and its end 6.6 µs.

A lock for each mailbox, 2026-11-08, on 69694be5f5a1 and its change, the same conditions.

| Workers | Ping-pong, µs a round | `work`, ms | `pairs`, µs a round in all |
| ---: | ---: | ---: | ---: |
| 1 | 3.6 | 1097 | 3.6 |
| 2 | 5.4 | 548 | 2.6 |
| 8 | 5.9 | 138 | 2.1 |
| 24 | 6.3 | 56 | 3.8 |
| 48 | 6.3 | 35 | 4.2 |

With 48 workers, spawning a million processes took 3.9 µs each, and a message to one and its end 5.4 µs.

Once a receive from a thread that runs no process used the same counts as its senders, `pairs` was 2.3 µs at 8 workers and 4.0 at 48. Ping-pong with the GPU, the median of 5, against the numbers before the mailbox locks: `cpu-local` 7.1 (7.0), `gpu-local` 21.3 (19.5), CPU and CPU between nodes 24.6 (22.7), GPU and CPU 37.7 (35.0), GPU and GPU 48.4 (44.8) µs a round. Run alternately with the runtime before the mailbox locks, 9 runs each, the GPU paths matched (`gpu-local` 21.6 and 20.7, GPU and GPU 49.0 and 49.1), so the difference was what else the machine was doing.

Fragments that start at 256 bytes and double, 2026-11-08, on 8d0d2c124bb6 and its change, the same conditions with the machine otherwise idle. A 4 KB chunk for every message had cost a page fault each: 975,000 faults and 1.73 of 2.05 s in the kernel for 16 pairs on one worker; 75,000 and 0.13 of 0.28 s with 256 bytes.

| Workers | Ping-pong, µs a round | `work`, ms | `pairs`, µs a round in all |
| ---: | ---: | ---: | ---: |
| 1 | 0.57 | 1098 | 0.56 |
| 2 | 1.26 | 550 | 0.56 |
| 8 | 1.47 | 138 | 0.52 |
| 24 | 1.40 | 55 | 0.45 |
| 48 | 1.50 | 41 | 0.64 |

A ping-pong round leaks 0.64 KB. Ping-pong between mailboxes, the median of 5, µs a round: `cpu-local` 4.2, `gpu-local` 13.2, CPU and CPU between nodes 18.2, GPU and CPU 32.1, GPU and GPU 40.4. Linear scaling of `pairs` would be 0.035 µs at 16 pairs; what limits it now has not been looked at.

The checks that only preemption needs, 2026-11-08, on 9a2c1eb88c40 and its change, `-O2`, 5 alternating runs, median seconds without and with them: nrev 18.25 and 18.50, naive Fibonacci 1.12 and 1.17, a tail-recursive loop of a billion iterations 3.42 and 3.42.

`request`, 2026-11-08, on 8fd150e06965 and its change, `-O2`, one process asking one server 300,000 times, with a new monitor and two new mailboxes each time: 1.8 µs at 1 worker, 2.7 at 2, 3.7 at 8, 3.6 at 48. Single runs. It leaks about 1.1 KB a request, which the collector is to take care of.

A target region on the blocking pool, 2026-11-08, on ae27a5dcd1f5 and its change, `-O2`, a loop of nearly empty regions, 7 alternating runs, median µs a region, in a program that has not spawned (no move) and one that has: on the GPU 27.0 and 34.4 with the default workers, 26.8 and 34.3 with one worker; falling back to the host 0.03 and 8.7, and 0.03 and 6.8 with one worker.

Thread-local addresses read by an asm the backend cannot move, 2026-11-08, on ae27a5dcd1f5 and its change, `-O2`, alternating, median, before and after: nrev 18.80 and 18.65 s, naive Fibonacci 1.12 and 0.97 s, a tail-recursive loop 3.42 and 3.42 s; `work` with 48 processes 1155 and 968 ms on one worker, 151 and 121 on 8, 42 and 35 on 48; ping-pong 0.58 and 0.58 µs on one worker, 1.88 and 1.72 on 8; 16 pairs 0.73 and 0.59 µs on 48. Not hoisting the address may free a callee-saved register, which would explain the gain; that is not established.

Ping-pong with processes, 2026-11-08, on d7d31eff1a49, `-O2`, 20000 rounds, 9 runs, machine otherwise idle, median µs a round: `cpu-process` 5.83 (0.63 with one worker), `gpu-process` 13.26 (12.82), `cpu-local` 3.68, `gpu-local` 13.27, CPU and CPU between nodes 18.47, GPU and CPU 31.77, GPU and GPU 40.74. So a GPU answering a process of its own node costs the same as answering a thread.

A receive with a timeout, on the same commit: `cpu-process` with `after(10000)` and with `infinity`, median µs a round, 6.03 and 1.77 with the default workers, 6.03 and 1.69 with 8, 1.98 and 1.08 with 2. A timer for each receive costs about 4 µs across workers, which `request` with a timeout pays too. The cause was the order in `Block`: the timer was added before the next process was taken off the local queue, and a spinning worker stole it meanwhile. With 8 workers 81% of blocks then found nothing local, against 16% for `infinity`. Wakes of sleeping workers (0.16 futex calls a round) and false sharing of the timer's variables (separating them changed nothing) were tried and are not the cause.

Taking the next process first, 2026-11-08, on top of 06f2d398af98, `cpu-process`, 20000 rounds, 9 runs, median µs a round, `after(10000)` before and after: default workers 6.01 and 3.81, 8 workers 5.83 and 3.55, 2 workers 2.15 and 1.55, 1 worker 0.65 and 0.64. `infinity` with the change: 1.67, 1.51, 1.73 and 0.54. `gpu-process` did not move (13.13 and 13.33). About 2 µs of the cost remains with several workers, and has not been explained.

The send ring's prototype against the RPC, 2026-11-08, on a5efa6819f86, `-O2`, 20000 rounds, 9 alternating runs, median µs a round, RPC and ring: `gpu-local` 13.29 and 13.48, `gpu-process` 13.22 and 12.61, GPU and CPU between nodes 31.96 and 26.41, GPU and GPU between nodes 41.39 and 36.37. The ring is not built; see INPROGRESS, The send ring.


A token between a kernel's teams through device memory, 2026-11-09, on 51e6275ebfde, a C++ OpenMP probe at `-O2`, gfx1030 (RX 6950 XT), 200,000 one-way hops, 3 runs, machine otherwise idle apart from the desktop on the same GPU, ns a hop on the device's clock, polling teams: 2 teams 200, 4 to 64 teams 240, 256 teams 264, 1024 teams 520, 2048 teams 1125. Teams of 64 idle threads cost the same as teams of one (2 and 16 teams). One word with a cache line to itself, not yet a message with a queue; proposed in `INPROGRESS.md`, "Processes as teams".

### Heap pointers in an address space of their own

Measured on 2026-11-08, commit 559bb24685ad against 5df05bc20023, the commit
before the change, and 38f0d96813d1 between them. nrev of 400 elements, 30000
rounds, about 2.4 billion cells, at `-O2` on one pinned core; user time in seconds,
median of 5 runs.

| | Before | Terms in address space 1 | Words too |
| --- | --- | --- | --- |
| Monomorphic | 4.05 | 4.01 | 3.97 |
| Polymorphic | 4.01 | 3.97 | 4.00 |

The spread of the runs of one build, about 0.2 s, is larger than any difference, so
the change costs nothing here. User time only: until the collector, a process
never reuses its heap, so the run takes 37 GB and spends a further 15 s in the
kernel, which hides the code.

A hole as the cell's pointer and an offset, measured the same way on 2026-11-08
against 559bb24685ad, with the change not yet committed: 4.00 s against 4.00 s
monomorphic, and 4.01 s against 4.02 s polymorphic. The extra argument costs nothing
here.

The `deopt` bundles of what is live at each safepoint, still unused and taken off
before code generation, measured the same way on 2026-11-08, commit 0de549515905
against 4a8a4296f7f2: 3.99 s against 4.06 s monomorphic, and 4.02 s against 3.99 s
polymorphic. Within the spread of the runs. nrev has little for an optimizer to
inline or remove, so this says little about code that does.

Statepoints and stack maps, measured the same way on 2026-11-08, commit 91cf88a7b539
against 0de549515905: 4.00 s against 3.90 s monomorphic, and 3.97 s against 4.03 s
polymorphic. Within the spread of the runs, with the differences in both directions.
nrev's work is in append, whose loop has safepoints only on allocation's slow path, so
this says little about code with many calls and live values across them. The program's
text grew from 150 KB to 164 KB, library included, with 94 KB of stack maps.

Every root's type and `type_info`s in the bundles, measured on 2026-11-08, commit
7c4ca12cb257 against 91cf88a7b539, nine runs of each on one core. These are medians
of elapsed time, kernel included, so they are not comparable with the user times above:
18.635 s against 18.606 s monomorphic, and 18.607 s against 18.566 s polymorphic.
The monomorphic program's text grew from 298 KB to 318 KB, and its stack maps from
94 KB to 107 KB, with 0.6 KB of root table. nrev has few live values at its safepoints,
so as before this says little about code with many calls.

### The collector

Measured 2026-11-08 at commit d1d4d913b2f8, -O2, one process pinned to one core
of an otherwise idle machine, five runs each, median seconds. "No collection" is
the same program linked with a runtime whose threshold is never reached, so its
chunks are only freed at the end.

| Program | Collecting | Peak | No collection | Peak |
| --- | --- | --- | --- | --- |
| nrev, monomorphic | 3.01 | 6 MB | 18.84 | 36.9 GB |
| nrev, polymorphic | 3.03 | 5 MB | 18.81 | 36.9 GB |
| Garbage under 300,000 frames | 16.25 | 41 MB | 4.58 | 8.4 GB |
| Garbage with a live list of 128 MB | 18.72 | 499 MB | 9.34 | 17.5 GB |

The spread between runs was under 1%.

- **nrev is six times faster collecting.** Almost nothing survives, so a
  collection costs little, and reusing a few MB avoids the page faults of touching
  37 GB of new memory. The polymorphic program costs the same.
- **A deep stack is the worst case: 3.5 times slower.** A collection each 4 MB
  walks all 300,000 frames, whatever survives. A stack watermark, which skips
  frames that have not run since the last collection, needs generations: here
  every collection moves every live object, so every frame must be updated.
  The threshold below pays for the walk instead.
- **A large live heap costs twice the time.** Each collection copies what is
  live. Generations are what would remove this, so they stay deferred until a
  program that matters looks like this one.
- Comparing with no collection flatters neither side simply: its page faults
  are part of its time, so the collector's own cost is more than the difference
  in the last two rows shows.

With the next collection due only after allocating four times the bytes of
stack just walked, as well as 4 MB and twice what is live (2026-11-09, at commit
b4d7d9366528, same method):

| Program | Collecting | Peak | Before the change | Peak |
| --- | --- | --- | --- | --- |
| nrev, monomorphic | 3.01 | 6 MB | 3.01 | 6 MB |
| nrev, polymorphic | 3.03 | 6 MB | 3.03 | 5 MB |
| Garbage under 300,000 frames | 3.47 | 96 MB | 16.25 | 41 MB |
| Garbage with a live list of 128 MB | 18.69 | 499 MB | 18.72 | 499 MB |

The deep stack now runs faster than with no collection (4.58), for 55 MB more
at its peak, and the others do not change. The factor of 4 is a round number, not
tuned. Frames whose size the stack map does not give are not counted.

## Target applications

Recorded on 2026-11-05. This is judgement, not a result: the only performance evidence so far is the bandwidth benchmark, whose program is no simpler than its C version (`benchmarks/bandwidth/README.md`). That the code is simpler has still to be shown on a program with real logic in it.

### Where the language could win

The best fit is a program that combines complicated decision logic with large numeric or data-parallel kernels. Such programs are usually C++ plus Python, or a rule engine and GPU code joined together; here they could be one typed program, whose source runs sequentially, on the host's threads or on a GPU.

| Application | Mercury side | GPU or parallel side | Typical today |
| --- | --- | --- | --- |
| Combinatorial optimization: scheduling, routing, branch and bound | search, backtracking, constraints | bounds and candidate scoring | C++, or Python around a solver |
| Chip design tools: synthesis, placement, verification | netlist rewriting, rules | placement and timing calculations | C++ and Tcl or Python |
| Query engines | plan rewriting and search | column scans and joins | C++, Rust, Scala |
| Robot planning | task planning | collision checks and trajectory scoring | C++ and Python |
| Rule-heavy live services: fraud, trading risk, network policy | rules | batch scoring, isolated processes | Java, Erlang or Elixir plus native code |
| Tensor and HPC compilers | graph rewriting | the host compiler's own heavy analyses | C++, Python, Rust |

What gives it the edge:

- **Modes and determinism.** The compiler checks data flow and how many answers a predicate can produce, where rule-heavy C++ tends to hide bugs.
- **Unique arrays.** In-place updates with no garbage, and a declared single owner gives the compiler aliasing facts that C needs `restrict` for.
- **Directives as comments.** The sequential build is a reference for the parallel one, as the OpenMP tests already use it.
- **Lightweight processes** (designed, not built). Erlang-style isolation for numeric code.

### Compared with today's languages

| Language | Merkurius could do better on | That language stays better on |
| --- | --- | --- |
| Python plus C++ | one language, one type system, no boundary between the two | libraries, prototyping, hiring |
| C++ | symbolic code, memory safety, fewer logic bugs | control, ecosystem, tooling |
| Prolog | static checking, speed, GPU use | interactive and metaprogramming work |
| Erlang/Elixir | numeric speed, static types | a proven runtime; hot code loading and distribution |
| Haskell | predictable in-place updates and performance, built-in search | abstraction, libraries |
| Rust | backtracking and symbolic code without ownership friction | systems programming, libraries |
| Julia | compile-time guarantees, predictable compiled code | numeric libraries, interactive work |
| Fortran | everything around the kernel | array notation, existing numeric code |

It is unlikely to win for operating systems and drivers, GUIs, quick scripts, or purely numeric programs where the kernel is the whole program. C or Fortran are already simple enough there, and the C comparison in the benchmark's README shows Mercury has no automatic speed advantage.

What could stop it:

- **Missing language pieces.** Typeclasses, existential types, processes and parts of the standard library are not there yet.
- **Unchecked parallel promises.** The OpenMP directives are promises the compiler does not fully check, so "safe" does not yet cover data races in a kernel.
- **GPU limits.** Kernels work on flat arrays and cannot allocate, so symbolic data has to be converted to arrays before it reaches the GPU. If that is awkward, much of the simplicity is lost.
- **Adoption.** Modes and determinism are hard to learn, and there are no numeric libraries yet. A niche where the benefit is obvious is more realistic than general use.

### Compiling AI kernels: the state of the field

The user pointed out that knapsack-like problems matter for compiling AI kernels, distributing code and finding communication patterns, and were far from solved 3 to 4 years ago. A quick search on 2026-11-05 found much progress but no complete solution: each sub-problem has practical methods, mostly applied separately. It read mostly search excerpts rather than full papers, most of the numbers are self-reported, and it found no recent head-to-head survey, so this is a starting point and not a literature review.

| Problem | Where it stands | Still open |
| --- | --- | --- |
| Distributing a model across devices | Propagation systems such as GSPMD and Shardy are the compiler infrastructure. Alpa uses integer programming to choose each operator's partitioning and minimize communication, and PyTorch/XLA's auto-sharding option uses it. | TOAST (2025) combines static analysis with Monte Carlo tree search and claims to beat industrial methods, which suggests search scale is still a live problem. It is a preprint. |
| Memory and scheduling inside a kernel | Constraint solvers are in real use: NXP's eIQ Neutron compiler uses constraint programming, and a 2025 backend generator uses a SAT-based solver for memory allocation. TelaMalloc (2022) did this earlier for production accelerators. | Joint scheduling, memory and rematerialization. One 2026 paper shows recomputation produces a knapsack-cover constraint and is strongly NP-hard even in restricted cases. |
| The solvers themselves | GPUs now help noticeably. NVIDIA's cuOpt runs its search heuristics on the GPU, using the PDLP method for LP relaxations. COPT and HiGHS also use GPU PDLP. | cuOpt still runs branch and bound on the CPU, and its documentation says proving optimality is still under development. |
| Learned approaches | RL has beaten fixed pass orders such as `-O3`, and AlphaEvolve-style heuristic discovery has appeared (Magellan, 2026, seen by title only). | No good comparison against integer programming was found, so whether they have overtaken it is unknown. |

What this means for Merkurius:

- **Plain knapsack is the wrong demonstration.** A single-constraint knapsack is largely easy in practice, with dynamic programming or a good approximation, even though knapsack constraints appear throughout these problems.
- **The hard version is the coupled one.** Choosing partitions, schedule and spills together, with memory limits per device and communication costs between them, looks like the part that remains open. That is a reading of the results above, which do not state it outright.
- **The architecture suits it.** cuOpt's split, with the GPU doing bulk evaluation and the CPU managing the tree, is what Merkurius does naturally: search logic on the host, flat-array kernels on the GPU.
- **It tests the language too.** Partitioning rules and legality constraints are what Mercury expresses well.

### A possible benchmark

A small tensor-partitioning solver, instead of plain knapsack. Not planned yet; which problem it targets should depend on which of them hurts most in practice.

- **Input.** An operator graph, each device's memory limit and the communication costs.
- **Choices.** A partitioning for each operator, and which values to recompute instead of keeping in memory.
- **Search.** Branch and bound in Mercury, scoring batches of candidates on the GPU.
- **Baselines.** An integer programming solver for solution quality, and a C++ version for code size and speed.

A mature solver could beat a small custom one easily. The claim to test is that domain-specific search is simple to write and quick on large graphs, not that it beats solvers in general.

## Phased plan

Seven milestones take the project from today's experiments to processes and parallel loops in one program, and then across machines. Each ends with a working compiler and a gate measured against the Experiments baseline.

&#91;embedded content: phased plan · 6 milestones, each with its gate\]

M3 and M4 both build only on M2, so their order can be swapped if processes matter more than parallel loops. M6, distribution, builds on M4 and can start before M5. M1 begins with the front end and its unparser, whose round-trip tests then guard every later change to them.

## Open questions and risks

Seven decisions remain open. The two that most constrain the runtime are the nondeterminism strategy and the stack model, so they come first. The designs being worked out for them, and the plans for the pieces under way, are in `INPROGRESS.md`; each moves here when it is decided and implemented, with the options it rejected and why. The ones ticked at the end were decided between 2026-10-27 and 2026-11-04.

**Open questions**

- [ ] Nondeterminism: continuation passing with stop flags, or an explicit choice-point stack.
- [x] Process stacks: a fixed size given at spawn, in lazily committed slots, with a limit check on entry to procedures on cycles (Runtime system, "Processes and scheduler").
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
- [x] Process declarations: `:- process Kind(Types) receives Types [on gpu]`, expanded into the kind's predicate and `spawn_Kind`, behind `-fprocess-declarations` (Language extensions, "Processes and mailboxes").

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
- Target applications, excerpts from a search on 2026-11-05:
  [TOAST](https://arxiv.org/abs/2508.15010),
  [PyTorch/XLA SPMD](https://pytorch.org/blog/pytorch-xla-spmd/) and its [auto-sharding RFC](https://github.com/pytorch/xla/issues/6322),
  [a technical review of Alpa](https://www.zhongzhuzhou.org/blog/2026-03-26-2026-03-26-Alpa-technical-review-en/) (secondary),
  [eIQ Neutron](https://arxiv.org/pdf/2509.14388),
  [generating compiler backends from tensor IR](https://arxiv.org/pdf/2510.09932),
  [TelaMalloc](https://dl.acm.org/doi/pdf/10.1145/3567955.3567961),
  [recomputation and knapsack-cover constraints](https://arxiv.org/html/2610.23328v1),
  [NVIDIA cuOpt](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) and its [26.02 release notes](https://docs.nvidia.com/cuopt/user-guide/26.02.00/release-notes.html),
  [GAMS on cuOpt](https://gams.com/blog/2025/09/gpu-accelerated-optimization-with-gams-and-nvidia-cuopt/),
  [AutoPhase](https://ar5iv.labs.arxiv.org/html/1901.04615),
  [CORL](https://arxiv.org/pdf/2008.08951),
  [target-independent XLA optimization using RL](https://arxiv.org/pdf/2308.14364),
  [Magellan](https://arxiv.org/pdf/2601.21096).
- Measurements in this document were made with Melbourne Mercury 22.01.9 (`hlc.gc`), gcc 13.3, SWI-Prolog 9.0.4 and CPython 3.13 on a 2-vCPU cloud VM.
