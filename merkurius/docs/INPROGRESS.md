# In progress

Designs being worked out and the plans for carrying them out. When a
design is decided and implemented, it moves into the design document
(`MercuryMLIRCompilerandRuntimeDesign.md`). There it states what was chosen
and why the alternatives were rejected, with the measurements that decided
it, above all for how values are represented and how the collector works.
This file then keeps only what is still open.

## The order of the next pieces

Three chunks remain before processes and memory management are done: terms
on a heap, then processes and messaging, then garbage collection. The first
is shared: messaging needs terms (a message is a value of a user type,
`mailbox(M)` is polymorphic, `spawn` takes a closure, and `send` deep-copies
a term), and so does a collector, which traces them with the same layout
information. Messaging comes before collection: done with the invariants
below, it needs no rework when the collector arrives.

1. **Terms and allocation (shared foundation).** Constructors and lists,
   tagged pointers, polymorphism with `type_info`, closures, and all
   allocation through `mrt` operations with a heap-limit check from the
   start, even while the slow path only takes another chunk and never
   frees. The cell layout (Runtime system, "Memory management", options A
   and B) is decided here, so that neither messaging nor the collector has
   to change it later.
2. **Processes and messaging, leaking.** Per-process heaps as bump
   allocation into chunks, the scheduler and mailboxes, `send` copying into
   a fragment as the design has it, and preemption by lowering the heap
   limit, with `mrt.yield_check` in loops that do not allocate. Each process
   has a large `mmap`ed stack with a guard page instead of a growable,
   copied one, which needs stack maps.
3. **Garbage collection.** Statepoints and stack maps, minor and major
   collection, the remembered set for destructive updates, received
   fragments as roots, and growable stacks.

**Invariants that keep chunk 2 from being redone in chunk 3:**

- Every allocation goes through `mrt.alloc` and its limit check, so the
  collector hooks into the slow path.
- A mailbox handle is a value, never a pointer into another heap.
- A message lives in a fragment of its own, which the collector merges,
  rather than pointing across heaps.
- The runtime's C++ code holds no raw heap pointer across anything that
  could become a safepoint.

What the collector adds beyond that is statepoint lowering of calls, a
change to lowering that leaves messaging's design alone. Aliasing and open
terms are apart from all three: they are variable cells and the trail, and
messages can be ground terms at first.

## Chunk 1: terms and allocation

### Where it starts

Code generation handles primitive types, strings, `io.state` and arrays of
word-sized values. It reports as not yet implemented any other type
("values of type ..."), polymorphic procedures and calls with hidden
arguments, disjunctions, and unifications other than assignments and tests
of primitive values. The `merk` dialect has no term type and none of
`merk.construct`, `merk.deconstruct`, `merk.unify_test`, `merk.closure` or
`merk.call_closure`. There is no `mrt` dialect: `merk` lowers straight to
the LLVM dialect. Arrays are allocated by the runtime with `malloc` and
never freed. The library's array predicates are built into code generation
"until the library can be compiled with type_infos".

### Decisions to make

Each is decided before the step that depends on it. The design document
records the choice and the rejected options.

**D1. Cell layout. Decided 2026-11-04: headerless cells (option A),**
traced by type. The design document has the decision, the rejected header
word and pages of one shape, and experiment E7's numbers
(`experiments/layout`). Still to do: implement it in step 3, and measure
nrev again on our compiler in step 7.

**D2. Where the heap pointer and limit live.** Every allocation reads and
bumps them, and the scheduler lowers the limit to preempt.

- A thread-local current-process pointer. A process runs on one worker at a
  time, so the worker's thread-local is its process. It costs a TLS access
  per allocation, which LLVM may keep in a register within a function.
- A hidden context argument on every procedure, as GHC's BaseReg is in
  effect. One register, no TLS, but every call and closure carries it, and
  foreign code that calls back needs it.
- A pinned register: fastest, but it needs a calling convention per
  target, which LLVM supports only partly.

To be decided with chunk 2's context switch in mind; the allocation fast
path is written so that only where the pair is read changes.

**D3. `type_info` and type descriptors.** A `type_info` for a type with no
arguments is the address of its static descriptor (`mrt.rtti`); for an
applied type, such as `list(int)`, a cell of the descriptor and the
arguments' `type_info`s, static when the type is known at compile time.
The descriptor gives each constructor's name, arity, argument types and
layout, for `unify` and `compare` at polymorphic types, `io.write`, `univ`,
the collector and the message copy (D1). Melbourne's representation is not
used as a reference (Clean-room rules); the reference manual's
`type_info` and `pseudo_type_info` notions and E4's observations (one hidden
argument per type variable) are.

**D4. Polymorphic arguments.** On 64-bit targets every value fits a word,
so a polymorphic argument is a word, a float its bits, with no boxes (E4
saw mmc box `list(float)` elements in a 16-byte cell). 32-bit targets
would need boxes for `float`, `int64` and `uint64`; they are not
supported.

**D5. Which of Melbourne's optimizations to keep, and when.** From the
design document's table: enums as small integers, constants among
functors as tagged small integers, primary tags, the secondary tag
in the cell's first word, notag types and static ground terms, from the
start. Packed enum fields later, since they change field access, not the
dialect.

**D6. Static ground terms and the collector.** A ground term known at
compile time is a global. The collector must not copy it; with headerless
cells (D1) it knows by the address: a pointer outside the process's heap
chunks is static or a received fragment's.

**D7. Closure layout and calls.** A closure is a cell with the code
pointer, the number of captured arguments and the captured arguments; a
call passes the closure and the remaining arguments. The question is
whether a closure call goes through a generic entry, which unpacks the
cell and calls the procedure, or the closure's code is a wrapper made per
`merk.closure` site, which is what `spawn`'s `pred(mailbox(M), io, io)`
needs to be fast.

### Steps

Each is a commit with its tests, run and output-checked as the existing
`test/CodeGen` tests are, with mmc's output where mmc can compile the
program.

1. **Done:** D1's measurement, in C, and the decision written into the design
   document with the rejected options.
2. **Done:** monomorphic types in `merk`. `!merk.term<"T">`,
   `merk.type_def`, `merk.construct`, `merk.deconstruct`,
   `merk.has_functor`, `merk.switch` on constructors, and unifications of
   terms as calls of a `unify.T` procedure generated for each type, in
   place of `merk.unify_test` (the design document has why). Enums,
   constants mixed with functors, notag types, tuples, and user types and
   `list(T)` at known `T`, lowered straight to LLVM, with cells from
   `merk_alloc`, a `malloc` that never frees. Left for later steps: static
   ground terms (step 3), unifications of terms with `string` or array
   arguments, and the library's `list`, `maybe` and `pair` predicates
   (step 6).
3. **The `mrt` dialect.** `mrt.alloc` with a bump pointer and limit check,
   whose slow path is `merk_alloc_slow` in the runtime, taking chunks that
   are never freed; `mrt.tag`, `mrt.untag`, `mrt.field_load`,
   `mrt.field_store`, and D1's headerless cells. `merk` now lowers through `mrt`, and
   the arrays move onto it. Static ground terms as globals (D6).
4. **Polymorphism.** Hidden `type_info` arguments first, as the calling
   convention says, static type descriptors and `mrt.rtti` (D3),
   polymorphic `unify` and `compare` through the descriptors, and
   polymorphic procedures.
5. **Closures.** `merk.closure` and `merk.call_closure` (D7), higher-order
   calls, and the library's higher-order predicates that need them.
6. **The library on terms.** `list`, `maybe` and `pair` compiled, and the
   array predicates compiled with `type_info`s instead of built into code
   generation; `io.command_line_arguments`, which needs lists.
7. **The gate.** nrev at E1's sizes on our compiler against E1's C and
   mmc, and the design document's sections on representation, the calling
   convention and the `merk` and `mrt` operations updated.

Not in chunk 1: disjunctions and nondeterminism (an open question of its
own), type classes, existential types, `io.write` (it needs D3's
descriptors but can come after), and the collector.
