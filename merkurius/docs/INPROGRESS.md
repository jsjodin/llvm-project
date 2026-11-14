# In progress

Designs being worked out and the plans for carrying them out. When a
design is decided and implemented, it moves into the design document
(`MercuryMLIRCompilerandRuntimeDesign.md`). There it states what was chosen
and why the alternatives were rejected, with the measurements that decided
it, above all for how values are represented and how the collector works.
This file then keeps only what is still open.

## The order of the next pieces

Decided 2026-11-07: the work heads for messages between processes on GPUs
of different hosts, and the route to it comes before the rest of chunk 1
(design document, Distribution and wire format, GPU processes, "The route
there"). What follows is the order as it stood before; the parts of it on
that route keep their place in it.

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
   has a stack of a fixed size given at spawn, with the limit check on entry
   to procedures on cycles (Runtime system, "Processes and scheduler").
3. **Garbage collection.** Statepoints and stack maps, minor and major
   collection, the remembered set for destructive updates and the holes of
   last call modulo constructor, and received fragments as roots. A
   `!merk.ref` becomes the cell's pointer and the hole's offset, so that
   the cell is a base the statepoints relocate (design document, Lowering,
   "GC metadata and safepoints").

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

(As of 2026-11-04, before step 1; steps 1 to 3 are done.) Code generation
handles primitive types, strings, `io.state` and arrays of word-sized
values. It reports as not yet implemented any other type
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

OpenMP favours the first: a team thread's thread-local is null until its
first allocation attaches it to the region's process (Runtime system,
"OpenMP cooperation"), where a hidden argument would have to pass through
outlined functions and a pinned register be set on each team thread.

To be decided with chunk 2's context switch in mind; the allocation fast
path is written so that only where the pair is read changes. Until then
(step 3) the pair is thread-local, `merk_heap` in `runtime/heap.cpp`, which
the first option's cost is, with a thread's heap where the first option
has a process's: only `mrt.alloc`'s lowering and `runtime/heap.h` read it.

**D3. `type_info` and type descriptors. Decided 2026-11-04,** with the tag
in the `type_info`, strong symbols for descriptors, and generated unify and
compare; moved into the design document with step 4.
The layout is ours: the manual's `type_info` and `pseudo_type_info` notions,
the RTTI paper (Dowd et al., 1999) and E4's observations are the references,
and Melbourne's representation is not copied (Clean-room rules).

*A descriptor* is static data, one for each type constructor, not for each
applied type, since a term's layout does not depend on its type arguments:

```c
struct MerkTypeCtor {
  uint8_t kind;           // below
  uint8_t flags;          // PointerFree, OpenSame, VariableArity
  uint16_t arity;         // type parameters
  uint32_t functorCount;
  const char *module, *name;
  MerkUnify unify;        // i1 (type_info, word, word)
  MerkCompare compare;    // i64 (type_info, word, word)
  const MerkFunctor *functors;   // in declaration order, compare's order
  uint16_t byTag[8];      // Cells: a primary tag's functor, or the shared mark
  const uint16_t *constants;     // Cells: a constant's number to its functor
  const uint16_t *shared;        // Cells: a secondary tag to its functor
};

struct MerkFunctor {
  const char *name;
  uint32_t arity;         // fields, without a secondary tag's word
  uint8_t tag;
  int32_t secondary;      // -1 if none
  int64_t constant;       // a constant's word
  const MerkFieldType *fields;
};
```

The kinds are the integers with their width and sign, `Float`, `Char`,
`String`, `Array`, `Dummy`, `Enum`, `NoTag`, `Cells`, `Tuple`, `Closure`
(D7), `TypeInfo` and `TypeclassInfo`, and later `Open` and `Closed`. A new
kind changes no existing descriptor. `PointerFree` means a value holds no
heap pointer if its type arguments hold none, so the tracer can skip it.

*A `type_info`* is one word, a pointer with a tag in its low bits:

| Tag | Points to | Example |
| --- | --- | --- |
| 0 | A descriptor, for a type of arity 0 | `int` |
| 1 | A cell: the descriptor, then its arguments' `type_info`s | `list(int)` |

A tuple's or closure's constructor (`VariableArity`) has the number of
arguments after the descriptor. A `type_info` with no type variables is a
static cell, one for each type in a module; `list(T)` in polymorphic code
is an allocated cell. Two `type_info`s of one type can have different
addresses, so `univ` and the like compare them by structure.

*A field's type* (`MerkFieldType`) is a word of the same form, with two more
tags: 2 for a type parameter of the term's type, its number in the high
bits, and 3 for an existential variable, whose `type_info` is in the cell's
hidden fields (design document, "Type descriptors"). So `[|]`'s fields are
`var 1` and the static cell `[list, var 1]`. A walker cannot allocate, least
of all the collector, so it does not make a `type_info` for each field: it
carries the field's type and a pointer to the frame that gives its
variables' `type_info`s.

*Finding a value's functor:* `Enum`, by the word; `NoTag`, no functor, the
same word at the argument's type; `Cells`, `byTag[word & 7]`, then with the
shared mark the secondary tag in the cell's first word, and with tag 0 and
constants in the type `constants[word >> 3]`. A cell's size is its functor's
arity, plus one for a secondary tag. That is all the collector and the
message copy need of the cell: they read no header.

*Unify and compare* in polymorphic code are `merk_unify(ti, x, y)`: load the
descriptor, whichever the tag, and tail call `unify` with the same
arguments. Lowering generates that entry for each type, which loads the
`type_info`s of the type's arguments and tail calls `unify.T`; `unify.T` for
`list(T)` takes `T`'s `type_info` and calls `merk_unify` on the heads. A
unification at a known type keeps calling the monomorphic `unify.T`
directly, as today, so monomorphic code does not change. `compare` follows
the same pattern. A type with user-defined equality has its predicate there.
The collector, the message copy, `from_open` and printing are walkers in
the runtime, in C++.

*Where descriptors live.* A strong symbol in the module that defines the
type, since a module importing an abstract type cannot make it. The
primitive types', arrays' and tuples' are `linkonce_odr` in each module using
them, not in the runtime as first proposed: their unify entries are `tailcc`
procedures, which C++ cannot define, and generating them keeps one path for
all. Until step 6 compiles `list`, `maybe` and `pair`, the same goes for the
uncompiled library's types, marked temporary. Nothing compares descriptors by
address, so a duplicate would be harmless.

*Words.* Polymorphic code requires that every value be one canonical word:
narrower integers and chars zero-extended, a float its bits, a dummy value 0.
Fields are so already (Type representation). Zero extension needs no
signedness, and a descriptor says how to compare the bits. `merk` gets
conversions between a value and its word at polymorphic calls and fields;
only the narrow types and floats generate code for them.

*Rejected:*

- **A descriptor that doubles as a `type_info`, told by its first word,**
  as the RTTI paper describes: testing needs a load, and mispredicts on a
  type mix, where the tag is known from the pointer. The paper itself
  has a switch to turn this off. Not measured; the choice is confined to
  `merk_unify` and the walkers, so it can be revisited at step 7's gate.
- **A descriptor per applied type,** which would hold each field's concrete
  type: polymorphic code would need to make them at run time, and they would
  be needed for every type used in a program.
- **Unify and compare as walkers only,** without generated procedures: less
  code, but slower on the path E4 measured at 3×, with no way to specialize
  or inline them, and user-defined equality has to call out anyway.
- **A generated trace procedure per type** in place of tables: possibly a
  faster collector, but more code per type, and the copy, printing and
  `from_open` need the tables anyway. It can be added beside them later.

*What open types need from this* (see "Open types", below):

- **One variable tag for every type.** `deref` and `bind` must not need a
  `type_info`, so tag 7 is the variable tag in every open layout. A type
  needing tag 7 in its ordinary layout moves one more functor under the
  shared tag in its open layout.
- **Whether the two layouts coincide is known only at run time** for
  `list(T)`: `T = tree` shares its layout, and `T = int` does not. `OpenSame`
  says a type's layouts coincide if its arguments' do, so that polymorphic
  `to_open` can decide from the `type_info`. Answering no for a type whose
  layouts do coincide only makes `to_open` copy.
- **An open layout is a fixed function of the ordinary descriptor,** so
  that the runtime can derive it for a type whose module never mentions
  `open`, which polymorphic code can receive. The design document's "each
  type used under `open(_)` gets a descriptor" cannot see such a type, and
  generating one for every type would charge programs that do not use open
  types, against the cost rule. Walkers therefore take a layout, ordinary or
  open, beside the descriptor.

**D4. Polymorphic arguments.** On 64-bit targets every value fits a word,
so a polymorphic argument is a word, a float its bits, with no boxes. mmc
does the same on 64 bits in `hlc.gc`: a `list(float)` built through a
polymorphic predicate takes the same memory as a `list(int)`, 16 bytes an
element, and its generated C's `MR_box_float` allocates nothing there
(measured 2026-11-04). 32-bit targets
would need boxes for `float`, `int64` and `uint64`; they are not
supported.

**D5. Which of Melbourne's optimizations to keep, and when.** From the
design document's table: enums as small integers, constants among
functors as tagged small integers, primary tags, the secondary tag
in the cell's first word, notag types and static ground terms, from the
start. Packed enum fields later, since they change field access, not the
dialect.

**D6. Static ground terms and the collector.** A ground term known at
compile time is a global (done in step 3). The collector must not copy it;
with headerless cells (D1) it knows by the address: a pointer outside the
process's heap chunks is static or a received fragment's.

**D7. Closure layout and calls.** Decided 2026-11-06 and done in step 5:
an entry shared by every closure of a procedure with as many captured
values, a lambda expression as a procedure of its own, and word copies for
tail calls (design document, Lowering, "Closures", with the rejected
alternatives).

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
3. **Done:** the `mrt` dialect. `mrt.alloc` with a bump pointer and
   limit check, whose slow path is `merk_alloc_slow` in the runtime, taking
   chunks that are never freed; `mrt.tag`, `mrt.untag`,
   `mrt.primary_tag`, `mrt.field_load`, `mrt.field_store`, and D1's
   headerless cells. Terms lower through `mrt`, and the runtime's arrays
   allocate on the same heap. Static ground terms as globals (D6), one for
   each cell's contents. The heap pointer and limit are thread-local until
   D2. Left for chunk 3: arrays' element loads and stores as `mrt`
   operations, for the write barrier.
4. **Polymorphism.** Done: canonical words for narrow fields, hidden
   `type_info` arguments, polymorphic procedures on terms and arrays, and
   descriptors with their functor tables, generic `unify` and `compare`, and
   `io.write` and `io.print` as the first walker, declared in `io.m` as
   polymorphic `foreign_proc`s, and typeclass_infos, with direct calls of
   known instances (design document, Lowering, "Type classes"), and
   existentially typed predicates and functions, and existentially typed
   constructors, whose cells have their `type_info`s and `typeclass_info`s
   first (design document, Lowering, "Existentially typed constructors").
   A constrained variable whose `type_info` is only inside a constraint's
   argument's type is "not yet implemented". Cloning a polymorphic procedure for a known dictionary
   (Optimizations, "What polymorphism costs") and `pragma inline` are in no
   step yet. Types with polymorphic recursion, whose instances
   have no end, such as `grow(T) ---> stop(T) ; more(T, grow(list(T)))`, are
   "not yet implemented" too.
   Values of an abstract imported type are pointers, unified and compared
   through the descriptor of the module that defines it, which reports as
   "not yet implemented" a type exported abstractly whose layout does not
   pass as a pointer does, such as a notag float, which travels in a float
   register (design document, Lowering, "Type representation").
5. **Done:** closures. `merk.closure`, `merk.closure_word` and
   `merk.call_closure` (D7), lambda expressions, closures of named and
   built-in procedures, higher-order calls in constant stack through word
   copies, `pred/N` and `func/N` type_infos, and `io.write_list` (design
   document, Lowering, "Closures"). Left: closures of methods, tail calls through imported procedures' closures,
   `io.write_list` to a stream; the library's other higher-order
   predicates, such as `int.fold_up`, come with step 6.
6. **The library on terms.** `list`, `maybe` and `pair` compiled, and the
   array predicates compiled with `type_info`s instead of built into code
   generation; `io.command_line_arguments`, which needs lists. In five
   commits: (a) `require`, and `-fstub-unimplemented` for the library's
   build; (b) `pair` and `maybe`; (c) `list`; (d) the array predicates;
   (e) `io.command_line_arguments`. Done: (a), (b), (c), and (e), ahead of
   (d) because TCP between two nodes needs an address from the command
   line (design document, "The route there"). Part of (d) is done
   (2026-11-07), for step 3 of the route's test: `generate`, its folds and
   the foldl family are compiled from `array.m`, through the sound stub,
   whose mode numbers programs call; `CompiledArrayCall` in code
   generation lists them. The rest of (d), the other predicates that are
   not built in and then those that are, waits for
   the code generation that comes before it (below), so that `list` is
   written plainly, with every mode working, rather than around stubs;
   (c) came once clauses for one mode and nondeterminism worked, before
   the collection of all solutions, which needs it.

   *Temporary, until code generation makes every procedure of the library*
   (decided 2026-11-06): the library's build passes `-fstub-unimplemented`,
   with which a procedure whose body code generation cannot make, such as a
   nondet mode of `list.member`, gets a body that calls the runtime's
   `merk_not_implemented`, which reports it and exits with status 1, rather
   than failing the module; one without a signature code generation can
   make is left out, as no call of it can be generated either. Rejected: leaving them all out,
   which a path to one would make a link error rather than a message; and
   removing the nondet and multi modes from the library's interfaces, which
   would make programs that mmc accepts mode errors. With the option, a
   predicate without clauses is not reported either (design document,
   "No clauses"), as the library has predicates that code generation
   builds in: `float`'s and the integer modules' arithmetic, and
   `solutions.unsorted_aggregate`, `unsorted_aggregate2` and `do_while`.
   When the option goes, those need a mark that code generation implements
   them, such as a merkc pragma or a list the check consults; which is
   decided then. Also temporary:
   `require.report_lookup_error`, whose message needs the key's type name
   and a writer to the standard error, says it is not implemented.
7. **The gate.** nrev at E1's sizes on our compiler against E1's C and
   mmc, and the design document's sections on representation, the calling
   convention and the `merk` and `mrt` operations updated.

Not in chunk 1: the collector.

### Before the rest of step 6: code generation's gaps

Decided 2026-11-06: code generation's gaps come before the rest of the
library and the gate, so that the library is not written around them and
the design has longer to be tested; working around them, as stubs or
library code written for what code generation can make, would be undone
later. In this order, each its own commit or commits:

1. **Disjunctions and nondeterminism,** by continuation passing (design
   document, Lowering, "Nondeterministic code"): first disjunctions that
   bind nothing used after them, and those in committed-choice contexts,
   which become chains of if-then-elses; then nondet and multi procedures
   and calls, through success continuations that return `i1` "stop"; then
   commits, where one solution of a nondet goal is wanted, in `cc_multi`
   calls, conditions, negations and the collection of all solutions; then
   nondet lambda expressions, closures, methods and word copies. Done:
   disjunctions with at most one solution or in committed-choice contexts,
   and `cc_nondet` procedures; nondet and multi procedures, their calls,
   disjunctions, switches and if-then-elses with more than one solution,
   commits in committed choice procedures and goals with no outputs,
   conditions with more than one solution (soft cuts), and nondet and
   multi lambda expressions, closures and higher-order calls. Next, in
   lock step with the library they need (decided 2026-11-06): clauses for
   one mode (item 2) and `list` (step 6 (c)), then last call modulo
   constructor, at every optimization level, so that `list` runs on long
   lists as mmc's does (design document, Lowering, "Deterministic and
   semidet code", "Last call modulo constructor"), and the collection of
   all solutions, `merk.aggregate` and the library's `solutions` (design
   document, Lowering, "Nondeterministic code", "Collecting solutions"),
   all done, and then nondet methods and word copies of nondet procedures,
   done too, which ends this item; next is item 2. Closures of methods are
   still reported as not yet implemented.

2. **What mode analysis does not check yet,** which code generation now
   reports: clauses for one mode, trace goals, and `try` and `atomic`
   goals. Done: clauses for one mode, taken before the collection of all
   solutions (decided 2026-11-06), as `list` needs them for modes such as
   `reverse(out, in)`, and `solutions/2` needs `list`; and trace goals,
   but for their `state(M, !S)` parameters, which wait for mutables
   (design document, Mode analysis, Determinism analysis and Lowering,
   "Trace goals"). Next: `try` and `atomic` goals.
3. **Open types,** `open(T)` and `closed(T)` (below): the `any` inst,
   variable cells and the trail. After nondeterminism, as backtracking
   undoes a binding through the trail.
4. **Exceptions,** `throw` and `catch` with `invoke` and landing pads
   (design document, Lowering, "Exceptions"), which also undo through the
   trail.
5. **Polymorphism's leftovers:** calls with hidden arguments, existential
   types in methods, closures and `foreign_proc`s, closures of methods,
   and unifications of terms with `string` or array arguments.

**The `char` module** is compiled (2026-11-07), and `string-errors.m`
checks the string module's errors for NUL and surrogate code points.

Then steps 6 (d), (e) and 7; (c), `list`, came within item 1, before the
collection of all solutions. `-fstub-unimplemented` goes when the library
needs it no longer.

## Remote handles and TCP

Route step 1, "Between them" (design document, "The route there"): node 1
sends to a mailbox of node 2 over TCP, and node 2 replies to a handle that
came in the message. Decided 2026-11-07: a registry of names, and matching
types first. Erlang's distribution has decades of commercial use, so it is
followed wherever typed mailboxes do not call for something else.

**What Erlang does** (Erlang reference manual, "Distributed Erlang"; ERTS
User's Guide, "Distribution Protocol"):

- A node is named `name@host`. On each host a daemon, EPMD, on port 4369,
  maps node names to the ports where the nodes listen.
- Connections are made implicitly, the first time a node's name is used,
  and are transitive by default: a node connects to the nodes its peers
  know. Hidden nodes connect only explicitly.
- Authentication is a shared cookie, never sent: each side sends a random
  challenge and checks the other's MD5 digest of the cookie and the
  challenge. The manual calls it "not entirely safe", vulnerable to
  takeover, and points to TLS distribution. Traffic is in clear text
  otherwise.
- Registered names are local to a node. `{Name, Node} ! Msg` sends to one
  with no lookup, as a `REG_SEND` control message followed by the payload in
  the external term format. Frames have a 4-byte length after the
  handshake. `global` registers names across the cluster.

**What we take, and what differs:**

- **Lookup instead of sending to a name.** A mailbox is typed, so
  `lookup` checks the message type's fingerprint on node 2 and returns a
  handle, which sends use from then on. A mismatch is an error when the
  mailbox is obtained, not a failed decode later. Erlang needs no check, as
  any term can go anywhere.
- **An address instead of EPMD,** for now: `connect("host:port", ...)`. Node
  names and discovery come with node ids mapped to transports (design
  document, Handles, routing and places).
- **Explicit connections, not transitive.** A handle of a node this one is
  not connected to, which a third node sent, is dropped by `send` for now.
  Erlang's implicit and transitive connections need discovery first.
- **A cookie, and secrets too.** As in Erlang, a connection is accepted only
  if both sides prove they know the cookie, by answering each other's random
  challenge, and the cookie is never sent. The digest is HMAC-SHA-256 rather
  than Erlang's MD5, which its manual calls "not entirely safe". The cookie
  comes from `MERK_COOKIE`, or else `~/.merkurius.cookie`, made at random
  if it is missing, as `~/.erlang.cookie` is. A registered name is public to
  the nodes that pass, as in Erlang; any other mailbox is reached only
  through its handle's secret. TLS is a later addition to the handshake,
  which carries a version for it.
- **Ticks,** as Erlang's: a connection with nothing to send for a quarter of
  the tick time sends a tick frame, and one that hears nothing for the whole
  tick time, 60 seconds by default as Erlang's `net_ticktime`
  (`MERK_TICK_TIME`), is closed as lost. TCP alone may not notice a peer
  that vanished for hours.
- **Simultaneous connections** end with one, as Erlang's do, but decided
  differently. Erlang decides during the handshake, by the names the nodes
  connect by; our ids are learned only in the handshake, and a node that
  lost its connection must be able to make a new one before the other has
  noticed. So the node with the greater id decides, after the handshake:
  the newest connection is current, whoever started it, and it says so
  with an `up` frame, which the other answers with `ack`; each then stops
  writing on the connections decided before it, leaving any not decided
  yet, which the deciding node is about to make current (stopping those
  lost 2 of 1200 rounds of the test under load to nodes that disagreed on
  the current connection), and a message on the current one is
  delivered only once the older ones are read to the end, so the order
  between two nodes holds across the change (`runtime/node.cpp`).
- **Frames** have a 4-byte length and a kind, as Erlang's, and the
  payload's encoding is ours (design document, Wire format).

**The library** (`process`):

```mercury
:- type node.
:- pred this_node(node::out, io::di, io::uo) is det.
:- func node_of(mailbox(M)) = node.
:- pred register(string::in, mailbox(M)::in, maybe_error::out,
    io::di, io::uo) is det.
:- pred unregister(string::in, io::di, io::uo) is det.
:- pred lookup(node::in, string::in, timeout::in,
    maybe_error(mailbox(M))::out, io::di, io::uo) is det.
    % Step 3:
    % Listens on "host:port"; port 0 picks one, which is returned.
:- pred listen(string::in, maybe_error(int)::out, io::di, io::uo) is det.
:- pred connect(string::in, maybe_error(node)::out, io::di, io::uo) is det.
```

`register` is an error, as Erlang's `register/2` is, if the name is taken,
if the mailbox has a name already, or if it is closed or another node's;
closing a mailbox removes its name. `lookup` fails with an error for an
unknown name, a type mismatch, and from step 3 a lost connection or the
timeout. Until step 2's fingerprints, a local lookup compares the
type_infos.

**The handle** becomes `mailbox(NodeHigh, NodeLow, PlaceAndSlot,
Generation, Secret)`, five words: a 128-bit node id chosen at random when
the runtime starts, the place in the top 16 bits of a word whose other 48
are the slot, the generation, and a 64-bit secret chosen at random when the
mailbox is made. A local handle is a value as before (the invariant of
chunk 2). The incarnation number goes until node ids outlive a run.

**The runtime** (`runtime/node.cpp`, POSIX sockets and threads, no C++
standard library):

- **Threads.** `listen` starts a thread that accepts connections. Each
  connection has a reader thread, which decodes frames and queues each
  message in its mailbox, as a local send does. A `send` to a remote handle
  encodes in the sending thread and writes the frame under the connection's
  lock, so a full socket holds the sender back.
- **Handshake.** Magic `MERK`, version 1, the node id and a random
  challenge of 32 bytes from each side; then each sends the HMAC-SHA-256,
  under the cookie, of a label, the other's challenge, its own id and the
  other's, and checks the other's answer to its own challenge, in constant
  time. The ids in the answerer's order keep an answer from serving on
  another connection, as when an attacker relays a node's challenge back
  to it. A mismatch refuses the connection, "the other node has another
  cookie". Each side learns the other's id; a connection to
  oneself is refused, and of two connections between the same nodes one is
  kept, as above.
- **Frames.** `message` (the whole destination handle, node id included,
  then the payload), `lookup` (a request number, the name, the type's
  fingerprint), `found` (the request number, a handle or an error),
  `tick`, and `up` and `ack` for simultaneous connections. A message's header has a fixed size, so the device can write it.
- **Routing** has one entry, a destination handle and an encoded payload,
  which a remote `send` calls after encoding, a connection's reader calls
  for each message, and the device's bridge will call for the device's
  messages: a mailbox of this node's host is decoded into its queue, one of
  another node goes to that node's connection, and one of a place this node
  does not have is dropped, as for a closed mailbox. A local `send` on the
  host still copies without encoding.
- **Delivery.** A message for a closed mailbox, or with the wrong secret,
  is dropped, as a local send to a closed one is. A lost connection fails
  the lookups waiting on it and drops later sends to its node; `down`
  messages come with monitors. A frame that does not decode closes the
  connection.

**The wire format, first version:**

- **Encoding.** A value is walked by its type_info, as the copy is, into
  items of 8 bytes, little-endian: an int, of any size, or a char as its
  value, a float as its bits, a string as its length and then its bytes,
  padded to 8, an array as its length and then its elements, or, for a
  narrow array ((s2) narrow array elements), its elements' bytes, padded
  to 8, and a
  constructor, a tuple's too, as its functor number and then its arguments.
  So no value takes less than an item, which bounds an array's length by
  the bytes left. No addresses, no word size, no tag layout. Done in step 2
  (`runtime/encode.cpp`); `test/CodeGen/encode-checks.m` has golden
  encodings, which a device's encoder must give too. Rejected: varints, as Erlang's small
  integers are, which save space on small numbers but make a GPU branch on
  every element, and keep an array of numbers from being one run of bytes
  (below).
- **Decoding** walks the receiver's type_info for the mailbox, checks every
  functor number, length, integer's range, string and padding against it,
  and builds the cells in a fragment, as a local send does. The encoding
  has one form for each value, so what decodes encodes to the same bytes,
  which the unit tests check on random input. A value may nest 10000 deep,
  but for a constructor's last argument of its own type, as a list's tail,
  which is a loop; step 3's reader threads need the stack for that.
- **Fingerprints.** A type's fingerprint is SHA-256, cut to 128 bits, of a
  description of it in the encoding's items: each type constructor by its
  module (empty for the builtin types), name and arguments, and the first
  time one of constructors is met, its definition, each functor's name,
  arity and argument types, with type parameters by number, through every
  type reachable from it. Types are told apart by name, as a program may
  have more than one descriptor for one. `lookup` sends it, so functor
  numbers mean the same on both sides; it is computed at each lookup,
  which is rare, rather than cached. Type tables and
  translation by functor name, which let definitions differ, come later as
  a connection record, without changing how a value is encoded.
- **Not yet:** existentially typed constructors and closures, which the
  local copy rejects too, open terms and back-references.

**What the GPU's side needs from this** (design document, GPU processes,
Handles, routing and places, Send and receive in a target region). The
device is the first slice's sender, so the formats are made for it now:

- **Encoded once, at the source.** A message from a GPU of node 1 to node 2
  is encoded on the device, by code that generation specializes at the
  message type, and the host's bridge and every later hop forward the bytes
  without decoding them (Opaque payloads at relays). So the encoding is
  what specialized device code can write: items of fixed size, with no
  varints. The runtime's encoder, driven by type_infos, and the compiler's
  specialized one must give the same bytes, which a test compares.
- **Arrays of numbers are one run.** An `array(float)`, or an
  `array(uint8)`, encodes as its length and then its elements as they are
  in memory, on the little-endian
  machines we target, x86-64 and AMD and NVIDIA GPUs, so encoding and
  decoding it is one copy. The benchmarks' messages are such arrays, and
  later a payload in device memory goes to the NIC by RDMA with no copy at
  all, which needs it contiguous.
- **The frame names the place.** The device's bridge, or node 2's host for a
  mailbox on its GPU, routes by the handle's place, so the message frame
  carries the whole handle, and its node id too, so that frames can go
  through relays later without changing.
- **One routing entry,** above, which the bridge calls as a TCP reader does.
  Until the bridge exists, a message for a GPU place of this node is
  dropped, as one for a place it does not have.
- **Secrets and generations of the device's mailboxes.** A device has no
  source of random numbers, so the host gives a region a random key at
  launch, from which the device makes its mailboxes' secrets, by a keyed
  hash of a counter. Generations must not repeat between launches, as
  mailboxes of an ended region must stay dead (Send and receive in a target
  region), so the host keeps each place's next generation. Designed with the
  device's runtime; nothing here rules either out.
- **Handles are flat,** five words, so they map into a region `to` as any
  value, and the device compares their node and place with its own, which
  the host passes at launch.
- **The transport between device and host,** from a survey of LLVM's
  offload library (2026-11-07):
  - libc's GPU RPC, which the plan builds on (design document, GPU
    processes), is started by the device only, and carries 64 bytes per
    lane per round trip. A program can add its own operations to it, with
    `__tgt_register_rpc_callback`, but a handler runs on the host's one
    server thread, which also serves `printf` and `malloc` for every
    device, so it must not block. So RPC carries control only: "a message
    is ready", "make a mailbox", and the like.
  - Messages go through rings in host memory that both sides map, from
    `llvm_omp_target_alloc_host`: fine-grained on AMD GPUs, pinned on
    NVIDIA ones, and coherent with a running kernel under system-scope
    loads and stores with fences. A ring with one writer and one reader
    and separate head and tail needs no atomic operations across PCIe.
  - So a device place's mailbox queues live in that host memory, not in
    the device's: the host's bridge reads and writes them directly, and a
    message from the host to a device mailbox is written there with no
    RPC at all. The device polls them with `__gpu_thread_suspend` between
    looks.
  - gfx1030 has no XNACK, so there is no unified memory: everything both
    sides touch must come from that allocator.
  - Still open, for the GPU step: whether a doorbell that wakes the
    host's thread needs PCIe atomics on a discrete gfx1030, when a
    registered callback takes effect, and the throughput of one lane. The
    host can poll the rings until then.

  None of it changes the formats above: the encoding is items of fixed
  size, which a ring holds as bytes, and the routing entry is the bridge's
  as it is a TCP reader's.

**Tests:**

- Unit tests of the encoding: round trips of the descriptors' kinds, and
  hostile input (truncated, wrong functor numbers, huge lengths) rejected
  without a crash; later a fuzzer on the decoder.
- Runtime tests: two programs over loopback, started by a lit script that
  passes node 2's port to node 1, ping-pong with reply handles, a lookup
  of an unknown name and of a mismatched type, and node 2 closing its
  mailbox.
- `process-checks.m`'s model and round trips, run again with the mailbox
  on another node.

**Steps,** each its own commit:

1. Handles with node, place and secret, still local; `register` and a
   local lookup.
2. The encoding and fingerprints, with unit tests, used by nothing yet;
   SHA-256, which the fingerprints need, checked against FIPS 180-4's
   examples. Done.
3. `listen`, `connect`, `lookup` and remote sends, with ticks and
   simultaneous connections; the loopback tests. Done:
   `test/CodeGen/process-remote.m`, whose driver runs two nodes over
   loopback, and stops one with SIGSTOP to see it lost within the tick
   time, and runs twenty rounds of two nodes connecting to each other at
   once; `lookup`'s errors besides step 1's are "not connected to the
   node", "the connection to the node was lost" and "the node did not
   answer in time".
4. The cookie: HMAC-SHA-256 in the runtime, checked against RFC 4231's
   test vectors, and the challenges in the handshake, with a test
   of a node with the wrong cookie refused. Done: `HmacSha256Test` has
   RFC 4231's seven cases, and `process-remote.m` a node with another
   cookie, the cookie file made with mode 0400 and used, its first line as
   the cookie, and the errors of a file others can read, an empty one, an
   empty `MERK_COOKIE` and no `HOME`, which `listen` and `connect`
   report. The driver gives each node a temporary `HOME`. Not yet: TLS,
   which the version leaves room for.

## The device's side of the first slice

Done, on 2026-11-08: the design and why alternatives were rejected are in the
design document, "The device's communication, as built", and its numbers in
"Measurements". Comments in the code cite these steps and decisions:

- Decision 1, the transport: opcodes of our own. Decision 2, the mailbox on the
  host, later shown through a ring. Decision 3, scalar messages first, then
  arrays. Decision 4, a receive takes the first message. Decision 5, encoding
  on the host.
- Steps: 1 handles into regions; 2 send; 3 receive; 4 the first slice; 5 the
  first benchmark; 6 arrays of scalars, with the arena and the bandwidth
  benchmark; 7 the ring in host-pinned memory, the spin, and GPU to GPU.

Still open:

- **Two real machines,** two GPUs and a real network. So far two processes over
  loopback on one GPU.
- **Several GPUs in a node:** regions on them at once, through `omp.parallel` or
  Merkurius processes with the scheduler, with `device(N)`; each GPU as a place,
  with a place table and a stable identity; a path between GPUs of a node that
  does not go through the host.
- **A send that does not wait for the host,** about 12 µs of a round from the
  GPU with the wait on the ring; needs another way to keep sends in order
  (The send ring, below).
- **Streamlining a node's start-up.** `listen` and `connect` are explicit, with
  a "host:port", and the benchmarks print a port for a script to pass on. As in
  Erlang, a node could have a name, a discovery step like `epmd`, and connect on
  first use. The handle has a node id rather than an address, so this goes
  underneath `send` and `receive`, and `listen` and `connect` stay as the layer
  it is built on.
- **The second copy of a sent array** on the host.
- **From the device straight to a network card,** not designed.
- **`target nowait`** is not supported by merkc; it comes with the CPU's
  scheduler, since a region that does not block its thread needs the process
  that started it to wait for it somewhere.
- **A GPU runtime of our own,** persistent kernels with a scheduler of GPU
  processes, is to be designed.
- **Unified shared memory** for arrays (design document), and the ring's cost on
  an APU, not measured.

## The send ring

Proposed 2026-11-08; not agreed yet. A one-word send from a device is an RPC
round trip, as it waits for the host to queue the message. That is 7.5 µs at
the floor, and most of the 12 µs that a GPU costs over a CPU in a round between
nodes, with the wait on the receive ring.

**Why a send waits today:** so that a region does not end before its messages
are queued, and so that two sends are not served out of order on two ports.

**Proposed:**

- **A ring for each device, in host-pinned memory,** that the device writes and
  the host drains. One ring keeps a device's sends in order. A slot has the
  handle's five words, the type's key, the message, and a full flag written
  last. The device reserves a slot with an atomic add, as several teams may
  send; a full ring makes the sender wait.
- **A send returns when its slot is written.** That is about half of the 2 µs
  round trip through host memory, where the RPC took 7.5.
- **A drain thread on the host,** started with the first region that sends. It
  calls `merk_mailbox_send`, so the encoding and the TCP write leave the
  plugin's server thread too.
- **The drain thread spins and then sleeps,** as the host's other waits do. A
  device cannot wake a host thread directly, so the ring has an `asleep` word,
  and a sender that finds it set makes one RPC (a new opcode) to wake the
  thread.
- **The end of a region drains.** `merk_region_exit` waits until the region's
  device's ring is empty, so messages are queued when the region has ended, as
  now.
- **An array send drains first.** Arrays stay RPCs, and the callback empties the
  device's ring before it handles one, so a word and an array keep their order.

**What it costs:** a host thread for each device with sends, up to a core while
it spins. **What it gains** is not measured: perhaps 5 to 8 µs of a round for
each GPU, and fewer jobs for the one server thread. A prototype should measure
that before the rest is built.

**To decide:** whether a send may return before the message is queued. A send
to a closed or unreachable mailbox is dropped either way, so a program cannot
see the difference, apart from the order against what the device reads from
host memory by other means.

**Prototype, measured 2026-11-08, and put off.** One device's word sends through
a ring in host-pinned memory, with a drain thread that spins without sleeping.
Five alternating runs, 20000 rounds, µs a round, the median with the RPC and
with the ring: GPU and a thread of its node 21.8 and 18.2; GPU and node 2's
`main` 38.2 and 32.9; GPU and GPU between nodes 49.4 and 39.0. About 5 µs for
each GPU. The user decided to hold it until the scheduler is designed, since
that decides how the host's threads are used, and draining may become one of its
jobs.

**Measured again 2026-11-08, on a5efa6819f86, with the scheduler.** The same prototype,
nine alternating runs, 20000 rounds, median µs a round with the RPC and with
the ring: GPU and a thread of its node 13.29 and 13.48; GPU and a process of its
node 13.22 and 12.61; GPU and node 2's `main` 31.96 and 26.41; GPU and GPU
between nodes 41.39 and 36.37. So nothing within a node, and 5 µs between
nodes, 12 to 17%, with a host core spinning. Not yet decided. That only the
rounds between nodes gain suggests that what helps is the TCP write leaving the
plugin's server thread, not the ring; that is not tested. A cheaper test of it:
the server's callback hands a send to another node to a thread, with no change
on the device.

**That test, 2026-11-09, on b4d7d9366528: the TCP write is not what helps.** The
server's callback queued a word send to another node for one thread, which made
the `merk_mailbox_send` call; the device still waited for the RPC's answer. Nine
alternating runs, 20000 rounds, median µs a round, with today's path, a thread
that spins, and a thread that spins for 50 µs and then sleeps: GPU and node 2's
`main` 31.73, 30.70 and 31.00; GPU and GPU between nodes 40.75, 44.65 and 44.10.
So at most 1 µs against a CPU partner and 3.5 to 4 µs worse between GPUs, and no
reason to move the write. What the ring gains between nodes is therefore not
explained, and the ring stays undecided.

## Before the collector

What has to be done before garbage collection (design document, Memory
management, and Lowering, "GC metadata and safepoints"), and what the scheduler
left open (design document, Processes and scheduler).

**Needed first.** Decided 2026-11-08 to leave these until processes ran:

- Freed so far: when a process ends, its heap's chunks and the fragments of the
  messages it read; a mailbox that closes frees its unread messages, unless a
  device's ring may have shown them; a dropped send frees its copy. Still kept:
  everything a live process has allocated or read, which is the collector's
  job, and a monitor's copied function, an array a device gave, messages a
  device's ring took, and what a thread that runs no process allocates.

**Left open by the scheduler:**

- A compile option that leaves out what only preemption needs, for programs
  that do not spawn.
- A monitor across nodes or places reports that it is not supported yet.
- A send to another node holds its worker while its socket is full.
- Regions on several devices at once need a machine with two GPUs to run.
- About 2 µs of the cost of a receive with a timeout remains, with several
  workers, unexplained (design document, Measurements).

## Heap pointers in an address space of their own

Proposed 2026-11-08; not agreed yet. The collector's next step (design document,
Lowering, "GC metadata and safepoints"). `RewriteStatepointsForGC` finds roots by
type, so a heap pointer has to look different from the other pointers a frame
holds. Nothing runs differently: this change must not alter a program's
behaviour, and it is measured against nrev because it could change what LLVM's
optimizer does.

**Checked 2026-11-08,** with hand-written IR through `opt -O2`, the statepoint
rewrite and `llc`, on `x86_64` with `ni:1` in the data layout. All of these came
through correctly:

- A tag added by a GEP is relocated as a pointer derived from the cell. The
  rewrite rematerialises the GEP after the call.
- `llvm.ptrmask` removes a tag and tests one.
- An integer in a word by `inttoptr` is its own base, so a word may hold one.
  The rewrite's source says this is deliberate.
- A phi of a heap term and a static term, the latter an `addrspacecast` constant.
- `musttail` with `"gc-leaf-function"` stays a tail call.

Found by reading the rewrite: a non-constant `addrspacecast` to a heap pointer
is an assertion ("unsupported addrspacecast"). So nothing may be cast into the
address space at run time. The ways in are a load, a call's result, an
`inttoptr` and a constant.

**What changes:**

- **The types.** A term, an array, a string, a closure, a type_info, a
  typeclass_info and a word that may hold one of those are `ptr addrspace(1)`.
  Continuations, environments, frame slots, descriptors and the runtime's own
  records stay in address space 0. One function in lowering says which, and it
  answers address space 0 on a device: there, address space 1 means something
  else (AMDGPU's global memory), and a device has no collector.
- **Allocation.** The heap's `next` is loaded as a heap pointer, so that the
  bump is a GEP from it. `merk_alloc_slow` returns one.
- **Tags.** Adding a tag is a GEP and removing it a GEP or `llvm.ptrmask`, not a
  round trip through an integer. Reading a tag may stay `ptrtoint`, as the
  integer is not made into a pointer again.
- **Words.** `merk.word` lowers to `ptr addrspace(1)`. An integer goes in by
  `inttoptr` and a float through an `i64`.
- **Static terms.** A reference to one in code is a constant `addrspacecast`.
  Static data holds plain addresses in address space 0, which are only bits
  there.
- **Calls into C.** A foreign call, which may be inlined from the module's
  foreign code, and `strcmp`, which the optimizer knows by name, take address
  space 0, by an `addrspacecast` on the way out. A heap pointer coming back is
  an `inttoptr` of its bits. The runtime's own functions are in other objects,
  so they are declared with heap pointers and need no casts.
- **OpenMP on the host.** A region's and a loop's outlined arguments and maps
  stay in address space 0, cast at the boundary as for C, since Flang's and
  OpenMP's lowering assume it. Collection during a region is a separate
  question (below).
- **The data layout** gets `ni:1` on the host (D1). The back end insists on its
  own layout, so `EmitCode` puts that back after the optimizer.

**Built, pieces 1 to 3, 2026-11-08.** Piece 2 went into piece 1, since nothing
compiled without it. In piece 3:

- A word on the host is a heap pointer; on a device it stays an `i64`.
- A number in a word is an `inttoptr` of its bits, which the rewrite takes as its
  own base.
- An array of numbers still loads and stores its elements as integers, so that
  loops over arrays do not go through pointers. Rejected: elements as words,
  which would have put an `inttoptr` in every array loop.
- A foreign call takes and gives a word as an integer, as C's `MR_Word` is one, so
  that the call matches the definition it is linked with.

**Not in this:** statepoints themselves, `!merk.ref` as a cell and an offset, and
the frame slots.

**Pieces,** each its own commit, with nrev measured after 1 and after 3:

1. The types, allocation and tags, with words still `i64`: a word is
   `ptrtoint` of a term and `inttoptr` back, marked Temporary.
2. Static terms and the C and OpenMP boundaries, which piece 1 needs in part.
3. Words as heap pointers, which removes the Temporary conversions.

About 42 test files name a pointer in their CHECK lines, and those lines change
spelling; piece 1 is mostly that.

**To decide:**

- **D1, decided 2026-11-08: use `ni:1`.** It tells LLVM that it may not turn these pointers
  into integers on its own, as by rewriting a load and a store of a pointer as
  integer ones, which would hide a root from the rewrite. It may also stop some
  folds, which nrev would show. Proposed: use it, since a missed root is a
  memory error that shows up rarely, and a lost fold is a number that can be
  measured.
- **D2, decided 2026-11-08: a process does not collect while it is in a parallel
  or target region,** and its heap grows instead. Regions rarely allocate, what
  they use is nearly always arrays, and other parallelism is done with
  processes, which the collector handles.

**Risks:**

- **Code quality.** Vectorization and some of InstCombine treat non-integral
  pointers more carefully. Polymorphic code is the most exposed, since a float
  or an integer in a word goes through a pointer.
- **The size of piece 3.** Every place that treats a word as an integer has to be
  found, and an `i64` that remains is a root that is missed once the collector
  exists: nothing in this change would show it. The collector's stress mode has
  to cover that, by collecting at every safepoint.

## Statepoints and stack maps

Agreed 2026-11-08. The collector's next step, after heap
pointers got their address space. The design document's "GC metadata and
safepoints" settles roots in frame slots, `musttail`, holes and walking the
stack. This section adds what it leaves open.

**The gap: the type of a root.** Cells have no headers, so the collector traces
a root by its type (Memory management). The design says that type comes from the
stack maps, but a stack map says only where a value is. `RewriteStatepointsForGC`
decides what is live after the optimizer has run, when the merk types are gone.

**Proposed: each safepoint call carries a `deopt` bundle of what is live across
it, with a type before each value, made while the merk types are known.** A
backward liveness pass over the structured merk form, before lowering, annotates
each call and allocation. Lowering emits `[ "deopt"(i64 type, ptr addrspace(1)
value, ...) ]`. After the rewrite, a pass turns each statepoint's bundle into a
table that says, for each entry of `gc-live`, its type, and for a polymorphic
root, which `gc-live` entry or frame slot is its `type_info`. The values
themselves leave the bundle, so nothing extra is kept in the stack map.

Checked 2026-11-08 with hand-written IR through `opt -O2`, the rewrite and `llc`:

- Inlining joins the bundles, the caller's first, which is what the frames that
  the inlined call stands for need.
- The values in a bundle are relocated like any other live value.
- A tagged term can reach `gc-live` as its own base, so the collector takes
  tagged roots.

**Inlining needs every safepoint call to have a bundle, even an empty one.** The
inliner adds the caller's bundle only to calls that already have one (checked in
`InlineFunction.cpp` and with `opt`). A call in tail position that is inlined stops
being a tail call, and the caller's frame is live under it, so:

- A `musttail` call gets an empty bundle, `[ "deopt"() ]`, and so does any other
  safepoint call with nothing live across it.
- Leaf marking happens after the optimizer, on the calls that are still `musttail`,
  and the bundle comes off those. Marking a `musttail` before the optimizer would
  leave an inlined one with no safepoint at all, and the caller's roots lost.
- A bundle holds a `type_info` as a value, not an index, as inlining a polymorphic
  callee turns it into the caller's value or a constant. A constant is not in
  `gc-live`, so the table must be able to name a static `type_info` too.
- After the rewrite, a root in `gc-live` that no bundle names is a compile error,
  not a guess. That catches what this list has missed.

Rejected or worse:

- **An identity intrinsic, `merk.typed(value, type)`, at every definition.** Every
  load and call result would be opaque to the optimizer, on the hot paths. A
  bundle is only on calls, whose values are live anyway.
- **Metadata on loads and calls, recovered from each root's definition.** GVN and
  others drop metadata they do not know, and a moving collector cannot be
  conservative about a root with none.
- **Liveness of our own and hand-made statepoints at lowering.** LLVM's optimizer
  would then see statepoints from the start, which it handles badly.
- **Cell headers,** rejected by experiment E7.

**Which calls are safepoints.** A process's heap is collected only by that process,
so a call is a safepoint only if a collection can happen under it:

- A call of Mercury code, known by `tailcc`, direct or through a pointer.
- `merk_alloc_slow`, which is also where preemption takes effect.

Every other call is marked `"gc-leaf-function"`: foreign code, the runtime's
functions and libomp's. A runtime function that allocates, as `merk_array_new`
does, grows the heap and does not collect. Every `musttail` is a leaf too. Blocking
in a receive is safe, since other processes do not touch this heap. Nothing in C
calls Mercury except a process's entry and OpenMP's outlined regions, so the walk
stops at those.

**In regions (D2):** outlined functions get no `gc` attribute, so they are not
rewritten. A process in a region has collection turned off, and a team thread has
no process to collect.

**Pieces,** each its own commit:

1. Built 2026-11-08: `!merk.ref` as the cell's pointer and the hole's offset, with
   no runtime change. nrev did not change.
2. Built 2026-11-08: the liveness pass (`lib/CodeGen/Liveness.cpp`) and the bundles
   on calls and slow paths, still unused, so that their cost to the optimizer
   shows up alone. A `merk.live` before each safepoint holds the values, so that
   lowering's replacements follow them. It runs after last call modulo
   constructor, which looks for a call next to the cells it fills. `EmitCode`
   takes the bundles off for now. nrev did not change.
3. Built 2026-11-08, nrev did not change: the statepoint rewrite after the optimizer,
   frame pointers, leaf marking and each root's type in the stack map
   (`lib/CodeGen/Statepoints.cpp`). What it took beyond the plan:
   - The rewrite does not take an aggregate holding a heap pointer. Such a result
     crosses as i64, and the structs are split into scalars first.
   - It makes roots of a call's own arguments. One the bundle does not list is
     dead after the call, and gets index -1, which the collector ignores. It is
     not added to the bundle, which inlining would keep alive in the callee.
     What is loaded from static data gets -1 too.
   - A root the optimizer substituted for a listed one (the untagged cell for the
     tagged term, a phi over them) takes its type from the bundles of the whole
     function, through phis and relocations.
   - An integer word must not cross a safepoint as a pointer. `to_word` is opaque
     to the optimizer (`merk.word_of_bits`, an inttoptr again at the end of
     `OptimizeLLVMIR`). Rejected: `freeze`, which the optimizer moves onto the
     pointer and into aggregates; ignoring unlisted roots, which turns a missed
     real pointer into heap corruption instead of a compile error. Comparisons
     and `ptrtoint`s of pointers no bundle names read the bits at the definition.
   - Calls in OpenMP regions have the bundle `(i64 -2)` and are leaves, as an
     outlined region can be inlined into a function with safepoints. The
     collector must not run while a region does (D2), which piece 5 must enforce.
   - Programs link with `-no-pie`: the stack maps hold code addresses in a
     read-only section. `-z notext` keeps PIE, at the cost of a text relocation.
   - The table from a type's index to the type is emitted in piece 5.
   - Tail calls that LLVM found on its own are lost here, so any that matter must
     be `musttail` from lowering first.
4. Built 2026-11-08: environments in the frame as Direct locations. A statepoint's
   deopt state is the number of roots, a type's index for each, and then an index
   of a layout and an address for each environment. One listed is one the frame
   made (`merk.env_new`); one that is an argument is in a caller's frame. A call
   given an environment lists it, as the callee uses it. Slots `merk.env_set` has
   not filled hold null. Output slots are not listed: `merk.succeed` fills them and
   returns, and the caller reads them before any safepoint. Being in a bundle stops
   SROA taking apart an inlined environment; not timed.
5. In the runtime: the table built at startup, the stack walk, recording the last
   Mercury frame on calls into C, and a test mode that lists roots at a
   safepoint and checks each against its type. No collection yet.

   Split into 5a and 5b, agreed 2026-11-08.

   5a, built 2026-11-08: the table, the walk and the test mode, with monomorphic
   types (`runtime/roots.cpp`). `MERK_CHECK_ROOTS` walks the stack at each slow
   path, and checks each root against its type.
   - **D6, finding a module's table, decided 2026-11-08, changed when built.** Each
     module's table is in section `merk_root_types`, found by `__start_` and
     `__stop_`, and points at that object's own `__LLVM_StackMaps`, so link order
     does not matter and no record count is needed. A module with no statepoint has
     no table, or the symbol would be undefined.
   - **D7, the last Mercury frame, decided 2026-11-08: not recorded.** C++ enters
     Mercury only at the bottom of a stack (`CallClosure`, `merk_main`), so a walk
     follows frame pointers from `merk_alloc_slow` and ends at the first return
     address with no record. A foreign export would change this.
   - A statepoint's stack map has a pair for each relocation still used, in the
     order of `getGCRelocates`, so the type indices follow those, not `gc-live`.
   - A frame's stack pointer is computed from its frame pointer and the function's
     stack size, as a tail-calling callee may have popped arguments.
   - A function with a real bundle is Mercury code whatever its convention: main
     is inlined into `merk_main`.
   - A monomorphic term's entry is a static `type_info` built from its name. A
     word and a term with a type variable take `type_info`s (5b).

   5b: a polymorphic root's `type_info`. Built 2026-11-08.
   - MLIRGen emits `merk.root_type` notes, read and erased by `MarkSafepoints`; a
     bundle entry's header has the number of `type_info`s above the type's index,
     and they follow its value.
   - Instruction selection adds a location pair for each heap pointer in the deopt
     state that is not relocated, after the relocations, so the collector can
     update those `type_info`s. The type indices cover those pairs too.
   - A note goes in the region that defines the value. Outlining captures the
     `type_info`s with the value and notes it again in the continuation; a value
     converted to or from a word shares its note.
   - An environment's layout holds, for a polymorphic field, the positions of the
     `type_info`s among its captures. Accumulator and result slots get theirs from
     outlining (`slot_infos`).
   - `ListLiveRoots`, before the LLVM rewrite: inlining can keep a value live past a
     safepoint whose bundle does not have it. It is listed there, with `type_info`s
     from another bundle of the function. Where branches had different ones, a phi
     or select of them is made.
   - A root the rewrite adds that no bundle lists, of a polymorphic type, is a base
     kept for a pointer into the same cell. Its index is -2: the collector updates
     it and does not trace it, as the pointer into the cell is live there and says
     what the cell holds. The compiler does not check that this pointer is live; it
     follows from how the rewrite picks bases.
   - D5's compile errors, all "not yet implemented": a polymorphic root or
     environment field with no `type_info`s, a root the runtime cannot describe,
     and a live root with no type. The runtime check treats an undescribed root as
     a bad root, so no root is left unresolved.
   - Rejected: letting a missing `type_info` through and counting it at run time.
     It hid real gaps, such as the built-in `aggregate`'s accumulators.
   - These are consistency checks. Whether a description is right is tested by the
     root check, and by the collector running at every allocation.
   - **D5, naming a word's type variable, decided 2026-11-08.** MLIRGen puts the
     type variable on the operations and arguments that make a `!merk.word`, as an
     attribute, and the procedure says which value holds each variable's
     `type_info`. A missing one is a compile error. Rejected: `!merk.word<"T0">`, as
     a callee's `T0` is not its caller's, so every polymorphic call needs a cast.

The collector and the barrier, agreed 2026-11-08, each step its own commit:

1. Built 2026-11-08. A copying collector, one generation per process, with no
   barrier, and `MERK_GC_EVERY_ALLOC` to collect at every allocation at -O0 and
   -O2, or at every N-th with `MERK_GC_EVERY_ALLOC=N`, for programs too long for
   the first.
2. A nursery and an older generation, the remembered set, and the barrier on
   `merk.ref_store` and `array_set`. `bind` does not exist until open types do,
   and will use the same barrier.
3. The large object space, built 2026-11-08 ahead of step 2: an allocation with
   a block of its own is marked where it is, traced in place, and freed if
   unreached, so its address does not change. Then timing, with a call-heavy
   benchmark.

- What step 1 fixed about roots, which steps 2 and 3 rely on:
  - A stack map location is at the frame pointer in a frame that pushes call
    arguments, and at the stack pointer otherwise. Any other register is an error.
  - A value the bundle lists that is dead after the call has no relocation, but
    can be the only pointer with a type to a live cell. It gets a pair of its own,
    after the type_infos in the deopt state. The word for the number of pairs
    has the number of those above bit 32.
  - A root that is the untagged cell of a term has the term's tag in bits 56 to 58
    of its type word. A base that nothing lists is update-only, and those are
    updated after every typed root has gone.
  - A field whose type's arguments are all type parameters does not add to the
    chain of environments, or a long list costs its length squared. Polymorphic
    recursion, as `foo(T)` holding `foo(list(T))`, can still lengthen the chain.
  - A monitor's function is always copied into chunks of the monitor's, as it
    outlives the collections of the process that made it.
- **Step 2, deferred 2026-11-08.** Generations lose no correctness by waiting,
  and arrays, used destructively, stay put once large objects do. Timing will
  say when they are worth building. Barrier-needing features added before
  then (`bind`, mutables) should keep D11's rule, a type at each store. The plan
  as written: each process has a nursery, one
  contiguous block, and an older generation, the chunk list that step 1 collects.
  A minor collection copies what is reachable in the nursery into the older
  generation, with no aging. A major collection is step 1's. It runs when the
  older generation has doubled since the last major.
  - **D11, the barrier records a type.** Cells have no headers, so a remembered
    slot means nothing without the type of what it holds. The record is the
    slot, a count, the root table entry of the stored value's type, and its
    type_infos, as a safepoint has. Only the slow path uses them. Inline, the
    barrier tests that the value is in the nursery and the slot is not, two
    compares on the bounds in `merk_heap`. A store of a non-pointer type has no
    barrier. A polymorphic store with no type_infos in reach is a compile error,
    as a root is. Rejected: a type_info in each array's header, which changes
    every array's layout, the device's included, and does not help holes.
  - **D12, the same record for runtime array operations.** `fill`, `copy`,
    `resize` and `from_list` can write young pointers into an array that is not
    in the nursery. The compiler adds a range record after the call: the
    elements, their count and their type. Rejected: making the next collection
    major, which costs a major each time for arrays of terms.
  - **D13, a nursery that cannot be collected.** If it fills where no collection
    can run (no safepoint, a pinned process, an OpenMP team, or C++), the
    allocation takes an ordinary chunk and the next collection is major. Such
    a chunk's cells can hold nursery pointers no barrier saw, and the bounds
    do not recognize them as young. The end of an OpenMP team also makes the next
    collection major, as other threads' `merk_heap` is not the process's.
  - **D14, message chunks.** A minor collection leaves them alone, unpinned or
    not. A nursery cell or an older one may point into them, and a major frees
    them. Their size counts towards the next major. Rejected: treating them as
    young, as the barrier's bounds do not see them.
  - **Nursery size.** It starts small and doubles at each minor up to 1 MB, as
    the chunks do, so a process that allocates little costs little.
  - **Known cost, for step 3's timing:** each minor walks the whole stack, so
    deep recursion pays for its depth at each one. A stack watermark is the
    fix if the call-heavy benchmark shows it.

- **D8, copying first, decided 2026-11-08.** A barrier bug and a root bug both show
  as heap corruption, so the roots are proven with no barrier involved. Rejected:
  generations at once, which saves some code that is replaced but makes failures
  much harder to tell apart.
- **D9, allocation from C++, decided 2026-11-08.** It only grows the heap, as C++
  frames hold raw pointers. The next allocation from Mercury code collects.
  Array creation has a safepoint of its own before the runtime call, so that a
  loop that makes only arrays collects. A loop that allocates only in other C++,
  such as string concatenation, still does not collect until Mercury allocates.
  Rejected: registering C++ roots at each site, where one is easily missed.
- **D10, messages still in a mailbox, decided 2026-11-08.** The chunks of a message
  read but not taken are pinned until it is removed. It points only within its own
  fragment, so it needs no tracing. Values taken from it are copied out at the
  first collection after that. Rejected: the mailboxes as roots, which makes the
  collector walk the queues.
- A closure capture that cannot be described is a compile error, as the collector
  and the message copy walk a closure by its captures. There is no "unknown" mark
  any more.

**Decided:**

- **D3, the bundle, decided 2026-11-08.** As above. It is what the pieces rest on.
- **D4, how a bundle names a type, decided 2026-11-08.** Monomorphic types have a
  static descriptor, which a bundle names by an integer index into a module table,
  as a pointer would be a relocation in every stack map entry.

**Open, for piece 5: which `type_info` is a word's.** A `!merk.word` does not say
which type variable it is of, and a polymorphic term names its variable
(`list.list(T0)`) without saying which `type_info` holds it. MLIRGen knows both.
The word's type needs its variable, or each `merk.live` needs the pairing from
MLIRGen. For now a polymorphic root keeps all of the procedure's `type_info`
arguments live, which is the most the collector could need, so that the timing
of piece 2 is not too kind.

**Risks:**

- **Optimizer cost of bundles.** A call with `deopt` cannot be marked read-only or
  removed if unused. Pure Mercury calls that LLVM could delete stay. Piece 2
  measures that.
- **Lost tail calls** in piece 3, which would use stack rather than time.
- **Compile time and code size** of the table, one entry per live root per call.

## Collectives and direct GPU transfers

Proposed 2026-11-08, from a side discussion; not agreed, and nothing is built.
It adds to the design document's GPU processes (What the hardware offers;
Places say what they can do; What choosing among links adds), which already
cover moving a message between GPUs. What they do not cover is collectives.

**Checked in this tree, 2026-11-08.**

- MLIR has the `shard` dialect, with `all_reduce`, `all_gather`,
  `reduce_scatter`, `all_to_all`, `send` and `recv`, and an `mpi` dialect.
- LLVM's AMD offload plugin is built on ROCr, not HIP:
  `hsa_amd_memory_async_copy` and `hsa_amd_agents_allow_access`. Its
  `isDataExchangable` is always true, and `omp_target_memcpy` between devices
  reaches `dataExchangeImpl`.
- Our runtime has no peer mapping yet. It uses `omp_target_memcpy` only between
  host and device. Every queue is on the host.

**From the side discussion, not checked here.** To be confirmed against the
sources before anything depends on them.

- *APIs, lowest first:* ROCr; HIP peer access (`hipDeviceEnablePeerAccess`,
  `hipMemcpyPeer`, IPC handles); RCCL, with NCCL's API, over PCIe or xGMI within
  a node and InfiniBand, RoCE or TCP between nodes; rocSHMEM, put and get from
  inside a kernel, with GPU Direct Async or a host proxy between nodes;
  GPU-aware MPI.
- *Data center:* MI300X to MI355X are 8 GPUs per node, all-to-all over Infinity
  Fabric, about 1,075 GB/s per GPU on MI355X. MI455X moves GPU-to-GPU traffic to
  UALink over Ethernet. On MI250X, kernel loads and stores to a peer beat
  `hipMemcpyPeer`, as the copy engines cannot fill a multi-link connection.
- *Consumer Radeon:* PCIe only, with no bridge on current cards. Peer transfers
  need CPU-attached x16 slots and Resizable BAR, sometimes `iommu=pt`. Two R9700s
  measured about 28 GB/s one way. RCCL's peer path was reported to crash through
  the root complex on dual R9700s, and staging through the host about halves
  bandwidth. rocSHMEM lists Radeon cards as experimental.
- *MSCCL++:* four device-side primitives, put, signal, wait and flush, with the
  algorithms written above them. Reported up to 3.8 times RCCL's speed for small
  AllReduce on MI300X, after about seven weeks' work for one developer.
- *Licences:* MSCCL++ is said to be permissive; check before reusing it. The
  clean-room rule covers only Melbourne Mercury.

**RCCL's tables,** skimmed on GitHub 2026-11-08 (`ROCm/rccl`, whose default
branch is `develop_deprecated`, so development may have moved). `LICENSE.txt` is
BSD-style, with NVIDIA's, AMD's and an MIT-licensed part of Microsoft's;
`src/graph/rome_models.cc` has an MIT-style notice. Reuse keeps the notices.

- `src/graph/tuning.cc` has 8 tuning models: latency by hardware, algorithm and
  protocol (`hwLat`), `bwRatio`, tree and ring correction factors by the log of
  the message's size, and thresholds for protocols and channel counts. They
  calibrate RCCL's cost model for its own kernels and protocols, so the numbers
  do not carry over. The shape of the model does, and is close to the route
  table's lookup by source, destination and size class. The numbers can serve as
  a rough check on where our own crossovers fall.
- `src/graph/rome_models.cc` has known server layouts. They describe hardware
  rather than RCCL's speed, so they may be of use, for the servers AMD listed.
- `ext-tuner/example` has a tuner configuration for gfx950 and a script that works
  from performance data. Ours would likewise be measured on each target machine
  and generated.

**The proposal.** Build on what OpenMP offload and ROCr provide, and add what
RCCL would have provided, with no dependency on RCCL. It stays the benchmark to
beat. Most of RCCL is topology handling, tuning, the network path and robustness,
and most of that is either expressed by processes already or not needed yet.

- **Logical topology is the program's.** Who talks to whom, who holds which tile
  and who takes part are mailboxes and places. A group is a list of mailboxes,
  and a position in it gives what a rank gives, including the order that
  operators that do not commute need. A participant's end is seen by monitors.
- **Physical topology is a query.** Which GPUs share a link is a fact of the
  hardware, which fills in the place table's "which GPUs it can reach, over which
  link". ROCr is believed to report the link's type and bandwidth for a pair of
  agents; not checked.
- **Algorithms are chosen from few cases:** direct exchange over all-to-all
  Infinity Fabric, a ring or tree over PCIe, and within a node first, then between
  nodes. The operator and element type are known at compile time. The size may
  not be, so the choice is partly at run time, by size class, from a table
  measured on each target machine.
- **Processes plan, kernels carry the data.** A process message between GPUs costs
  about 40 µs today, through the host. Collectives over messages are the correct
  first version; the fast one has the processes set up the pattern, and kernels
  do peer stores.
- **What is added below OpenMP is small:** peer-mapped buffers, signal and wait
  with system-scope ordering, and the filled-in place table. OpenMP has no
  collectives across devices, and does not appear to expose mapping one GPU's
  memory for another's kernels. The plugin calls `hsa_amd_agents_allow_access`
  internally, so this needs ROCr called directly or a small plugin extension.
- **Collectives are in the language and in `merk`.** A parallel loop across devices
  can do its cross-GPU reduction in its own kernel, as tiles finish, which no
  library can do. Uniqueness makes in-place safe.
- **The network path is later, and is ours.** Without RCCL there is no fallback to
  it: the design's plan is libfabric, UCX or verbs under the node id, with TCP.

**What the first version of this proposal got wrong, and limits.**

1. *A collective on one participant is not pure.* `all_reduce(sum, A0, A)` with
   `A0::di` and `A::uo` returns something that depends on other GPUs' data, so
   it is not a function of `A0`, and `promise_pure` would be false. Either it
   threads `io`, as `send` and `receive` do, or it is an operation over the whole
   distributed array, inside a loop across devices, whose sequential meaning is
   an ordinary reduction. Only the second can be pure, and it is also the one
   that can be fused.
2. *Lowering to RCCL first was rejected* (the side discussion's first step). RCCL
   is a HIP library addressed by rank, and we have neither: our regions are OpenMP
   offload over ROCr, and programs name mailboxes, not ranks. Buffers from
   `omp_target_alloc` would have to be usable from HIP, which is unchecked.
   Collectives on messages over today's route are slower, but correct, need no
   new dependency, can be tested on this machine, and get faster when the route
   does.
3. *Use ROCr rather than HIP for peer access.* We already sit on ROCr through the
   plugin, and it is the layer that HIP's peer access is built on. Whether being
   one OS process avoids the crash reported on dual R9700s is a guess.
4. *The primitives are the route table's "peer store" hop.* Put, signal, wait and
   flush are what a mailbox ring in a peer's memory needs too (What choosing among
   links adds: where the queue lives, several senders, order, size). So build one
   transport, for messages first, as GPU-to-GPU messaging is the goal, and
   collectives on it (One message transport, with backends, below).
5. *Fusion needs loops across devices,* which do not exist. Regions run on one
   device each, and `device(N)` regions at once are scheduler step 8.
6. *A participant can die.* The design's recovery benchmark kills a GPU process
   during a halo exchange. A collective must end with an error rather than hang,
   which monitors allow and rank-based libraries do not offer.
7. *`shard` is built around sharded tensors,* and has no route to AMD peer
   memory. Worth a look as a level below `merk`, not to be assumed.

**Suggested order, for discussion.**

1. Collectives over messages, as library code, once GPU processes run with the
   scheduler (step 8). The halo exchange is the first user.
2. Peer-mapped rings between GPUs on one node, as the route's peer-store hop.
3. Loops across devices, and collectives as part of them, with fusion.
4. Compare with RCCL on real hardware, with a table measured there. Only then
   decide what to do about the network path.

**What we can test.** This machine has one RX 6950 XT, so no peer transfer can
run here. Steps 2 to 4 need at least two GPUs: two consumer cards for PCIe, or an
Instinct node for Infinity Fabric.

## One message transport, with backends

Proposed 2026-11-10, from a discussion; not agreed, and nothing is built. The
design document's route table has five kinds of hop (GPU processes, Places say
what they can do), and Collectives and direct GPU transfers, above, says to
build one transport for messages and put collectives on it. This makes that
concrete: one interface that every hop implements, a backend for each, and a
table measured on each machine that picks among them. No library is chosen once
for the whole program. RCCL's device API becomes one candidate backend among
several.

**RCCL's GPU-initiated networking** (RCCL 2.30 documentation, "Device API:
GPU-initiated networking", read 2026-11-10; not tried here).

- It is NCCL 2.30.3's device API, and marked experimental. A kernel holds an
  `ncclDevComm` and builds an `ncclGin` on one of its contexts.
- `gin.put(team, peer, dstWindow, dstOffset, srcWindow, srcOffset, bytes,
  ncclGin_SignalInc{n})` writes into the peer's window and increments a signal
  there. `gin.flush(ncclCoopCta())` makes the source reusable, but does not mean
  the peer can see the data: the peer waits on the signal for that.
  `ncclGinBarrierSession` is a barrier with a timeout.
- The host allocates with `ncclMemAlloc`, registers windows with
  `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`, and makes the device
  communicator with `ncclDevCommCreate`. All three are collective.
- On AMD the only validated backend is a host proxy (`NCCL_GIN_TYPE=2`): the
  kernel queues the operation and a CPU thread posts it to an InfiniBand or RoCE
  NIC. That is the same shape as our device's RPC to its host, so it brings
  RCCL's network handling but no lower latency than a proxy of our own.
- It needs a NIC and driver that register GPU memory by DMA-BUF. Windows are
  single-segment. `ncclDevComm`'s layout is versioned, so device code is rebuilt
  for each RCCL release.
- It is HIP C++ (`ncclCoopCta`, `cuda::memory_order_acq_rel`). We would call it
  through a shim of `extern "C"` device functions compiled to AMDGPU bitcode and
  linked into the device image, as `device.cpp` is. Its host side needs the HIP
  runtime in the same process as liboffload. Both sit on ROCr; untried.
- Not on that page: how a receiver waits on a signal or counter, whether the
  device side is header-only, and the load-store API within a node ("LSA").

**The interface.** MSCCL++, rocSHMEM and GIN agree on four device-side
primitives: put, signal, wait and flush. A mailbox needs two more, a slot to put
into and a way to give it back. So, between a sender and a mailbox's ring at the
receiver:

- **reserve** a slot, atomically when there can be several senders;
- **put** the message's bytes into it: a store, a DMA copy or an RDMA write;
- **signal** the receiver, ordered after the put;
- **wait**, at the receiver, until a counter reaches a value, or a deadline
  passes, which `receive`'s `after(T)` needs;
- **flush**, at the sender, until the source can be reused;
- **credit** back a slot, from the receiver, as a signal in the other direction,
  so that a sender learns of free slots without reading remote memory. Today's
  rings have their own flow control; this replaces it with one scheme for all.

`send` is reserve, put and signal. `receive` is wait, read and credit.

**Messages of variable size.** A whole array, a string and a packed message
(flat records, above) have a size known only when they are sent. So a ring's
entries are of variable length: reserve takes a number of bytes, not one slot.
A message above a threshold goes by rendezvous instead: its bytes are put into
a buffer at the receiver, and only their place and size go through the ring,
as the design document's size classes have it (What choosing among links adds).
Fixed-size slots are then an optimisation for a message whose size the compiler
knows. The host
keeps the control path, as the design document has it: it opens a route between
a sender's place and a mailbox, registers memory and exchanges keys, closes it,
and tells the mailbox's monitors when a peer has gone.

**Backends,** by the route table's hops, with what can run here:

1. *The host's queue,* within a place's host process. Built.
2. *Between a host and its devices,* the RPC and the rings in pinned memory.
   Built.
3. *Peer stores between GPUs of a node,* with ROCr's
   `hsa_amd_agents_allow_access`. Needs two GPUs.
4. *Host-posted RDMA between nodes,* over verbs or libfabric, with the GPU's
   memory registered with the NIC. Needs machines with RDMA NICs.
5. *GPU-posted RDMA,* the GPU ringing the NIC's doorbell itself, as rocSHMEM's
   GDA backend does. Needs a NIC that supports it.
6. *RCCL's GIN,* through the shim above. A proxy today, perhaps GPU-posted
   later. Needs RDMA NICs.
7. *TCP,* the fallback between nodes (Remote handles and TCP). Built for the
   host.

On the device, a mailbox handle's route is fixed when the device first uses it,
and compiled code calls one entry per primitive, which dispatches on it. Where
the specializer knows the place at compile time, the call goes straight to the
backend. On the host, a backend is a table of functions, as the runtime has no
C++ standard library.

**Where this can do more than RCCL.** The user's point: a framework of our own
can be much more flexible.

- *Mailboxes, not ranks.* RCCL's communicators and windows are made
  collectively, by every rank at once, and membership is fixed. Here a route is
  opened on a pair's first send, processes come and go, and no step needs
  everyone.
- *A route per pair, not per communicator.* One program can use peer stores
  within a node, RDMA between nodes and the host's queue for CPU processes, each
  pair on its fastest link.
- *Types known at compile time.* GPU messages are flat, so most sends have a
  size known at compile time. The choice between a store into the ring and a DMA
  copy can then be made by the compiler, and a ring's slots can be exactly the
  message's size. RCCL chooses its protocol at run time from the byte count.
- *Communication fused with computation.* A loop across devices can send a
  tile's part of a reduction as the tile finishes (Collectives and direct GPU
  transfers). A library called between kernels cannot.
- *Failure is a message.* A peer that dies ends the waits on it with an error,
  through monitors. GIN's barrier has only a timeout.
- *CPU and GPU processes alike.* Both send to the same mailboxes. RCCL is for
  GPUs only.
- *No lock-in.* GIN's version coupling and HIP dependency stay inside its
  backend. Where it is fastest, the measured table picks it; where it is not,
  nothing else changes.
- *Balancing and stealing* (Distributing work, below) use the same messages.

**What it costs.** We own the hard parts that RCCL has already done: ordering at
system scope across PCIe and the NIC, flow control, and robustness on many NICs.
Order between one sender and one mailbox holds only if they keep one route,
which the design document already requires (What choosing among links adds).
Each backend is a set of tests on hardware we may not have.

**Suggested order, for discussion.**

1. Define the interface, and move backends 1, 2 and 7 behind it with no change
   in behaviour. Check that messages are no slower, with the benchmarks in
   Measurements.
2. Credits, and reserving slots for several senders, in the rings. Today's rings
   have one producer; peer stores will need several. The host and the device as
   two senders to one ring can test it here.
3. Calls straight to a backend where the place is known at compile time.
4. Peer stores within a node (backend 3), once there are two GPUs.
5. Between nodes: a host proxy over verbs first (backend 4), then GIN's shim and
   GPU-posted RDMA (5, 6), measured side by side against RCCL's collectives.
   Waits for the machines, as the halo exchange does.
6. Collectives on the transport (Collectives and direct GPU transfers, steps 1
   and 2).

**Open questions.**

- GIN: the receiver's wait, whether the device side is header-only, and LSA.
- Whether GIN can register memory that liboffload allocated, or only memory from
  `ncclMemAlloc`. If only the latter, a mailbox's ring would be allocated by
  RCCL when its route is GIN.
- Which NICs rocSHMEM's GPU-posted backend supports outside Instinct systems.
- Whether one OS process running HIP and liboffload side by side has problems on
  ROCr.

## Distributing work: balancing and stealing across GPUs

Proposed 2026-11-10, from a discussion; not agreed, and nothing is built. It
waits for messages between GPU processes (GPU processes without `omp target`,
step 5 and after) and adds to Collectives and direct GPU transfers, which
cover the operations whose shape is known in advance.

**Why.** Collectives move data whose split is fixed before they start. Much of
the work that the goal serves is not split that way. Running large language
models is the main case:
- **Serving** takes requests that arrive at any time, with prompts and outputs
  of very different lengths. Servers batch continuously, adding and removing
  requests at each step, and newer ones run the prompt on some GPUs and the
  generation on others, moving each request's KV cache between them, as the
  scheduler decides at run time.
- **Mixture of experts** routes each token to the experts that its router
  picks, so the all-to-all carries uneven amounts, and the GPU of an expert
  in demand holds the others up. Systems balance it by copying busy experts,
  limiting each expert's tokens, or adding a balancing term to the loss.
- **Training** splits evenly by design, but sequences of different lengths,
  pipeline bubbles and slow GPUs leave some waiting at each step.
- **Within a GPU** the same idea is persistent kernels that take tiles from
  a shared counter, Stream-K matrix multiplication, and split-KV attention.

**What RCCL gives,** from the discussion, not checked against its sources:
collectives and grouped `send`/`recv`, NCCL's API, queued on streams from the
host. Every rank calls the same collectives in the same order, and a `send`
needs a `recv` that knows its size. So it has no work stealing and no
balancing: an idle GPU cannot ask for work or take from another's queue.
That needs writes and atomics on another GPU's memory, started from a
kernel, which NVSHMEM, rocSHMEM and MSCCL++ (put, signal and wait) give.
Serving systems move KV caches with libraries of their own (NIXL, Mooncake's
transfer engine, UCX), and DeepSeek's mixture-of-experts inference uses
all-to-all kernels that the GPU starts (DeepEP). Recent NCCL adds calls
that a kernel makes, and RCCL 2.30 has them too, experimentally (One message
transport, with backends, above).

**What we have.** The host's scheduler already steals between its workers
(design document, How the scheduler was built). Mailboxes are asynchronous,
and a receiver need not know what it is sent, which is the shape that
dynamic work needs, as Erlang's pools of workers use. Mercury's own parallel
conjunctions steal work too (Bone's thesis, a paper we may read).

**Proposed.** Two layers, the second this section's:
1. **Collectives** for what is split in advance: all-reduce and all-gather
   for tensor parallelism, and all-to-all with uneven sizes for experts
   (Collectives and direct GPU transfers).
2. **Work that moves at run time,** built from processes and mailboxes:
   - **Pulling from a pool first.** A dispatcher process holds the work, and
     each worker, on the host or on a GPU, asks it for more when it is
     idle. It needs only messages, so it works as soon as GPU processes
     can message each other and the host.
   - **Stealing between GPUs later.** Each GPU keeps its queue in its own
     memory, and an idle GPU takes from another's with an atomic on that
     queue, over xGMI or PCIe between peers, or an RDMA atomic between
     hosts. A ring on the receiving GPU (The send ring) is close to such a
     queue already, once peers can write to it.
   - **Balancing as a library.** Pools, dispatchers and stealing are library
     code over processes and mailboxes, not the runtime's, so a program can
     choose its policy.
   - **Data shared, not copied, within a GPU:** a pool's workers on one GPU
     read their parts of a shared array in place, counted, and send back
     results; across places without shared memory, the parts are copied and
     sent (design document, "Messages and memory on the device").

**Steps, for discussion.**
1. **A pool on one GPU, as the first measure of GPU messaging.** A host
   dispatcher and GPU worker processes on this machine's one GPU, with work
   items of uneven cost, measured against a static split. It serves
   handling a single GPU well first.
2. **Workers that are GPUs' teams,** taking items from a counter in device
   memory, within one kernel, measured against a static split of the teams.
3. **Pulling between GPUs,** when messages go between GPU processes of
   different GPUs, and then of different hosts.
4. **Stealing between GPUs,** when peers can write to each other's memory.
5. **Mixture of experts and moving KV caches** as the tests of the first
   four, with an all-to-all of uneven sizes.

**Open.**
- Whether a pool's dispatcher may run on a GPU, which needs messages
  between GPU processes, or stays on the host.
- How much of it the language says: a `parallel` library over processes,
  or directives as OpenMP's loops have (Data parallelism, and a
  `parallel` library, in GPU processes without `omp target`).
- What this machine can measure: one GPU, so stealing between GPUs waits
  for the user's other machines, as the halo exchange does.
- On a GPU, the workers are waves and teams of the one kernel of a phase's
  processes (GPU processes that run loops, decided 2026-11-11), which take
  work without the host and can move between the phase's processes.

## Choosing among the ways: experiments for GPU code

Written 2026-11-10 at the user's request: there are several ways of doing
each thing, and which are worth having is not known. This lists them by the
question each answers, and orders the experiments that decide them, cheapest
and most deciding first. Each experiment says what it measures, against
what, and what result chooses what. Results go in the design document's
Measurements, with date and commit; decisions here, with the numbers that
made them, and then to the design document as usual. The bar is code
written by hand in HIP or C, not our own earlier versions.

**This machine** (checked 2026-11-10): one RX 6950 XT (gfx1030, wave32, 16
GB, a 128 MB Infinity Cache), which drives the display; ROCm 7.2's HIP and
ROCr are installed (`hipcc`, `libamdhip64`, `libhsa-runtime64`); BAR 0 is
16 GiB, the whole of the card's memory, so resizable BAR is on and the host
can write into all of the GPU's memory. No peer GPU, no xGMI, no second
node: what needs them waits, as the halo exchange does. Every run is small
first, with a timeout, and the user is told before benchmarks, so that
they can close what else uses the GPU.

### The ways, by question

- **Q1. How a GPU process computes in parallel.**
  - *One thread* of one team: as built. A sliver of one compute unit.
  - *OpenMP within its team:* the process's kernel is a generic-mode
    OpenMP kernel (`__kmpc_target_init`, its `KernelEnvironmentTy`, and the
    launch environment's argument, which `libompaccsupport/device.cpp`
    makes for libomptarget and we would make), with OpenMP's device runtime
    linked into the image. `parallel` and worksharing loops across the
    team's threads.
  - *Our own SPMD kernel* without OpenMP's runtime: every thread runs the
    process, with the sequential parts guarded.
  - *A pool:* one persistent cooperative kernel
    (`OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE`) that fills the GPU.
    Processes run in some of its teams or waves; a parallel loop is posted
    as a work item that the worker teams take in chunks from a counter,
    each chunk run with OpenMP's `parallel for` within the team. Processes
    share the teams. Work functions are outlined and dispatched by a switch
    over the image's known ones, not by pointer.
  - *Regions for the process:* the host launches a target region for each
    parallel loop that the process asks for, by RPC. About 27 µs a region
    as measured.
- **Q2. Where processes run.** A kernel and a queue for each, as built; many
  processes in one persistent kernel (the pool); processes as the teams of
  one region (Processes as teams, below).
- **Q3. How a message reaches a GPU process from the host** (5c). Its ring in
  host-pinned memory with an RPC for what is not one word, as built; the
  host writing into a queue in the GPU's memory through the large BAR; a
  DMA for each message into that queue; slots in host memory that the GPU
  copies from once.
- **Q4. How a message leaves a GPU process for the host.** libc's RPC, as
  built; the GPU writing into a queue in host-pinned memory with
  system-scope atomics, which the host polls.
- **Q5. Between processes on one GPU** (5d). Through the host, as built; a
  queue in device memory, reserved by an atomic on its tail.
- **Q6. How a GPU process waits.** Polling with `s_sleep`, as built, with
  other backoffs; in a pool, a worker that waits gives its team to other
  work.
- **Q7. How work is split** (Distributing work). A static split; chunks
  from an atomic counter within a GPU; a dispatcher process that hands out
  items by message; stealing between GPUs, later.
- **Q8. How data is shared.** Copied by value (`send`, `send_slice`,
  `send_tile`); a shared array, counted, read in place; a unique array
  handed over whole within a GPU.
- **Q9. How a tile is copied.** A rectangular DMA (ROCr's, below
  liboffload); a strided copy kernel; a layout stored tile by tile.
- **Q10. How fast our kernels are** (Specialized kernels for loops). The
  `omp` lowering as built, against C and HIP, at sizes beyond the Infinity
  Cache, which the measurements of 2026-11-04 did not pass.

### The experiments, in order

**E1. Baselines and floors, in HIP and C, with no compiler change** (Q3 to
Q6, Q10). Small programs under `benchmarks/gpu-floors/`, each a number that
our runtime is later held to:
- (a) A word from the host to a kernel and back: through pinned host
  memory (2.0 µs on 2026-11-08), through device memory written over the
  large BAR, and by a DMA for each message (`hipMemcpyAsync` of 8 bytes).
  Then 64 B, 4 KB and 1 MB messages each way.
- (b) Two workgroups of one kernel, and two kernels on two queues, passing
  a word through device memory: the floor for Q5.
- (c) Launching an empty kernel through liboffload and through HIP, and
  through `omp target`: what a kernel or queue per process costs (Q2).
- (d) BabelStream in HIP, and ours and C's with OpenMP, at 2^25 and 2^27
  elements, beyond the cache.
- (e) A persistent cooperative kernel of all the teams that fit: the
  latency from a post in device memory to a worker team taking it, and
  chunks taken per second from one atomic counter by all teams.

Decides: Q3 by the latency and the host's CPU time at each size, the large
BAR if it is within 20% of the best for small messages; Q4 likewise; Q5's
floor; whether a pool's post is ten times cheaper than a launch (Q1, Q2).

**E2. OpenMP within a GPU process's team** (Q1). A process kernel in C,
compiled as a generic-mode OpenMP kernel, launched through liboffload with a
launch environment we make, running `parallel for` over an array: it
works or it does not, and its parallel region's overhead against the same
in a target region, and against one thread. Small, and needed whatever the
pool decides. Decides: whether process kernels become OpenMP kernels.

**E3. The pool against a launch per loop** (Q1, Q2, Q6). A pool in C or HIP,
with two processes posting loops to worker teams, against launching a
kernel for each loop, over loops of 10^3 to 10^8 iterations, with the
registers of the largest work function varied to see occupancy fall.
Decides: the pool, if it beats a launch per loop by five times or more
for loops that take under 100 µs and costs under 5% for long ones;
otherwise a launch per loop, with OpenMP within the team.

**E4. Splitting work** (Q7). Synthetic loads in the pool of E3, or in a
region if E3 rejects the pool: costs that are even, that follow a power law,
and token counts routed as a mixture of experts routes them. A static
split, chunks from a counter at several chunk sizes, and a dispatcher by
message. Measures the time to finish, against a perfect split. Decides:
counters within a GPU and their chunk size; the smallest item that a
dispatcher by message should hand out.

**E5. Sharing against copying** (Q8, Q9). A reduction of a large array by N
workers on one GPU, each sending back a scalar: copying each worker's part,
against reading a shared array in place, whose count costs two atomics a
worker. A tile of a matrix sent by a rectangular DMA, by a strided copy
kernel, and packed by the program, at several shapes. Decides: whether
shared arrays come before `send_tile`, and how tiles are copied.

**E6. Across GPUs and nodes** (Q5 between GPUs, Q7 stealing, Q8 peers,
collectives against RCCL). Waits for the user's machines with several GPUs;
until then nothing here is chosen.

**What comes first in code meanwhile.** 5a (the checks by type) and 5b
(the place table) depend on none of these. 5c waits for E1, and the pool
for E3. E2 comes before OpenMP is put in process kernels. The experiments
are run in this order, and a result may reorder what follows.

## Processes as teams, and messages within a GPU

Proposed 2026-11-09; steps 1 to 5 below are built. The design document
(The device runtime, "Who communicates") has a process for each team in an SPMD
kernel as a later step. This adds messages between such processes that do not go
through the host.

**Why.** Separating a step's communication from its computation with several
target regions, so that several teams can compute, goes back through the host
for each launch. A kernel whose teams are processes stays running. And a halo
exchange between teams needs no second GPU to develop.

**Proposed.**

- A kernel starts N teams. Each is a process, with its own mailbox and arena.
- A handle has node, place and slot; the team is the slot, so `send` chooses the
  route as it does now.
- **A send within a GPU** writes to a ring in device memory, which the receiver
  polls, using device atomics and no call of the host.
- **An array within a GPU moves by its pointer.** A send hands over a unique
  array, and both teams can address it, so nothing is copied.
- One thread of each team communicates. The others wait in the state machine or
  at a barrier until there is parallel work, with `masked` and a barrier for a
  message the whole team reads. Those bind to the team; nothing in OpenMP
  synchronizes between teams, apart from the end of the teams region.

**Decided 2026-11-09: a teams region first.** A program starts the processes with
`%$omp target teams num_teams(N)` before a goal that threads `io`; each team runs
the goal as a process. Rejected for now: spawning by a `:- process` declaration's
kind name, which the design document has for GPU processes, as those declarations
need semantic analysis first; spawn comes later (below).

```mercury
%$omp target teams num_teams(4)
worker(!IO)

worker(!IO) :-
    team_number(Me, !IO),
    number_of_teams(N, !IO),
    team_mailbox((Me + 1) mod N, Next, !IO),
    send(Next, token(Me), !IO),
    team_mailbox(Me, Mine, !IO),
    receive(Mine, any, after(1000), Got, !IO),
    ...
```

- **As Flang lowers `target teams`:** a generic kernel, `omp.teams` in
  `omp.target`, its `num_teams` a `host_eval` value on the host and read from its
  map on the device. A generic kernel runs the goal on each team's initial thread,
  so the first version needs no `masked` or barrier.
- **`num_teams(N)` asks; the program asks back.** As in OpenMP, a program may get
  fewer teams, so it uses `number_of_teams`, not N.
- **Its sequential meaning** is OpenMP's: without `-fopenmp`, the goal runs once,
  as team 0 of 1. A ring of `number_of_teams` teams is then one team sending to
  itself. With `-fopenmp` and no device, libomp runs the teams as host threads.
- **A teams region binds nothing used after it,** as each team would bind it. Its
  results leave in messages or output.
- **Team mailboxes.** `team_mailbox(T, Box, !IO)` gives team T's mailbox for Box's
  message type, in the current teams region: one for each team and type, made when
  first used, in the region's memory, living as long as the region. Only the teams
  of the region send to them at first; a send from elsewhere is reported as not
  yet implemented.

**Steps.**

1. `target teams` with `num_teams` (an integer, or a variable bound before), and
   `team_number` and `number_of_teams`, on the host, sequentially and on a GPU.
   Implemented 2026-11-09 (`CodeGen/omp-target-teams.m`): with OpenMP the two
   become `omp_get_team_num` and `omp_get_num_teams`, on the host and in the
   device's runtime (the library's C since 2026-11-10: design document,
   Library code on the device); a number of teams that is not from 1 to 2^31 - 1 is a
   compile-time error for a literal and a run-time error before the region for
   a variable; `thread_limit` is not yet implemented. A variable used only in
   the clause is not bound, as for `device` (mmc reads a clause as a comment).
2. Team mailboxes for one-word messages: a ring for each team and type, which any
   team appends to with an atomic add and its team takes from, with no call of the
   host. The same code in merkrt for the host's teams and the sequential program.
   Implemented 2026-11-09 (`CodeGen/omp-target-team-mailbox.m`, and its
   `-gpu.test` at -O0 and -O2; also right with 64 teams, by hand):
   - **On a device**, each (team, message type) has a ring of 256 slots in the
     device's arena, made when first used. A slot holds a sequence and a message,
     as in Vyukov's bounded queue. A sender takes a position with an atomic add
     on the tail, and writes the slot if its sequence says it is free.
   - **A send never waits.** If the slot is still full, the sender pushes the
     position and the message onto a lock-free list for that slot, and the
     receiver takes the position from there when it reaches it. So the order of
     positions is kept, and a host mailbox has no bound either. Rejected: *a
     send that waits for a free slot.* The first GPU run showed why: team 0 sent
     300 messages to itself before receiving any, and waited forever. Any
     program in which a team sends more than a ring holds before it receives
     would deadlock, and the bound would be a property of the device, not of
     the program.
   - **Only the owning team receives,** which the device checks: a receive from
     another team's mailbox ends the program with an error. The host does not
     check it.
   - **Regions.** The host gives each region an id. The first team to enter
     makes the region's table of rings in the arena, with a compare and swap, and
     each team keeps the table in LDS. A device handle names the team in its
     slot, with place `kTeamsPlace`, and the region by its id. Using it in
     another region is an error. Calling `team_mailbox` in a target region
     without teams is not yet implemented. Such a region says it has no teams
     at its start, since LDS starts undefined. Only modules that use team
     mailboxes do so, which holds while a region on a device calls only its
     own module's procedures.
   - **On the host and sequentially,** a team mailbox is an ordinary mailbox with
     no owner, from a table for the region that is closed when the region ends.
     Outside a region there is one table, of team 0 of 1, for the whole program.
   - **Errors on the device** reach the host by a new RPC opcode, `rpcFail`. The
     host reports them as it reports the same error on the host, and ends the
     program.
   - **Not yet implemented,** reported when compiled or run
     (`CodeGen/omp-target-team-mailbox-errors.m`): messages that are not one
     word or an array of them, and sends from the host or another node. A
     team handle cannot leave the device, since a device cannot yet send a
     mailbox in a message.
   - **An LLVM bug** showed up here first, at -O2 on the GPU, and its fix is
     a separate commit. OpenMPOpt's heap-to-stack turned the receive's
     out-slot into an alloca but kept the call's `tail` marker. LICM then
     hoisted the slot's load out of the receive loop, so every message read
     as the first. Upstream fixed it in llvm/llvm-project#231611 and #239059,
     which we applied ahead of a rebase.
3. Arrays between teams by their pointer. Implemented 2026-11-09
   (`CodeGen/omp-target-team-arrays.m`, and its `-gpu.test`;
   also right with 64 teams, by hand):
   - **On a device**, a team mailbox of arrays of one-word elements holds the
     arrays' pointers in its ring, as a mailbox of words holds words. Every team
     of the region can address an array, whether made in the arena or mapped
     for the region, and the arena lasts as long as the region. So the region
     owns the arrays, which answers the question of an owner within a region.
     An array that must outlive its region is still open (below).
   - **Sharing keeps the meaning of a copy.** `send` takes its message with mode
     `in`, so the sender's array is shared after the send, and `receive` gives
     a shared array. Neither team can then update it in place: a team that
     wants to change it copies it first, as on the host. So the program means
     the same on the host, where team mailboxes copy, and sequentially.
   - **Not the hand-over of a unique array** that "Proposed" has, which would
     let the receiver update the array in place. That needs a send whose
     message has mode `di`, with a receive that gives a unique array, and
     belongs with the pieces of one array (below).
4. A check at launch that the teams fit on the GPU at once (risk 1), reported as an
   error, since a team that is not running deadlocks the others. Implemented
   2026-11-09 (`CodeGen/omp-target-teams-wait.m`,
   `omp-target-teams-too-many-gpu.test`), as a start barrier, which the user
   chose:
   - **The teams of a region whose code uses team mailboxes wait until all
     have started.** Each team adds itself to a count in the region's table and
     waits, with `s_sleep`, until the count is the number of teams. Teams that
     fit start within microseconds. If no more have started for the stall
     time, ten seconds or `MERK_TEAMS_STALL_MILLISECONDS`, one team reports
     "Mercury runtime: target teams: only M of the N teams ran at once on the
     device, …" and the program ends. (A second from each team's start at
     first, which a full test run broke: design document.)
   - **Only regions with team mailboxes wait.** Code uses them if it calls
     `team_mailbox`, or refers by a symbol to code of the module that does,
     at any depth. Other teams regions may still run more teams than fit, a few
     at a time: 2000 teams without team mailboxes run. The host's teams are
     threads, which all run, so the host does not wait.
   - **Measured, 2026-11-09, RX 6950 XT.** A generic kernel's teams have 256
     threads, 8 waves. With 124 VGPRs, 157 of 400 teams ran at once. A rough
     estimate from the kernel's registers alone gives about 160, so something
     else, probably the display, held a few slots. A smaller kernel fit 197 of
     2000. The token ring experiment above, with almost no registers, fit 320
     at 8 waves.
   - Rejected: *a table for each GPU,* since how many teams fit depends on the
     kernel's registers and LDS as much as on the GPU. *An occupancy estimate in
     the offload plugin before launch,* from the kernel's metadata and the
     device's compute units: it gives an error before launch, but cannot see
     what else holds the GPU, so 158 to 160 teams would pass it and deadlock.
     It would also be a local change to the plugin. *A probe at startup* has
     the same problem, and depends on the kernel too.
   - **Later:** fewer threads per team, by `thread_limit`, which is not yet
     implemented. One wave per team would let about eight times as many teams
     run at once, when the team has no parallel work.
5. A barrier from messages, measured against the estimates below. Done
   2026-11-10 (`benchmarks/team-barrier`; the numbers are in the design
   document's Measurements, "Barriers between teams"):
   - **Two shapes,** as in the estimates: `central`, where every team reports to
     team 0, which releases each, and `tree`, a binary tree. Each message carries
     its round, so a wrong order is counted. None was wrong.
   - **About 5 to 10 times the estimates** on the GPU: 153 µs for central and
     29 µs for the tree at 64 teams, against 30 and 3. A hop of a real message
     costs about 1.2 µs, against the 240 ns of a word that the estimates
     assumed. Where the rest goes is not measured yet.
   - **The barrier is the program's code,** not the library's, since library
     code is not yet compiled for the device; nor is `list.length`, which the
     benchmark works around (marked Temporary). The barrier moves into the
     library, as `process.team_barrier`, when the library is compiled for the
     device.

**Later:** SPMD kernels with `masked` and a team barrier; team mailboxes reachable
from the host and other nodes; and spawn. Spawning on a GPU may need a first
process there, a persistent kernel that the host starts, perhaps when the node
starts, which then spawns into free team slots. Whether that needs a call of its
own before `spawn` (a `spawn_init`), or the first `spawn` to a GPU starts it, is
open.

**Risks, the first being the largest.**

1. **Forward progress.** A GPU does not promise that every team makes progress.
   A team that spins for a message from a team that is not running deadlocks. It
   must be known how many teams run at once, and whether that is a limit a
   program can stay within. Threads of one wavefront must not spin on each other
   either (design document).
2. **A fixed set of processes.** One kernel gives the processes it starts with.
   Spawning needs persistent kernels and a scheduler of GPU processes, not
   designed. A halo exchange needs only a fixed set.
3. **The compiler** needs `target teams` with Mercury code in each team, `masked`
   and `barrier`, and an arena for each team.
4. **Failure.** A trap ends the whole kernel, so one team cannot fail alone as a
   monitor expects. That weakens recovery within a kernel.

**The experiment (2026-11-09, RX 6950 XT, gfx1030, 80 compute units).** A C++
OpenMP program whose teams pass a token around a ring in device memory, each wait
bounded, one team giving up ending the others. Each team first ran a `parallel`
that used its threads. No GPU resets.

| Threads per team | Teams that worked | Teams that gave up | Hops before stopping |
|---|---|---|---|
| 1, 8, 32 | 2560 | 4096 | 2561 |
| 64 | 1024 | 2048 | 1281 |
| 128 | 512 | 1024 | 641 |
| 256 | 256 | 512 | 318 |
| 320 | 128 | 256 | 238 |

The hops show how many teams were on the GPU at once: 2560, 1280, 640, 320 and 240.
That fits `80 * floor(32 / waves)`, with 32 threads to a wave. The 32 wave slots for
each compute unit are read from these results, not from a specification.

- **The limit is sharp and can be computed,** and a program within it made progress
  in every run. A team beyond it does not start until a team on the GPU ends, so a
  cycle of waits through such a team deadlocks.
- **No OpenMP query gives it.** `omp_get_num_teams()` reported 65536 teams launched,
  when 2560 could run together. The compute units are in the device's information.
- **It is observed, not promised.** One GPU; other users of the GPU, such as the
  display, could lower it; waits must be bounded in the runtime, or a mistake hangs
  the GPU.
- **Latency** (Measurements): a one-way hop takes 200 ns between 2 teams, 240 ns up
  to 64 teams, 520 ns at 1024 and 1125 ns at 2048, so a round trip between two teams
  is about 0.4 µs, against 7.5 µs for one RPC round trip. Idle threads in a team cost
  nothing. This is one word, not a message: a real send adds a queue's bookkeeping.
  The growth with the number of teams, while only one team has work, suggests that
  polling itself loads the memory system; that was not tested.

**Coordinating teams with messages.** OpenMP has nothing that makes one team wait
for another inside a teams region, and code in `teams` outside `distribute` runs
once in every team, so a `receive` there would run in each. With a process for each
team, `send` and `receive` give the coordination:

1. Team 0 receives from the network.
2. It sends the array to the other teams, by its pointer.
3. Each team computes its slice and sends its part to team 0.
4. Waiting for those replies is the barrier.

Rejected for this:

- *`omp_get_team_num() == 0` and a flag that the other teams spin on:* a barrier
  written by hand for each program, and only team 0 would have the array.
- *Three target regions in a row* (receive, compute with `target teams distribute
  parallel do`, send): the end of a region is the barrier, but each step returns to
  the host for a launch of about 30 µs, and a received array would have to outlive
  its region.

The cost of a barrier, in which every team reports and then is released, estimated
from the hops measured above at 240 ns. Hops of a tree overlap, so what counts is
the hops in sequence. Not measured, and a real message costs more than a word.

| Teams | Shape | Hops in sequence | Estimate |
|---|---|---|---|
| 4 | team 0 with each | 6 | 1.5 µs |
| 64 | team 0 with each | 126 | 30 µs |
| 64 | tree | 12 | 3 µs |

What it needs, beyond risk 1:

- **Every team that takes part must be on the GPU.** The kernel must not start more
  teams than fit, and no OpenMP query gives that number. Decided in step 4: the
  teams wait for each other to start, and the program ends if they do not all
  run at once.
- **An owner for the array.** If it is in team 0's arena, team 0 must not reclaim it
  until every team has replied. The replies guarantee that only if the program keeps
  to the protocol, so it needs a rule that the runtime or compiler can check. Memory
  shared by the teams, with a lifetime of its own, may be cleaner. Open.
- **Bounded waits.** A team blocked in `receive` keeps its wave slot and polls; a
  bound in the runtime is what keeps a mistake from hanging the GPU.

Written with `send` and `receive`, a barrier or a gather is ordinary Mercury library
code, so the same predicates could serve the collectives between nodes, at another
cost. The first test, once teams are processes, is such a barrier, measured against
the estimates above.

**Pieces of one array, in place.** A large array is not copied to each team: each
team gets a piece and updates it where it is. All teams of a GPU address the same
device memory, so a piece is a pointer, an offset and a length. Within one region,
`target teams distribute parallel do` already gives each team a range of the same
array; messages are for pieces that teams keep across steps, as a halo exchange's.

- **Split and join.** A unique array is split into disjoint unique slices, each
  consuming its part of the original. A slice is sent to a team, which updates it
  in place and sends it back; joining the slices gives back the whole array, as
  Rust's `split_at_mut`. Uniqueness means no two teams can hold the same element,
  so there is no data race, and mode analysis checks it.
- **A read-only array** is shared by every team at once, with no transfer of
  ownership. Its memory must outlive the readers.
- **Between nodes,** a sent slice copies only itself: the halo exchange's
  decomposition. The same program runs on one GPU or several, at another cost.

Open:

- **Representation:** a slice type, or the array type with an offset, and the
  library's split and join. mmc has no slices, so for running on mmc the library
  can split and join by copying, with the same meaning.
- **Lifetime:** the memory under the slices must live until they are joined. On
  the host, a large array has a block that a collection does not move
  (`runtime/heap.cpp`), so a pointer into it stays valid; the block must still not
  be reclaimed while a slice is out. On a device, a region's cells are freed when
  it ends, so slices that outlive a region need memory with a lifetime of
  its own: the question of an owner for the array, above.
- **Boundaries** on multiples of 256 bytes: a cache line is 128, so that two
  teams do not write to one line, but a stream from elements not on 256 bytes
  is up to a quarter slower on gfx1030 (design document, Measurements, "Dense
  stencils", where an array's elements start). The same holds for where an
  array's own elements start, which its size word puts 8 bytes off.

**A later use: streaming on the GPU.** Teams as the stages of a pipeline, passing
arrays by pointer. It has no cycle of waits, but a stage that is not on the GPU
still stalls the one before it when the ring between them fills.

### What teams as processes are for, beside the specialized kernels

Proposed 2026-11-10, from a discussion; not agreed. The user was unsure how a
process per team relates to the optimized reductions of Specialized kernels for
loops, below. The two rest on opposite assumptions about teams, so they are for
different work.

**Two models of a team.**

- *The paper's: teams are anonymous, short-lived and independent.* No-loop and
  big-jump kernels want many more teams than run at once, and the hardware's
  dispatcher runs them in waves, which hides memory latency and balances the
  load. The cross-team reduction never waits for another team: each team reduces
  within itself, by wave shuffles and then LDS, writes its value to a slot and
  increments a counter, and the team that sees the last count combines the
  slots. Teams may finish in any order, and need never run at the same time.
- *Teams as processes: named, long-lived actors that wait for each other.* All
  of a region's teams must run at once (step 4), which the scheduler of a shared
  GPU does not promise: the teams tests failed so under a full test run, when
  more programs had queues than it holds (design document, the teams' stall
  time). How many fit depends on the kernel: 157 of 400 at 124 VGPRs, above. And
  such a kernel holds its compute units for as long as it runs, away from
  data-parallel kernels.

**Proposed.**

1. **Data-parallel loops and their reductions use the paper's kernels,** never
   messages between teams. `reduction` already lowers to the cross-team
   reduction; no-loop and big-jump are planned. A barrier from messages cost
   29 µs at 64 teams (step 5), and the paper's reduction needs no barrier.
2. **Within a process, the paper still applies.** A parallel loop in a team's
   goal runs on the team's threads, and its reduction is the paper's within a
   team.
3. **Between processes, the same pattern.** A slot and a counter, where the last
   to arrive combines, is put and signal in One message transport, with
   backends, above. A reduction across GPU processes, or across GPUs, puts each
   partial value into a slot at the root and signals; the one that sees the
   count reach N combines. No participant waits for another to be running, so it
   needs neither co-residency nor a barrier. Collectives and direct GPU
   transfers, above, would use it.
4. **Few processes, which coordinate.** One process, or a few, per GPU, holding
   data and exchanging messages. The large loops they are handed run as
   specialized kernels, with more teams than fit and every compute unit. The
   host launches them when a process's message asks; a running GPU process
   queueing a kernel itself, by writing a dispatch packet to a queue, may be
   possible on AMD hardware, but is not explored.
5. **A pool of anonymous workers as the default for work handed out.** Workers
   take tasks from a queue in device memory. No worker waits for a particular
   other, so a pool of which only some are running is slower, not deadlocked,
   and it needs no co-residency. It suits processes that are handed the data
   they work on (the user, 2026-11-10). Work stealing (Distributing work, above)
   is its extension across GPUs.
6. **Named teams as processes, where a pattern needs them:** a halo exchange
   between tiles of one GPU with no relaunch, a pipeline (above), irregular work
   whose teams talk to particular others. They keep the check of step 4 and its
   report.

**Open.**

- Whether a pool replaces team mailboxes for most programs, or sits beside them:
  a library over team mailboxes, or a construct of its own.
- How a process hands a loop to a specialized kernel and learns that it has
  ended, through the host at first, and what that costs per launch, to measure.
- Whether a persistent process should use fewer threads per team
  (`thread_limit`, step 4's "Later"), so that it holds less of the GPU while
  kernels run.

## Library code on the device: later

Built 2026-11-10 and moved to the design document (Library code on the
device), with its measurements (Measurements, "Barriers between teams").
What is left:

**The next piece: closures and type classes on a device.** Most of `list` is
higher-order: `foldl`, `map` and `filter` make or call closures, which a device
rejects. The specializer makes a call with a known closure a direct call, so
with the bodies the device's compile copies in, it makes those predicates run
on a device (The specializer: known closures and dictionaries).

**Strings and formatted output on a device** (the user, 2026-11-10: "will
strings be okay with GPU code if we want to print things?"). Step 5a lets
GPU code make strings, as whole values, and a GPU process prints literals
and integers through LLVM libc's stdio. What is missing, each a step:
1. **The string helpers on the device:** `++`, `int_to_string`,
   `float_to_string` and the others whose `foreign_proc`s call runtime
   functions (`merk_string_append`, `merk_string_from_int`) that the device
   library lacks, which the device's compile reports today ("its foreign
   code calls merk_string_append, which is not on the device"). Each
   allocates its string in the arena, with LLVM libc's `snprintf` where it
   formats.
2. **A call of `format` whose format and argument list are written in it,**
   as nearly all are, compiled to one `printf` of libc's: the list of
   `poly_type`s is never made, which step 5a's check rejects in a GPU
   process, and one call writes the whole line, so that lines from many
   lanes do not interleave within one. Parsing the format at compile time
   also gives mmc's warning for a call whose arguments do not fit its
   format (`io-format.m`, whose mismatch mmc reports when compiling), on the
   host as well. A format that is not known at compile time stays an error
   on a device.

**Streams on the host,** next in the library: `io.m`'s stream types are
abstract and every predicate with a stream argument is only declared:
`stdout_stream`, `stderr_stream`, `stdin_stream`, `open_input`,
`open_output`, `close_input`, `close_output`, the stream forms of the
`write_*` predicates, `format/5`, `write_strings/4`, `print/4` and
`read_line/4`. A call of one is an error at compile time; writing to the
standard error is the case programs need most.

**Open.**

- Whether host compiles use the same bytecode, for inlining and specialization
  across modules. If they do, the interface file's optimization section becomes
  `merk` bytecode too, and the design document changes to say so.
- **A `teams` reduction or a library predicate.** The other pattern between
  teams in the paper on specialized kernels: each team leaves its value, and the
  last to finish combines them, so no team waits. In OpenMP that is `reduction`
  on `teams`, which upstream lowers to the same cross-team call as our
  `reduction` on `distribute` does. It could also be a library predicate, such
  as a `team_sum` that gives `yes(Total)` to the last team and `no` to the
  others. The clause is the feature and needs no new library; the predicate
  fits a region that computes its value late.
- **The program's other modules on a device** (`omp-target-other-module.m`):
  their bytecode could come from the build driver.

## The specializer: known closures and dictionaries

Proposed 2026-11-10; step 1 is built. The `merk` pass that the design
document's Optimizations section names for higher-order and type
specialization and for `pragma inline`. This piece is its first part: calls
given a closure or a dictionary known at the call become direct calls.

**Why now.** On a device, a closure or a dispatch on a type class is an error
(`RejectOnDevice`), so most of `list` cannot run in a region: `foldl`, `map`
and `filter` take closures. The `parallel` library proposed in GPU processes
without `omp target` (option a) is predicates of that kind too. On the host,
a closure call costs about three times the monomorphic loop over an array
(design document, "What polymorphism costs"), and blocks vectorization.

**What the code looks like now.** `scaled(K, L) = foldl(func(X, A) = A + K * X,
L, 0)` gives:

```mlir
%1 = merk.closure @ho.scaled_f2_0$lambda1$closure1(%0) captures [["int/0"]]
%8 = merk.call @list.foldl_f3_0(%3, %4, %1, %6, %7) : (...) -> !merk.word
```

and `list.foldl_f3_0` calls `merk.call_closure %arg2(...)` for each element,
then itself with `%arg2` again. The closure's entry takes the closure first
and reads `K` with `merk.closure_word`. A dictionary is the same:
`merk.typeclass_info @instance(type_infos)` in the caller, and
`merk.typeclass_info_word` and `merk.call_method` in the callee.

**Proposed: specialize on values made by known operations.**

- **Known values.** An argument is known when one of these makes it:
  - `merk.closure @entry(captured)`: known as its entry; the captured words
    are not, and stay values.
  - `merk.typeclass_info @instance(arguments)`: known as its instance, with
    its arguments known or not in the same way.
  - `merk.type_info` of constants, which are static already.
  A known value's pattern is its operation with each unknown word as a hole,
  such as `closure @e(_)` or `typeclass_info @show(int)(type_info "int/0")`.
- **A call with a known argument is cloned.** The clone is the callee with,
  for each known argument, a parameter for each hole in place of the
  argument, and the operation rebuilt from them at its entry. Clones are
  private, named `<callee>$spec<N>`, and kept by callee and patterns, so the
  same call elsewhere gets the same clone.
- **Then the clone folds:**
  - `merk.call_closure` of a `merk.closure @e` becomes `merk.call @e` with the
    closure and the inputs, and `merk.nondet_call_closure` becomes
    `merk.nondet_call`. That call has a known argument, so the entry is
    cloned with the captured words as parameters, and its
    `merk.closure_word`s fold to them.
  - `merk.typeclass_info_word` of a `merk.typeclass_info` becomes its
    argument.
  - `merk.call_method` of a `merk.typeclass_info @i` becomes a call of the
    procedure in the method's slot of `@i`'s table: the dispatch procedure,
    which takes the dictionary. Cloned in turn, it calls the instance's
    method procedure with the dictionary's words.
  - A call in the clone that passes the rebuilt value on, such as
    `foldl`'s call of itself, has the same patterns, so it calls the clone:
    the loop calls its body directly.
  Afterwards the rebuilt `merk.closure` and `merk.typeclass_info` that nothing
  uses are removed. One that something still uses stays, as when it is put
  in a term or passed to a callee without a body. On the host that is
  correct and only slower; on a device it is the error it is now, at that
  use.
- **One mechanism for all three.** A closure's entry and a dispatch
  procedure are cloned by the same rule as `foldl`, because each takes the
  known value as an argument. The direct calls of known instances that MLIR
  generation makes now (design document, Lowering, "Type classes") remain,
  as they need no dictionary at all; the specializer handles the rest.
- **Limits, as mmc has** (`--higher-order-size-limit`,
  `--higher-order-arg-limit`): a pattern no deeper than a few levels, such as
  a closure that captured a known closure; a limit on clones of one callee;
  and polymorphic recursion, whose patterns grow at each call, stops at the
  limit. Then the call stays as it is. On the host nothing is reported yet
  (`pragma inline`'s remarks come later); on a device what is left is the
  error it is now.
- **Where it runs.** On the `merk` module after MLIR generation, after
  `CopyDeviceLibrary`, and before `MarkRegionsForTeamMailboxes`, so that the
  calls these follow are the specialized ones. Only procedures with bodies
  are cloned: the module's own, and on a device the library's copies.
- **When.**
  - **On a device, always,** for what target regions reach, at every
    optimization level. It is what lets a region run `list.foldl`, so a region
    that compiles at `-O2` also compiles at `-O0`, as `pragma inline` has it.
  - **On the host, at `-O2`,** within the module first. Across modules it
    needs the library's bodies in the host's compile. That is the open
    question kept from Library code on the device (step 5).

**Steps.**

1. **Closures within a module.** The pass, known closures, cloning and
   folding of `merk.call_closure`, `merk.nondet_call_closure` and
   `merk.closure_word`, and the limits. Tests: `merk` checks of the clones
   for a module's own fold, an entry with captured values, a recursive
   higher-order predicate, a closure that escapes into a term, and the depth
   limit; each also runs at -O0/-O2 against mmc's output. Host only, at -O2.
   Built 2026-11-10 (`lib/CodeGen/Specializer.cpp`, `SpecializeKnownClosures`):
   - **The closure stays a parameter.** A clone takes the closure as well as
     what it captured, rather than making it again at its start, which
     would allocate on each call of a recursive clone. Once nothing is left
     to clone, a clone's parameters that nothing uses, or that it only
     passes on to itself in the same place, are removed from it and its
     calls, and then the closures and conversions that the specializer left
     unused. Only those: MLIR generation leaves unused conversions as notes
     of a word's type, which the collector reads.
   - **Captured values' types, for the collector.** A captured `type_info`
     or `typeclass_info` is a parameter of that type. A word whose type the
     closure's `captures` describes is followed by a `type_info` parameter,
     which the caller makes from the description and the clone's
     `merk.root_type` names. Made in the clone, a `type_info` that is not
     static would be a safepoint at which the captured value has no type yet.
     A recursive call passes its own `type_info` parameters on.
   - **Two fixes the clones needed.** The safepoints and outlining find the
     type of a word that a continuation's environment holds from the typed
     value it was converted from, as they did for a root. Before, a closure
     held such a word, and the environment held the closure.
   - **Useless clones.** A clone that comes out the same as the procedure it
     was copied from is removed, and its calls call that procedure, as for a
     callee that only puts the closure into another closure (`compose`).
   - **Not cloned:** a procedure without a body, with foreign code, which may
     not be duplicated, or with a target region, whose kernel would be.
   - **Limits:** closures captured by closures are followed three levels
     deep, and a procedure gets at most 16 clones.
   - **Tests:** `specialize-closures.m`, with a fold, a predicate with `io`,
     a closure in a term, `compose`, a nondet closure, a chain of four
     closures, a captured value of a type variable, and two closures
     captured by closures. Every runtime test also builds at -O2, so the
     whole suite runs the specializer.
2. **Dictionaries.** Known `typeclass_info`s, `merk.typeclass_info_word`,
   and `merk.call_method` through a table slot's dispatch procedure.
   Instances with constraints, so a dictionary inside a dictionary. Tests as
   step 1, plus a superclass's method.
3. **On a device.** Running it on a device's module at every level. The
   copier copies an instance's dispatch procedures when the specializer
   reaches its table, as it copies procedures now: the two alternate until
   nothing new is reached. The device's errors then name only what is left.
   Tests: a region with `list.foldl`, `map` and `filter` with lambdas that
   capture values, and a type class method in `list.foldl`'s closure, each
   with a `-gpu.test`; an error test for a closure in a term.
4. **Measured.** The array loop of "What polymorphism costs", closure and type
   class versions at -O2, against the monomorphic one, before and after.
5. **Across modules on the host,** once that question is decided: host
   bytecode of the library, and the specializer loading the body of an
   imported procedure that a call gives a known value.

**Later:** specializing on `type_info`s alone (type specialization), which
turns `merk.unify_word` and `merk.compare_word` of a known type into calls of
its entries, and then lets a device unify and compare polymorphically; and
`pragma inline`'s promise and remarks.

**Rejected alternatives.**

- **Inlining instead of cloning.** A recursive callee such as `foldl` cannot be
  inlined completely; a clone that calls itself is the loop. Inlining small
  clones is left to LLVM, which does it once the calls are direct.
- **LLVM's devirtualization.** By then a closure is a cell in memory and its
  call an indirect call through a load, and the closure would have to be
  proved constant across the recursion. gcc did not do it even for a constant
  closure (design document, "What polymorphism costs").
- **Specializing on the source or the semantic form,** as mmc does on its
  HLDS. The `merk` form has the closures and dictionaries as values and calls,
  and the device's copies of the library exist only in this form.
- **Clones that keep the closure cell and pass it,** with only the call
  inside made direct. A device cannot make the cell, and the captured values
  would stay loads from memory.

## Specialized kernels for loops

Proposed 2026-11-10; nothing is built. From Chakrabarti et al., "Specialized
Kernels for Optimizing GPU Offload in OpenMP" (WACCPD 2023, AMD's ROCm
compiler), which generates three kinds of kernel for `target teams distribute
parallel for` at code generation, rather than leaving a general one for
OpenMPOpt to trim.

| Kernel | What it is | Merkurius now |
| --- | --- | --- |
| Big-jump loop | Each thread starts at its index in the grid and steps by the grid's size, so the threads of a wave read neighbouring elements | Missing: no `dist_schedule`, as in Flang, so each team has one block of iterations |
| No-loop | As many threads as iterations, each runs one, with a bounds check and no back edge | Missing |
| Cross-team reduction | Each team reduces within itself and writes its value to a slot in device memory, then increments a counter; the team that sees the last count combines the slots | Have it: `reduction` lowers to upstream's `__kmpc_gpu_xteam_reduce_nowait` |

**Why.** `benchmarks/bandwidth` measured Copy at 312,000 MB/s and Mul at 271,000
against C's 532,000 and 456,000 (2026-11-04). Its README names clang's chunks of
256 iterations against our one block per team as the larger part of that gap.
Those numbers predate the rebase onto upstream, so they are measured again
first.

**Proposed.**

1. **Big-jump:** give `distribute` a static chunk of the team's thread count when
   no schedule is given, as clang does. Measure with `benchmarks/bandwidth` and
   `compare-c.sh` before and after. Compare with Flang's output for the same
   loop first, which has the same gap; a fix there may be worth taking upstream.
2. **No-loop:** set `spmd_no_loop` under Flang's conditions
   (`canPromoteSPMDToNoLoop`): no `num_teams`, no `reduction`, and the two
   oversubscription promises.

**What Mercury can prove that C must promise.** The paper's conditions are
mostly about what a loop's body might do: no nested parallelism, and no calls
that read or set an ICV. C has flags for them, `-fopenmp-assume-no-thread-state`
and `-fopenmp-assume-no-nested-parallelism`. A Merkurius parallel loop's body is
det and pure (`MLIRGenOpenMP.cpp`), so it cannot call `omp_set_` functions,
which would need `io`. merkc can check both itself. The trip count is known on
the host at launch, so the choice between no-loop and big-jump could be made
there, rather than by the promise that the trip count fits the grid. The paper
does not do this ("We currently don't multi-version kernels").

**Not in it:** the paper's rule against environment variables that set teams or
threads (`-fopenmp-target-ignore-env-vars`). That stays a flag.

## GPU processes without `omp target`: liboffload and libc

Proposed 2026-11-10; steps 1 to 3 are done. Decided by the user the same
day:
- **GPU processes launch without `omp target`.** A process on a GPU is a
  kernel that our runtime launches, not a target region.
- **OpenMP stays the programming model for parallelism:** target regions,
  teams and parallel loops keep their directives and their lowering through
  the `omp` dialect, following Flang.
- **libc over our own code.** Where LLVM's GPU libc does something, the
  runtime uses it rather than code of ours (as for output: design document,
  Output from a region).

**Why.** A GPU process lives as long as its work, sends and receives, and runs
beside the host's processes and the GPU's other kernels. A target region is the
wrong shape for that:
- **Its host thread waits.** `omp target` returns when the region ends, unless
  it is a `nowait` task, which libomp runs on a host thread of its own.
- **Its arguments are OpenMP's.** They are maps of host variables, with a
  kernel environment and launch arguments that libomptarget makes. A process
  is given a mailbox handle and its arguments, by value.
- **The decided form needs a kernel by name.** The design document has a GPU
  process spawned by its kind's name from its `:- process` declaration, as a
  persistent kernel (GPU processes, "The restricted first version").

**What LLVM has.** Read in the tree on 2026-11-10:
- **liboffload** (`offload/liboffload`, the `ol` API) is the
  language-neutral layer for devices. It covers what we need:
  - `olCreateProgram` loads an image.
  - `olGetSymbol` finds a kernel or a global by name.
  - `olLaunchKernel` launches a kernel with arguments by value, on a queue
    if given, without the host waiting.
  - `olMemAlloc`, `olMemAllocHost` and `olMemcpy` handle memory.
  - `olPlatformRegisterRPCCallback` takes a callback for RPC opcodes before
    libc's. It is what `__tgt_register_rpc_callback` calls.
- **libomptarget is built on liboffload.** Upstream finished removing the
  plugin interface from it (#239525), so only liboffload makes the plugins.
  But libomptarget links its own static copy (`LLVMOffloadStatic`) and its
  export list hides the `ol` functions. A program that also loads
  `libLLVMOffload.so` has two copies, each with its own plugin and RPC
  server (found by the probe, step 1).
- **Other languages use it too.** `offload/languages` is a CUDA and HIP
  runtime on liboffload (`LLVMOffloadKernel`), registering images and
  launching kernels by name. `llvm-gpu-loader` is built on it.
- **`gpuintrin.h`** (clang's headers) gives the GPU's ids and sizes, lanes
  and syncs: `__gpu_block_id`, `__gpu_num_blocks`, `__gpu_thread_id`,
  `__gpu_sync_threads`, `__gpu_ballot` and so on. LLVM's GPU libc is
  written on it. It needs no OpenMP.
- **LLVM's GPU libc** gives `stdio` over its RPC, `clock_gettime`, `malloc`
  and the maths library. The device's image links it already.

**Proposed.**
- **Launching:** the runtime calls liboffload itself. It loads the image of
  the program's GPU processes with `olCreateProgram`, finds each kind's
  kernel with `olGetSymbol`, and spawns a process with `olLaunchKernel` on a
  queue of its own. The kernel's arguments are the mailbox handle, the
  process's place and its arguments. The host does not wait. The process ends
  when its kernel returns, which the runtime learns from the queue or from
  a message.
- **Its image is a separate one.** merkc compiles the GPU processes of a
  module as plain kernels for the GPU, without OpenMP. They are linked with
  `libmerkdevice`, libc and libm, and their foreign code as a region's is
  (design document, Library code on the device). The image is embedded in the host's binary, where
  the runtime finds it. OpenMP's regions stay in OpenMP's image. The two are
  separate programs on the device, each with its own globals. So a process
  and a region share only what goes through the host or through device
  memory that a handle names, as between any two processes.
- **RPC:** the runtime registers its opcodes with
  `olPlatformRegisterRPCCallback` itself. Then they work with or without
  libomptarget in the program.
- **The device runtime without OpenMP's API.** `libmerkdevice` uses
  `gpuintrin.h` and libc instead of `omp_get_team_num`,
  `omp_get_num_teams`, `omp_get_wtime` and `omp_get_device_num`, so that it
  works in a process's kernel and in a region's alike. A device's place
  number comes from a global the runtime sets at load, or from the
  kernel's arguments. The `__kmpc_` calls stay in the code that OpenMP's
  lowering makes. Library C under `__AMDGPU__` does the same:
  `process.team_number` becomes `__gpu_block_id_x`, which is OpenMP's team
  number in a region's kernel on an AMD GPU (step 2 checked it).
- **The host's device calls behind one interface** in `runtime/offload.cpp`.
  `omp_target_alloc`, `omp_target_memcpy` and the rest are used for regions
  today. The interface would let the process side use liboffload, and later
  the region side too, where nothing of OpenMP's is needed.

**Data parallelism, and a `parallel` library.** OpenMP stays the model. The
open question is a library form, such as `parallel.fold(F, Range, Acc0, Acc)`
or `parallel.map` over an array, beside loops under directives:
- **(a) Library predicates written with directives:** a `parallel` module in
  Mercury whose bodies are loops under `%$omp target teams distribute parallel
  do` with `reduction`. A call's closure becomes a direct call through the
  specializer, the next piece (The specializer: known closures and
  dictionaries). Then the loop is an ordinary OpenMP loop. It gets the
  cross-team reduction, and big-jump and no-loop when they come (Specialized
  kernels for loops).
- **(b) A builtin that lowers to the `omp` dialect,** as Flang lowers `do
  concurrent`, a construct of the language rather than of OpenMP, onto
  OpenMP's operations (`DoConcurrentConversion`). This needs no specializer,
  but puts each predicate in the compiler.
- **(c) A runtime of our own:** work split over teams and threads with
  `gpuintrin.h`, and a reduction of our own. That would redo what OpenMP's
  device runtime and OpenMPOpt do, and get none of its improvements.

Recommended: (a), with (b) only where a predicate cannot be written as a loop.
Its semantics is the loop's: the operator of a fold must be associative, and
the call is the programmer's promise, as a directive is (design document,
Parallel loop annotations). Not decided: the predicates and their names, and
whether a GPU process may run a parallel loop itself. A process's kernel has no
OpenMP device runtime, so a loop there would be (c), or a target region that
the process asks the host to launch. The first version has neither.

**Steps.**

1. **Probe coexistence.** In a program that runs an OpenMP region, the runtime
   calls `olInit`, loads a small kernel image of its own, launches it on a
   queue while the region runs, and serves an opcode of ours from it through
   `olPlatformRegisterRPCCallback`. It checks:
   - that the devices `olIterateDevices` gives can be matched to OpenMP's
     device numbers;
   - that a persistent kernel and a region's kernel run at once;
   - that one RPC server serves both programs' clients.
   Run small first, with a timeout, as the GPU drives the display.

   Done 2026-11-10, a probe outside the tree: a C++ program with one OpenMP
   region, and a kernel built without OpenMP from `gpuintrin.h` and libc,
   launched through `libLLVMOffload.so`. Each kernel waits for the other's
   flag, in host memory from `olMemAllocHost`, for at most five seconds.
   Found:
   - **The two copies coexist,** with liboffload or libomptarget
     initialized first. Both RPC servers serve our opcode, liboffload's to
     the process and libomptarget's to the region. libc's `printf` works in
     the process. So the runtime registers its callback with both:
     `olPlatformRegisterRPCCallback` for its own copy, and
     `__tgt_register_rpc_callback` for libomptarget's.
   - **Devices match by UID.** `OL_DEVICE_INFO_UID` gives `AMDGPU-0`, and
     OpenMP 6.0's `omp_get_device_from_uid` maps it to device 0.
   - **The process and the region run at once.** `olLaunchKernel` returns
     at once, and each kernel saw the other's flag.
   - **The image must export `__llvm_rpc_client`.** The plugin finds the
     client by its dynamic symbol. Without it, nothing initializes the
     client, and the first send reads address 0. With LTO, the client is
     exported only if an object given to the link defines it. libc's and
     `libompdevice`'s definitions come from archives and are internalized.
     A weak, protected definition in the image's own object, as OpenMP's
     device runtime has, keeps it (merkc's images get one from the module).
     A process's image needs one too.
   - **A region's body must be the same in the host and device compiles.**
     Device-only code under `#ifdef __AMDGPU__` inside the region changed
     its kernel's arguments, and a `declare target` variable that only the
     device declares added an offload entry the host lacks. Both faulted.
     Device-only operations go in functions declared `declare target`, with
     host versions, as `libmerkdevice`'s are.
2. **The device runtime off OpenMP's API:** `gpuintrin.h` and libc in
   `runtime/device` and in the library's C under `__AMDGPU__`. The existing
   tests check that regions behave the same.

   Done 2026-11-10. All tests pass, the GPU's included:
   - **Team number and number of teams:** `__gpu_block_id_x` and
     `__gpu_num_blocks_x`, in `device.cpp` and in `process.team_number` and
     `process.number_of_teams`. In a region's kernel on an AMD GPU they are
     OpenMP's, as the team mailbox tests show.
   - **Time:** libc's `clock_gettime(CLOCK_MONOTONIC)`, in nanoseconds, for
     the receive deadlines and the teams' start. libc takes the clock to be
     100 MHz unless the loader sets `__llvm_libc_clock_freq`, which the
     offload plugin does not. That is right from gfx9 on. Before, deadlines
     would be up to four times too long. `omp_get_wtime` had the plugin's
     frequency.
   - **The device's number:** a global, `merk_device_number`, which starts
     at -1. A region's image keeps that and takes `omp_get_device_num`,
     through a weak reference, so an image without OpenMP's device runtime
     still links. Step 4 sets the global when it loads a process's image.
     That image must define it in its own object, or LTO internalizes it, as
     it did libc's RPC client (step 1).
   - **What stays OpenMP's:** `merk_device_shared_alloc` and `_free` call
     `__kmpc_alloc_shared`. Code generation calls them only in a parallel
     region's code, which a process's kernel does not have, and LTO drops
     them from an image that does not. The `omp_get_num_teams` that a teams
     region's code calls is OpenMP's lowering, as its `__kmpc_` calls are.
3. **The host's device calls behind one interface,** and the RPC callback
   registered through liboffload.

   Done 2026-11-10, but for the registration. `runtime/devices.h` has the
   calls `offload.cpp` makes: memory on a device, host memory a device
   reads, copies each way, and the callback's registration. A device is
   OpenMP's number of it, which the device's runtime sends with its opcodes.
   `devices.cpp` puts them over libomptarget, through weak references, as
   `offload.cpp` did. The behaviour is the same, and all tests pass.

   The registration through liboffload moved to step 4. The probe found
   that the runtime's liboffload is a copy of its own, beside
   libomptarget's (step 1). Registering with it serves only images it
   loaded, and it loads none before step 4.
4. **A GPU process's kernel:** merkc compiles a process kind to a kernel, a
   separate image linked and embedded, and the runtime spawns it on a GPU.
   The runtime calls `olInit` when it loads the first such image. It
   registers its callback with `olPlatformRegisterRPCCallback` then. It
   matches liboffload's devices to OpenMP's numbers by UID, and sets the
   image's `merk_device_number`. `devices.cpp` uses liboffload for the
   devices of processes, so that a program with GPU processes and no
   regions does not need libomptarget. The image defines the RPC client and
   `merk_device_number` in its own object, so that LTO exports them.
   The kernel sends to and receives from the host's processes. `:- process`
   declarations need semantic analysis first; until then, which predicate is
   a GPU process's entry is an open question for review.

   Designed 2026-11-10, in parts, each a commit of its own. The entry is
   the kind's predicate from its `:- process` declaration, as the design
   document has it (Processes and mailboxes), so the declarations come
   first. Nothing parses them yet: `process` and `receives` are not
   operators.

   - **4a. `:- process` declarations, on the host.** Parsed as `:- mutable`
     is, checked, and expanded into the kind's `pred` declaration and
     `spawn_<kind>`, which calls `process.spawn` with a closure that calls
     the kind's predicate. In the first version a kind receives one message
     type; more report "not yet implemented". `spawn_<kind>_on` needs
     remote spawn, which does not exist, so a call of it reports "not yet
     implemented" too. Tests: the declaration's errors, and programs that
     spawn kinds, with the output they print. mmc has no `:- process`, so
     the expected output is written by hand.

     Done 2026-11-10, behind `-fprocess-declarations`, as the design
     document records (Processes and mailboxes, "Process declarations as
     implemented"). `process`, `receives` and `on` are operators only in an
     item that begins with `:- process`, so that importing the module
     `process` still reads. The converter makes the expansion while the
     declaration's terms exist, and `ExpandProcesses` puts it into the
     items, so later passes see ordinary declarations and clauses. A kind
     marked `on gpu` is parsed, and reports "not yet implemented" until 4b
     ("Temporary" in `lib/Semantics/expand-processes.cpp`).
   - **4b. A GPU kind's kernel.** For each kind that runs on a GPU (decision
     D1), merkc makes a kernel `merk_process_<kind>`, which takes the
     mailbox's handle and the kind's arguments by value. In the first
     version the arguments are scalars and mailboxes, and the kernel is one
     team, whose initial thread runs the kind's predicate (design document,
     "What a process is on a GPU"). A driver job compiles the module for the
     GPU without OpenMP. Clang links it with `libmerkdevice`, libc and libm,
     into an image of its own. Code generation defines the RPC client and
     `merk_device_number` in the module, weak and protected, so that LTO
     exports them (steps 1 and 2). The image goes into the host's object as
     a global, which a start-up call registers with the runtime, as message
     types are registered. Not the linker wrapper's offload binary: the
     wrapper links only OpenMP's, CUDA's and HIP's kinds.

     Done 2026-11-10, as the design document records (Driver, Linking and
     offloading, "Images of GPU process kinds"). What differs from the plan:
     - **The start object defines the RPC client and `merk_device_number`,**
       not code generation: `merk-process-start-amdgcn.o`, from
       `runtime/device/process-start.cpp`, given to the link as an object.
       Rejected: code generation defining them, which would copy
       `rpc::Client`'s size and layout into the compiler.
     - **A device mode without OpenMP:** `TargetOptions::IsDevice()` is true
       for OpenMP's device compile and for `-fprocess-kernels`, and the code
       that a device does without (roots, statepoints, the collector,
       descriptors' entries, the specializer) checks it. The kernels are the
       roots that the library's procedures are copied from, and
       `FilterKernelProcs` keeps what they reach, without OpenMP's
       declare-target passes. Errors say "in a GPU process". Rejected:
       compiling the kernels as an OpenMP device module, which would bring
       OpenMP's attributes and its target regions into an image without
       OpenMP's device runtime.
     - **Everything but the kernels is hidden,** so that the image, a shared
       object, exports only them (`HideAllButKernels`).
     - **The Lowerer's own stack slots are in the stack's address space,**
       as region slots were: OpenMP's translation had fixed them for
       regions.
     - **The image's link is at the `-O` given, or else at `-O2`,** as the
       linker wrapper's for OpenMP's device code. AMDGPU's code generation
       fails at `-O0` on a `musttail` call in divergent control flow (design
       document, LLVM bugs to report), which is left as LLVM's bug, not
       worked around. Decided by the user the same day, after a first
       version linked at `-O2` whatever `-O` was: an `-O` that is given is
       respected; `-O2` is the default for code generation for an AMD GPU,
       in the image's link, OpenMP's link (which now passes `-O` on, as
       Clang's and Flang's do) and `-fc1 -S` and `-emit-obj`.
     - **The host registers message types** for a module with GPU kinds,
       as for one with target regions.
     - **Arguments:** a kind that is polymorphic, or takes a value other
       than an integer, a float, a character or a mailbox, is reported as
       not yet implemented in the kernels' compile. So is a message type
       the device's receive does not take yet, as a closure through the
       library's `receive`, as in a region's callee.
     - **The runtime records each image** (`runtime/process-images.cpp`),
       by module and GPU, for step 4c to load.
     Tests: `Driver/process-images.test`, `CodeGen/process-kernels.m` (the
     kernel, the registration, the image's exports, and the program on the
     host at `-O0` and `-O2`), and `CodeGen/process-kernel-errors.m`.
   - **4c. Spawning on a GPU.** `spawn_<kind>_at` (decision D2) spawns the
     kind on a GPU. On the first such spawn, the runtime calls `olInit`,
     registers its RPC callback, and matches liboffload's devices to
     OpenMP's numbers by UID, or numbers them in liboffload's order without
     libomptarget. It loads the image on the device when first needed and
     sets its `merk_device_number`. Each process gets a queue of its own.
     A record on the host stands for the process: it owns the process's
     mailbox, whose place is the GPU. It ends when the kernel does, which a
     host function queued after the kernel (`olLaunchHostFunction`) tells
     it, and its mailbox then closes as any process's does. `devices.cpp`
     uses liboffload for these devices.
   - **4d. Memory of a running process.** The devices' arenas and rings are
     freed when no region runs (The device's side of the first slice). A
     GPU process lives longer than any region. So a running GPU process
     counts as a running region for them, until a process has an arena of
     its own (design document, "Its heap"). There is no collector on the
     device, as for regions, and MERK_DEVICE_ARENA_BYTES limits it.

     4c and 4d done 2026-11-10, together, as the design document records
     (GPU processes, "Spawning on a GPU as implemented"). What
     differs from the plan:
     - **Numbers:** a GPU's number is OpenMP's for it, found by its UID with
       `omp_get_device_from_uid`, so that a region and a process name a
       device alike. A GPU that libomptarget has no number for, or every GPU
       without libomptarget, is numbered after OpenMP's devices, in
       liboffload's order. liboffload has no device numbers, only handles in
       `olIterateDevices`' order, and libomptarget numbers devices in the
       order that compatible images are first registered, so neither order
       alone stays OpenMP's. Messages between processes and regions go
       through the host's mailboxes, so the numbers only choose memory.
     - **No record on the host owns the mailbox.** The runtime makes an
       unowned mailbox (`merk_mailbox_new_unowned`) and closes it from the
       host function queued after the kernel. The handle's place stays the
       host's, 0, until step 5 routes by place ("Temporary" in
       `runtime/gpu-processes.cpp`). The queue is destroyed at the next
       spawn, not on liboffload's thread, which runs the host function.
     - **Arguments are words,** one each, and five for a mailbox, which
       `process.gpu_argument`'s instances give, so that an argument of
       another type is a type class error at compile time (because: "a
       process kind that runs on a GPU takes integers, floats, characters
       and mailboxes so far"), not the kernels' compile's "not yet
       implemented". AMDGPU's plugin copies the arguments by the kernel's
       metadata, ignoring the sizes passed, so each is an 8-byte word.
       `array.from_reverse_list` got code generation for it
       (`merk.array_from_list reversed`).
     - **4d:** a running GPU process counts as a running region
       (`GpuProcessStarted` and `GpuProcessEnded` in `runtime/offload.cpp`).
       Arenas are freed when the next region ends, not on liboffload's
       thread, and the program ends at once if main ends while one runs.
       Since replaced: each GPU process has an arena of its own, which a
       thread of the runtime's frees when it ends (the leaks after 4e).
     - **Two images on one GPU have arenas of their own:** `ArenaOf` matched
       a device alone, so a process image's allocation could take a region
       image's arena. It matches the device and the arena's address.
     - **The start object defines `omp_get_device_num`,** returning -1:
       libc's code refers to it weakly, and HSA's loader does not load an
       image with that undefined symbol ("HSA_STATUS_ERROR_VARIABLE_UNDEFINED").
     - **liboffload's API is declared in `runtime/liboffload.h`,** weakly, as
       its `.td` files define it. Rejected: including the generated
       `OffloadAPI.h`, which the runtimes' build makes after merkrt's.
       `devices.cpp` uses libomptarget for OpenMP's devices and liboffload
       for the others. A program with GPU kinds links `-lLLVMOffload`.
     - **At `-O0`** the image's link fails on AMDGPU's bug, as in 4b.
     Tests: `CodeGen/process-spawn-at.m` (the host, the software errors),
     `CodeGen/process-spawn-at-gpu.test` (on the GPU, a program ending
     while its GPU process waits, and an image for another GPU),
     `CodeGen/process-spawn-at-O0.m` (XFAIL), `Semantics/desugar/processes.m`,
     `CodeGen/process-kernel-errors.m` and `CodeGen/array-from-list.m`.
   - **4e. Tests on the GPU.** A host process spawns a counter on the GPU,
     sends it `add(N)` and `get(Reply)` messages, and prints its answers.
     Then two GPU processes exchange messages through the host. Run small
     first, with a timeout, as the GPU drives the display.

     Done 2026-11-10, as the design document records (GPU processes,
     "Spawning on a GPU as implemented", Messages). What differs from the
     plan:
     - **Messages of any type.** `get(Reply)` is a term with a mailbox in
       it, which the device did not send or receive: only one-word messages
       and arrays of them. Decided by the user the same day, as recommended:
       such a message is copied by its `type_info`, with the host's copy,
       which the GPU's runtime runs too (`runtime/copy.h`), through host
       memory that both sides address. "If this is an inefficient way of
       doing things we will have to revisit and come up with something more
       optimal, but more functionality now is important." Rejected: a copy
       that code generation specializes for each type, as the design
       document had it, which is much more work, repeats the terms' layout
       and would have covered flat types first; and fixed words for flat
       types only, which every later type would have had to extend.
     - **The device keeps its descriptors' functors,** by which the copy
       walks, though nothing unifies or compares there.
     - **What cannot cross is refused when copied:** a closure, and an
       existentially typed constructor's hidden fields, which name the
       other image's procedures and descriptors. A message type with a
       higher-order argument is reported at compile time.
     - **Target regions get it too,** as they send and receive the same
       way: strings, terms and arrays of non-word elements, which were "not
       yet implemented" in regions. A team mailbox still holds only words
       and arrays ("Temporary" in `lib/CodeGen/MLIRGenOpenMP.cpp`), as
       sharing a term by its pointer between teams is not tested yet.
     - **`descriptor.h` includes `stdint.h`,** not `cstdint`, so that the
       GPU's runtime, which has no C++ library, includes it.
     - **The host's copy for a ring is freed when its mailbox closes,** if
       the device has moved the head past its message. Found in review: a
       first version dropped it at close, which leaked a chunk of pinned
       memory for each GPU process whose last message had cells, as the
       ring is not trimmed after the last receive. A copy that the device
       may still be reading is freed with the ring, when its last user ends.
       `MERK_CHECK_PINNED` reports at the end how many chunks are left;
       without the fix, two hundred such processes left two hundred.
     - **Not on the GPU yet:** the string library's operations that call
       the runtime, such as `++` (`merk_string_append`), which the compile
       reports, so the test's GPU process sends its string back as it got
       it; and making closures, so it sums a list without `foldl`.
     Tests: `CodeGen/process-gpu-messages.m` and `-gpu.test`,
     `CodeGen/omp-target-term-messages.m` and `-gpu.test`, and the region
     error tests, which keep the higher-order case.

     **Leaks found meanwhile, in code before 4e, each to be fixed as a
     commit of its own after it:**
     - Fixed 2026-11-10, together, as messages that went through a
       device's ring: `TrimRing` unlinked the messages that a device had
       taken but never freed them, for every message any device received;
       `merk_mailbox_message` gave the chunks of a message that the host's
       side of `rpcTakeArray` read to a reader, and the plugin's server
       thread, which runs no process, kept them for ever; and `Close` kept
       the messages a ring had shown, as `TakeArray` copied a message after
       the mailbox's lock was released. The host now copies an array for a
       device with the lock held (`merk_mailbox_copy_array_for_ring`, since
       merged into `merk_mailbox_copy_for_ring`), as
       it copies a term, and answers whether the mailbox was open; then
       `TrimRing` and `Close` free the messages. `MERK_CHECK_MESSAGES`
       counts the queued messages that no process read and no mailbox
       freed; without the fix, the two hundred GPU processes of
       `process-gpu-messages-gpu.test` left four hundred.
     - Fixed 2026-11-10: in a program with GPU processes but no target
       regions nothing freed the rings, the arenas or the retired copies,
       since only the end of the last region did, and a GPU process's end
       runs on liboffload's thread, which must not call liboffload. Running
       that cleanup when the last GPU process ends would not have been
       enough: the GPU processes of one image shared its arena, which was
       freed only when nothing ran on any device, so a server whose GPU
       processes overlap would never have freed it. So each GPU process
       now has an arena of its own, as 4d planned: the host allocates its
       words on the device and passes them as the kernel's first argument,
       which the device's runtime reads from the kernel's dispatch packet
       in a process's image (`merk_device_process_image`, defined by the
       start object), so that no state is set up on the device. When the
       kernel ends, the host function on liboffload's thread closes the
       mailbox and takes its ring and the copy there
       (`merk_mailbox_close_ended`), and a thread of the runtime's, the
       reaper, frees them, the arena and its chunks, and the queue, and then
       counts the process as ended; the last of the regions and GPU
       processes to end runs the cleanup when nothing runs. A GPU process
       still counts as a running region, so that a ring of another
       mailbox that it reads stays. `MERK_CHECK_DEVICE_MEMORY` reports how
       many arena chunks and rings are left at the end; without the fix,
       the two hundred GPU processes of `process-gpu-messages-gpu.test`
       left two hundred of each.
     - Fixed 2026-11-10: `merk_mailbox_resolve` dropped the fragment of a
       monitor's message if the mailbox had closed meanwhile. Found with
       it: the message's new chunks were given to the reader without being
       counted as held, so `MERK_CHECK_MESSAGES` went below zero for each
       message a monitor made, five in `process-monitor.m`; and the check
       counted messages still queued in open mailboxes, which the processes
       that the program does not wait for may not have read yet. It now
       leaves those out, looking again if a message changes hands while it
       looks.

   **Decisions,** made by the user on 2026-11-10, each as recommended:
   - **D1. Which kinds get a kernel: those whose declaration says so,**
     such as `:- process counter(int) receives counter_msg on gpu`.
     merkc reports at compile time what a marked kind cannot do on a GPU.
     It is explicit, so its errors are reported where they belong.
     Rejected: every kind the device can compile, with the others skipped,
     and the kinds that some `spawn_<kind>_at` call names. The first
     compiles every kind twice and skips silently, which the rule against
     skipping known-unsupported features rules out. The second cannot see a
     spawn in another module.
   - **D2. How a place is named:** `spawn_<kind>_at(Place, Args, Mailbox,
     !IO)` (since 2026-11-11 `spawn(Place, Kind(Args), Mailbox, !IO)`, see
     "spawning names the kind" below) with `:- type place ---> host ; gpu(int)` in `process`, and
     `process.gpus(Count, !IO)` for how many there are. Spawning at `host`
     is `spawn_<kind>`. A GPU that does not exist, or a kind without a
     kernel, is a software error at the spawn. Rejected:
     `spawn_<kind>_on_gpu(N, ...)`, which names no place, though handles and
     routing are by place (design document, Handles, routing and places).
   - **D3. One message type for each kind in the first version,** and
     scalar and mailbox arguments for a GPU kind. Several message types,
     and other arguments, come later.
5. **Flat messages, and a GPU process's queue on its GPU.** Decided by the
   user on 2026-11-10, in place of "spawning from the GPU's other places,
   and the place table's identities", which said too little to plan: a GPU
   process's mailbox lives at its GPU's place, and its handle says so; and
   messages to and from GPU code are flat, and a GPU process makes no cells
   but arrays (design document, "Messages and memory on the device"). That
   replaces turns, decided earlier the same day, which the design document
   keeps among the rejected. Performance on the GPU comes first: what can
   be decided at compile time is, and the receiving side copies nothing.

   **Why the queue moves.** Today the host's list is the queue, and the GPU
   reads it through a ring in host memory, asking the host for each message
   that is not one word. Every message to a GPU process crosses PCIe on the
   GPU's request, two GPU processes on one GPU talk through the host, and
   the handle's place, the host's, cannot tell a sender that the receiver
   is on a GPU. The goal, messages between GPUs, needs the queue where the
   receiver is, so that a sender, the host or a peer, writes into it. Flat
   messages make that a write of bytes, with no pointer to fix up.

   **Steps:**
   - **5a. The checks by type, at compile time.** Built 2026-11-10 (design
     document, "Messages and memory on the device", As built): a message
     type that is not flat in a GPU kind's or a target region's `send` or
     `receive`, and a goal of a GPU kind's code that makes a value of a type
     that is not flat, are errors that name the type. The GPU tests of 4e
     send flat messages now.
     Moved to flat records' steps 1 and 2, decided while building 5a: the
     end of step 4e's term messages (the ring of terms, `rpcTakeTerm`,
     `rpcSendTerm`, `rpcPinnedChunk` and the device's use of `copy.h`), and
     strings as one-block messages. A flat record of several words is still
     a tree of cells on the host, and on the device until step 1, so its
     bytes are not yet the message; until then the `type_info` copy carries
     every message that is not one word or an array, marked Temporary.
   - **Flat records,** with 5a, since its check says what is flat (design
     document, "Messages and memory on the device"; the user, 2026-11-10:
     arrays may hold structs, as in OpenMP's custom reductions, with an
     internal array of fixed size and scalars, and temporaries that do not
     live across loops or recursions). The check accepts a type of one
     constructor, with no existential type or constraint, not recursive,
     whose fields are flat, and a tuple of flat types. Then, each a step:
     1. **By value in GPU code.** A flat record is an LLVM struct, in SSA
        values, never `merk_device_alloc`ed: construction builds the value,
        deconstruction reads its fields. A record received or passed in as a
        cell on the host's side is read into a value at the start.
     2. **Arrays of flat records, inline,** with C's layout, so that mapping
        an array to a region, or sending it, is one copy, and C or HIP code
        reads the same bytes. On the host too, which is the decision below.
        A message then travels as its bytes, which a wave's lanes copy
        together between the arena and host memory; arrays are copied word
        by word by one lane until then, since a DMA copy into or out of a
        running device's memory can meet stale lines of its L2 cache
        (design document, "The device's communication, as built").
     3. **Fields that are arrays of a fixed size,** with element reads by a
        variable index, as a histogram's bins are.
     4. **Custom reductions:** `reduction(F : V)` where `F` is a function of
        `(R, R) = R` on a flat record `R`, with an initializer, as OpenMP's
        `declare reduction` has, lowered to `omp.declare_reduction`, as
        Flang lowers one of a derived type.

     **Step 1's design, 2026-11-12: records by value everywhere.** Not
     only in GPU code: the user's (a) puts records inline in host arrays
     too, and one layout on both sides spares conversions at every map and
     message.
     - *Which types:* one constructor of two or more fields, no hidden
       ones, not exported abstractly, every field an integer, a float, a
       character, an enumeration that is not exported abstractly, a notag
       of one of those, or such a record. The rule reads only the
       `merk.type_def`s, so every module that sees a type's constructors
       decides alike; one that sees a field type only abstractly has no
       `merk.type_def` for it and decides it is not one, as does the module
       that exports it abstractly.
     - *`!merk.record<"name">`:* the value, an LLVM struct of its fields'
       types, a nested record inline. `!merk.term<"name">` keeps its meaning,
       the cell, which is what the runtime, polymorphic code and other cells
       see: a record is boxed into it by `merk.deconstruct` of the record
       and `merk.construct` of the term, and unboxed the other way, the
       existing operations, so liveness and safepoints need nothing new.
       Not a root.
     - *Where:* MLIRGen is unchanged. After lowering has chosen the layouts,
       before last call modulo constructor and safepoints, a rewrite makes
       every value of such a term a record: procedures' signatures, block
       arguments, results, environments' layouts. It boxes where a cell or
       a word is wanted: `merk.to_word`, a field of a cell, an array's
       element (until step 2), a foreign call's input, a cast to a pointer
       for a target region's map; and unboxes at the reverse. Every
       indirect call passes words already (closures, unify and compare
       entries, methods), so signatures change only for direct calls.
     - *Lowered:* construct to `insertvalue`s, deconstruct to
       `extractvalue`s, a test of the functor to true.
     - *Rejected:* boxing during lowering, as `ToBits` would: it allocates
       after safepoints are placed, so a collection there would miss live
       values. Records in GPU code only: two layouts of one type, and a
       conversion at every boundary between the host's and the device's.
       A record type in MLIRGen: every place there that handles terms
       would need a second case, where the rewrite has a few boundaries.

     **Step 1, built 2026-11-12** (design document, Type representation, and
     Lowering, "Records"). As designed above, with:
     - *Tuples* of flat types are records too, as the check has them.
     - *The choice* walks a type's fields with the set of types on the path
       to it, so a type in two fields, as a particle's two points, is
       checked twice, and only a type in its own fields is recursive.
     - *A record too large for registers* among a procedure's outputs is
       returned in memory, as any outputs are.
     - *Still boxed* until step 2: a record in an array, in a message
       (`send`, `receive` and the copies walk its `type_info`), and in a
       target region's map, an allocation on the host at each region. A map
       by value, the struct's bytes, is open for step 2.
     - *Tests:* `CodeGen/flat-records.m` (signatures, the boundaries where
       a record is boxed, `solutions`, results in memory) and
       `CodeGen/process-records.m` with `process-records-gpu.test` (records
       as GPU processes' messages and state, by value in their code).

     **Step 2's design, 2026-11-12: arrays of records inline.** Two
     parts, each a commit: 2a, arrays; 2b, messages and maps as bytes.
     - *C's layout.* A record's struct has each field at its C size and
       alignment: `int8` to `int32`, `char` and enumerations (4 bytes, as
       an array holds them and as C's `enum` is) at their sizes, `int`,
       `float` and the 64-bit integers at 8. Step 1 gave an enumeration a
       word; that changes, with a `trunc` and a `zext` at construction and
       deconstruction.
     - *The header.* A fifth width code, 4, for inline records, and a second
       header word, the element's bytes (C's `sizeof`, a multiple of its
       alignment); the elements start at word 2. The size stays in the low
       56 bits of word 0, so `merk.array_size` is unchanged, and the whole
       array's words come from the two words alone, which the collector,
       the copies and the maps need. *Rejected:* the element's bytes in
       the header's spare bits, which caps them at 255 or takes bits from
       the size and makes every size read depend on the code; a pointer to
       the descriptor in word 1, which a copy to a device would carry as a
       host address.
     - *The descriptors.* A record's, and an enumeration's and a notag's
       that is a flat field, get a flag, which `LowerTypeCtors` sets by the
       same rule as `ChooseRecords`, so that the runtime agrees with the
       compiler on what is inline, and on a tuple, whose descriptor all
       tuples share, from its arguments. The runtime computes a record's
       layout from its fields' kinds by C's rules, which are LLVM's for a
       literal struct on both targets; no table of offsets.
     - *Monomorphic code* reads and writes an element in place: a load or
       store of the struct at word 2 plus `i` times its bytes. Making,
       filling and resizing pass the element's bytes by pointer to the
       runtime (`merk_array_new_bytes` and the like), which copies them.
       `from_list`, whose elements are cells, passes the type_info.
     - *Polymorphic code* gets the element's type_info, as narrow arrays'
       making operations already do (MLIRGen's `ElementTypeInfo`), on
       `merk.array_get`, `array_set`, `array_fill` and `array_resize` too.
       A get tests the code: a record is boxed by the runtime, by the
       type_info, after a `merk_gc_poll`, since the runtime's allocation
       only grows the heap; any other is `merk.array_load` as now. A set,
       fill, resize or making unboxes a cell into the bytes, which
       allocates nothing. *Rejected:* a get entry in the descriptor, a
       Mercury procedure the compiler makes for each record type: no
       runtime code to box, but a second way to reach an element, and
       the runtime's boxing is needed anyway for `io.write`'s and the
       decoder's sake.
     - *The runtime's walkers.* The collector and the copies move an
       inline array as one block and scan nothing in it. `io.write` and
       the encoder read the fields in place; the decoder copies the bytes
       and checks each character and enumeration, as it does a narrow
       array's.
     - *2b, messages and maps:* a record message, and an array of records,
       travels as its bytes, and a target region's map of a record is its
       struct's bytes, not a cell. Designed when 2a is built.

     **Step 2a, built 2026-11-12** (design document, Type representation,
     and Lowering, "Arrays" and "Records"). As designed above, with:
     - *One rule in one place.* `lib/CodeGen/Records.h` decides which types
       are records from a module's `merk.type_def`s; lowering's choice, the
       descriptors' flag and MLIRGen's target regions all ask it, so they
       cannot disagree.
     - *Target regions* map an array of records by its words, as an array
       of numbers, in both directions, and the device makes such arrays
       too. That was planned for 2b, but needed nothing of 2b's.
     - *The decoder* also checks a record's padding is zeros, so that equal
       records keep equal bytes, and names what is wrong with a record:
       its padding, a character or an enumeration.
     - *Messages* of arrays of records between host processes are copied
       by their `type_info`s, as one block; a GPU kind still rejects them
       as messages, until 2b.
     - *Not built:* `array.sort`, `binary_search`, `foldr` and the others
       of `array.m` without clauses are not yet implemented for any element
       type, records included.
     - *Tests:* `CodeGen/record-arrays.m` (every array procedure on records,
       monomorphic and polymorphic, a field read alone in a loop, mmc's
       output, the collector at every allocation), `record-imports.m` (a
       record type of another module, and abstract ones, which are not
       records), `omp-target-record-arrays.m` and its GPU run (the mapped
       sizes), `process-record-arrays.m` (host processes), and the record
       arrays in `encode-checks.m` (golden bytes, round trips, hostile
       input).
     - *The oracle:* `generate-layout-programs.py --arrays` makes a quarter
       of its types of one constructor of two to four fields, and puts
       each type's values in arrays, read and set in monomorphic and
       polymorphic code (`CodeGen/oracle/layouts-arrays-1.test`). Of 200
       modules (seed 503), 110 with arrays of records, all print what mmc's
       print at -O0 and -O2, and the same with the collector at every
       allocation; so do the 460 modules of the earlier layout batches.

     **Decided by the user, 2026-11-12:** (a), inline on the host as on
     the GPU; and a fixed size as a library type, `fixed(T, n8)`, whose
     size is an ordinary type the library declares, so that the type
     checker keeps sizes apart and mmc still runs the programs.

     **Were open, before step 2:**
     - **How the host holds an array of flat records.** (a) Inline on the
       host as on the GPU, with a lookup boxing the element into a cell, but
       for a lookup whose result is only deconstructed, which reads the
       fields in place, the usual case; polymorphic array code then needs
       the element's size, from its `type_info` or from the specializer.
       (b) Boxed on the host, as now, and packed or unpacked at each map or
       send: a copy and a pass over the array each time, against the bar
       of C, which copies once. (c) A type of its own, such as
       `record_array(T)`, inline everywhere, leaving `array(T)` as it is.
       Recommended: (a), since the array stays the one the program names,
       and the copy is one block, as C's is.
     - **How a fixed size is written,** since Mercury's types have no
       numbers: a pragma on the type naming the field and its size; a
       library type whose size a type-level marker gives, `fixed(T, n16)`;
       or tuples, which have a size but no indexing by a variable. Wanted
       first: an example from the user's OpenMP codes, to choose by.
     - **Records of several constructors** (tagged unions, as messages
       already may be) inside arrays: later, if a program needs them.
     - **Arrays whose length is known only at run time,** in records
       (discussed 2026-11-10). An array of scalars or enumerations is flat
       today only as a whole message; a record or tuple with an array field,
       or an array of arrays, "has no fixed size". Two ways, not exclusive:
       - *Packed messages.* A record with array fields may be a whole
         message, as one block: its fixed part first, each array field as an
         offset from the block's start and a length, and the elements after.
         Offsets from the block, not pointers, so a store, a DMA copy or an
         RDMA write moves the bytes unchanged, and the receiver reads the
         message where it lies, with no unpacking. Such a record is never an
         array's element, as its size varies. It covers metadata with data,
         such as a halo message `{tile, step, cells}`, which C also sends as
         one buffer. Ragged arrays are one with two fields, the elements and
         their offsets, as in CSR. Recommended, with step 2, as both need
         the same layout.
       - *Bounded arrays:* a capacity fixed in the type and a length field,
         such as `bounded(T, n64)` (the spelling as for fixed sizes, below).
         Of a fixed size, so a field, an element or a value in registers, at
         the cost of the space up to the capacity, as GPU codes' neighbour
         lists with a maximum count have it. Step 3 with a length; when a
         program needs it.
       - *Not planned: temporaries of variable length inside records.* The
         user, 2026-11-10: processes are handed the data they work on, so
         they are unlikely to need array temporaries, and the limit is
         acceptable for now. A GPU process can still make an array in its
         arena, as a value of its own.
     - **Narrow fields' layout:** `int8` to `int32` and enumerations at their
       C sizes, so a record of small fields is small; a float is 64 bits, as
       Mercury has no 32-bit float, which GPU codes use widely, to be
       raised separately.
   - **5b. The place table.** Place 0 is the host and each GPU of the node
     has a place of its own, with its kind, its architecture and its
     identity: liboffload's UID where the GPU has a UUID, and otherwise its
     PCI address, from the kernel's topology in sysfs, as liboffload gives
     none (design document, "What the plugins lack"). A GPU process's handle
     carries its GPU's place. Nodes exchange their tables at the handshake.
   - **5c. The queue in the GPU's memory.** The mailbox's memory, on the GPU,
     owned by the mailbox: a sender reserves space in it and writes the
     message's bytes, and the receiver reads it there. A message that a
     selective receive passes over is moved to the host's list for the
     mailbox, so that it does not hold the queue's space.
   - **5d. Routing by place.** `send` to a handle of a GPU place: from the
     host, into the GPU's queue; from a GPU process on the same GPU, straight
     into the queue, with no host on the way; from another GPU, through the
     host until peers write to each other's memory, which this machine's one
     GPU cannot test.
   - **5e. Measurements,** in the design document: a ping-pong between two
     GPU processes on one GPU, through the host before 5d and direct after
     it, and from the host to a GPU process, before and after 5c. Each
     beside the same written directly in HIP or C, a ping-pong through
     pinned memory and a queue in device memory, so that the gap to code
     written by hand is measured, not guessed: that is the bar.

   **After step 5,** each a step of its own:
   - **Arrays freed at their last use** in GPU processes, where uniqueness
     shows it (compile-time garbage collection), so that a GPU process that
     lives long does not grow.
   - **`send_slice` and `send_tile`,** copying a range or a tile straight
     into the receiver's queue: a rectangular DMA, which liboffload does not
     expose, or a strided copy kernel.
   - **Multi-dimensional arrays and views** as strided descriptors, following
     `memref`.
   - **Shared arrays,** counted, in a heap of their own from libc's `malloc`
     on the device, with views; for reductions and pools of workers
     (Distributing work).

   **Open, to decide before the step that needs it,** by the experiments
   of Choosing among the ways, above:
   - **How the host writes into the GPU's memory (5c),** by E1. Copying each message
     with a DMA from pinned memory costs a copy's latency per message.
     Writing through host-visible device memory, a large BAR, which a
     consumer Radeon may not offer. Or the queue's slots in host memory, as
     now, with the GPU copying each message into its memory once. To be
     measured on this machine before choosing.
   - **What a GPU sender on the same GPU reserves with (5d):** an atomic on
     the queue's tail in device memory, and how a full queue spills.
   - **Whether a type with a string field is a message,** such as
     `greet(string)`: not one block, so not in the first version; a header
     and a tail would allow it.

   **Papers,** downloaded 2026-11-10 from mercurylang.org's papers page to
   `~/Downloads/Mercury/` (Execution, Memory, Parallelism), and what each
   serves:
   - *The execution algorithm of Mercury* (Somogyi, Henderson and Conway,
     JLP 1996): the type, mode and determinism information is used for
     speed, not paid for with it. The stance of the whole step.
   - Mazur's thesis, *Compile-time garbage collection for the declarative
     language Mercury* (KU Leuven 2004), and *Practical aspects for a
     working compile time garbage collection system* (ICLP 2001): freeing
     and reusing arrays at their last use, after step 5.
   - *Runtime support for region-based memory management in Mercury* (Phan,
     Somogyi and Janssens, ISMM 2008), Phan's thesis (KU Leuven 2009) and
     *Path-sensitive region analysis* (Phan and Janssens, PPDP 2009): a
     region's arena is a region in their sense; their analysis is the way
     to finer ones, if regions' code ever needs them. The ISMM paper
     measures regions about 25% faster than Boehm's collector on their
     benchmarks.
   - *Run time type information in Mercury* (Dowd et al., PPDP 1999): the
     generic operations that step 4e's walk is, which flat messages leave
     to the host.
   - *Compiling Mercury to high-level C code* (Henderson and Somogyi, CC
     2002), for code that the target's compiler optimizes well, and
     *Accurate garbage collection in an uncooperative environment*
     (Henderson, ISMM 2002), the shadow stack, for the host's collector in
     chunk 3, not the device.
   - Bone's thesis, *Automatic parallelisation for Mercury* (Melbourne
     2012): work stealing and parallelising only where profiling says it
     pays, for Distributing work.

**Rejected alternatives.**
- **The whole program on the GPU,** run by `llvm-gpu-loader` as libc's tests
  are. Half of the runtime lives on the host: the scheduler, TCP between
  nodes, the collector, and mailboxes that hosts and GPUs share. The loader
  serves only libc's opcodes. It stays an option for testing device code.
- **A process as a `nowait` target region.** It is still `omp target`, with
  OpenMP's maps and a libomp task on a host thread for as long as it lives.
- **A process's kernel in OpenMP's image,** launched by liboffload. The two
  would share globals, but libomptarget does not give the program handle that
  it made, and launching a kernel that OpenMP's lowering made would need its
  kernel environment. It could come later, if libomptarget gives the handle.
- **The CUDA and HIP runtime's registration** (`__llvmRegisterFunction`).
  That is the interface clang's generated code uses for those languages, and
  not one for other languages.
- **HSA directly.** It would redo the plugin, and liboffload covers CUDA and
  Level Zero devices too.

## The halo exchange

Proposed 2026-11-09; nothing is built. Implementation waits until several
machines are available, as real numbers need them. It is the design document's
second acceptance criterion (GPU processes): a stencil on a grid split across
GPUs, against an MPI+HIP program doing the same, to show whether communication
overlaps computation.

**First version, without overlap.**

- A 2D grid split in one dimension, each piece owned by one target region, with
  a left and a right neighbour. On one machine, several nodes on one GPU over
  loopback, as the ping-pong has.
- **One mailbox for each direction,** "from left" and "from right", made by the
  region that receives from it. A device's receive takes the first message
  (The device's side of the first slice, decision 4), so mailboxes do what MPI's
  tags do, and the type of a mailbox's messages is checked.
- **Each step:** copy the edge rows into arrays, send them, receive the
  neighbours', compute. A send hands over a unique array, so a buffer cannot be
  reused before the send has finished.
- This gives the baseline that overlap has to beat.

**Second version, with overlap:** start the interior, and fill in the edges when
they arrive.

**Open.**

- **A send from a device waits for the host,** so a region cannot compute during
  it unless the computation is in separate teams or regions. This is the main
  question for overlap, and it may bring back the send ring (The send ring), for
  overlap rather than latency.
- **Array speed,** 0.8 to 1.5 GB/s with a second copy on the host, limits a large
  halo.
- **The recovery benchmark** (a GPU process killed during the exchange) needs
  monitors across nodes, a way to report a dead GPU process, and a receive that
  does not wait forever.
- **The MPI+HIP program** to compare with is not written.
- **Decomposition in two dimensions** and corners are left for later.

## GPU processes that run loops: cooperative kernels

Proposed 2026-11-11 for step 3 of the demo (A demo: a stencil whose active
region moves). Decided by the user the same day: a GPU process runs its own
data-parallel loops, as a kernel of many teams launched cooperatively, rather
than a host process driving the GPU with target regions ("If we need to
implement something to make the sparse tiles work, we should implement
that"); and development is on the gfx1030, which drives the display, with
the process limited to some of its compute units by a CU mask.

**Why.** A GPU process today is one team, whose initial thread runs the
kind's predicate (GPU processes without `omp target`, 4b). A place that owns
many tiles must step them all with the whole GPU, between its messages, as
the design document has it: "Between messages it may run data-parallel
loops." Rejected: *the process launching each loop as a kernel from the
GPU,* by writing the launch packet into a queue, which would repeat the
plugin's arguments and environment for an OpenMP kernel on the device; and
*the process asking the host to launch each loop,* which costs a round trip
and a launch, 15 to 40 µs, every loop.

**What LLVM has,** read on 2026-11-11:
- **liboffload's API has cooperative launch:** the launch property
  `OL_KERNEL_LAUNCH_PROP_TYPE_IS_COOPERATIVE`,
  `olGetKernelMaxCooperativeGroupCount` and
  `OL_DEVICE_INFO_COOPERATIVE_LAUNCH_SUPPORT`, with unit tests
  (`offload/unittests/OffloadAPI/kernel/olLaunchKernelCooperative.cpp`).
  CUDA's and Level Zero's plugins implement it. AMDGPU's refuses it:
  "cooperative kernel launch not supported for AMDGPU".
- **ROCr** has cooperative queues (`HSA_QUEUE_TYPE_COOPERATIVE`, by GWS,
  the global wave sync), but not on gfx1030: its agent reports
  `HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES` false (probe, 2026-11-11), and 80
  compute units, all of them fully enabled for cooperative dispatch. Queues
  take a CU mask (`hsa_amd_queue_cu_set_mask`, in pairs on processors with
  work group processors, as gfx10's), ANDed with the process's
  `HSA_CU_MASK`.
- **No barrier across teams** in OpenMP's device runtime or ours.

**Design.**

- **(a) Cooperative launch in AMDGPU's plugin,** for upstream, as CUDA's.
  - *The largest count of teams* that are resident at once: the compute
    units the queue may use, times the teams a compute unit holds, from the
    kernel's metadata, its waves a team, VGPRs, SGPRs and LDS, against the
    processor's limits, as LLVM's AMDGPU occupancy computation counts them.
    At most `HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT` compute
    units.
  - *A launch of more teams than that is refused,* as CUDA's is.
  - *Residency:* on a processor with cooperative queues, the launch goes to
    one, which guarantees that the teams run together. Without them, as on
    gfx1030, teams that fit are resident unless other work holds the compute
    units, and then they become resident when it ends; what never ends, such
    as another cooperative kernel on the same units, can deadlock it. That
    is documented, as HIP's is for such processors (to check).
  - *Tests:* liboffload's existing unit tests, which skip without the
    support, and a kernel that crosses a barrier across teams.
- **(b) A CU mask.** For development on a GPU that drives the display, the
  process's GPU work is limited to some of the compute units, so that the
  display always has free ones. First by `HSA_CU_MASK`, which ROCr applies
  to every queue of the program, set by the tests that run cooperative
  kernels and by `utils/gpu/with-gpus.py` when asked. The count of (a)
  takes the queue's mask (`hsa_amd_queue_cu_get_mask`). Later, if places
  share a GPU (Several places on one GPU, open, in the demo), a mask a
  queue, which liboffload's API does not have yet.
- **(c) A barrier across teams,** in the device runtime: each team's thread
  0 arrives after a barrier of its team, on a counter in device memory with
  a sense that flips each time, and waits with `s_sleep`; then the team's
  barrier again. Its cost, measured, against a kernel launch.
- **(d) A process's kernel runs its loops on its other teams, by a schedule
  of ours.** Decided by the user 2026-11-11, replacing the first design,
  "a process's kernel as an SPMD kernel of OpenMP's", which could not be
  built:
  - *The omp dialect has no such kernel.* An SPMD `omp.target` must capture
    an `omp.loop_nest` ("SPMD kernel must capture an 'omp.loop_nest'
    operation"), a tight `target teams distribute parallel do`; sequential
    code with loops between messages is a generic kernel, in which every
    team's main thread runs the sequential code.
  - *One thread of a wave cannot run the process while the others wait.* A
    team's barrier (`s_barrier`) counts waves, so every lane of a wave must
    reach it on the same path; a wave whose lane 0 runs the process and
    whose other lanes wait elsewhere would execute each barrier twice and
    deadlock the team. OpenMP's generic mode keeps the main thread's wave
    out of the work for the same reason.
  Rejected with it: *emulating an SPMD kernel of OpenMP's by hand,* with the
  kernel environment, `__kmpc_target_init` and `__kmpc_parallel_60` around
  the process, so that `__kmpc_distribute_for_static_loop` runs the loops,
  which copies OpenMP's internal ABI into merkc and the runtime, and breaks
  when it changes, as the device environment's size did (2026-11-11); and
  *a kernel kind of its own in the omp dialect and OMPIRBuilder,* upstream,
  the most correct, but a large project before any tile; it could replace
  the schedule later.

  The design:
  - *The kernel* stays a plain kernel, launched by liboffload, now
    cooperatively, with as many teams as are resident at once, of 256
    threads unless the kind says otherwise.
  - *Team 0 is the process's.* Lane 0 of its first wave runs the kind's
    predicate, as the one team's initial thread does now; the other lanes
    of that wave end, and its other waves only take part in the barriers
    across teams. Every other team is a worker. Rejected: *team 0's other
    waves as workers too,* which a barrier after each round of a loop would
    deadlock, as the process's wave does not take part; one team of the
    launch is the cost, about 1% at 256 threads.
  - *A loop:* merkc outlines a loop's body in a process's predicate into a
    procedure of the kernel's module, with the values it uses passed in a
    block of device memory. At the loop, the process writes the loop's
    number, its bounds and the block into the process's control block, a
    barrier across teams starts the workers, and a second one ends the loop.
    Each worker team waits at the barrier, reads the control block, and
    calls the loop's procedure through a `switch` that merkc makes over the
    module's loops, so there are no function pointers. The process ending
    tells the workers to end.
  - *The schedule,* in `libmerkdevice`: the worker teams' threads take
    iterations in rounds, a block of consecutive iterations to each team
    and one to each thread, with a team barrier after each round, as the
    barrier after each round does for OpenMP's loops (Dense stencils).
    Reductions: each thread's partial result, combined in the team, then
    across teams after the barrier that ends the loop, by the process.
  - *The directive:* `%$omp distribute parallel do`, without `target teams`,
    as the process's kernel is the teams; a loop under another directive in
    a process's predicate is reported. Its checks are a parallel loop's.
  - *Memory:* the control block and the values' blocks are in the process's
    arena; arrays the process makes are in device memory, which every team
    addresses, so a loop reads and writes them in place.
  - *Sharing a GPU between its processes,* decided by the user 2026-11-11.
    Two processes that each took as many teams as fit would hang: the
    second cannot become resident while the first waits for it, as pinger
    and ponger of `process-gpu-messages.m` would. So each process's queue
    has a CU mask of its own, and the runtime partitions each GPU's compute
    units (of those `LIBOMPTARGET_AMDGPU_CU_MASK` leaves): a kind whose
    module has no process loops runs as one team on a small set of compute
    units that such processes share; a kind with loops takes compute units
    of its own, by default all those free, and a spawn that finds none
    fails with an error that says what holds them, rather than hang. Its
    teams are counted on its queue's mask, so they are resident at once.
    Rejected: *a budget of teams on queues that share the GPU,* whose
    kernels' work groups need not fit together, as the hardware places
    them and kernels differ in registers, so a hang stays possible; and *at
    most one process with loops on a GPU,* which would allow one tile
    process a GPU. Replaced 2026-11-11 by one kernel for a GPU's
    processes (below), as queues of their own run out and kernels that
    share a queue may run one at a time.
  - *Parts,* each a commit of its own: (d1a) queues with a CU mask of their
    own in liboffload (`olCreateQueueWithComputeUnitMask`, as HIP's
    `hipExtStreamCreateWithCUMask`), each on an HSA queue of its own, and
    the cooperative count on a queue's mask
    (`olGetKernelMaxCooperativeGroupCountOnQueue`), an upstream candidate;
    (d1b) the runtime's partition of each GPU's compute units; (d1c) the
    kernel's teams, the control block, the workers' loop and the end,
    without loops, the existing GPU process tests passing; (d2) loops without
    reductions, outlined, with the schedule; (d3) reductions; (d4) the cost of a
    loop and of the barrier across teams, measured, against the same loops in
    target regions.

**(d1a) done 2026-11-11:** queues with compute units of their own.
- `olCreateQueueWithComputeUnitMask` gives a queue a stream on an HSA queue
  under `hsa_amd_queue_cu_set_mask`, which the queues of one mask share, made
  for the first and destroyed with the last. An HSA queue for each first
  oversubscribed the GPU's scheduler when lit ran processes in parallel
  ("Runlist is getting oversubscribed", 107 times in a minute, 2026-11-11), as
  each process has a queue of its own. Kernels on one HSA queue run at once: a
  kernel packet does not set the barrier bit, and a stream pushes a barrier
  packet, which holds up the whole HSA queue, only behind an operation of its
  own, which a process's kernel, the first on its stream, never is; a host
  function is a signal handler, not a packet. But a kernel that uses scratch
  may run alone (One kernel for a GPU's processes, below), which the API's
  text says since. A queue destroyed while busy is
  never reused by another, but released once done, when a queue is next
  created or destroyed, or with its context. A mask must set some of the units
  that `olGetDeviceComputeUnitMask` gives: `LIBOMPTARGET_AMDGPU_CU_MASK`'s, or
  else ROCr's for its queues (none known under `HSA_CU_MASK`, the ROCr 7.2
  bug, which is reported).
- `olGetKernelMaxCooperativeGroupCountOnQueue` counts on the queue's own
  mask by the KFD model of (b); a cooperative launch on the queue is
  refused above it.
- Measured on the gfx1030 with a kernel that records each work group's
  HW_ID1: all 80 units ran on 40 WGPs, units 0-39 on 20 (5 in each SE),
  0-3 on SE 0's and SE 1's first WGPs, 0-1 on SE 0's alone, 2-3 on SE 1's
  alone: the mask reaches the hardware, dealt as (b) models it.
- Found on the way, fixed first as commits of their own: a queue destroyed
  while busy was never reused (`getOutstandingQueue` tested an `Expected`
  for an error, not its value), and a queue destroyed while idle lost its
  `__tgt_async_info`, about 110 bytes each.
- Tests: OffloadAPI unittests `olGetDeviceComputeUnitMask`,
  `olCreateQueueWithComputeUnitMask` (64 busy limited queues destroyed in
  turn, with plain queues between), and
  `AllGroupsResidentOnLimitedQueue`: on half the units the count is below
  the device's, every work group meets at the barrier, and one more is
  refused.

**Decided 2026-11-11: one kernel for a GPU's processes,** replacing (d1b),
(d1c) and the partition of compute units between processes. The user: "I
think we should go with one kernel per gpu", with a kernel for each phase of
a computation, as "having too wide of a variety would perhaps affect
performance because of register pressure". Found while testing (d1b), which
gave each process a kernel of its own on a queue limited to the compute
units that processes share:

- *Queues run out.* KFD maps 8 compute queues of the gfx1030 to its hardware
  (`num_cp_queues` in the KFD topology), for all programs together, and as
  many processes as it has VMIDs. Beyond them it time-slices the queues
  ("Runlist is getting oversubscribed"), by preemption, which failed and
  reset the GPU that drives the display on 2026-11-11 ("qcm fence wait loop
  timeout expired ... unsuccessful queues preemption"). A GPU test
  program uses 3 HSA queues at most (sampled from
  `/sys/class/kfd/kfd/proc/*/queues`); the four GPU tests that lit runs at
  once (`MERK_LIT_GPU_JOBS`) need 12. A queue for each process with loops
  would allow a few of them a GPU here.
- *Kernels that share an HSA queue may run one at a time.* A dispatch that
  needs more scratch than its HSA queue has makes ROCr give the queue more;
  if ROCr does not keep it, the scratch serves one dispatch at a time
  (`AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE`, read in ROCr 7.2.0's
  `amd_aql_queue.cpp`, not measured). Measured with two kernels that each
  wait for the other on one HSA queue (a timeout of about 2 s makes ones that
  run in turn report it), at 3181ee73b1fc:

  | The kernels | Ran at once |
  | --- | --- |
  | no scratch, or a fixed frame of 260 bytes | yes |
  | a dynamic stack of 1024 to 1792 bytes a lane | yes |
  | a dynamic stack of 1920 bytes or more, or the plugin's 16 KiB | no, in either order, also beside a kernel without scratch |
  | 16 KiB after a kernel that grew the queue's scratch, or with `HSA_SCRATCH_MEM=16384` | no |
  | 1792 bytes after kernels with scratch on 1 other queue / on 3 | yes / no |

  So whether two processes run at once depends on the scratch of the
  program's other queues: no stack size is safe. `process-gpu-messages-gpu`
  hung that way under (d1b): its kinds use a dynamic stack, and pinger waited
  behind ponger.
- *Where the hardware places kernels is not known.* Kernels that differ in
  registers might fragment a SIMD's register file, so a count of what fits
  need not hold for kernels of several kinds.

The robust shape is one dispatch that never ends on each HSA queue, and so
one kernel for a GPU's processes.

- *The kernel and phases:* `merk_place`, one in each module's process image
  (as each module has one), with a `switch` over the module's kinds and
  loops. A module is a phase: the loops of one computation, such as a
  stencil's, go in one module and share its kernel, and a computation with
  phases that want different code starts each phase's main process from a
  module of its own. The runtime launches a module's kernel cooperatively at
  the first spawn of one of its kinds on a GPU, on a queue limited to the
  compute units that `LIBOMPTARGET_AMDGPU_CU_MASK` leaves, with as many teams
  of 256 threads as are resident at once there
  (`olGetKernelMaxCooperativeGroupCountOnQueue`), and waits for them all to
  start before any spawn uses it.
- *Scratch can hold teams back* (found 2026-11-11, gfx1030, ROCm 7.2.0).
  The cooperative count takes the kernel's registers and local memory, not
  its scratch. A kernel with a dynamic stack gets the plugin's size, 16 KiB
  a lane unless `LIBOMPTARGET_STACK_SIZE` says otherwise, and of a kernel of
  four teams the GPU ran one: the others, held back until a team ended,
  never started, and a process in their slots never ran. With 1 KiB all
  four ran. A call of a function that is not `norecurse` makes the stack
  dynamic, and `llvm.trap` is not `nocallback`, so LLVM infers `norecurse`
  for no function that traps: the device runtime's `Fail`, which every
  kernel calls, ended with a trap that it never reaches, as the host ends
  the program first. It ends the wave instead (`s_endpgm`), and a kernel
  without recursion has a fixed frame (`process-kernels.m` checks it).
  Code that recurses, or a call of a generated procedure that traps and is
  not inlined, still makes the stack dynamic, as `process-gpu-messages`'
  kinds and `process-gpu-arrays`' at -O0 do. So the runtime launches the
  kernel with a stack of its own, 1 KiB a lane, now 1536 bytes (below)
  (`MERK_GPU_PROCESS_STACK_BYTES`), through liboffload's launch property
  `OL_KERNEL_LAUNCH_PROP_TYPE_STACK_SIZE`, which we added to the AMDGPU
  plugin for it. And each team counts itself started
  (`PlaceTable::started`), and the launch waits for them as a target
  region's teams wait for each other (`MERK_TEAMS_STALL_MILLISECONDS`): a
  spawn that launches a kernel whose teams do not all start fails, naming
  the counts (`process-teams-gpu.test`).
- *The stack is checked* (2026-11-11). A GPU's stack overflows silently,
  and a kernel's fixed frames take part of it (`process-gpu-messages`'
  took 1056 bytes, more than its 1 KiB). So:
  - *At compile time,* each call in GPU process code of the module that may
    recurse and is not a tail call is warned of, after lowering to the LLVM
    dialect, where tail calls are known (`WarnGpuRecursion`).
  - *At load,* the host's compile has read what the kernel's calls take
    outside any recursion from the image's metadata
    (`.private_segment_fixed_size`, if `.uses_dynamic_stack`), and
    registered it. The runtime refuses an image whose calls take more than
    the stack, naming `MERK_GPU_PROCESS_STACK_BYTES`, as nothing checks code
    that does not recurse, and otherwise sets the image's
    `merk_device_stack_bytes` to the stack.
  - *At run time,* each procedure on a cycle of calls that are not tail
    calls, or that makes an indirect call, checks at its entry, after the
    optimizer (`AddDeviceStackChecks`), that the stack pointer, after its
    frame, plus what its calls may take below it fits in the stack, and
    otherwise ends the program with "stack overflow in a GPU process". What
    the calls may take is the backend's: `llvm.amdgcn.callee.stack.size`,
    which we added to the AMDGPU backend for it, is the function's resource
    usage symbol `.L<f>.private_seg_size` (its frame and its callees' most,
    leaving out calls back into its cycle) less its frame, a literal that
    the asm printer emits once the frame is known. A first version compared
    the stack pointer with one limit, the stack less the kernel's fixed
    frames, which is safe but fires falsely at a check on the kernel's
    deepest path, whose frames above it count twice:
    `process-gpu-messages`' first copy of a message failed so with 1536
    bytes, and would have needed about twice its 896 fixed bytes.
  - *The optimizer's inferences are dropped* where the checks make them
    false: it had found the recursive procedures, and those that call
    them, to touch no memory and need no flat scratch (`amdgpu-no-*`), so
    the kernel did not set flat scratch up, and the report of an overflow,
    which builds its RPC port on the stack through a flat pointer, faulted
    at 768 bytes a lane (0x6000, 768 × 32 lanes). `ForgetInferredAbove`
    drops them from the checked procedures, their callers and their calls;
    the device's link infers the `amdgpu-no-*` again.
  - *The runtime's copy of a message checks too:* it recurses on the
    message's structure (`copy.h`, `CopyTyped`, a Memory's `CheckStack`,
    inlined into the copy, with `llvm.stacksave` and the callee stack size
    as above), and its scalar fields take no call, so that a flat record
    does not recurse. The device's `Fail` sends its text in two parts, which
    the host joins, rather than build it in a 160-byte buffer on the stack,
    which was the copy's deepest path: `process-gpu-messages`' fixed frames
    went from 1056 to 896 bytes, 544 of them `merk_process_dispatch`'s
    (device runtime code that LTO inlined, and saved registers: AMDGPU's
    calling convention has a callee save the VGPRs v40-v47, v56-v63, ...
    that it uses, 4 bytes a lane each; `CopyTyped`'s 176 bytes are all
    saves).
  - *The default stack is 1536 bytes:* a copy level takes 176 bytes, and
    with 1 KiB a message with a mailbox in it had 128 bytes below the fixed
    frames, where it had overflowed silently before the checks.
  - *How large a stack can be:* ROCr allocates scratch for every wave slot
    of the GPU, 80 CUs × 32 slots × 32 lanes on the gfx1030, and keeps it
    bound to a queue, reused, only up to `HSA_SCRATCH_SINGLE_LIMIT`, 140 MiB
    by default: 140 MiB / 81920 lanes is 1792 bytes, the boundary measured
    above. Above it, the scratch serves one dispatch, which holds a kernel's
    teams back. Without async reclaim, all such bound scratch is limited to
    an eighth of the scratch pool too. The hardware allows about 256 KiB a
    lane. Raising `HSA_SCRATCH_SINGLE_LIMIT` from the runtime is not yet
    measured.
  Open: the measurements above say no stack larger than about 1792 bytes is
  safe beside other queues' scratch, and 1536 is not measured beside three;
  `merk_process_dispatch`'s 544 bytes; `nocallback` on `llvm.trap` upstream;
  the same checks in target regions' code, whose stack is the plugin's.
- *Phases take turns on a GPU's compute units.* A kernel ends when its last
  process ends, which frees its units for the next phase's. A spawn of a
  kind of another module while a kernel holds them fails, naming the module
  that holds them, rather than waiting, which could deadlock if the running
  processes wait for the new one. So a program ends a phase, as by its main
  process ending once its workers report, and then spawns the next phase's
  main process. A kernel launched after another on the same HSA queue waits
  for it to end, which is safe as it does end, whatever ROCr does with
  scratch. Two computations at once on one GPU are an exception: kernels on
  disjoint compute units, each on a queue of its own, as few as the
  hardware's queues allow.
- *Waves and teams:* a process without loops takes a wave, whose lane 0 runs
  the kind's predicate, so a team of waves that never cross a team barrier
  holds 8 of them on wave32; a process with loops takes a wave for itself and
  whole teams as workers, which cross team barriers together and a barrier
  across the process's teams (c), at which the process's wave arrives alone.
- *Spawn:* the host's runtime finds free waves and teams, writes the
  process's slot (kind, arguments, arena, mailbox) and releases its
  assignment word, which idle waves poll, sleeping. A spawn that finds too
  few fails, with what holds them, instead of hanging. The end of a process
  is written to its slot, which the host's bridge thread reads, as there is
  no kernel end to follow.
- *Scratch:* the kernel's one dispatch has all of it, at the kernel's private
  size for every resident wave, 512 KiB a wave at the plugin's 16 KiB a lane,
  640 MiB over 40 compute units; a per-launch stack size in liboffload would
  make it the program's choice.
- *Balancing work.* Which worker takes which work, from a pool or by
  stealing, stays library code over processes and mailboxes (Distributing
  work, above), with either shape. But one kernel gives every GPU waves
  that are already running, which can take work from its own queue, a
  dispatcher or a peer's without the host; a spawn that is a write to a
  slot, microseconds, which fits or fails, instead of a launch of 15 to 40
  µs and a guess at residency; and worker teams that move between the
  phase's processes between chunks of a loop, if loops take chunks from a
  counter rather than a static split. Kernels of their own fix each
  process's teams for its life, and a GPU's free compute units belong to no
  kernel that could take work.
- *The cost:* the kernel's registers, and so its occupancy, are those of the
  module's heaviest kind or loop, measured with (d4) against kernels of
  their own, which phases in modules of their own keep to what a phase
  needs; the kernel holds its compute units while its phase has processes,
  idle or not; a phase runs only the code of its module; and a kernel
  launched again for the next spawn after its last process ended costs a
  launch, to measure in (d4), with a short wait before a kernel ends if it
  shows. A fault in a process ends them all, as ROCr ends the program on a
  GPU memory fault anyway.
- *Rejected:* *kernels of their own on a queue the processes share* ((d1b)
  above), by the measurements; *a small stack for process kernels,* whose
  safe size depends on the rest of the program; *an HSA queue for each
  process,* which runs out at 8 here; *raising ROCr's limits*
  (`HSA_SCRATCH_MEM`, `HSA_SCRATCH_SINGLE_LIMIT`), which apply to every
  queue of the program, each then with scratch for all 80 compute units,
  2.7 GB at 16 KiB a lane; *all of a program's kinds in one kernel,* linking
  the modules' process images into one, whose registers would be those of
  the heaviest kind of any phase; and *phases' kernels sharing compute units
  at once,* which brings back what fits and scratch.
- *What (d) above keeps:* the cooperative launch, the barrier across teams
  (c), the outlined loops and their `switch`, the schedule and the
  directive. What changes: a process takes a wave, not team 0, and its
  teams are those the runtime gives it in the kernel, not the whole launch.
- *Parts,* each a commit of its own: (p1) `merk_place` with processes
  without loops, launched at a module's first spawn on a GPU and ended with
  its last process, with spawns that do not fit, or that find another
  module's kernel, failing, the GPU process tests passing; (p2) processes
  with loops: worker teams, the control block and the barrier across a
  process's teams; then (d2), (d3) and (d4) as above. Of (d1b), the
  runtime's lists of each GPU's compute units stay.
- *Open:* how data outlives a phase, as a process's arena is freed when it
  ends: the ending phase sends what the next needs, or it lives in memory
  that the program owns rather than a process; whether the queues of one
  mask still share an HSA queue, which only one queue a GPU would use.

**(p1) done 2026-11-11.** The device's compile makes a start
`merk_process_<kind>`, which loads the kind's arguments and mailbox handle
from the process's words, and `merk_process_dispatch`, which starts a kind
by its number, its place among the module's GPU kinds; the host registers
the kinds' names with the image. `merk_place` (`runtime/device/
process-start.cpp`) is the image's one kernel. `runtime/place.h` is its
table, in host memory: a slot for each wave, with state, kind, arena and
words, and a flag that ends the kernel. Lane 0 of each wave polls its slot
with `s_sleep`, clears the process's arena, runs it and sends
`rpcProcessEnded`. The host (`runtime/gpu-processes.cpp`) launches the
kernel cooperatively on one queue per GPU, with as many teams of 256 threads
as are resident at once, and frees each launch's table after it ends. A
spawn fails, rather than waits, if another module's kernel runs or no slot
is free. What differs from the plan:
- *The kernel's compute units.* On every unit that the mask leaves, the
  kernel holds every team slot, so target regions beside a GPU process
  never became resident: `omp-target-region-rings-gpu` hung. For now the
  kernel takes the first pair of units, as (d1b)'s processes of one team
  did (Temporary in `Ready`). Open below.
- *Tests:* `%gpu-env` in `test/lit.cfg.py` sets
  `LIBOMPTARGET_AMDGPU_CU_MASK`, `0-39` unless the environment gives one,
  for programs that spawn GPU processes. `process-places-gpu.test`: a
  module's processes one after another, a spawn of another module's kind
  while the first module's kernel runs, and spawns until no slot is free,
  which fails. Its last run, which fills every slot, has not been run on
  the gfx1030.
- *A phase that follows another* needed a monitor of a GPU process, done
  below: a process answers before its wave tells the host that it ended,
  so a spawn of the next phase's kind right after the answer could still
  find the kernel running.

**Monitors of GPU processes, done 2026-11-11.** A GPU process has a record
on the host (`NewGpuProcessRecord` in `runtime/process.cpp`), which owns its
mailbox and is never run, so that `monitor` and `demonitor` work on it as
on a host process's. When the device reports the end (`rpcProcessEnded`),
the host frees the slot, ends the kernel if it was the last, closes the
mailbox, fires the record's monitors (normal; a fault in a GPU process ends
the program) and frees the record, in that order: a watcher that spawns the
next module's kind on the down message finds the kernel ending, and the
spawn launches the next kernel after it. A monitor made after the end says
noproc, by the closed mailbox. A host process that receives from a GPU
process's mailbox is now an error, as for any other process's, where it
used to take the mailbox over. Tests: `process-places`, cases `phases` and
`steal`. Rejected: monitors on the mailbox itself, which a close before the
end, or the slot's reuse, would leave pointing at another mailbox.

Found while testing, each fixed in a commit of its own first: the device's
compile copied a library descriptor without those its fields name
(`process.down_info`'s); and the runtime's errors ended the program with
`std::exit`, whose handlers freed host memory that a running kernel read,
so the GPU faulted.

**(p2) done 2026-11-11: loops across a process's teams.**
- *The directive:* `%$omp distribute parallel do` on a `do` loop, in a GPU
  kind's code or a predicate of the module that it calls. The driver passes
  `-fprocess-loops` to the kernels' compile when the program is compiled
  with `-fopenmp`, which reads the directives without OpenMP; without
  `-fopenmp` they are comments and the process runs its loops itself. On
  the host, as in a kind spawned there, it is a parallel do. check-gpu-code
  reports any other directive in a GPU kind's code, the loop outside such
  code, and the loop in a target region, as not yet implemented: there it
  would be OpenMP's, under `target teams`. Reductions are not yet
  implemented (`process-loops-errors.m`).
- *The compile:* each loop's body is outlined, numbered, and called from
  `merk_process_loop` by its number with the counter and the loop's words
  (`kProcessLoopWords`, 32), which the process writes into its control
  block (`merk_device_loop_values`) before `merk_device_process_loop`. The
  host's registration says which kinds run loops (`loops`, a `'0'`/`'1'`
  string by kind number).
- *The runtime:* a spawn of a kind with loops reserves as its worker teams
  every team of the module's kernel whose slots are all free, other than
  its own, and fails, saying so, if there is none (`ReserveWorkers`). Under
  `set_worker_teams(Place, share(K))` it takes only the first of them up to
  a K'th of the kernel's teams, at least one, so that K such processes run
  loops at once (step 4 of the demo, (m1), 2026-11-12). The
  reserved slots record the process's slot, the team's number among the
  workers and how many; the process's wave makes its control block in its
  arena and sets them working. The waves of a working team take
  iterations in rounds (a team's threads a block of consecutive ones, the
  teams apart by all of their threads), with a barrier across the team
  after each round (d2, 2026-11-11), so that its waves stay together, as
  the device runtime's loops do for stencils, and a
  barrier between the process and its workers (`ProcessArrive`,
  `WorkerArrive`) starts and ends each loop. When the process ends its
  workers set their slots reserved again before it tells the host, which
  then sets them idle. Every lane of a wave now stays in `merk_place`, so
  that a worker team's whole wave works; lane 0 alone runs a process.
- *Tests:* `process-loops.m` (the compile, and runs at the host, parallel
  and sequential) and `process-loops-gpu.test`: loops over arrays of one
  to a million elements, a loop through a predicate, a negative step,
  floats, an empty loop, an echo beside the worker, a second worker that
  finds no team, and the arena's memory freed.
- The kernel took the first pair of compute units then, so a worker had
  few teams; the percentage below replaced it.

**(d3) done 2026-11-11: reductions in loops across a process's teams.**
- *The compile:* a reduction's value before the loop goes into a word of
  the control block's results (`merk_device_loop_results`), and
  `merk_device_process_loop` takes the number of reductions, at most 8
  (`kProcessLoopReductions`); after the loop the results' words are the
  reductions' values. Beside `merk_process_loop`, which now also takes
  the thread's partial results, loading each reduction's value from them
  and storing the body's next value back, the compile makes
  `merk_process_loop_start`, which sets them to the operators'
  identities, and `merk_process_loop_reduce`, which combines two sets by
  the operators; both switch on the loop's number. The identities and
  combiners are those of the target regions' `omp.declare_reduction`
  (`ReductionIdentity`, `ReductionCombine`).
- *The runtime:* each worker thread keeps its partials on its stack. After
  its last round a team combines them: in each wave by exchanging them
  with the lane half a wave, then a quarter ... away (`__gpu_shuffle_idx`),
  then the waves', by the team's first thread, from the team's local memory
  (`wavePartials`), into the team's words after the control block. The
  process, after the barrier that ends the loop, combines the teams' words,
  in team order, into the results. Floating-point sums so depend on the
  number of teams, as a target region's do on its launch.
- *Tests:* `process-loop-reductions.m` (the compile, and runs at the host,
  parallel and sequential) and `process-loop-reductions-gpu.test`: a sum
  from a value other than the identity, a minimum and a maximum in one
  loop, floats, an exclusive or, a count beside an array updated in place,
  a product, a maximum of floats and a loop with no iterations, of one to a
  million iterations.

**(s1) done 2026-11-12: loops across a team's threads, in the bodies of
distribute loops,** for the demo's sparse tiles (A demo, step 3), whose
cells a team steps.
- *The language:* `%$omp distribute` before a loop in a GPU process's code
  runs its iterations across the process's worker teams, a team's first
  wave running each iteration's body. A loop under `%$omp parallel do` in
  that body, or in a predicate of the module that it calls, runs across the
  team's threads, with reductions. As OpenMP's `distribute`, the outer loop
  takes no reduction clause; its body updates arrays in place, each
  iteration its own elements. On the host, as in a kind spawned there,
  `distribute` is a parallel do, and a `parallel do` in its body a nested
  one, which OpenMP runs on one thread unless nesting is enabled.
- *The checks* (`check-gpu-code.cpp`): each body is followed in one of four
  contexts: a region's, the process's lane, a distribute loop's body, and a
  loop's body across threads. Errors, each naming its case:
  - a parallel do on the process's lane;
  - a loop under distribute or distribute parallel do in the body of
    another loop;
  - a parallel do in the body of a loop across threads.

  The runtime checks the same, and fails with a message rather than hang,
  for code the checker cannot see.
- *The compile:*
  - A distribute loop goes to `merk_device_process_distribute`. Its rounds
    have no barrier, as only a team's first wave runs them.
  - A team loop's words go into the team's local memory
    (`merk_device_team_values`, `merk_device_team_results`), and
    `merk_device_team_loop` runs it.
  - Team loops' cases are in `merk_team_loop`, apart from
    `merk_process_loop`'s. One function would make the kernel's calls
    recurse, from a distribute body through the runtime back into the
    switch, which makes the kernel's stack dynamic; ROCr's dynamic scratch
    serializes dispatches.
  - The compile finds the bodies that distribute loops reach
    (`CollectTeamBodies`), so that a parallel do elsewhere in the module,
    in the host's code, stays as it is.
- *The runtime* (`Distribute`, `TeamShare`):
  - The first wave of each worker team runs the team's iterations,
    `workers` apart. All 64 lanes run with the same values, in lockstep, so
    an update of an element that the body read is one load and one store.
  - The other waves wait at the team's barrier. At a team loop, the first
    wave writes the loop into the team's local memory and crosses the
    barrier. Every thread takes its rounds, with a barrier after each, as in
    a loop across the process's teams; the reductions are combined across
    the team's waves (`ReduceAcrossTeam`); and the team crosses a barrier,
    after which the first wave reads the results. When its iterations end,
    the first wave says so and crosses once more.
  - The branch between the first wave and the others is on a value that
    the compiler knows is the wave's own (`readfirstlane`), so that no wave
    runs the other's barriers. AMDGPU's backend also keeps the skip of a
    block with no active lanes around barriers and calls
    (`SIInstrInfo::hasUnwantedEffectsWhenEXECEmpty`).
- *Cells made in a distribute loop's body:* the first wave's lanes make one
  between them, which their first lane bumps (`merk_device_alloc`,
  `teamMains`). With one each, a team loop filled only the array of the
  lane whose words won the team's local memory, and a lane reading its own
  saw it unfilled. The test shows it: a team loop's bound is read from an
  array that the body made and another team loop filled.
- *Rejected:*
  - *OpenMP's generic mode,* one thread of a team running the sequential
    code while the others wait in a state machine. On AMD a wave crosses a
    barrier as a whole, so the thread's wave could not join the team loops'
    barriers while the thread ran apart, and the wave would sit them out.
  - *Every wave running the body:* waves do not run in lockstep, so two
    waves' updates of an element that each read race, and both add to it.
- *Tests:* `process-team-loops.m` (the compile, and runs at the host at -O0
  and -O2, parallel and sequential) and `process-team-loops-gpu.test`, from
  one tile to 400, more than the GPU has teams; the errors in
  `process-loops-errors.m`.

**(s2) done 2026-11-12: narrow array elements,** for the demo's tiles of
`uint8` cells (A demo, step 3), which had been a word each.
- *The layout* (`runtime/array.h`): an array's first word is its header,
  the size in the low 56 bits and the elements' width in the top byte.
  Width 0 is a word each, every type's but these, so a word array's header
  is its size, and its layout is as before. Widths 1, 2 and 3 are 1, 2 and
  4 bytes each, for `int8` and `uint8`, `int16` and `uint16`, and `int32`,
  `uint32`, `char` and types of constants only, as C holds them. The
  elements take whole words, the last padded with zeros.
- *Signedness is not recorded.* As a word, a narrow integer is
  zero-extended wherever it is one (`ToBits`), so a narrow element read as
  a word is the word that boxing it gives, and the element's type, which
  every reader that needs its value has, a type_info or a known type, says
  how to extend it.
- *Polymorphic code* has only the pointer, so the width is in the array:
  - A get or set of a type variable's elements branches on the header's
    width, in `merk.array_load` and `merk.array_store`, which lowering adds
    to the module, always inlined.
  - An array made for a type variable's elements, `merk.array_new` and
    `merk.array_from_list`, takes the type_info of what the variable stands
    for (`width of %ti`), and the runtime's `merk_array_width` reads the
    width from its descriptor's kind.
  - Another module's abstract type is treated as a type variable, since
    lowering does not see its layout, but its descriptor may say it is an
    enumeration.
  - The width of a known type follows the descriptors' kinds, the runtime's
    rule, so the two cannot disagree: a notag of an `int8`, an `i8` to
    LLVM, is a word, as its descriptor is a notag's.
- *Everything that copies an array* copies its words, as the header says:
  the collector, which follows no narrow element; messages' copies, local
  and to and from a device, whose rings carry the header; target regions'
  maps, over `merk.array_words`; and the device's copies to the host.
- *On the wire* a narrow array is its length and then its elements' bytes,
  padded to 8, as a string's are, so that it is one run, as arrays of numbers
  are meant to be (The wire format, first version, below). Each character
  must be a code point and each constant one of its type's.
- *Also needed:* arrays of enumerations may now cross into target regions;
  `array.map` and `array.to_list` are compiled.
- *Rejected:*
  - *The width from a type_info at every access* in polymorphic code: every
    get and set would need the type_info, and the runtime's copies, which
    have none for a root array, could not size an array.
  - *A second header word* for the width: two words for every array, and
    the word arrays' layout would change for nothing.
  - *Signedness in the header:* nothing needs it (above).
  - *One byte for enumerations:* a type of more than 256 constants would
    need a second rule; four bytes hold any, and `uint8` is there for cells.
  - *One item per element on the wire,* as the first version had: a `uint8`
    array would take eight times its bytes, and not be one copy.
- *Tests:* `narrow-arrays.m` (every narrow type, written, unified and
  compared, through the library's polymorphic code and the test's own, kept
  by the collector at every allocation, with mmc's output);
  `omp-target-narrow-arrays.m` and its GPU run, which checks the mapped
  sizes; `process-narrow-arrays.m` and its GPU run, arrays of 0 to 70
  elements both ways; `Dialect/Merk/arrays.mlir`; the encodings in
  `encode-checks.m` and `EncodeTest.cpp`.
- *Not a bug, though it looked like one:* `io.write_line(array.sort(A),
  !IO), io.write_line(first(A), !IO)`, which mmc accepts, is rejected. merkc
  reorders as mmc does, `first(A)` before the `sort` that clobbers `A`, but
  `first` takes `A` `in`, which shares it under the sound array stub, so the
  `sort` cannot clobber it (design document, "Arrays are unique by default";
  declaring `first` `array_ui` is the fix). With `-fmmc-uniqueness` it is
  accepted. The error is at the `write_line` that waits for the `sort`'s
  result, the first goal left waiting.

**(d4) measured 2026-11-11: Bandwidth's kernels in a GPU process.**
`benchmarks/bandwidth/bandwidth-process.m` runs the five kernels across a
process's teams; the numbers are in the design document, Measurements,
"Memory bandwidth".
- A loop costs about 2 µs (Dot, with its reduction, 6.5), a target region
  15 to 17 (Dot 36).
- At 2^25 elements the kernels were 6 to 42% slower than target regions,
  because `merk_place` takes 205 VGPRs, the most of anything inlined into
  it, and so has 4 waves a SIMD resident, where a stream wants more. Capped
  at 128 by hand, Copy passed the target regions and the rest came within
  14%.
- The barrier after each round (d2) gains 8 to 17% even here.
- *Open:* how the loops get the waves they need. Ways: a VGPR budget for
  `merk_place` (`amdgpu-num-vgpr` or `amdgpu-waves-per-eu`), which spills
  in the process's code (195 spills at 128); keeping the cold paths
  (messages, copying terms, failure) out of line behind calls, so that only
  what they use counts at their call; or the loops in a kernel of their own,
  which the user's "one kernel per GPU" ruled out. Also open: Dot's 14%
  left, where each iteration loads and stores the partials through private
  memory, and each loop generated twice, for the kind's procedure and its
  `$words` variant.

**GPU experiments, 2026-11-11 night: what limits a GPU process's loops.**
Kept for the multi-GPU machine (8× gfx90a), where the numbers will differ:
wave64, a unified register file of 512 VGPRs and AGPRs a lane, 110 compute
units a die, HBM. The methods below are what to repeat there. The numbers
are in the design document, Measurements, Memory bandwidth.

*Findings on the gfx1030 (RX 6950 XT, 80 compute units, wave32).*
- `merk_place`, the processes' one kernel, had 205 VGPRs: every kind's code,
  the runtime and the loops' bodies were inlined into it, and a kernel's
  count is the most of anything it runs. That left 4 waves a SIMD (1024
  VGPRs a SIMD for wave32), where the target regions' kernels (8 VGPRs, Dot
  43) run 16, and Bandwidth's kernels were 6 to 42% slower than target
  regions.
- Capping it by hand (`"amdgpu-num-vgpr"` on `merk_place`): 128 gave 8
  waves, with 195 spills in the process's code, and recovered most of it;
  96 and 91 measured the same as 128. 8 waves a SIMD are enough for a
  stream here.
- Splitting instead of capping, by `noinline`: with the kinds' code
  (`merk_process_dispatch`) and the loops (`merk_process_loop`) out of line,
  `merk_place` took 128 with no spills; with `Work`, `RunProcess` and
  `CombineTeamPartials` out too, 95 with no spills. Each piece alone: the
  kinds' code 91 (`CopyTyped`'s, the floor), `Work` 85, the team combine 37,
  a loop like Copy's 12.
- A call for each iteration would cost a stream too much, so each loop's
  rounds became a generated procedure (`<body>$rounds`, tail recursion that
  becomes a loop), one call a thread a loop, with the body inlined and the
  barrier after each round (`merk_device_sync_threads`); in the ISA each loop
  is then a tight loop with no calls and no scratch. Every function of a
  GPU-process compile is `convergent`, as clang makes a GPU language's, so
  that the barrier is not duplicated into branches. Result: faster than
  target regions on every kernel, Copy at 95% of the rated bandwidth.
- The barrier after each round (d2) is worth 8 to 17% even for a stream.
- A loop costs about 2 µs, a target region 15 to 17 µs.
- Dot at 1024 elements took 25 µs (6.5 before the split). Skipping the
  team combine gave 4.4 µs, and the cost grew with the number of teams.
  Moving the cross-team step into the last team to finish, its threads
  loading the teams' partials in parallel (`stored`, `lastToStore`), did
  not change it. The combine did 214 scratch accesses: the partials were in
  arrays whose addresses reached the generated `merk_process_loop_reduce`
  and `_start` as integers. As pointers, with all 8 words handled so that
  the indices are constants, 20.
- That gave wrong sums, 3/16 of the right ones at every size and number of
  teams, and a page fault with few teams. Each step across a wave's lanes
  added the thread's first value, not its running sum: 6 lanes of 32. The
  bitcode was right going into the device link's LTO and wrong after
  OpenMPOpt, which runs the Attributor on OpenMP device code (the device
  library is). AAPointerInfo skipped a write in a callee in a loop: it asked
  whether the dominating write before the loop could reach the callee
  without passing the load, which in a loop it cannot, though the call's
  value comes back round to the load. Fixed in 232e9ac212a1 (the load is
  excluded only when it cannot be in a cycle), with a reproducer in
  `llvm/test/Transforms/Attributor/value-simplify-pointer-info-loop-call.ll`.
  Bisecting LTO: `opt -passes='lto<O2>' <img>.0.2.internalize.bc
  -print-changed -filter-print-funcs=<f>`, `-openmp-opt-disable` to rule
  OpenMPOpt out, `-debug-only=attributor`.
- Dot at 1024 is now 12 µs. Cut-short variants (design document,
  Measurements, Memory bandwidth) put most of the rest after the team's
  barrier, but combining the waves' values in parallel did not remove it.
  Open: with the combine returning at once Dot still takes 6.4 µs at 100%
  of the units and 2.6 at 50%, against Copy's 2.2: something in a loop with
  reductions grows with the number of teams. Next step: timestamps inside
  the kernel (`s_memtime`) rather than more variants, whose noise is
  1 to 2 µs.
- At 2^25 elements Dot is 8% below `bandwidth.c`; the combine is under 1%
  of a run there, so that gap is in the loop.
- `with-gpus.py` twice printed its GPU listing instead of running the
  program; a retry worked. Cause not looked at.

*Methods.*
- The process image and its metadata: `merkc ... -save-temps=obj` keeps
  `<name>-<arch>-processes.elf`; `llvm-readelf --notes` gives
  `.vgpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`,
  `.uses_dynamic_stack`, `.group_segment_fixed_size`.
- Each function's registers: relink with `clang --target=amdgcn-amd-amdhsa
  -mcpu=<arch> -flto -O2 <processes.o> lib/merk-process-start-amdgcn.o
  lib/libmerkdevice-amdgcn.a -lc -lm -Wl,--save-temps`, then
  `opt -O2 -force-attribute=<f>:noinline ...` on `.0.2.internalize.bc`
  (or the link with `-Wl,-mllvm,-force-attribute=<f>:noinline`), `llc
  -mtriple=amdgpu10.30-amd-amdhsa -mcpu=<arch>`, and read the `.set
  <f>.num_vgpr` lines. `-pass-remarks-analysis=kernel-resource-usage`
  reports kernels only. `-inline-threshold` below 0 asserts. The device
  library's own functions are inlined when it is built; to split them,
  `[[gnu::noinline]]` in `device.cpp`.
- A register cap: `llvm-dis lib/merk-process-start-amdgcn.o`, add
  `"amdgpu-num-vgpr"="<n>"` to `merk_place`'s attribute group, `llvm-as`,
  and replay `merkc -###`'s steps with that object in the device link
  (there is no `-amdgpu-num-vgpr` option).
- A runtime variant: edit `runtime/device/device.cpp`, `ninja merkdevice
  merkurius-device-library`, build the program, restore the file and
  rebuild. A variant that breaks the results is timed with a copy of the
  benchmark whose check does not stop it (`if Valid = Valid`).
- A loop's cost: `BANDWIDTH_ARRAY_SIZE=1024 BANDWIDTH_NUM_TIMES=1000`, µs
  from the bytes over the MB/s; teams halved with `MERK_GPU_PROCESS_UNITS`.
- The ISA: `llvm-objdump -d` of the image, counting `buffer_load`/
  `buffer_store` (scratch on gfx10), `s_swappc_b64` (calls), `s_barrier`.

*Committed 2026-11-12:* the rounds in generated procedures, `convergent`
and `noinline` marking (`Lowering.cpp`, `MarkKernelFunctions`),
`Work`/`RunProcess`/`CombineTeamPartials` out of line, and the last team's
parallel combine; all GPU tests and the non-GPU suite pass. Then, 2026-11-12,
the combine's partials as pointers with constant indices, and the waves'
combined in parallel (Dot 25 to 12 µs). Next: the reduction loops' cost that
grows with the teams (above), then the design document's description of
the loops.

**Open after (p1).**
- *Which units the processes' kernel takes,* as it holds every team slot
  of them. **Decided 2026-11-11:** a percentage of the available compute
  units, those `LIBOMPTARGET_AMDGPU_CU_MASK` leaves, 100% by default (the
  user: "Saying a % of the total available compute units would be the
  easiest for the user"), set by the environment or by the program before
  its first spawn on the GPU, with the rest for target regions and other
  kernels; a target region launched while GPU processes hold all the units
  fails, saying why, rather than hangs. **Done 2026-11-11:**
  - `process.set_gpu_units(gpu(N), Percent, !IO)`, beside spawn rather than
    an argument of it, as the units are the kernel's, which only the spawn
    that launches it could use, and a phase may want its own: it applies to
    the GPU's next kernel, so a program calls it before its first spawn
    there or between kernels, also while the last kernel is ending, as it
    is when a monitor says that kernel's last process ended. While a
    kernel runs processes, for the host, or for a share that is no
    percentage, it is a software error. `MERK_GPU_PROCESS_UNITS` sets the
    default, 100. (The user chose this over an optional argument of spawn,
    2026-11-11.)
  - The kernel takes the lowest pairs of the GPU's usable units, the
    percentage of them rounded up (`compute-units.h`, `KeepShare`). Each
    launch makes a queue with that mask if the GPU's latest queue has
    other units; the reaper destroys an old queue once no kernel on it is
    left. One queue a GPU is in use at a time, apart from a kernel that is
    ending, so KFD's queue slots are not at risk.
  - Before a target region the host calls `merk_check_region_units` with
    the region's device, or -1 for OpenMP's default: on a GPU whose
    running processes' kernel holds every usable unit, the region would
    never start, and it is a software error. A spawn that launches the
    kernel while regions hold units still fails as before, its teams not
    starting.
  - *Idle waves back off:* with the kernel on all of the units, the waves
    that wait for processes, each reading its slot in host memory every
    3.5 µs, slowed a running process tenfold (`process-loops-gpu`'s
    sequential run took over 120 s). The wave's first lane alone reads the
    slot, and an idle wave doubles its sleep up to about 225 µs, resetting
    when its slot changes; the run takes 2.3 s (design document,
    Measurements, Idle waves of GPU processes' kernels).
  - *Scratch:* with a 16 KiB stack, a kernel on one pair of units has a
    team held back, as before, but on 20 or 40 units all its teams started;
    why is not known. `process-teams-gpu.test` keeps the held case at 1%.
  - Tests: `compute-units.test` (the shares), `process-places-gpu.test`
    (phases alternating 100% and 50%, a change while a kernel runs, for
    the host, of 0, a bad `MERK_GPU_PROCESS_UNITS`, and the slot-filling
    case at 1%, one pair), `omp-target-region-rings-gpu.test` (regions
    beside a kernel at 50%, and the error at 100%), and
    `region-arena-beside-process.m`, which calls `set_gpu_units`.
- **Decided 2026-11-11: spawning names the kind,** as
  `spawn(adder(Reply), Mailbox, !IO)` and
  `spawn(gpu(0), adder(Reply), Mailbox, !IO)`, one name for both, the place
  first, as Erlang's `spawn(Node, Fun)` (the user: "spawn for both. It
  looks uniform and no need to remember two forms"). Implemented
  2026-11-11:
  - Desugaring makes a call of `spawn/5`, unqualified or as
    `process.spawn`, whose second argument is a name or a name applied to
    arguments, a call of the kind's `'spawn_at$<kind>'` with the place, the
    kind's arguments and the rest. Name resolution then finds the kind,
    module-qualified or not, and a name that is no kind's of that many
    arguments, or a kind that receives several message types, is an error
    there: "spawn's second argument must be a process kind applied to its
    arguments". The kind, whether it runs on a GPU and its arguments'
    types are known at compile time, as GPU spawn needs.
  - The predicates that a declaration generates are `'spawn$<kind>'` and
    `'spawn_at$<kind>'`, names that an unquoted name cannot spell, so a
    program does not call them and a predicate of the program named
    `spawn_<kind>` no longer clashes. A kind declared twice reports its
    predicate's duplicate alone. Software errors at the spawn name
    `process.spawn`, the predicate that the program wrote.
  - `spawn/4` stays the library's closure spawn, which a kind written out
    already is: `adder(Reply)` is a closure of `adder/4`. The library has
    `spawn/5` for a closure in a variable: at `host` it is `spawn/4`, and at
    a GPU a software error that says to write the kind out. Not a compile
    error, as the place is a value.
  - A module's own `spawn/5` with a name applied to arguments as its second
    argument is taken for the spawn form too. A module that wants such a
    predicate has to name its own module in the call (`m.spawn(...)`).
  - Compared with Erlang (OTP 29, module erlang): the place is Erlang's
    `Node`, a mailbox stands for its pid, and the kind written out is the
    `Module, Function, Args` form, checked at compile time where Erlang
    returns a pid whose process exits with `undef`. A GPU with no room is a
    software error at the spawn, as `spawn_request`'s `system_limit` is an
    error rather than a process that never runs. `spawn_monitor` is
    deferred until a program needs it (`process.monitor` after the spawn
    does the same, as a kind's process cannot end before its first
    receive); links do not exist.
  Rejected: `spawn_at` beside `spawn`, a second name to remember; a closure
  for GPU spawn, which names host code and captures cells of any type, so
  neither the kind nor its arguments' words are known; rewriting after type
  checking, as goals are numbered in a table and the polymorphism pass
  adds a polymorphic kind's type_infos.
- *Spawning from a GPU process:* a wave that writes a free slot of its own
  kernel's table, without the host. A piece of its own, later.

- **(e) Waiting.** While team 0 waits in a receive, the other teams wait at
  the barrier across teams, sleeping, so the process holds its compute
  units and little else. Between messages the process uses no other
  resource, but it holds its units, which is why (b) comes first.
- **(f) Tests on the GPU,** small first and with a timeout, under a CU mask
  that leaves half the compute units free: a process that sums an array it
  receives with a loop across teams and replies; then one that runs many
  loops between messages, against the same loops in target regions.

**Steps,** each a commit of its own: (a) and (b) together, as they are
tested together; (c); (d) in parts; (e) and (f). Then sparse Life (demo,
step 3) on them.

**(a) and (b) done 2026-11-11,** in `offload/plugins-nextgen/amdgpu`. What
differs from the plan:
- **The mask is the plugin's own, `LIBOMPTARGET_AMDGPU_CU_MASK`,** a list
  of compute units and ranges, such as `0-39`, or a mask in hexadecimal, as
  `HSA_CU_MASK` takes for one GPU. The plugin applies it to each queue it
  creates (`hsa_amd_queue_cu_set_mask`). `HSA_CU_MASK` does confine the
  work (a probe of the work group processors that groups ran on, by
  `HW_ID1`), but ROCr up to 7.2 does not record a mask it applies:
  `SetCUMasking` returns after applying it without updating the mask that
  `hsa_amd_queue_cu_get_mask` reports, which is then empty (ROCr's main
  branch records it). The plugin takes the queue's mask when it applied
  none, and refuses a cooperative launch, saying why, when that mask is
  empty. A mask that `HSA_CU_MASK` reduces is an error, as the plugin
  would count compute units the queue lacks.
- **Counted by shader engine.** The hardware deals work groups to the
  shader engines in turn and waits for one that is full, so an engine with
  fewer compute units than the others limits them all. Measured: with
  compute units 0-19, which the kernel driver deals in pairs to the four
  engines in turn (6, 6, 4 and 4 compute units; `mqd_symmetrically_map_cu_mask`
  in amdkfd), 260 groups of two waves were resident at once, against 320
  that the compute units hold; with 0-39 (10 each), all 640. So the count
  is the engines the mask uses, times the fewest compute units any of them
  has, times the groups a compute unit holds. It assumes the compute units
  are spread evenly over the shader arrays and one XCC, and refuses a
  cooperative launch otherwise, saying why.
- **Groups a compute unit holds,** by LLVM's limits, from
  `AMDGPUTargetParser`: waves a SIMD holds, VGPRs (with AGPRs on processors
  with a unified register file), SGPRs before gfx10, 16 barriers for
  groups of more than one wave, and LDS. A group's waves are taken to share
  the SIMDs of one compute unit, which counts fewer than a work group
  processor holds, never more (one-wave groups: 16 a compute unit counted,
  32 fit).
- **Not done:** cooperative queues (`HSA_QUEUE_TYPE_COOPERATIVE`), which
  gfx1030 lacks; `utils/gpu/with-gpus.py` setting a mask, as the
  environment variable is enough.
- **Tests:** liboffload's unit tests (`olLaunchKernelCooperative.cpp`), with
  `AllGroupsResident`: the most groups that a cooperative launch allows, of
  32 to 1024 threads, meet at a barrier across the grid, each waiting a
  bounded time, so a group that is not resident fails the test instead of
  hanging the GPU. It passes on the whole GPU and with the masks 0-39,
  20-39, 0-19, 0-21, 0-7 and 0x3. The unit tests are built by hand here, as
  the build also wants their SPIR-V device code, which needs a SPIR-V target
  this build does not have. Offload's lit suite is not run on the gfx1030:
  its tests that crash kernels on purpose reset the GPU that drives the
  display (2026-11-11), so it waits for the shared machine.

**(c) done 2026-11-11:** `merk_device_grid_barrier` in
`runtime/device/device.cpp`, which every thread of every team calls. Each
thread fences its writes at device scope, the team synchronizes, its first
thread arrives at the barrier's counts (the last to arrive ends the round,
as `process.team_barrier`'s teams do, which shares the code), the team
synchronizes again, and each thread fences to acquire. The counts are two
words in device memory, zero at the launch, one pair for each launch, so
that several GPU processes of one image do not share them. No time limit,
for (e). Its cost is to measure with (d).
- **Tests:** `test/Runtime/grid-barrier-gpu.test`: a kernel linked as a GPU
  process's image is, launched cooperatively through liboffload with the
  most teams that are resident at once, of 64 to 1024 threads, under
  `LIBOMPTARGET_AMDGPU_CU_MASK=0-39`, at `-O2` and `-O0`. In each of 1000
  rounds every thread writes its slot, crosses the barrier, and checks a
  slot of the next team and of the one before. The same kernel with a
  barrier of each team in its place reported 40,957,700 wrong slots of
  65,536,000, so the test sees a barrier that does not hold. `%merk-lib-dir`
  is the build's lib directory, for linking device code by hand;
  `runtime/liboffload.h` declares cooperative launch and `olSyncQueue`.

**Open.**
- *Which teams a loop uses:* all the kernel's, or fewer for a small loop.
- *Arrays a process makes* are in its arena, in device memory, which every
  team addresses, so a loop reads them in place.
- *HIP on processors without cooperative queues:* how it keeps cooperative
  launches resident, from its documentation, to follow or to differ from.

## A demo: a stencil whose active region moves

Proposed 2026-11-10, from a discussion; not agreed, and nothing is built. A demo
of what a decomposition fixed in advance, as MPI and RCCL programs have, cannot
do: work that grows, moves and shrinks while the program runs, placed on GPUs as
it appears, onto more GPUs when it needs them, and off them when it no longer
does. It extends The halo exchange, above, whose grid is split once.

**The problem.** A grid of tiles, of which only the active ones are computed.
Each step updates every cell from its neighbours, so a tile needs its
neighbours' edge cells, its halo. Two versions:

- *A reacting front, the demo:* diffusion between cells, a cheap stencil, and in
  each burning cell a stiff system of chemical reactions, hundreds to thousands
  of operations a cell and step with no communication. The front advances, tiles
  ahead of it start, and tiles behind it burn out, go quiet and end. The
  computation is heavy exactly where the front is, as in combustion codes, where
  chemistry takes most of the time (AMReX's Pele codes), while the halos stay
  the diffusion's. The user, 2026-11-10: the demo wants heavy computation, as a
  plain stencil's communication costs as much as its computation.
- *Conway's Game of Life,* with a glider gun, as the first step: the simplest
  stencil, exact in integers, easy to see, and growing without bound, so it runs
  out of one GPU. Its computation is light, so it tests growing and shrinking,
  not speed.

**Where the pattern occurs,** from general knowledge, not checked here: adaptive
mesh refinement, which refines around flames, shocks and collapsing gas (AMReX,
Chombo); tsunami and flood models, which compute only wet cells (GeoClaw);
narrow-band level sets, which compute only near a moving surface, for fluids in
graphics and for segmenting images; sparse voxel grids for smoke and fire in
visual effects (OpenVDB); seismic waves spreading from their source; phase-field
models of solidification, around a growing crystal's surface; and wildfire
models that couple a fire front to the weather (WRF-SFIRE).

**Design.**

- **A process for each place, owning many tiles,** not a process for each tile.
  Each GPU process is a kernel today, and hundreds of kernels oversubscribe the
  GPU's scheduler (design document, the teams' stall time). A place's process
  holds its tiles' arrays, handed to it, and steps them all with one
  data-parallel kernel, a big-jump loop over tiles and cells (Specialized
  kernels for loops; What teams as processes are for). Rejected for now: *a
  process for each tile,* which needs processes far lighter than a kernel on the
  GPU.
- **Halos.** Between tiles of one place, a copy within the GPU. Between places,
  one packed message per neighbouring place and step, `{step, edges}` with each
  tile's coordinates and edge cells (flat records, Arrays whose length is known
  only at run time), so that many tiles' edges cost one message. No global
  barrier: a tile's step `s` needs only its neighbours' edges of step `s`, which
  the message's step number names, so places may be a step apart.
- **Computation against communication.** A plain stencil does a few operations
  for each value it loads: it is bound by memory, and a step's halo messages
  cost as much as its computation. A message between GPU processes costs about
  40 µs today, through the host, so a step must compute for much longer than
  that. The chemistry gives that. **Deep halos** add to it: a halo of width k
  sent every k steps, which is k times fewer messages for some computation
  repeated at the tiles' edges. The halo of a tile grows with its edge and its
  work with its area, so larger tiles help too.
- **A directory process** maps a tile's coordinates to the place that owns it.
  Only the directory creates tiles, so two places that find activity at the edge
  of the same missing tile get the same owner.
- **Growing.** A tile whose edge becomes active next to a tile that does not
  exist asks the directory, which gives the new tile to the place with the most
  room, the asker's own if it has room. When every place is full, the directory
  starts a process on another GPU, of this node or another (`spawn_<kind>_at`).
- **Shrinking.** A tile that has been quiet for some steps, with quiet
  neighbours, is dropped, and its last state goes to the directory. A place left
  with no tiles ends, which frees its GPU. A missing neighbour's halo is quiet.
- **Balancing.** The directory moves tiles from busy places to idle ones: the
  tile's arrays go to the new owner in a message, and its neighbours learn of
  the move from the directory. Work stealing (Distributing work, above) is the
  same move, asked for by the idle place.
- **Output.** Every few steps each place sends a reduced picture of its tiles to
  an output process, which writes frames, and the number of tiles on each place,
  which charts the growing and shrinking.
- **Checking.** The final grid against a dense sequential run on the CPU. Exact
  for Life; for the front, integers or a fixed order of floating-point sums.
- **Failure, later.** Kill a place's process: the directory sees it by a monitor
  and gives its tiles to other places from their last pictures, which are
  checkpoints. That is the halo exchange's recovery benchmark.
- **Comparison.** An MPI+HIP program that splits the grid in advance, either
  computing every tile or only the active ones without moving them: time to
  solution and GPU time used as the pattern grows. AMReX, which regrids at fixed
  intervals, is a stronger comparison, later.

**A stepping stone: a deep Mandelbrot zoom.** The cost of a tile varies by
orders of magnitude near the set's boundary, unpredictably, so stealing beats a
split in advance. No tiles start or end, so it tests balancing alone, and is
simple to write and check.

**Steps.** Decided 2026-11-10 (the user): the work heads for this demo, and
its reference is in C or Fortran.

1. **References,** for Life and then for the front: a program in C or
   Fortran with OpenMP offload, the bar to measure against, and the same
   program in Mercury, dense, on one GPU. The two together say whether
   Mercury's code is slower, which the process-oriented version would have to
   make up for, before any process is involved. Life done 2026-11-11
   (`benchmarks/life`: `life.m`, `life.c` with cells of 8 or 1 bytes). The
   front done 2026-11-11 (`benchmarks/front`: diffusion, a loss of heat, and
   two stiff reactions in 100 linearly implicit substeps where a cell burns;
   checked to the bit against Python). merkc does not fuse multiplies and
   adds on the GPU where clang does, which the comparison will show.
2. **Two places over TCP, to try things out:** two nodes on one machine, over
   loopback, each with the GPU (Remote handles and TCP), the tiles split
   between them and halos as messages. It needs no second GPU. Life done
   2026-11-11 (`benchmarks/life/life_two.m`): each node owns half of the
   rows; after each step a target region sends its edge row and takes the
   other's into its halo. It showed that a target region that allocates
   costs about 45 µs more than one that does not, since its arena's chunk
   is allocated, freed and its slot cleared every time, through
   `omp_target_alloc`, `omp_target_free` and `omp_target_memcpy`. Fixed
   2026-11-11 (design document, "The arena"): the host keeps chunks, and
   the device clears the slot by its generation; such a region costs about
   12 µs more now. A scalar output of a region still costs about 27 µs, a
   copy from the device and a wait, which the exchange's `Ok` pays every
   step: open, for instance outputs written to host memory that the device
   can reach, read after the kernel ends.
3. Sparse tiles on one place: tiles start and end, one GPU process, which
   steps its tiles with loops across its teams (GPU processes that run
   loops: cooperative kernels, above). First, dense, done 2026-11-12:
   `benchmarks/life/life-process.m`, Life's steps as loops across a GPU
   process's teams, 20% faster than `life.c` with cells as wide at 4096²
   and 26% at 1024² (design document, Measurements, Dense stencils).
4. The Mandelbrot zoom with stealing between several places.
5. Several places with a directory, growing, shrinking and moving tiles, on one
   GPU at first (below).
6. Several GPUs, on the shared machine (below), and then several nodes,
   against the MPI+HIP program.
7. Recovery from a killed place.

**Step 3, sparse tiles on one place: design.** Agreed 2026-11-12; nothing
is built yet. The user: "If features are missing we should implement
them." So the program is written as it should be, and every feature it
needs and the compiler lacks is a step to build first (Features to build,
below), never a limit the design accepts.

One GPU process steps the active tiles of a Life world on one GPU. Tiles
start where life reaches them and end where it dies out. There are no
messages and no directory process: those are steps 4 and 5. This step
settles the layout and the step that the later ones send halos and move
tiles for.

*The world.* Bounded at first:
- `LIFE_WORLD` cells square, 8192 by default, dead beyond its edge, as
  `life.m`'s border is.
- Tiles of `LIFE_TILE` cells square, 64 by default, a power of two; 32 and
  128 to be measured.
- It starts from a soup of side `LIFE_SOUP`, 256 by default, at its centre,
  alive by `life.m`'s hash of the cell's index in the world.

A soup burns for a few thousand steps. It sheds gliders that fly off across
empty tiles to the world's edge, and it leaves ash. Tiles start in front of
each glider and end behind it, and the soup's own tiles end as it settles.
Unbounded worlds need a map from coordinates to tiles that grows, a hash
table in arrays; later.

*Cells* are `uint8`, one byte each, as C's are. Dense, C's 1-byte cells were
29% faster than 8-byte ones at 4096² (design document, Measurements, Dense
stencils), and a step moves nothing but cells.

*Layout.* The process makes its arrays once:
- *The pool:* two arrays of `Capacity * T * T` cells, the current step and
  the next, swapped as `life-process.m`'s grids are. Tile `I`'s cells are
  `I * T * T` onwards, by rows. Tile 0 is the *dead tile*: always all
  zero, never stepped.
- *The directory:* one int for each tile position of the world, `(W / T)^2`
  of them, holding the pool index of the tile there, or 0 for the dead
  tile. The world is bounded, so this is a dense array, not a hash table.
- *The tiles' records:* an array of `Capacity` flat records, inline, one
  for each pool tile, as C's array of structs:
  ```
  :- type tile
      --->    tile(
                  x :: int32, y :: int32,      % position, in tiles
                  neighbours :: fixed(int32, n8), % pool indices, 0 if none
                  live :: int32,               % from the last step
                  edges :: uint8,              % live edges and corners
                  quiet :: uint8               % steps without life near
              ).
  ```
  The dead tile stands in for a missing neighbour, so the step never
  branches on one. How a fixed size is written is open (Features to build,
  flat records); `fixed(int32, n8)`, as the user decided 2026-11-12.

  *As built, separate arrays* (the user, 2026-11-12, deferring flat
  records' 2b and 3): one array for each field, `Capacity` long, and the
  neighbours in one `int32` array of `Capacity * 8`, tile `I`'s at
  `I * 8 + D`. A team reads its tile's fields once a step, against the
  4096 cells it steps, so the layout costs nothing measurable; an array of
  records waits for `fixed` fields, which the neighbours need.
- *The active tiles:* their pool indices packed, with their number, and a
  free list of pool indices.

A pool made once is what C does too, and a long-lived process then never
allocates as it goes.

*The step: a team for each tile, its threads over the tile's cells.*
```
%$omp distribute
( for(J, 0, Active - 1) do
    Tile = Actives ^ elem(J),
    %$omp parallel do reduction((+) : Live) reduction(\/ : Edges)
    ( for(K, 0, T * T - 1), ... do
        % The cell's eight neighbours, from this tile or, at its edges,
        % from the neighbour tile that the tile's record names.
        ...
    ),
    !Tiles ^ elem(Tile) := (Tiles ^ elem(Tile)
        ^ live := Live) ^ edges := Edges
)
```
- *Halos are read in place.* A cell at a tile's edge reads its neighbour
  tile's cell through the neighbour indices, so there is no copy pass
  between the tiles of one place.
- *Ghost tiles.* When places exchange edges (step 5), a remote neighbour's
  edge cells arrive into a ghost tile of the pool that the neighbour index
  names, and the step does not change.
- *Per-tile results.* The team's reductions give the tile's live cells and
  its edges bitmask, which is all the bookkeeping needs, with no pass over
  the cells of its own.

*Bookkeeping, after each step, on the process's lane:*
- An edge bit set towards a dead-tile neighbour makes that neighbour: taken
  from the free list, zeroed, entered in the directory, and linked to its
  eight neighbours both ways.
- A tile with no live cells, whose neighbours have no live cells on the
  edges facing it, for `Q` steps (4), ends: it is unlinked, goes back on
  the free list, and leaves the actives.
- The cost is linear in the active tiles, on one lane, against a step that
  is linear in their cells: about a thousand tiles' bookkeeping against a
  step of 4 million cells, to be measured. If it shows, the bookkeeping
  runs every `k` steps with tiles made a ring ahead, or in parallel with a
  scan.

*Output:*
- The population and checksum, the sum of the live cells' indices in the
  world, from a loop with two reductions over the active tiles' cells.
- The time of each step, and the number of active tiles after it, for a
  chart of the active region growing and shrinking.

*Checking.* The same world run dense by `life.m` and `life.c`, given the
world's size and the soup (a `LIFE_SOUP` setting for both: the soup at the
centre and the rest dead), must give the same population and checksum.
Exact, as Life is.

*The C reference,* `life-sparse.c`: the same tiles, records and step, in C
with OpenMP, as a C programmer would write it. The host keeps the tile
table and launches one kernel a step: `target teams distribute` over the
active tiles, with `parallel for` over their cells. It does the
bookkeeping after copying back the tiles' counts. It is the bar for the
GPU process, which keeps the bookkeeping on the GPU and starts no kernel a
step. Dense `life.c` over the whole world is the bar that sparseness must
beat, as a decomposition fixed in advance computes every tile.

*Features to build,* each before the program step that uses it:
- *A loop under `distribute` with a loop under `parallel do` in it, in a
  GPU process:* teams take the outer iterations and each team's threads
  the inner one, with reductions in the inner loop. Today a GPU process has
  only the combined `distribute parallel do`. The runtime has most of it:
  the team barrier after each round, `ReduceAcrossTeam` for a team's
  reduction, and the process's loop control. What is new:
  - dealing outer iterations to teams;
  - running the inner loop's rounds over a team's threads;
  - leaving the inner reduction's result with the team for the rest of the
    outer iteration's body.
  Clang and Flang lower the same nest to `omp.distribute` around
  `omp.parallel` and `omp.wsloop`.
- *Narrow array elements.* Every array element is one 8-byte word today
  (`Lowering.cpp`, `LowerArray`). An array of `int8` to `int32`, `uint8` to
  `uint32`, `char` or an enumeration should hold its elements at their C
  sizes, on the host and the GPU alike. Then mapping an array to a region,
  sending it, or handing it to C reads the same bytes as C's array. This
  is the narrow-fields layout of flat records (step 5, above), for arrays'
  elements, and independent of the rest of flat records.
- *Flat records' steps 1 to 3* (step 5, Flat records, above): records as
  values in GPU code, arrays of flat records inline, and fields that are
  arrays of a fixed size. The user decided 2026-11-12: arrays of flat
  records inline on the host too, (a), and `fixed(int32, n8)`, a library
  type with the size as a type.
- Multi-dimensional arrays and views (After step 5, above) would spare the
  pool's index arithmetic. They change how the program is written, not its
  speed, and the step does not wait for them.

*Steps,* each a commit:
- (s1) `distribute` with `parallel do` inside, in GPU processes, with
  reductions in the inner loop. Tests on the GPU as for the combined
  loops, sequential builds too. Done 2026-11-12 (GPU processes that run
  loops, (s1)).
- (s2) Narrow array elements. Tests for each element type, on the host and
  the GPU, sent and mapped. Done 2026-11-12 ((s2) narrow array elements,
  below). Then dense Life in a GPU process with `uint8` cells, measured
  against `life.c`'s `uint8_t`: done 2026-11-12, `life-process8.m`, 37.3
  Gcells/s against C's 31.8 at 4096², 32.8 against 23.6 at 1024² (design
  document, Measurements).
- (s3) Flat records' steps 1 to 3 (decided 2026-11-12: (a), and
  `fixed(T, n8)`). Steps 1 and 2a done 2026-11-12. The user, 2026-11-12:
  defer 2b and start the tiled Life, since the tiles' records can be
  separate arrays (below); 2b and 3 wait for a program that needs them.
- (s4) The `LIFE_SOUP` setting in `life.m` and `life.c`, for the dense check
  and the bar. Done 2026-11-12: a soup of side `LIFE_SOUP`, the whole grid
  by default, from `(LIFE_SIZE - LIFE_SOUP) / 2` in rows and columns, each
  cell alive by the hash of its index in the grid; both programs, and an
  independent one in Python, agree (`Benchmarks/life.test`, and the GPU
  runs in `life-gpu.test` and `life-c-gpu.test`).
- (s5) `life-sparse.m`: the world, tiles starting and ending, checked
  against the dense programs. Done 2026-11-12, as designed above, with:
  - *Nine neighbours,* the middle one the tile itself, so that the cell at
    row R and column C, each from -1 to T, is in neighbour
    `((R >> Log) + 1) * 3 + (C >> Log) + 1`, with no branch.
  - *The soup's tiles* start with every edge live, so that the first
    bookkeeping starts the ring around them.
  - *New tiles' cells* are zeroed in the cells the next step reads, a loop
    across the teams, only when tiles started; the step writes all of an
    active tile's cells in the other array.
  - *The pool running out* stops the steps and is reported, rather than
    giving wrong cells.
  - *No compiler feature was missing.* The -O0 build with
    `--offload-arch` hits the known AMDGPU "failed to annotate CFG" bug on
    musttail calls in divergent control flow, so the -O0 test builds
    without it, as `process-records.m` does.
  - *Checked* against `life.c` on 22 worlds at the host (tiles of 4 to 64,
    soups from none to the whole world, a side of 200, gliders that reach
    the edge, up to 3000 steps), sequential and with the loops on the
    host's threads, and on the GPU at 256², 200², 2048² and the default
    8192²: all the same (`Benchmarks/life-sparse.test`,
    `life-sparse-gpu.test`).
  - *First timings, not yet measurements* (s6 measures): at the default
    world, about 45 active tiles, a step took 72 µs on average against
    dense `life.c`'s 2.0 ms with `uint8_t` cells; but 47 µs of it was the
    bookkeeping on the process's one lane, about 1 µs a tile, a chain of
    dependent loads from global memory.
  - *Parallel bookkeeping,* the design's fallback (above, bookkeeping),
    done 2026-11-12: `decide`, a loop under `distribute parallel do` over
    the active tiles, finds each tile's quiet steps and its fate, from the
    step's live cells and edges alone, as bits by its place in the
    actives: the directions where its live edges face no tile inside the
    world, and a bit that ends it; it reduces the count of tiles with a
    fate. Only when that is not zero does the lane end the tiles that end
    (`finish`, moving the last tile's fate with it) and start the tiles
    the fates ask for (`grow`, which skips a direction a tile started
    earlier in the same pass already fills). The default world's step now
    takes 34 µs on average, the bookkeeping 12 µs, most steps no more
    than the one loop.
- (s6) `life-sparse.c`, the C reference. Measurements against it and
  against dense C, with the active tiles over time. Done 2026-11-12:
  `life-sparse.c` keeps the tiles' tables on the host, a kernel a step,
  the results copied back by the active tiles' slots and their neighbours
  forward only when tiles change; it gives the same cells and the same
  active tiles at every step as `life-sparse.m` (`Benchmarks/
  life-sparse.test` at the host, `life-sparse-c-gpu.test`), and
  `compare-sparse.sh` runs the three programs in turn. At the default
  world the GPU process's step is 33 µs, C's sparse one 62 µs and dense
  C's 2.0 ms; 64² tiles beat 32² and 128² (design document, Measurements,
  Dense stencils). Step 3 is done.

*Open.*
- The tile size. 64² cells of one byte are 4 KiB. Smaller tiles follow the
  gliders more closely; larger ones waste less on teams and bookkeeping.
  Measured 2026-11-12 (s6): 64² is the fastest at the default world, 33 µs
  a step against 42 µs for 32² and 59 µs for 128².
- Teams with no tile in a step: with fewer active tiles than teams, the
  rest wait. Fewer teams for a small loop is open above (Which teams a loop
  uses).
- The cost of a loop with reductions grows with the number of teams (GPU
  experiments, above), and a short step pays it every time.

*Rejected.*
- *A flat loop over all active cells:* it gives no per-tile live count or
  edges without atomics or a pass on one lane, and no team shares a tile's
  cache.
- *Halo copies into tiles of side `T + 2`:* a copy pass each step that the
  neighbour indices make unnecessary on one place. Ghost tiles keep the
  step the same when edges come from other places.
- *A process for each tile:* a GPU process is one wave, 64 lanes, against a
  team of 256 threads for a tile of 4096 cells. A tile's work is a team's.
- *Working around missing features:* 8-byte cells, the tile table as
  parallel int arrays. First drafted that way on 2026-11-12, and rewritten
  at the user's direction with the features as steps to build.

**Step 4, the Mandelbrot zoom with stealing between places: design.**
Decided by the user 2026-11-12, from four questions:
- *Places* are GPU processes on this machine's one GPU, each with a share
  of the kernel's teams; the same program runs a place a GPU on the shared
  machine in step 6.
- *Tiles move by stealing:* each place starts a frame with a static share
  of its tiles, and an idle place asks another for half of what it has
  left. Measured against the static split alone and against a dispatcher
  that hands out tiles.
- *Messages go through the host,* as GPU processes' messages do today
  (step 5 of GPU processes without `omp target`): a steal moves a batch of
  tiles that take milliseconds, so 40 µs a message is small beside them.
  The direct queue in the GPU's memory (5c, 5d) stays a step of its own.
- *Doubles,* zooming to a width of about 10^-13, with the escape limit
  rising frame by frame. As built, to 5.5 * 10^-12 in 40 frames: at 2048
  pixels, a pixel's step is then some 25 units in the last place of the
  centre's coordinates, and below about 2 * 10^-13 pixels would collapse
  onto each other.

*Why stealing here.* A pixel's iterations run from 1 to the limit, and near
the set's boundary a tile's cost varies by orders of magnitude, from frame to
frame and unpredictably. A split made in advance leaves the places with the
cheap tiles waiting. No tile starts or ends, so balancing is tested alone.
On one GPU, one kernel over a whole frame is balanced by the hardware's own
scheduling of work groups, and places that split the GPU's teams can only
come close to it: the dense programs are the bar for what stealing loses
to that, and the static split for what it gains. Across GPUs, where no
hardware balances, the gain is the point (step 6).

*The zoom:*
- `MANDEL_SIZE` pixels square a frame (2048), in tiles of `MANDEL_TILE`
  square (128, a power of two that divides the size); `MANDEL_FRAMES`
  frames (40).
- Frame f's width is `3.0 * 0.5^f` (`MANDEL_RATIO`), centred on a point of
  the boundary, (-0.743643887037151, 0.131825904205330), as written.
- Pixel (px, py) of a frame is `c = (X0 + float(px - Size / 2) * Step,
  Y0 + float(py - Size / 2) * Step)`, with `Step = Width / float(Size)`.
- The escape limit of frame f is `256 + 64 * f` (`MANDEL_LIMIT`).
- An iteration, in this order, so that every program computes the same
  doubles: `X2 = X * X`, `Y2 = Y * Y`, escape if `X2 + Y2 > 4.0`, else
  `Y := (2.0 * X) * Y + Cy`, `X := X2 - Y2 + Cx`. C is built with
  `-ffp-contract=off`, as merkc does not fuse.
- The output: each frame's total iterations, as a checksum, and its time;
  the totals' sum; and for the places, each place's tiles and busy time per
  frame, and the steals.

*Checking:* an independent program in Python gives each frame's total for
small sizes, and every program must give the same totals, exactly.

*Programs:*
- `mandel.c`, the C reference: a kernel a frame, `target teams distribute
  parallel for` over the frame's pixels, `schedule(static, 1)` rows.
- `mandel.m`, dense, one GPU process: a loop under `distribute parallel do`
  over a frame's pixels a frame, all the kernel's teams.
- `mandel-places.m`: a host coordinator and `MANDEL_PLACES` place processes
  (4), each with a quarter of the teams. Per frame, the coordinator sends
  each place its static share of the tiles, a contiguous range of the
  frame's tiles by rows, as a split in advance would give it. A place steps
  one tile at a time, a loop across its teams over the tile's pixels with a
  reduction of its iterations, and between tiles takes any steal request
  from its mailbox (a receive that does not wait). With none left, it asks
  a victim, the next place round the ring that it has not asked this
  frame, which answers with the back half of what it has left or with
  none; after asking all, it reports idle. The coordinator ends the frame
  when every place has reported its tiles' totals and is idle.
  `MANDEL_POLICY` chooses `static` (no stealing), `steal`, or `pool`, where
  the coordinator holds the frame's tiles and an idle place asks it for
  `MANDEL_CHUNK` at a time.

*Features to build,* each before the program step that uses it:
- *A share of the teams for each place.* A spawn of a kind that runs loops
  reserves every free team as its workers today, so a second such process
  finds none. `set_worker_teams(Place, Teams, !IO)`, with `:- type
  worker_teams ---> all_free ; share(int)`, beside `set_gpu_units`: with
  `share(K)`, each kind with loops spawned there afterwards reserves at most
  a K'th of the kernel's teams, at least one, and fails as now if none is
  free. `all_free` is the default. A setting, not a spawn's argument, so
  that the compiler's expansion of spawns does not change.
- Messages between GPU processes through the host exist; a receive that
  does not wait (`after(0)`) on a GPU is to check.

*Steps,* each a commit:
- (m1) `set_worker_teams`, with tests at the host and on the GPU: two kinds
  with loops at once, each with half the teams, their loops' results right.
  Done 2026-11-12 (design document, GPU processes, "The library"):
  `process-worker-teams.m` runs 2 and 4 workers with `share(K)` at once,
  each a loop with a reduction, asked together so that their loops overlap;
  on the GPU a third worker under `share(2)` finds no team, and their
  arenas are freed. The share counts the kernel's teams, so the first
  worker takes a K'th and a later one may get a team fewer, as each
  worker's own wave sits in a team that is then not free.
- (m2) `mandel.c`, `mandel.m` and the Python check, measured. Built
  2026-11-12 (`benchmarks/mandel`): all three give the same counts, at the
  host and on the GPU (`Benchmarks/mandel.test`, `mandel-gpu.test`), and
  `compare-c.sh` runs the two programs in turn. Measured: the GPU process
  takes 3.45 s for the 2048² zoom, 38.8 Giterations/s, against C's 3.57 s,
  and 0.90 s against 0.98 s at 1024² (design document, Measurements,
  Uneven work).
- (m3) `mandel-places.m` with the three policies, checked against the
  dense totals, and measured: the time to finish the zoom, each policy
  against the others and the dense programs, and how even the places'
  busy times are.
  Done 2026-11-12 (`benchmarks/mandel`, design document, Measurements,
  Uneven work): every policy, with one to four places, gives the dense
  totals at the host and on the GPU (`Benchmarks/mandel-places.test`,
  `mandel-places-gpu.test`), and `compare-places.sh` measures them against
  the dense programs. The places' states call each other last with
  different arguments, which needed tail calls between types on the GPU
  (757f82d18450). Measured at 2048² in 128² tiles: static 4.28 s, balance
  1.23; steal 3.97 s, 1.04; pool 3.94 s, 1.02; dense `mandel.m` 3.45 s.
  Stealing evens the places and saves 7%, but splitting the GPU costs 9%
  before any balance, as each tile's loop waits for its slowest pixel, so
  the hardware's balancing of the dense programs wins on one GPU. Open: a
  place's teams taking the next tile while the last one ends, or larger
  tiles split for a thief, before stealing across GPUs (step 6).

**Step 5, several places with a directory: design.** Proposed 2026-11-13.
The user, 2026-11-13: the directory is a host process, and the places and
the directory agree on the tiles after every step, which can be made less
frequent later; a GPU process's memory is given back by freeing values at
their last use, (a) below.

Step 3's world, run by several places on one GPU, each a GPU process with
a share of the GPU's teams (`set_worker_teams`, step 4), each owning some
of the tiles. The program is `benchmarks/life/life-places.m`; its
population and checksum after each run must equal `life-sparse.m`'s for
the same world and steps, for any number of places.

*The directory,* a host process, holds the world map: for each tile
position, the place that owns it, or none. It alone changes the map. It
gives a new tile to the place that asked for it if that place has room,
else to the place with the most room; the soup's tiles go round the places
in rows at the start.

*A place* keeps step 3's pool, directory and neighbour links, with two
kinds of pool tile:
- *its own,* which it steps, as `life-sparse.m` does;
- *ghosts:* a copy of each neighbouring tile that another place owns,
  never stepped, whose edge cells arrive each step. A cell at a tile's edge
  reads a ghost through the neighbour links as it reads its own tiles, so
  the step does not change.
It also keeps a copy of the world map, which the directory's answers keep
up to date, so that it knows which place owns each neighbour.

*A step, s to s + 1, for each place:*
1. Step its own tiles, with the ghosts holding step s's edges.
2. Decide each tile's fate, as `decide` does: tiles to end, and missing
   tiles that life reaches.
3. Send the directory `changes(s, Ended, Wanted)`, the positions of its
   tiles that end and of the missing ones it wants, as one block.
4. Send each neighbouring place `halo(s + 1, Cells)`: the border cells, 4T
   of them, of each of its tiles next to that place's tiles. They are in
   the order of the tiles' positions, which both places know from their
   copies of the map, so the message holds cells alone.
5. Wait for the directory's `map(s + 1, Changes)`, the changes to the map
   from every place's step, and apply them: start the new tiles given to
   it, end its ended tiles, add and drop ghosts.
6. Wait for a `halo(s + 1, ...)` from each neighbouring place and copy its
   cells into the ghosts.

Messages are of a fixed part and one block (done 2026-11-13, 078c2cbf13ef:
the user chose it over arrays of flat records, flat records' step 2b). A halo goes
from a GPU process to another through the host's memory today, about 40 µs;
the directory's round trip costs about as much. Both are the price of the
every-step agreement that the user chose to start with.

*Features to build first:*
- *Memory that a GPU process gives back* (below), which the user chose
  2026-11-13 over (b), a receive into an array the process gives: (a) is
  general, where (b) would serve only programs written around buffers.
- *Copies of a ghost's cells into the pool* from a received array: a loop
  across the place's teams, as `zero` is.

**Memory that a GPU process gives back: design.** Agreed in outline
2026-11-13. A GPU process's arena only grows: the host gives it chunks,
and they go when the process ends (`runtime/device/device.cpp`,
`merk_device_alloc`). Every array it makes, every cell, as a message it
builds, and every message it receives stays, even the `message(...)` cell
of each receive. A place that lives for thousands of steps would grow
without bound. GPU processes make only flat values (design document,
Messages and memory on the device), which share nothing but the arrays in
a block message's last field, so the compiler can find where each value
dies, without the general analysis of Mazur's thesis.

*The allocator.*
- *Small blocks,* under 1 MiB, from LLVM libc's allocator on the GPU
  (`libc/src/__support/GPU/allocator.cpp`): slabs of 2 MiB, a block found
  by a random walk over a slab's bitfield with atomics alone, and `free`
  finds the slab by rounding the pointer down to 2 MiB. Following the
  user's preference for libc on the GPU.
- *Large blocks,* of half a slab or more, libc asks the host for over RPC,
  rounded up to 2 MiB, and gives back the same way: two round trips to the
  host a block. So the runtime keeps freed large blocks itself, by size, on
  a list for each GPU, and reuses one before it asks libc, up to a cap on
  the bytes it keeps (the user, 2026-11-13: "makes sense if we keep track
  of the big ones ourselves"). A place that receives a halo of the same
  size every step then allocates it once.
- *Each process's blocks* are on a list of its own, through a header
  before each block, and the list's blocks go when the process ends, as
  its arena does now: what the compiler does not free, such as a pool that
  lives as long as the process, is not lost.
- *A received message is one block,* copied by its type_info into a block
  of the message's whole size, so that freeing it frees its arrays too.
- *Static cells* of ground terms, which lowering makes, and string
  literals, go into a section of their own on the device, which `free`
  recognizes by its `__start_` and `__stop_` symbols and leaves alone, as
  it does a constant's tagged word.
- The runtime's own structures, mailbox handles, rings and team tables,
  stay in the arena.

*The compiler: values freed at their last use,* a pass on the merk
dialect, after MLIRGen, on the procedures of GPU kinds that run on their
process's lane:
- *Owned values:* what the procedure allocates (`merk.construct` of a
  cell that is not static, `merk.array_new`, an array's copy), what it
  receives, its `di` arguments, and a call's result that the callee's
  summary says is fresh. An `ite`'s or a `switch`'s result is owned when
  each branch gives an owned value or a constant.
- *Classes of values that live and die together,* by union-find: a field
  taken from a cell joins the cell's class, as does an array stored into
  a cell, `merk.array_set`'s result its array's, and a call's result the
  arguments that the callee's summary says it may return.
- *A class escapes* if one of its values is returned, passed in a tail
  call, as a place's pool is to its next state, or passed to a callee
  whose summary says it keeps or returns that argument, or to a foreign
  call other than a send, which copies. A class that escapes is not
  freed: the process's end frees it.
- *Summaries,* for each procedure of the module, by a fixpoint: which
  arguments its results may be, and whether its results are fresh.
  Library procedures that GPU code calls are foreign or lowered in place,
  with known summaries.
- *Where:* in each region, after the last op that uses a value of the
  class, nested regions counting as uses; where the last use is a branch,
  in each branch after its own last use, or at its start. The owned values
  of the class that are visible there are freed, each once on each path.
- Loops across a process's teams are left out at first: what their bodies
  make is a team's, and freed when the process ends (to do after f3).

*Steps of (a),* each a commit:
- (f1) The runtime: GPU processes' blocks from libc's allocator and the
  cache of large ones, each process's list, received messages as one
  block, `merk_device_free`. `MERK_CHECK_DEVICE_MEMORY` reports the bytes
  a program's processes left and the most they held at once. Nothing is
  freed before a process ends yet, so every test stays as it is. Done
  2026-11-13: `runtime/device/device.cpp`, `Block`, `NewBlock`,
  `merk_device_free`; the report is `MERK_CHECK_PROCESS_MEMORY`'s, since
  a process's blocks all go as it ends; test `process-blocks.m`. Two tests
  of target regions beside a GPU process changed: the process no longer
  takes a chunk of the arenas.
- (f2) `merk.free` and its lowering on the device: the cell's pointer
  from its tagged word, a constant's word left alone; static cells and
  string literals in their section. Done 2026-11-13: `merk.free` is lowered
  after the terms, while its value keeps its merk type, which says how:
  an array's or a string's pointer, a term's word without its tag, null
  for a constant's word; an enumeration's term and a record by value are
  not freed. Only a GPU process's compile frees; elsewhere the operation
  goes. The section is `merk_static`, which `merk_device_free` finds by
  `__start_merk_static` and `__stop_merk_static`. Tests:
  `test/Dialect/Merk/free.mlir`, and `test/CodeGen/process-free.m`, where
  a foreign_proc frees, until the pass does.
- (f3) The pass, with merk-level tests of where it frees, and a runtime
  test: a GPU process that receives a million messages, each with an
  array, and makes an array for each answer, in bounded memory, at the
  host and on the GPU. Done 2026-11-13: `lib/CodeGen/Frees.cpp`,
  `FreeAtLastUse`, run by lowering in a compile of GPU process kinds,
  before the terms are lowered; `-fdebug-dump-frees` prints the merk
  module with its frees. As built, beside the design above:
  - A procedure frees only blocks of its own, never an argument: its
    caller frees that after the call, when the callee's summary says it
    is not kept. So `di` arguments need no special case.
  - A block that is stored in one cell and used nowhere else, as a
    received message is in the `message(...)` cell that a receive makes,
    or that is the result of a branch, has no name where its class dies.
    It is freed through what holds it: a test of the functor it was stored
    with, and a `merk.deconstruct` of its field, before the cell is freed.
    Where parts of a branch disagree on what a functor's field holds,
    nothing is freed through it.
  - A term of a type whose one constructor has one field is that field,
    with no cell (Lowering.cpp, Layouts): the pass takes it as the same
    block as its field, as lowering does.
  - A class whose last use is a tail call, as the next state of a process
    is, is not freed there: the call would no longer end the procedure.
    It waits for the process's end.
  - A loop across the process's teams reads its values' words from a
    buffer; such a store counts as a use until the loop's call, which
    returns when the loop has ended. A free in a loop's body, of a
    worker team's block, does nothing.
  - A team mailbox shares an array or a term by its pointer, which freeing
    cannot allow for: a GPU process's wave that sends or receives one
    through a team mailbox now stops the program as not yet implemented
    (`runtime/device/device.cpp`, `TeamBlockFromProcess`).
  - What stays until the process ends: the results of library procedures
    the module does not define, such as strings', blocks made in an
    if-then-else's condition, and blocks that a branch makes and the code
    after it cannot name.
  - The runtime test sends 20000 messages, not a million: each round trip
    goes through the host, about 80 µs, and a million would load the
    display's GPU for over a minute. The bound does not depend on the
    count. `test/CodeGen/process-bounded.m`: an echo that receives
    `data(I, A)`, A of 1000 ints, and answers with a new array, held at
    most 16760 bytes of blocks at once on the gfx1030 (2026-11-13); kept,
    its messages and answers would hold over 300 MB. Merk-level tests:
    `test/CodeGen/frees.mlir`.
- (f4) Measurements, in the design document: a `malloc` and `free` of 4 KiB
  and of 4 MiB on the device, beside a message's time. Done 2026-11-13: `benchmarks/frees`;
  results in the design document, Measurements, "A GPU process's blocks".
  A 4 KiB block costs 4.0 µs against libc's 3.1 µs, a 4 MiB block from the
  cache 2.0 µs against libc's 61 µs, a message's round trip 26 µs.

*Open,* from f4's measurements:
- `array.init` in a GPU process fills from its one lane: 2.6 ms for 4 MiB.
  The process's wave has 32 lanes, or its worker teams could fill it.
- A small block costs 1 µs more than libc's `malloc` and `free`, in
  dependent loads and stores from one lane. Free lists of the process's
  own, by size, would take neither libc's atomics nor the list's.

*Steps,* each a commit:
- (p1) Memory that a GPU process gives back, f1 to f4 above. Done
  2026-11-13.
- (p2) `life-places.m` with fixed owners: the directory places new tiles but
  never moves them. Checked against `life-sparse.m` at 1, 2 and 4 places,
  at the host and on the GPU. Done 2026-11-13:
  `benchmarks/life/life-places.m`, `test/Benchmarks/life-places.test` (1 to
  7 places, -O0 and -O2, sequential and on the host's threads, a
  collection at every allocation, a pool that runs out) and
  `life-places-gpu.test` (the default world, no device memory left). As
  built, beside the design above:
  - *The directory decides every tile's fate,* from a summary of each
    place's tiles after the step: position, whether dead, live edges and
    corners. The design had each place decide and send `changes(s, Ended,
    Wanted)`. But a tile ends only if no neighbour's live edge faces it,
    and a neighbour's edges are known to its owner alone until its halo
    comes, which is after the decision. Deciding at the directory needs no
    second exchange.
  - *Halos carry their tiles' positions,* 4 bytes before each tile's 4T - 4
    border cells. The design ordered the cells by the positions that both
    places know from their maps. But a place may get a neighbour's halo
    before the directory's map, and a GPU process takes its messages in the
    order they come, so the receiver could not yet know the order. A halo
    that comes first starts the ghosts it needs.
  - *A ready handshake:* the directory sends each place every place's
    mailbox, `peer(J, Mailbox)`, which the place keeps in an array of
    mailboxes (the design document, Messages and memory on the device), and
    waits for each place's `ready` before the first map, so that no halo
    reaches a place before it knows its peers.
  - *A new tile goes to the first place that asked,* by place number. The
    design gave it to the asker if it had room, else to the place with the
    most room; a place whose pool is full reports it, and the program ends
    with status 1. Placing by room comes with p3's moves.
  - Messages are of a fixed part and one block: `map(Step, Changes)`, each
    change `Position << 8 \/ Owner`; `summary(Place, Step, Trouble,
    Tiles)`; `halo(Place, Step, Cells)`.
  - Found on the way, each fixed in its own commit: a pointer into an array
    that the statepoint rewrite kept across a collection, at -O2 with two
    places (design document, "Roots that only the optimizer makes"); C's
    `exit` while GPU processes ran, which faulted their kernels (design
    document, the program's end while a region runs); and `ui` arguments
    of multi-clause predicates, which became ground (the register).
  - Temporary: the places' calls take 1824 bytes of stack a lane, more than
    a GPU process's 1536, and 2560 without `-fopenmp`, so the GPU test sets
    `MERK_GPU_PROCESS_STACK_BYTES`; with p3's moves, over 2048, so 2304. It goes once their frames fit, which the
    work on scalar code is to address (the user, 2026-11-13). 2048 is above
    the 1792 bytes that the gfx1030 keeps bound to a queue (GPU processes
    that run loops, "How large a stack can be").
  - First numbers, 2026-11-13, gfx1030 on compute units 0-39, the default
    world (45 tiles at most), median of 3 runs of the average step:
    `life-sparse.m` 38 µs, `life-sparse.c` 60 µs, `life-places.m` 236 µs
    with 1 place, 1157 µs with 2 and 1423 µs with 4. One place's cost is
    the directory's round trip. Two places' 900 µs more was not a wait but
    the places' one lane: `array.init` of a byte array read the array's
    header for each byte, and the messages' copies to and from host memory
    waited for each word's load. With both fixed, 448 µs with 2 places and
    680 µs with 4 (design document, Measurements, "Life on several
    places", which has what is left).
- (p3) Moving tiles: the directory moves tiles from a place with more than
  its share to one with fewer, between steps, the tile's cells as a
  message. Checked as p2. Done 2026-11-13. The user chose, 2026-11-13: the
  directory decides the moves, as it already reads every place's summary
  each step, so a move costs no round trip, where an idle place that asks
  to steal would add one, and no place is idle with every-step agreement;
  and a move takes a tile next to the receiver's if the donor has one, so
  that regions stay together and halos small. As built:
  - After each step but the last, while the place with the most tiles has
    more than its share, the total rounded up over the places, and the one
    with the fewest has less, the directory moves a tile from the one to
    the other, up to `LIFE_MOVES` a step (4; 0 for fixed owners). Only
    tiles from before the step move, by position, so the moves are the
    same in every run, whatever order messages come in; a tile moves at
    most once a step.
  - A move is a change of the map like any other, a tile with an owner
    given another. On the map, the old owner sends each new owner
    `moved(From, Step, Cells)`, each tile's position and its T^2 cells,
    before it lets them go; the new owner makes each its own, from its
    ghost if it has one, and copies the cells across its teams. The cells
    may come before the map, as halos may, so the tile becomes its own
    then, and the map's change finds it so.
  - A place's halos wait until its moved tiles have come, as they read
    them; the moves go out on the map, unconditionally, so no place waits
    on another's halos.
  - No move after the last step: the outcome reads the live cells that a
    step counts, which a moved tile has not had counted.
  - Found on the way: merkc made every value used after a negation
    `mostly_unique` inside it, so `A ^ unsafe_elem(I) \= 0` with
    `A::array_ui` was an error (56864ff63891); and mmc misses an array's
    clobber in a negation (the register).
  - Moves on the default world: 15 at 2 places; on a 128-cell world whose
    gliders reach the edge, 509 to 734 in 2000 steps at 2 to 7 places, as
    tiles start and end throughout. Whether moving pays is p4's.
- (p4) Measurements against `life-sparse.m` on one place: what the
  agreement and the halos cost each step, and the active tiles of each
  place over time. Done 2026-11-13 (design document, Measurements, "What
  places agree on, and what it costs"): the agreement costs more than the
  step and grows with the tiles, as the place walks its tiles on its one
  lane several times a step, each tile a chain of loads; moving tiles
  balances them but does not pay yet, and gives places more neighbours.

*Open.*
- A place's walks of its tiles on its one lane (`summary`, `end_own`,
  `drop_ghosts`, `expected`, `select`, `apply_halo`'s), 0.3 to 1.3 µs a
  tile each (p4). `expected` as a `distribute` loop took 459 µs to 45 at
  346 tiles. How to run them on many lanes is to decide: `distribute` loops
  in the program, each a loop across the teams; or a loop that runs across
  the lanes of the process's own wave, with reductions, which process code
  has no directive for (a `parallel do` must be in a `distribute`).
- The directory moves tiles by their counts alone; the halos that a move
  makes cost more than the balance saves (p4).
- Every-step agreement costs a round trip to the host each step. Life moves
  at most one cell a step, so a tile asked for k steps ahead lets places
  agree every k steps (the user's second option, 2026-11-13); later.
- Moving tiles to balance, and stealing (step 4), are the same move; which
  one asks for it, the directory or an idle place, is open until p3.

**What it needs.** It uses GPU processes, `spawn_<kind>_at`, flat messages and
arrays, which exist. It also needs the place table and routing by place (GPU
processes without `omp target`, steps 5b to 5d), flat records and packed
messages, monitors across places, and for steps 6 and 7, several GPUs.

**Running on a shared machine.** Development moves to a machine of about
eight GPUs that others use too, of which 3 or 4 are ours (the user,
2026-11-10). So:

- `utils/gpu/with-gpus.py` runs a command on GPUs that no other process has
  memory on and that are idle, by KFD's files in sysfs, and sets
  `ROCR_VISIBLE_DEVICES` to them, which the ROCm runtime honours below
  OpenMP's offloading and liboffload alike. It holds a lock on each until the
  command ends, so two of one user's runs never share a GPU. Every test,
  benchmark and demo run goes through it.
- lit passes `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` to the tests,
  which it dropped before, and runs at most four `-gpu.test` files at once.
- **Target IDs with features,** such as `gfx90a:xnack-`, are reported by the
  driver for now. The device's bitcode carries no target features, and the
  image's link gives LTO only the processor, so a feature would be dropped.
  Carrying them means setting `target-cpu` and `target-features` on each
  device function, as Clang does. Code for the processor alone is for "any",
  and runs with xnack and sramecc on or off.
- **Its hardware,** from `rocminfo` (2026-11-10): eight gfx90a dies of 104
  compute units, four MI250 packages of two, each die a GPU of its own with
  64 GB of HBM, 8 MB of L2 and lines of 128 bytes; ISA
  `gfx90a:sramecc+:xnack-`, XNACK off. Waves of 64, not 32: 4 SIMDs and at
  most 32 waves a compute unit, 64 KB of LDS a workgroup, so a team of 256
  threads is 4 waves. Coarse-grained, fine-grained and extended fine-grained
  device memory; fine-grained gives the coherence and atomics at system scope
  that rings in a peer's memory need. The host: two EPYC 7713, 128 cores in
  16 NUMA nodes, about 1 TB, so builds and lit runs there give `-j`.
- **Which GPUs:** two whole packages, four dies, give both links, between a
  package's dies and between packages, for the route table's choice. Which
  dies share a package is to confirm with `rocm-smi --showtopo`.
- To expect: memory coherence unlike gfx10's, including the L2 cache that
  array copies work around, and other occupancy.

**Open.**

- *Several places on one GPU,* to develop steps 4 and 5 here: several GPU
  processes, or queues each limited to some of the compute units by ROCr's
  `hsa_amd_queue_cu_set_mask`, as "virtual GPUs". Untried. This GPU drives the
  display, so it starts small.
- *A GPU process's frames, after scalar registers* (noted 2026-11-12). Done
  2026-11-13: a process's code is marked `"amdgpu-single-lane"` and AMDGPU's
  code generation keeps its values in scalar registers (design document,
  calling convention, "A GPU process's code runs on one lane"; Measurements,
  "GPU processes' stacks"), and with constants kept in loops that make
  calls, the device runtime's process-only functions marked, and error texts
  built on the host, `life-places.m`'s places fit in 784 bytes a lane and the
  work-stealing example in 496. Open:
  - gfx90a: ROCr keeps scratch bound to a queue only up to
    `HSA_SCRATCH_SINGLE_LIMIT`, 140 MiB, over every wave slot of the GPU
    (scratch can hold teams back, above), 104 CUs × 32 slots × 64 lanes
    there, about 690 bytes a lane, against 1792 on the gfx1030, so the
    places' 784 may hold teams back, though the example's 496 should not. Raising the limit from the runtime before ROCr
    starts, 1 KiB a lane being 208 MiB of a die's 64 GB, is the other half.
    But tried 2026-11-13 on the gfx1030: the work-stealing example, 992
    bytes fixed and a dynamic stack over the whole GPU, ran with the limit
    at 1 MB, so the variable does not decide whether teams start, and the
    reading is unconfirmed. On the gfx90a, first look for held-back teams
    (`process-teams-gpu.test`, the example); the change to the runtime was
    reverted.
- *The size of a tile:* large enough that a step's kernel and messages are worth
  their cost, and small enough to follow the front closely. To measure.
- *The chemistry:* a real mechanism, such as hydrogen and air's, or a synthetic
  one of chosen size and stiffness, which makes the cost a cell a parameter of
  the demo. Its solver is per cell and implicit, so a cell's cost varies with
  how stiff it is.
- *Other heavy problems,* if the front does not suit: particles that cluster, as
  two colliding galaxies, which need rebalancing; or a service of matrix
  multiplications of mixed sizes, measured against rocBLAS.

## A showcase with more work per tile: shallow water

Proposed and decided by the user 2026-11-13, after step 5's p4 (A demo: a
stencil whose active region moves): Life's tile is about 0.65 µs of a GPU's
time a step, against a place's 200 to 450 µs of agreement a step, so
distributing it cannot pay. The user: "Are there other sparse computations
with more computation that might be a better showcase?" Chosen: the shallow
water equations with wetting and drying, whose wet region moves over a dry
bed as Life's live region does, with 20 to 50 times the work a cell.

**The model.** The 2D shallow water equations (Saint-Venant), in depth `h`
and momenta `hu`, `hv`, over an analytic bed `b`:
- *A first-order Godunov finite-volume scheme,* an HLL Riemann solver at
  each face, with hydrostatic reconstruction (Audusse et al. 2004) so that a
  lake at rest stays at rest and the depth stays non-negative at the
  shoreline; explicit Euler in time; no friction.
- *Cells of side 1 m,* g = 9.81. A cell at most `1e-6` m deep is dry: its
  momenta are zero, and it is not wet. The time step is 0.45 over the
  largest `|u| + |v| + 2 sqrt(g h)` of the wet cells, after each step for
  the next.
- *The bed* is a paraboloid bowl, `b = D0 (r^2 / R^2 - 1)`, `D0` 10 m,
  `R` 0.45 of the world, at the world's centre, computed at each cell's
  centre wherever needed rather than stored.
- *Two cases* (`SWE_CASE`): 0, a column of water 5 m deep with a flat top,
  of radius a sixteenth of the world, a fifth of the world east of the
  centre, which collapses and sloshes across the bowl, the showcase; 1,
  Thacker's oscillating paraboloid (1981), whose exact solution has a
  moving shoreline, with A = 0.4, for correctness: the programs print the
  relative L1 error of the depth against it.

**Tiles,** as Life's (Sparse tiles on one place): the pool, the directory,
nine neighbours a tile, a dry dead tile 0, fates decided after each step. A
wet cell on a tile's edge or corner starts the tile beyond it; a tile with
no wet cell and no wet edge facing it for 4 steps ends, and what water it
held, under `1e-6` m a cell, goes with it. A cell reads its four
neighbours, through the neighbour table. The tiles at the start are those
over the wet region's bounding square, with all their edges wet, so the
tiles around them start too.

**Three programs,** in `benchmarks/swe/`, as Life's sparse ones:
- `swe-sparse.m`, one GPU process, its step a `distribute` over the active
  tiles with a `parallel do` over each tile's cells, which reduces the
  tile's wet cells, wet edges and largest speed;
- `swe-sparse.c`, C with OpenMP offload, a kernel a step and the
  bookkeeping on the host, as `life-sparse.c`;
- `swe-sparse.hip`, the same in HIP, the bar a GPU programmer would set.

**Bit-identical results.** The three, at the host or on a GPU, give the same
bits: every operation in the same order, no contraction into fused
multiply-adds (`-ffp-contract=off` for C and HIP; merkc emits none),
correctly rounded division and square root (LLVM's, on AMDGPU too), and the
time step from a maximum, which no order changes. They print the simulated
time, the wet cells, and a checksum, the sum modulo 2^64 over the active
cells of `(bits(h) + 3 bits(hu) + 5 bits(hv)) (2 g + 1)`, `g` the cell's
index in the world and `bits` a double's plus `0.0`, so that `-0.0` counts
as `0.0`; the volume and the error, sums whose order differs, to 6 and 4
digits. An exclusive or of the bits was the first checksum, but the bowl's
symmetry cancelled pairs of cells in it. Mercury's `float.max(X, Y)` is `X >= Y ? X : Y` and `min` is
`X =< Y ? X : Y`; the C and HIP programs write them so.

**Settings:** `SWE_WORLD` (2048 cells square), `SWE_TILE` (32, a power of
two that divides it), `SWE_STEPS` (1000), `SWE_CASE` (0), `SWE_CAPACITY`
(the tiles of the pool), `SWE_TRACE` (each step's line).

**Done 2026-11-13:** the three programs, `compare-swe.sh`, and tests at the
host and on the GPU (`test/Benchmarks/swe-sparse*.test`). They agree bit
for bit, at the host and on the gfx1030. Building `swe-sparse.m` found that
a GPU process's foreign C was compiled with OpenMP's globalization, whose
`__kmpc_alloc_shared` its image lacks; fixed first, in its own commit
(`DeviceLibrary.cpp`, CUDA mode for process images). An untimed run at the
default size, with Firefox open: Mercury 477 µs a step, C 486, HIP 447, at
most 472 tiles.

**Timed 2026-11-13** (design document, Measurements, "Shallow water on
sparse tiles: the work a tile"): a tile of 64 cells square is 3 to 3.5 µs
a step, 4.6 times Life's, not 20 to 50, as the gfx1030's doubles run at a
sixteenth of single's rate. Mercury is within 5% of HIP. On the default
world one GPU's step (280 µs) is below a place's agreement; on an 8192-cell
world with tiles of 64 it is 1.4 ms with 520 tiles.

**Then:** the places version, as `life-places.m`, on the 8192-cell world
with tiles of 64, after a place's walks of its tiles run on many lanes
(Open, in the demo's section): its agreement grows by 1.3 µs a tile
against 3.5 µs of work.

## Topology: GPUs, NUMA domains and NICs within a node

Proposed 2026-11-09; nothing is built, and it is not agreed. Real numbers wait for
a machine with several GPUs, as the halo exchange does; the model, routing and
placement can be developed before (Developing without the machines).

**Why.** Inside a node, the route between two GPUs, or from a GPU to the network,
decides the bandwidth, and the fast route is often not through the CPU. Today a
GPU's messages go through host memory and a socket, whatever the machine.

**Two machines as examples.**

- **Supermicro AS-4124GQ-TNMI** (user's manual, 1.7 System Architecture, figure
  1-8, 4U):
  - Two CPU sockets, with three xGMI links between them.
  - Four MI250 OAM cards, each two GCDs, each GCD shown as 100 CUs.
  - Between the cards, xGMI links every card to every other: a square and both
    diagonals. Within a card, the two GCDs are linked.
  - Each card has x32 lanes to its own PCIe switch, which has x16 to a CPU (two
    switches on each socket), x32 to two NICs, and NVMe drives.
  - Not shown: which GCD of a card carries the PCIe link, and the xGMI bandwidths.
- **Frontier or Crusher node** (OLCF Crusher quick start guide, node diagram;
  Crusher was retired in 2024, Frontier's node is the same):
  - One 64-core CPU in 4 NUMA domains, each two L3 regions of 8 cores.
  - Four MI250X cards, two GCDs each, 8 GPUs to the program.
  - Each GPU has its own Infinity Fabric link to one L3 region, 36+36 GB/s. The
    numbering is not in order: L3 region 0 is GPU 4, and GPU 0 is in NUMA domain 3.
  - Within a card, 4 links, 200 GB/s. Between cards, links of 50+50 GB/s, unevenly:
    2 between GPUs 4 and 2 and between 6 and 0, 1 for 5–7, 4–6, 2–0, 3–1, 5–1 and
    3–7, and none for other pairs.
  - Four NICs of 25+25 GB/s, one on each card, attached to one GPU (5, 3, 7 and 1)
    over PCIe Gen4 ESM, 50+50 GB/s.

What follows for routes:

- On the Supermicro machine, a GPU and its NICs share a PCIe switch, whose link to
  the CPU is half as wide. Data for the network should go GPU, switch, NIC, not
  through host memory.
- On Frontier, GPU 4 has no NIC; its fast route out is over 200 GB/s to GPU 5,
  which has one. Between GPUs with one link, a large transfer may be faster split
  over a second route through a neighbour.
- Processes that talk to a GPU belong on its NUMA domain or L3 region. Slurm's
  `--gpu-bind=closest` does this for MPI ranks, and the guide warns that rank i is
  not GPU i.

**What RCCL does** (ROCm/rocm-systems, `projects/rccl/src/graph`):

- A graph whose nodes are GPUs, PCIe switches, CPUs (commented "Actually NUMA
  domains"), NICs and the network, with typed links and a bandwidth constant for
  each kind.
- It searches the graph for rings and trees, which become the channels of its
  collectives.
- `rome_models.cc` is 5,335 lines of ring orders written for particular machines,
  matched by a pattern string: detection alone was not enough.
- A collective is a kernel launched from the host, over a communicator fixed in
  advance.

**Where Merkurius could do better.**

1. **No host in the loop.** A launch for each collective costs tens of µs, which
   dominates small and medium messages. GPU processes that keep running send and
   receive themselves (Processes as teams). rocSHMEM and MSCCL++ start transfers
   from the GPU too, so they are the ones to compare with.
2. **The runtime uses the topology.** A GPU is a place with distances, so `send`
   chooses the route, such as through GPU 5's NIC on Frontier, and a spawn puts a
   process near its GPU.
3. **Patterns other than collectives.** A halo exchange and irregular point to
   point traffic, overlapped with computation; collectives written in Mercury,
   fused with the computation around them.

Not claimed: RCCL's bandwidth for large collectives, tuned for these machines.

**Proposed, in order.**

1. **Direct paths.** Peer access between GPUs of a node, and sends from GPU memory
   to the NIC (GPU-direct RDMA), so a message need not go through host memory.
2. **A model.** Places (GPUs, NUMA domains, NICs), links with bandwidth and
   latency, and which NIC belongs to which GPU. On Linux,
   `/sys/class/kfd/kfd/topology/nodes/*/io_links` gives link types and weights.
   A file can describe the machine instead, for what detection gets wrong, which
   is `rome_models.cc`'s lesson.
3. **Routing and placement** from the model.
4. **Measurements** against `rccl-tests` and an MPI+HIP program.

**Developing without the machines.**

- **The model is data.** With the topology read from a file (an environment
  variable naming it), the Supermicro and Frontier nodes are two files, and unit
  tests check routing and placement on them: from GPU 4 on Frontier, the network
  is through GPU 5. This needs no hardware.
- **Places on one GPU.** Several processes on the one GPU, each told by the file
  that it is a different GPU, check that messages take the routes chosen, as
  both-regions already runs two processes on one GPU.
- **Several OpenMP devices on the CPU.** LLVM's host offload plugin is built here
  and presents 4 devices (`NUM_DEVICES` in `offload/plugins-nextgen/host`), so
  `device(N)`, transfers between devices and regions on several devices can run
  on the CPU. Whether merkc can target the host as an offload device is not checked.
- **Links slowed on purpose.** A transport for tests that delays each message by a
  link's latency and bandwidth from the file shows whether routing picks the
  faster path. Only the choice is tested; the speed is made up.
- **APIs with stand-ins.** The runtime calls an interface of its own for
  transports, with real backends (HIP peer access, verbs or libfabric) and one
  over host memory or sockets. libfabric's tcp provider runs without special
  hardware. Soft-RoCE gives verbs over ordinary Ethernet but needs a kernel module,
  which is the user's to decide.
- **Not simulated:** real bandwidths, contention between links, and the cost of
  GPU-direct RDMA. Those wait for the machines.

**Open.**

- Which GPU of a Supermicro card carries the PCIe link, and the xGMI bandwidths.
- Whether to split a large transfer over several routes, and how to set its size.
- How a program sees the topology: not at all, as hints to spawn, or as places it
  can name.
- Whether the model also covers nodes, where the network's topology matters.

## The receive goal: later

The receive goal is done (design document, Processes and mailboxes, "The
`receive` goal"); two extensions remain.

Guards, `select(Box, Pattern), Guard -> Goal`, a pure `semidet` goal run in
the selector's predicate after the pattern matches (design document,
"Guards, a later extension").

With open types (Open types: `closed(T)`): each pattern variable's role by
its inst at the goal, which is Erlang's rule, instead of by where it
occurs. A ground variable is a test inside the selector, as now. Any
other, new, free or a hole, gets a fresh copy inside the selector, which
comes back through the result, and the arm unifies the variable with it
once the message is chosen; so a hole is filled only by the arm that took
the message, never by a selector that accepted a message which then lost to
an older one. This also lets a pattern bind a variable used after the
goal, which the first rule makes an input, and removes the puzzling error
of a tested variable bound only after the goal. Mode analysis knows the
insts, so it builds the selector then, from a receive goal kept until it.
Under the first rule a hole in a pattern is a mode error, as a lambda that
captures a variable that is not ground is, so nothing compiles now that
the second rule would change.

## Open types: `closed(T)`

Decided 2026-11-04: `closed(T)` comes with open types, not after them. It
lets a type say which parts are open, where `open(T)` alone is deep: every
argument at every depth becomes open, so filling a hole with a large ground
value means `to_open` on all of it, a copy if it has raw fields, and
`from_open` copying it back.

```mercury
:- type bundle ---> bundle(closed(big), closed(big), closed(big)).

    % open(bundle): a field may be unbound, but once bound holds a ground big.
Hole = closed(Chunk)    % one small box holding the pointer; Chunk is not touched
```

`closed(T)` is a library type, `:- type closed(T) ---> closed(T).`, so
outside `open(_)` it is a notag type: `T`'s word, at no cost. Under `open(_)`:

| Where | What changes |
| --- | --- |
| Type checker | `closed(T)`'s argument under `open(_)` stays `T` |
| Mode analysis | That argument is `ground`, under inst `any` |
| Open layout | A one-word box of `T`'s word, with no descent into `T` |
| Descriptors | The constructor's kind is `Closed`; no other field is needed |
| `to_open`, `from_open` | Add or remove the box; the value is shared |
| Unification | Both bound: `T`'s ordinary unify; the occurs check stops there |
| Collector, message copy, printing | The walker changes to the ordinary layout there |

Ground code, the heap and the trail do not change. A bound hole is a write
into an older cell, so it needs the remembered set, as every `bind` does.

- **`closed(open(U))`** cannot be ruled out in polymorphic code. The mode rule
  covers it, since a value with unbound variables is not `ground`, and the
  walker follows the `type_info`, back to the open layout.
- **A unique value put in `closed(T)` becomes shared,** so it can no longer be
  updated in place.
- **A message still copies the value:** `closed(T)` saves the copies inside a
  process. A hole cannot be filled from another process either, since open
  values are sent as `copy_term` does, so the process holding the hole binds it.
- **Not for building in order.** `foo(bundle(X, Y, Z)) :- recv(X), recv(Y),
  recv(Z).` needs neither: mode analysis constructs the bundle after the
  receives (checked with merkc and mmc), and values arriving in any order
  fit an ordinary term of `maybe`s. `closed(T)` is for a term that something
  must hold while it is still incomplete.
