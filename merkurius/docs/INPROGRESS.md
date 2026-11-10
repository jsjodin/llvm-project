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
  padded to 8, an array as its length and then its elements, and a
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
- **Arrays of numbers are one run.** An `array(float)` encodes as its
  length and then its elements as they are in memory, on the little-endian
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
   collectives on it.
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
   (`CodeGen/omp-target-team-arrays.m`, and its `-gpu.test` at -O0 and -O2;
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
     fit start within microseconds. If not all have started within a second, one
     team reports "Mercury runtime: target teams: only M of the N teams ran at
     once on the device, …" and the program ends.
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
  be reclaimed while a slice is out. On a device, the region's arena is freed when
  no region runs, so slices that outlive a region need memory with a lifetime of
  its own: the question of an owner for the array, above.
- **Boundaries** on multiples of a cache line (128 bytes), so that two teams do not
  write to one line.

**A later use: streaming on the GPU.** Teams as the stages of a pipeline, passing
arrays by pointer. It has no cycle of waits, but a stage that is not on the GPU
still stalls the one before it when the ring between them fills.

## Library code on the device: later

Built 2026-11-10 and moved to the design document (Library code on the
device), with its measurements (Measurements, "Barriers between teams").
What is left:

**The next piece: closures and type classes on a device.** Most of `list` is
higher-order: `foldl`, `map` and `filter` make or call closures, which a device
rejects. The specializer makes a call with a known closure a direct call, so
with the bodies the device's compile copies in, it makes those predicates run
on a device (The specializer: known closures and dictionaries).

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
       may still be reading is freed when no region or GPU process runs.
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
       device with the lock held (`merk_mailbox_copy_array_for_ring`), as
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
     !IO)` with `:- type place ---> host ; gpu(int)` in `process`, and
     `process.gpus(Count, !IO)` for how many there are. Spawning at `host`
     is `spawn_<kind>`. A GPU that does not exist, or a kind without a
     kernel, is a software error at the spawn. Rejected:
     `spawn_<kind>_on_gpu(N, ...)`, which names no place, though handles and
     routing are by place (design document, Handles, routing and places).
   - **D3. One message type for each kind in the first version,** and
     scalar and mailbox arguments for a GPU kind. Several message types,
     and other arguments, come later.
5. **Spawning from the GPU's other places,** and the place table's
   identities. liboffload gives a device's name and its PCI vendor id, but
   still no UUID or PCI address (design document, "What the plugins lack").

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
