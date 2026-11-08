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

## The scheduler: processes on one host

Route step 3 (design document, "The route there"), chunk 2. Proposed 2026-11-08;
not agreed yet. The design document decides most of it (Processes and
scheduler; Message passing; the part on parallel regions): stacks of a fixed
size that never move, a worker for each core with work stealing, preemption by
lowering the heap limit, and target regions off the workers. What is here is
the order, and what is left to decide.

**Where it starts.** A program is one process. Each OS thread has a heap in
thread-local storage (`merk_heap`). Mailboxes have one global lock and a
condition variable, and a receive blocks its OS thread. An uncaught exception
ends the program. `spawn`, links and monitors do not exist.

**Steps,** each its own commit, tested on the CPU first:

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
   libomp ties the region to the thread. To do: a run queue for each worker,
   stealing, timers for receive timeouts, and multi-producer mailbox queues with
   no global lock.
5. **Preemption.** Lowering the heap limit, and `mrt.yield_check` on loops that
   do not allocate.
6. **Ending a process.** An uncaught exception ends its process with a reason
   (route step 2), then links, monitors and `request`.
7. **What blocks.** Target regions on a blocking pool with the process
   suspended, the hand-off for parallel regions, and blocking I/O. Node
   connections stay OS threads, which wake processes.
8. **The GPU with processes.** A region's launch does not hold a worker,
   ping-pong between `main` and a GPU of its own node, regions on several
   devices at once with `device(N)`, and whether a worker drains the send ring
   (The send ring).

**Needed before the collector.** Decided 2026-11-08 to leave these until
processes run:

- `spawn` shares the closure and the handle's cell with the parent, and does not
  copy them, as a closure's cell does not say the types of what it captured. The
  compiler has to describe them, polymorphic ones too. That also lets closures
  be sent. It is safe until the collector moves or frees cells, as captured
  values are never unique, and so never updated.
- A process's heap chunks and a received message's fragments are never freed. A
  ping-pong round leaks about 8 KB.

**To decide:**

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
- **D3, the first scope.** Steps 1 to 3 are done; the rest is to be reviewed
  with the numbers from `benchmarks/processes`.

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
