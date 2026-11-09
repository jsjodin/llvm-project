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
