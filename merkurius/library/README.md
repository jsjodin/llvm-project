# Standard library

These are declaration-only stubs of standard library modules, so that the
front end can load the interfaces a program imports and resolve their
names before the library is written (design document, "Standard library,
compatibility and testing"). Each file has an interface section; most
have no implementation yet.

`io.m` implements its output predicates on the standard output,
`write_string`, `write_int`, `write_uint`, the sized integers' writes,
`write_char`, `write_float` and `nl`, with `foreign_proc`s for C, and
the narrower integers' writes in Mercury. `write`, `print` and their
`_line` forms are polymorphic `foreign_proc`s on `runtime/write.cpp`;
`write_list` is in Mercury, calling its closure for each element, on the
standard output only. `command_line_arguments` and `progname` read the
command line that the runtime's `main` keeps. `require.m` implements its errors with mmc's
messages, but `report_lookup_error`, which is not implemented yet.
`pair.m` and `maybe.m` implement all of theirs in Mercury. `list.m` implements all of its declarations in Mercury, every
mode, with mmc's behaviour where the manual leaves it open, as found by
running mmc's programs (design document, Code generation, "Library").
`solutions.m` implements all of its declarations: `unsorted_aggregate`,
`unsorted_aggregate2` and `do_while` have no clauses, as code generation
makes their procedures from `merk.aggregate`, and the others are written
in Mercury on them (design document, Lowering, "Collecting solutions").
`process.m`, a Merkurius extension with no counterpart in Mercury's
library, implements the mailboxes of one process, `new_mailbox`,
`close_mailbox`, `send`, `receive` on one mailbox and `any`, with
`foreign_proc`s on `runtime/mailbox.cpp` (design document, Language
extensions, "Processes and mailboxes").
`float.m` implements its
functions but `hash`, and `math.m` all of its
functions but `solve_quadratic`, with C's math library. The arithmetic
and comparisons of `int`, `uint`, the sized integers and `float` are
built into code generation, so those modules need no implementation for
them. The build compiles the modules that have an implementation into
`libmerkstd.a` (the list is in `CMakeLists.txt`), which the driver links
into every program. An exception the library throws, such as math's
`domain_error`, is reported by the runtime's `merk_uncaught_exception`
until exceptions can be caught. `write_float` writes as mmc
does, with the fewest significant digits, from 15 to 17, that read back
as the same float, and `.0` if that has no fraction or exponent; the
behaviour was found by running mmc's programs, not from its library. So
were the float functions' rounding and edge cases: `round(X)` is
`floor(X + 0.5)`, `max` and `min` are those of `>=` and `=<`, `pow`
multiplies by squaring, a float out of `int`'s range converts to
`min_int`, and math's checks throw `domain_error("math.Name")`.

The declarations are those of the Mercury 22.01 library reference manual,
which the clean-room rules allow as the source of the library's
behaviour and interface. The library's source and the files mmc derives
from it were not read. The manual's comments are not copied. A module's
interface must match the manual's for programs written for mmc to compile,
so the declarations are kept as the manual gives them.

What the stubs leave out:

- Declarations that need modules not stubbed yet, such as
  `pretty_printer`, `random`, `enum`, `stream`, and `set`, which
  `solutions_set` needs.
- In `list`, `io` and `string`, all but the commonly used predicates and
  functions.
- The `stm_builtin` module, which mmc provides for atomic goals and which
  the library reference manual does not document.

The build copies the modules to `lib/merkurius` beside `bin/merkc`, where
merkc looks for them; `-library-dir` names another directory.

## Sound stubs

The manual's `array` module works around mmc's missing unique-input
modes: its unique array insts are `ground`, as its own comments say, so
mmc checks no array's uniqueness, and the predicates that read an array
take it `in`. `make-sound-array.py` makes a sound stub from `array.m`, in
`lib/merkurius/sound`, with the insts the manual's comments give
(`uniq_array(I) == unique`), and with each mode that takes arrays `in`
preceded by copies that take them `array_ui`, so that reading a unique
array keeps it unique. merkc searches `sound` before the library's own
directory unless `-fmmc-uniqueness` is given. Edit `array.m`, never the
generated stub.
