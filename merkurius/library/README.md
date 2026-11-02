# Standard library

These are declaration-only stubs of standard library modules, so that the
front end can load the interfaces a program imports and resolve their
names before the library is written (design document, "Standard library,
compatibility and testing"). Each file has an interface section; most
have no implementation yet.

`io.m` implements its output predicates on the standard output,
`write_string`, `write_int`, `write_char`, `write_float` and `nl`, with
`foreign_proc`s for C. The build compiles the modules that have an
implementation into `libmerkstd.a` (the list is in `CMakeLists.txt`),
which the driver links into every program. `write_float` writes as mmc
does, with the fewest significant digits, from 15 to 17, that read back
as the same float, and `.0` if that has no fraction or exponent; the
behaviour was found by running mmc's programs, not from its library.

The declarations are those of the Mercury 22.01 library reference manual,
which the clean-room rules allow as the source of the library's
behaviour and interface. The library's source and the files mmc derives
from it were not read. The manual's comments are not copied. A module's
interface must match the manual's for programs written for mmc to compile,
so the declarations are kept as the manual gives them.

What the stubs leave out:

- Declarations that need modules not stubbed yet, such as
  `pretty_printer`, `random`, `enum` and `stream`.
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
