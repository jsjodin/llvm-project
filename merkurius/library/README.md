# Standard library

These are declaration-only stubs of standard library modules, so that the
front end can load the interfaces a program imports and resolve their
names before the library is written (design document, "Standard library,
compatibility and testing"). Each file has an interface section and no
implementation.

The declarations are those of the Mercury 22.01 library reference manual,
which the clean-room rules allow as the source of the library's
behaviour and interface. The library's source and the files mmc derives
from it were not read. The manual's comments are not copied. A module's
interface must match the manual's for programs written for mmc to compile,
so the declarations are kept as the manual gives them.

What the stubs leave out:

- Declarations that need modules not stubbed yet, such as `array`,
  `pretty_printer`, `enum`, `stream` and the sized integer types.
- In `list`, `io` and `string`, all but the commonly used predicates and
  functions.
- The `stm_builtin` module, which mmc provides for atomic goals and which
  the library reference manual does not document.

The build copies the modules to `lib/merkurius` beside `bin/merkc`, where
merkc looks for them; `-library-dir` names another directory.
