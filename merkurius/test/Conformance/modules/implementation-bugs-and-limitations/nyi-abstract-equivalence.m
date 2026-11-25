% MANUAL: Implementation bugs and limitations
% An equivalence type defined in a submodule and exported as an abstract
% type, for which the manual warns mmc may report spurious errors.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: nested submodules; nyi_abstract_equivalence.counter can be a separate submodule, in a file of its own, nyi_abstract_equivalence.counter.m, which the interface section of nyi_abstract_equivalence includes with `:- include_module counter.`

:- module nyi_abstract_equivalence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module counter.
    :- interface.
    :- type counter.
    :- func new = counter.
    :- func next(counter) = counter.
    :- func value(counter) = int.
    :- end_module counter.

:- implementation.
:- import_module nyi_abstract_equivalence.counter.

    :- module counter.
    :- implementation.
    :- import_module int.
    :- type counter == int.
    new = 0.
    next(C) = C + 1.
    value(C) = C.
    :- end_module counter.

main(!IO) :-
    io.write_line(value(next(next(new))), !IO).
