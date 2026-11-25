% MANUAL: Visibility rules
% A submodule sees all of the parent's declarations, those of its
% implementation too, and the interfaces of the modules the parent imports;
% the parent sees the child's after importing it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: nested submodules; nyi_visibility.child can be a separate submodule, in a file of its own, nyi_visibility.child.m, which the implementation section of nyi_visibility includes with `:- include_module child.`

:- module nyi_visibility.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module list, nyi_visibility.child.

:- type secret ---> secret(int).

:- func reveal(secret) = int.
reveal(secret(N)) = N.

    :- module child.
    :- interface.
    :- func peek = int.
    :- implementation.
    peek = reveal(secret(list.length([1, 2, 3]))).
    :- end_module child.

main(!IO) :-
    io.write_line(peek, !IO).
