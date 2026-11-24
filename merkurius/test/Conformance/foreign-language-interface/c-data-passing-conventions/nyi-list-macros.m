% MANUAL: C data passing conventions
% MR_list_head and MR_list_is_empty read a Mercury list from C.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: MR_list_is_empty on Merkurius lists

:- module nyi_list_macros.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.
:- pred first(list(int)::in, int::out) is semidet.
:- pragma foreign_proc("C", first(L::in, X::out),
    [promise_pure, will_not_call_mercury],
    "SUCCESS_INDICATOR = !MR_list_is_empty(L);
    if (SUCCESS_INDICATOR) X = MR_list_head(L);").

main(!IO) :-
    ( if first([3, 4], X) then io.write_int(X, !IO) else io.write_string("empty", !IO) ),
    io.nl(!IO).
