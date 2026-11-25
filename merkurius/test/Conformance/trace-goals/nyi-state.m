% MANUAL: Trace goals
% A state parameter, which gives a trace goal a mutable.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_state.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- mutable(count, int, 0, ground, [untrailed]).
main(!IO) :-
    trace [state(count, !C), io(!S)] ( !:C = !.C + 1, io.format("count %d\n", [i(!.C)], !S) ),
    io.write_string("end\n", !IO).
