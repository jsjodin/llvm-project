% A call that never succeeds ends its branch, which then need not bind what the
% other branch binds: an erroneous call, and a failure one. The output is mmc's.
%
% RUN: merkc %s -o %t && not %t | FileCheck %s --match-full-lines
% RUN: merkc -O2 %s -o %t.O2 && not %t.O2 | FileCheck %s --match-full-lines

:- module erroneous_branch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pragma foreign_decl("C", "#include <stdio.h>\n#include <stdlib.h>").
:- pred stop is erroneous.
:- pragma foreign_proc("C", stop, [will_not_call_mercury, promise_pure],
    "puts(\"stopped\"); fflush(stdout); exit(3);").
:- pred br(int::in, int::out) is det.
br(X, Y) :- ( if X > 0 then Y = X else stop ).
:- pred never(int::in) is failure.
never(_) :- fail.
:- pred sd(int::in, int::out) is semidet.
sd(X, Y) :- ( if X > 5 then Y = X else never(X) ).
main(!IO) :-
    br(8, T), io.write_int(T, !IO), io.nl(!IO),
    ( if sd(7, A) then io.write_int(A, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if sd(1, B) then io.write_int(B, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    br(-1, U), io.write_int(U, !IO), io.nl(!IO).
% CHECK: 8
% CHECK-NEXT: 7
% CHECK-NEXT: no
% CHECK-NEXT: stopped
