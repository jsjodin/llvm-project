% DESIGN: Loops / Shared iterator variables
% Two fromtos that have a variable in the same place are one accumulator,
% legal but almost always a mistake, so a warning names the variable and
% both iterators. Their arguments must agree, so the loop can fail where
% they do not, as its translation does. Other iterators that share a
% variable draw no warning, nor does a variable in different places: Y is
% a foreach's element and a fromto's Out.
% ORACLE: none
%
% RUN: merkc -fdo-loops %s -o %t 2>&1 | FileCheck %s --implicit-check-not=warning: --implicit-check-not=error:
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
:- module warn_shared_fromto.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred together(int::in, int::out, int::out) is semidet.
together(First, Sum, Count) :-
    ( foreach(X, [1, 1, 1]), fromto(0, S0, S1, Sum), fromto(First, S0, S1, Count) do
% CHECK: warn-shared-fromto.m:[[@LINE-1]]:{{[0-9]+}}: warning: the iterators fromto(0, S0, S1, Sum) and fromto(First, S0, S1, Count) share the variable S0, so they must agree in every iteration; give each iterator its own variables
% CHECK: warn-shared-fromto.m:[[@LINE-2]]:{{[0-9]+}}: warning: the iterators fromto(0, S0, S1, Sum) and fromto(First, S0, S1, Count) share the variable S1, so they must agree in every iteration; give each iterator its own variables
        S1 = S0 + X
    ).

main(!IO) :-
    ( if together(0, Sum, Count) then
        io.write({Sum, Count}, !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    ( if together(1, _, _) then
        io.write_string("yes", !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    ( foreach(X, [1, 2]), foreach(X, Copy) do true ),
    ( foreach(Y, [3, 4]), fromto(0, _, Y, Last) do true ),
    io.write({Copy, Last}, !IO), io.nl(!IO).
% OUTPUT:{3, 3}
% OUTPUT-NEXT:no
% OUTPUT-NEXT:{[1, 2], 4}
