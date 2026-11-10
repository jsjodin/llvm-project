% Closures as values: io.write writes a predicate's or a function's closure
% as mmc does, and a unification or comparison that reaches one, through a
% type_info or not, ends the program with mmc's message, through the
% descriptor of the higher-order type (design document, Lowering,
% "Closures"). `compare(R, [F], [F])` would give `(=)` in mmc, which merges
% the two lists (register, "Comparing terms that hold closures"), so the
% lists differ here.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.0.O2 && %t.0.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/^which = 0\./which = 1./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=UNIFY --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which = 0\./which = 2./' %s > %t.2.m && merkc %t.2.m -o %t.2
% RUN: not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=COMPARE --match-full-lines --strict-whitespace < %t.2.err
% RUN: sed 's/^which = 0\./which = 3./' %s > %t.3.m && merkc %t.3.m -o %t.3
% RUN: not %t.3 2> %t.3.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=COMPARE --match-full-lines --strict-whitespace < %t.3.err

:- module closure_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func which = int.
which = 0.

main(!IO) :-
    P = (pred(X::in, Y::out) is det :- Y = X + 1),
    F = (func(X) = X * 2),
    io.write(P, !IO), io.nl(!IO),
    io.write([F, F], !IO), io.nl(!IO),
    io.write({1, P}, !IO), io.nl(!IO),
    ( if which = 1 then
        ( if unify_any(F, F) then io.write_string("same\n", !IO) else true )
    else if which = 2 then
        compare(R, P, P), io.write(R, !IO), io.nl(!IO)
    else if which = 3 then
        compare(R, [F, F], [F]), io.write(R, !IO), io.nl(!IO)
    else
        true
    ).

:- pred unify_any(T::in, T::in) is semidet.
unify_any(X, X).

% OUTPUT:'<<predicate>>'
% OUTPUT-NEXT:['<<function>>', '<<function>>']
% OUTPUT-NEXT:{1, '<<predicate>>'}
% OUTPUT-EMPTY:
% UNIFY:Mercury runtime: called unify/2 for `pred' type
% COMPARE:Mercury runtime: called compare/3 for `pred' type
