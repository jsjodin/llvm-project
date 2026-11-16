% The implementation-defined literals (reference manual, "Implementation-
% defined literals"), as mmc gives them: $file, the source file; $line, the
% line; $module, the module's name; and $pred, the kind, qualified name and
% arity of the predicate or function whose clause it is in, a function's
% without its result, and within a lambda the clause's own. The output is
% mmc's for the same program, but for $file, which is the path that the
% compiler was given.
%
% $grade names one of mmc's grades, which Merkurius has no equivalent of,
% so it is not yet implemented.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/"no grade"/$grade/' %s > %t.grade.m
% RUN: not merkc %t.grade.m -o %t.grade 2>&1 | FileCheck %s --check-prefix=GRADE

:- module implementation_literals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    io.write_string($pred, !IO), io.nl(!IO),
    io.write_string($module, !IO), io.nl(!IO),
    io.write_string($file, !IO), io.nl(!IO),
    io.write_int($line, !IO), io.nl(!IO),
    io.write_string(f(1), !IO), io.nl(!IO),
    p(3, S), io.write_string(S, !IO), io.nl(!IO),
    L = list.map((func(_) = $pred), [1]),
    io.write(L, !IO), io.nl(!IO),
    io.write_string(g(2, 3), !IO), io.nl(!IO),
    io.write_string(zero, !IO), io.nl(!IO),
    io.write_string("no grade", !IO), io.nl(!IO).

:- func f(int) = string.
f(_) = $pred.

:- pred p(int::in, string::out) is det.
p(_, $pred).

:- func g(int, int) = string.
g(_, _) = S :- F = ((func) = $pred), S = apply(F).

:- func zero = string.
zero = $pred.

% OUTPUT:predicate `implementation_literals.main'/2
% OUTPUT-NEXT:implementation_literals
% OUTPUT-NEXT:{{.*}}implementation-literals.m
% OUTPUT-NEXT:28
% OUTPUT-NEXT:function `implementation_literals.f'/1
% OUTPUT-NEXT:predicate `implementation_literals.p'/2
% OUTPUT-NEXT:["predicate `implementation_literals.main\'/2"]
% OUTPUT-NEXT:function `implementation_literals.g'/2
% OUTPUT-NEXT:function `implementation_literals.zero'/0
% OUTPUT-NEXT:no grade
% OUTPUT-EMPTY:

% GRADE: error: not yet implemented: code generation for $grade
