% MANUAL: Syntax for explicit type quantifiers
% DESIGN: Sized floating point types
% The manual's declarations with explicit quantifiers: some on a pred and a
% func, a mix of existential and universal variables, and all before
% some. It is a template, run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_quantifiers_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_quantifiers_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:42
% OUTPUT-FLOAT-NEXT:{7 - "n", [7], ["n"]}
% OUTPUT-FLOAT-NEXT:"middle"
% OUTPUT-FLOAT-NEXT:1.5
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_quantifiers_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_quantifiers_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:42
% OUTPUT-FLOAT32-NEXT:{7 - "n", [7], ["n"]}
% OUTPUT-FLOAT32-NEXT:"middle"
% OUTPUT-FLOAT32-NEXT:1.5f32
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_quantifiers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, list, pair.

    % Here the type variable `T' is existentially quantified
:- some [T] pred foo(T).
:- mode foo(out) is det.
foo(42).

    % Here the type variables `T1' and `T2' are existentially quantified.
    % (list(T2) in place of the manual's set(T2).)
:- some [T1, T2] func bar(int, list(T1), list(T2)) = pair(T1, T2).
:- mode bar(in, out, out) = out is det.
bar(N, [N], ["n"]) = N - "n".

    % Here the type variable `T2' is existentially quantified,
    % but the type variables `T1' and `T3' are universally quantified.
:- some [T2] pred foo3(T1, T2, T3).
:- mode foo3(in, out, in) is det.
foo3(_, "middle", _).

:- all [T3] some [T2] pred foo4(T1, T2, T3).
:- mode foo4(in, out, in) is det.
foo4(_, 1.5@S@, _).

main(!IO) :-
    foo(X),
    io.write_line(X, !IO),
    P = bar(7, L1, L2),
    io.write_line({P, L1, L2}, !IO),
    foo3(1, Y, 'c'),
    io.write_line(Y, !IO),
    foo4(1, Z, 'c'),
    io.write_line(Z, !IO).
