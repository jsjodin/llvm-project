% Try goals (reference manual, "Exception handling"), as desugaring expands
% them (desugar.cpp, DesugarTry): code generation outlines the goal as a
% closure that gives the variables it binds that the goals after it use, as
% a tuple, calls exception.try or try_io with it in place of
% magic_exception_result, and switches on the result: the catches, the then
% part with the tuple's values, and the else part for a goal that can fail.
% OUTPUT is mmc's, but for the lambda expression's try goal, on which mmc
% aborts (test/Conformance/exception-handling/differs-lambda.m).
%
% RUN: merkc -fc1 -emit-merk %s -o %t.mlir
% RUN: FileCheck %s --check-prefix=MERK < %t.mlir
% RUN: FileCheck %s --check-prefix=MERK-IO < %t.mlir
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module try_goals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

% A semidet goal that binds two variables: try's semidet mode, a tuple of
% two, and the failed case.
% MERK-LABEL: merk.proc private @try_goals.halves_4_0(
% MERK:         merk.closure @try_goals.halves_4_0$lambda{{[0-9]+}}$closure
% MERK:         merk.call @exception.try_2_1(
% MERK:         merk.switch {{.*}} cases {{\[\[}}#merk.functor<"exception", 1>], [#merk.functor<"succeeded", 1>], [#merk.functor<"failed", 0>]]
% MERK:         merk.deconstruct #merk.functor<"succeeded", 1>
% MERK-NEXT:    merk.deconstruct #merk.functor<"{}", 2>
% MERK-LABEL: merk.proc private @try_goals.halves_4_0$lambda{{[0-9]+}}(
% MERK-SAME:    determinism = #merk.determinism<semidet>
% MERK:         merk.construct #merk.functor<"{}", 2>
:- pred halves(int::in, int::out, int::out, string::out) is cc_multi.
halves(N, H, R, What) :-
    ( try [] (
        N > 0, H0 = N // 2, R0 = N mod 2,
        ( if N = 13 then throw("13") else true )
    )
    then
        H = H0, R = R0, What = "halved"
    else
        H = 0, R = 0, What = "not positive"
    catch S ->
        H = -1, R = -1, What = S
    ).

% With an io parameter: try_io, whose io states have no bits, and a catch
% that continues from the io state before the goal.
% MERK-IO-LABEL: merk.proc private @try_goals.logged_3_0(
% MERK-IO:         merk.call @exception.try_io_4_0(
:- pred logged(int::in, io::di, io::uo) is cc_multi.
logged(N, !IO) :-
    ( try [io(!IO)] (
        io.format("logging %d\n", [i(N)], !IO),
        ( if N < 0 then throw(N) else true )
    )
    then
        io.write_string("logged\n", !IO)
    catch M ->
        io.format("caught %d\n", [i(M)], !IO)
    ).

% A polymorphic goal: the closure captures the type_info, and the tuple
% holds a value of the type variable's type.
:- pred first(list(T)::in, string::out) is cc_multi.
first(L, R) :-
    ( try [] H = list.det_head(L)
    then
        R = "first " ++ string.string(H)
    catch_any E ->
        R = "exception " ++ string.string(E)
    ).

% A goal that cannot succeed: its succeeded case is never taken, so the
% catch_any's variable is bound after the try.
:- pred thrown(T::in, io::di, io::uo) is cc_multi.
thrown(V, !IO) :-
    ( try [] throw(V)
    then
        io.write_string("then\n", !IO)
    catch_any E ->
        true
    ),
    io.write_line(E, !IO).

main(!IO) :-
    list.foldl(show_halves, [9, -2, 13], !IO),
    logged(1, !IO),
    logged(-5, !IO),
    first([[1, 2], [3]], F1),
    io.write_line(F1, !IO),
    first([] : list(int), F2),
    io.write_line(F2, !IO),
    thrown({1, "a"}, !IO),
    % A closure that a lambda expression's try goal calls.
    Divide = (pred(X::in, Y::out) is cc_multi :-
        ( try [] Z = 100 // X
        then Y = Z
        catch_any _ -> Y = -1
        )),
    Divide(4, A),
    Divide(0, B),
    io.format("%d %d\n", [i(A), i(B)], !IO).

:- pred show_halves(int::in, io::di, io::uo) is cc_multi.
show_halves(N, !IO) :-
    halves(N, H, R, What),
    io.format("%d: %d %d %s\n", [i(N), i(H), i(R), s(What)], !IO).

% OUTPUT:9: 4 1 halved
% OUTPUT-NEXT:-2: 0 0 not positive
% OUTPUT-NEXT:13: -1 -1 13
% OUTPUT-NEXT:logging 1
% OUTPUT-NEXT:logged
% OUTPUT-NEXT:logging -5
% OUTPUT-NEXT:caught -5
% OUTPUT-NEXT:"first [1, 2]"
% OUTPUT-NEXT:"exception software_error(\"function `list.det_head\\\'/1: Unexpected: empty list\")"
% OUTPUT-NEXT:{1, "a"}
% OUTPUT-NEXT:25 -1
% OUTPUT-NOT:{{.}}
