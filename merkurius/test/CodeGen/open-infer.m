% A type inferencer by unification (design document, "Open terms as built"): the
% types of a small functional language are open terms, whose holes are type
% variables, bound by unification with the occurs check (unify_oc). A let-bound
% name is polymorphic: each use infers its definition again, with new holes. A
% type error is a failure: a clash, an occurs check, or a lambda-bound name used
% at two types. The inferred types are written with their holes as _1, _2
% (open.write), and a ground one is taken out by from_open. mmc has no open
% types, so the outputs are checked by hand: each line says what it shows.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module open_infer.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open, string.

:- type expr
    --->    var(string)
    ;       lam(string, expr)
    ;       app(expr, expr)
    ;       let(string, expr, expr)
    ;       int(int)
    ;       bool(string)
    ;       plus(expr, expr)
    ;       if_then_else(expr, expr, expr)
    ;       pair(expr, expr).

:- type ty
    --->    int
    ;       bool
    ;       arrow(ty, ty)
    ;       pair(ty, ty).

    % A lambda-bound name has one type; a let-bound name, its definition
    % and the scope it is in.
:- type binding
    --->    mono(string, ty)
    ;       poly(string, list(binding), expr).

    % The environment's cells and bindings are never holes, as only infer
    % builds them, so the conditions that test them are logical, which
    % their promise_pure scopes say.
:- type env == open(list(binding)).

:- pred lookup(env::ia, string::in, open(binding)::oa) is semidet.
lookup(Env, X, B) :-
    Env = [B0 | Rest],
    promise_pure ( if B0 = mono(X, _) then
        B = B0
    else if B0 = poly(X, _, _) then
        B = B0
    else
        lookup(Rest, X, B)
    ).

:- pred infer(env::ia, expr::in, open(ty)::ia) is semidet.
infer(Env, E, T) :-
    (
        E = var(X),
        lookup(Env, X, B),
        promise_pure ( if B = mono(_, T0) then
            unify_oc(T, T0)
        else if B = poly(_, Env0, E0) then
            infer(Env0, E0, T)
        else
            fail
        )
    ;
        E = lam(X, Body),
        A = _ : open(ty),
        R = _ : open(ty),
        unify_oc(T, arrow(A, R)),
        infer([mono(X, A) | Env], Body, R)
    ;
        E = app(F, Arg),
        FT = _ : open(ty),
        AT = _ : open(ty),
        infer(Env, F, FT),
        infer(Env, Arg, AT),
        unify_oc(FT, arrow(AT, T))
    ;
        E = let(X, Def, Scope),
        % The definition must type, though each use types it again.
        infer(Env, Def, _ : open(ty)),
        infer([poly(X, Env, Def) | Env], Scope, T)
    ;
        E = int(_),
        T = int
    ;
        E = bool(_),
        T = bool
    ;
        E = plus(A, B),
        infer(Env, A, int),
        infer(Env, B, int),
        T = int
    ;
        E = if_then_else(C, Then, Else),
        infer(Env, C, bool),
        infer(Env, Then, T),
        infer(Env, Else, T)
    ;
        E = pair(A, B),
        TA = _ : open(ty),
        TB = _ : open(ty),
        unify_oc(T, pair(TA, TB)),
        infer(Env, A, TA),
        infer(Env, B, TB)
    ).

:- pred show(string::in, expr::in, io::di, io::uo) is det.
show(Name, E, !IO) :-
    io.write_string(Name ++ ": ", !IO),
    promise_pure (
        T = _ : open(ty),
        ( if infer([], E, T) then
            ( if impure from_open(T, G) then
                io.write(G, !IO),
                io.write_string(" (ground)", !IO)
            else
                impure open.write(T, !IO)
            )
        else
            io.write_string("type error", !IO)
        )
    ),
    io.nl(!IO).

main(!IO) :-
    Id = lam("x", var("x")),
    show("id", Id, !IO),
    show("const", lam("x", lam("y", var("x"))), !IO),
    show("compose", lam("f", lam("g", lam("x",
        app(var("f"), app(var("g"), var("x")))))), !IO),
    show("s", lam("x", lam("y", lam("z",
        app(app(var("x"), var("z")), app(var("y"), var("z")))))), !IO),
    show("apply", lam("f", lam("x", app(var("f"), plus(var("x"), int(1))))),
        !IO),
    show("let id", let("id", Id,
        pair(app(var("id"), int(1)), app(var("id"), bool("true")))), !IO),
    show("let twice", let("twice", lam("f", lam("x",
            app(var("f"), app(var("f"), var("x"))))),
        app(var("twice"), lam("n", plus(var("n"), int(1))))), !IO),
    show("lambda id", lam("id",
        pair(app(var("id"), int(1)), app(var("id"), bool("true")))), !IO),
    show("self", lam("x", app(var("x"), var("x"))), !IO),
    show("if", if_then_else(bool("true"), int(1), bool("false")), !IO),
    show("unbound", var("y"), !IO),
    show("swap", lam("p", let("q", var("p"), var("q"))), !IO).
% OUTPUT: id: arrow(_1, _1)
% OUTPUT-NEXT: const: arrow(_1, arrow(_2, _1))
% OUTPUT-NEXT: compose: arrow(arrow(_1, _2), arrow(arrow(_3, _1), arrow(_3, _2)))
% OUTPUT-NEXT: s: arrow(arrow(_1, arrow(_2, _3)), arrow(arrow(_1, _2), arrow(_1, _3)))
% OUTPUT-NEXT: apply: arrow(arrow(int, _1), arrow(int, _1))
% OUTPUT-NEXT: let id: pair(int, bool) (ground)
% OUTPUT-NEXT: let twice: arrow(int, int) (ground)
% OUTPUT-NEXT: lambda id: type error
% OUTPUT-NEXT: self: type error
% OUTPUT-NEXT: if: type error
% OUTPUT-NEXT: unbound: type error
% OUTPUT-NEXT: swap: arrow(_1, _1)
