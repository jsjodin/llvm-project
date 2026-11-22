% Mode and determinism analysis of open terms (design document, "Open
% terms"; docs/INPROGRESS.md, Open types), on the design document's
% examples. X = f(...) with X `any` is a get_structure, det for a type of
% one functor whose arguments are all free and open, and semidet otherwise; X = Y of
% two open values is a full unification, semidet. A free variable of an open
% type becomes a fresh unbound variable when no goal can run, the hole
% rather than the term around it, or when a construction takes it and it is
% not used again. A disjunction that may bind an open nonlocal keeps its
% solutions. closed(T)'s argument is ground. An open term is never unique,
% nor is what it holds.
%
% RUN: merkc -fc1 -fopen-types -fdebug-dump-modes %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s
% RUN: merkc -fc1 -fopen-types -fdebug-dump-determinism %s 2>&1 | FileCheck --check-prefix=DET --implicit-check-not=error --implicit-check-not=warning %s

:- module open_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module open, list, pair.

    % Two free variables unified: one is made fresh, and the others refer
    % to it. The recursive call runs before the construction that needs R.
:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.
make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).
% CHECK-LABEL: pred open_modes.make_open/3 mode 0: make_open(in, oa, oa) is det
% CHECK:          fresh HeadVar__2
% CHECK-NEXT:     assign T := HeadVar__2
% CHECK-NEXT:     assign V_2 := T
% CHECK-NEXT:     assign HeadVar__3 := V_2
% CHECK-NEXT:     final: HeadVar__1 ground, HeadVar__2 any, HeadVar__3 any
% CHECK:          call pred open_modes.make_open/3 mode 0: make_open(Xs, R, T)
% CHECK:          construct V_3 = '[|]'(X, R)
% CHECK:          final: HeadVar__1 ground, HeadVar__2 bound('[|]'(ground, any)), HeadVar__3 any
% DET: pred open_modes.main/2 mode 0: main(di, uo) is det, inferred det
% DET: pred open_modes.make_open/3 mode 0: make_open(in, oa, oa) is det, inferred det

    % to_open's result is ground, so Hole's unification with it is an
    % implied mode of the call.
:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.
close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    from_open(Front, List).
% CHECK-LABEL: pred open_modes.close/4 mode 0: close(ia, ia, in, out) is semidet
% CHECK:          call func open.to_open/1 mode 0 implied: Hole = to_open(Tail)
% CHECK-NEXT:     call pred open.from_open/2 mode 0: from_open(Front, List)
% DET: pred open_modes.close/4 mode 0: close(ia, ia, in, out) is semidet, inferred semidet

:- type dlist == open(pair(list(int), list(int))).

    % The second get_structure unifies B, which is no longer free.
:- pred append_dl(dlist::ia, dlist::ia, dlist::oa) is semidet.
append_dl(A - B, B - C, A - C).
% CHECK-LABEL: pred open_modes.append_dl/3 mode 0: append_dl(ia, ia, oa) is semidet
% CHECK-NEXT:   clause at
% CHECK-NEXT:     getstructure HeadVar__1 = -(A, B)
% CHECK-NEXT:     getstructure HeadVar__2 = -(B, C)
% CHECK-NEXT:     construct HeadVar__3 = -(A, C)
% CHECK-NEXT:     final: HeadVar__1 bound('-'(any, any)), HeadVar__2 bound('-'(any, any)), HeadVar__3 bound('-'(any, any))
% DET: pred open_modes.append_dl/3 mode 0: append_dl(ia, ia, oa) is semidet, inferred semidet
% DET-NEXT: getstructure HeadVar__1 = pair.-/2(A, B) [det]
% DET-NEXT: getstructure HeadVar__2 = pair.-/2(B, C) [semidet]

:- pred dl_to_list(dlist::ia, list(int)::out) is semidet.
dl_to_list(F - [], L) :-
    from_open(F, L).
% CHECK-LABEL: pred open_modes.dl_to_list/2 mode 0: dl_to_list(ia, out) is semidet
% CHECK-NEXT:   clause at
% CHECK-NEXT:     getstructure HeadVar__1 = -(F, V_2)
% CHECK-NEXT:     getstructure V_2 = []
% CHECK-NEXT:     call pred open.from_open/2 mode 0: from_open(F, L)
% CHECK-NEXT:     final: HeadVar__1 bound('-'(any, any)), L ground

    % Every goal waits at first; H1 and H2 are made fresh, not the lists
    % around them, so the ground parts are constructed as usual.
:- pred test(list(int)::out) is semidet.
test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).
% CHECK-LABEL: pred open_modes.test/1 mode 0: test(out) is semidet
% CHECK:          fresh H1
% CHECK-NEXT:     construct V_4 = '[|]'(V_5, H1)
% CHECK-NEXT:     construct V_2 = '[|]'(V_3, V_4)
% CHECK-NEXT:     construct D1 = -(V_2, H1)
% CHECK-NEXT:     fresh H2
% CHECK-NEXT:     construct V_8 = '[|]'(V_9, H2)
% CHECK-NEXT:     construct D2 = -(V_8, H2)
% CHECK-NEXT:     call pred open_modes.append_dl/3 mode 0: append_dl(D1, D2, D)
% CHECK-NEXT:     call pred open_modes.dl_to_list/2 mode 0: dl_to_list(D, L)
% CHECK-NEXT:     final: L ground

    % L is ground, so its deconstruction is the ordinary one.
:- pred first(int::out) is semidet.
first(X) :-
    test(L),
    L = [X | _].
% CHECK-LABEL: pred open_modes.first/1 mode 0: first(out) is semidet
% CHECK:          deconstruct L = '[|]'(X, V_2)

    % A hole not used again is made fresh at its construction, and a
    % closed field holds a ground value.
:- type big ---> big(int, int).
:- type bundle ---> bundle(closed(big), int).
:- pred fill(open(bundle)::oa) is det.
fill(bundle(closed(big(1, 2)), _)).
% CHECK-LABEL: pred open_modes.fill/1 mode 0: fill(oa) is det
% CHECK:          construct V_1 = closed(V_2)
% CHECK-NEXT:     fresh V_5
% CHECK-NEXT:     construct HeadVar__1 = bundle(V_1, V_5)
% CHECK-NEXT:     final: HeadVar__1 bound(bundle(bound(closed(bound(big(bound(1), bound(2))))), any))
% DET: pred open_modes.fill/1 mode 0: fill(oa) is det, inferred det

    % A get_structure of closed/1 gives its ground argument. It can fail
    % even if that argument is free: a closed(T) inside an open term may be
    % an unbound variable, which has no ground value to give.
:- pred unclose(closed(list(int))::ia, list(int)::out) is semidet.
unclose(closed(L), L).
% CHECK-LABEL: pred open_modes.unclose/2 mode 0: unclose(ia, out) is semidet
% CHECK:          getstructure HeadVar__1 = closed(L)
% CHECK-NEXT:     final: HeadVar__1 bound(closed(ground)), L ground
% DET: pred open_modes.unclose/2 mode 0: unclose(ia, out) is semidet, inferred semidet

    % A get_structure in each disjunct is no switch: X may be unbound, and
    % then each disjunct binds it.
:- pred either(open(list(int))::ia) is nondet.
either(X) :-
    ( X = [] ; X = [_ | _] ).
% CHECK-LABEL: pred open_modes.either/1 mode 0: either(ia) is nondet
% CHECK:          getstructure X = []
% CHECK:          getstructure X = '[|]'(V_1, V_2)
% DET: pred open_modes.either/1 mode 0: either(ia) is nondet, inferred nondet

    % A full unification.
:- pred same(open(int)::ia, open(int)::ia) is semidet.
same(X, X).
% CHECK-LABEL: pred open_modes.same/2 mode 0: same(ia, ia) is semidet
% CHECK:          unify HeadVar__2 = X
% DET: pred open_modes.same/2 mode 0: same(ia, ia) is semidet, inferred semidet

main(!IO) :-
    make_open([1, 2, 3], Front, Hole),
    promise_pure (
        if close(Front, Hole, [4, 5], List) then
            io.write_line(List, !IO)
        else
            io.write_string("hole already bound\n", !IO)
    ).

    % The negation waits for D, and cannot make it fresh itself, as that
    % would bind a nonlocal: D is made fresh outside it, once, and the
    % negation and both uses see the same variable.
:- pred hold(open(list(int))::ia) is det.
hold(_).
:- pred not_fill(int::out) is det.
not_fill(N) :-
    promise_pure ( if \+ D = to_open([7]) then N = 1 else N = 0 ),
    hold(D),
    hold(D).
% CHECK-LABEL: pred open_modes.not_fill/1 mode 0: not_fill(out) is det
% CHECK-NOT:      fresh
% CHECK:          fresh D
% CHECK-NOT:      fresh
% CHECK:          final:

    % A ground argument of a get_structure waits for a goal to bind it, so
    % the 1 is constructed first and unified with X's head, which binds an
    % unbound head, rather than taken out of the head and tested. Only when
    % nothing binds it is it taken out of its field, which fails if that is
    % unbound; closed(T)'s argument likewise.
:- pred head_one(open(list(int))::ia) is semidet.
head_one([1 | _]).
% CHECK-LABEL: pred open_modes.head_one/1 mode 0: head_one(ia) is semidet
% CHECK-NEXT:     clause at
% CHECK-NEXT:       construct V_1 = 1
% CHECK-NEXT:       getstructure HeadVar__1 = '[|]'(V_1, V_2)
% CHECK-NEXT:       final: HeadVar__1 bound('[|]'(bound(1), any))
:- pred head_out(open(list(int))::ia, int::out) is semidet.
head_out([H | _], H).
% CHECK-LABEL: pred open_modes.head_out/2 mode 0: head_out(ia, out) is semidet
% CHECK-NEXT:     clause at
% CHECK-NEXT:       getstructure HeadVar__1 = '[|]'(H, V_2)
% CHECK-NEXT:       final: HeadVar__1 bound('[|]'(ground, any)), H ground
:- pred closed_in(open(bundle)::ia, big::in) is semidet.
closed_in(bundle(closed(L), _), L).
% CHECK-LABEL: pred open_modes.closed_in/2 mode 0: closed_in(ia, in) is semidet
% CHECK-NEXT:     clause at
% CHECK-NEXT:       getstructure HeadVar__1 = bundle(V_1, V_3)
% CHECK-NEXT:       getstructure V_1 = closed(L)
% CHECK-NEXT:       final:
