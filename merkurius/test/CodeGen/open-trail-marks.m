% Trail marks only around goals that may bind a variable older than them
% (docs/INPROGRESS.md, Open types, step 4(d)): one that names a variable
% bound before it whose type mentions open types, or whose inst has `any`
% in it. A condition on ints in a procedure on open terms takes none, nor
% does a disjunction of tests; a condition or a negation on an open
% argument does, and so does a disjunction that binds one. Lowering takes a
% mark inline, and nest keeps 1000 live at once, past the runtime's first
% growths of the marks.
%
% RUN: merkc -fc1 -fopen-types -emit-merk %s -o - | FileCheck %s
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines

% CHECK-LABEL: merk.proc {{.*}}@open_trail_marks.build_3_0(
% CHECK-NOT: merk_open_trail_mark
% CHECK: merk.succeed
% CHECK-LABEL: merk.proc {{.*}}@open_trail_marks.head_is_3_1_0(
% CHECK: merk_open_trail_mark
% CHECK-LABEL: merk.proc {{.*}}@open_trail_marks.not_empty_1_0(
% CHECK: merk_open_trail_mark
% CHECK-LABEL: merk.proc {{.*}}@open_trail_marks.small_2_0(
% CHECK-NOT: merk_open_trail_mark
% CHECK: merk.succeed
% CHECK-LABEL: merk.proc {{.*}}@open_trail_marks.one_or_two_1_0(
% CHECK: merk_open_trail_mark

:- module open_trail_marks.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open.

    % I > N names no open variable: no mark.
:- pred build(int::in, int::in, open(list(int))::ia) is semidet.
build(I, N, L) :-
    ( if I > N then
        L = []
    else
        L = [I | T],
        build(I + 1, N, T)
    ).

    % The condition may bind L: a mark.
:- pred head_is_3(open(list(int))::ia) is semidet.
head_is_3(L) :-
    promise_pure ( if L = [3 | _] then true else L = [] ).

    % So may the negation's goal.
:- pred not_empty(open(list(int))::ia) is semidet.
not_empty(L) :- promise_pure ( \+ L = [] ).

    % Tests of ints in turn: no mark.
:- pred small(open(list(int))::ia, int::in) is semidet.
small(L, X) :- ( X = 1 ; X = 2 ), L = [X].

    % The disjuncts bind L, and with no outputs the first that succeeds
    % is enough: a mark.
:- pred one_or_two(open(list(int))::ia) is semidet.
one_or_two(L) :- ( L = [1], L = [2] ; L = [3] ).

    % Recurses inside the condition, so every level's mark is live at once.
:- pred nest(int::in, open(list(int))::ia) is semidet.
nest(N, L) :-
    promise_pure ( if N = 0 then
        L = []
    else if L = [N | T], nest(N - 1, T) then
        true
    else
        fail
    ).

:- pred show(open(list(int))::ia, io::di, io::uo) is det.
show(X, !IO) :-
    promise_pure (
        if impure from_open(X, G) then io.write(G, !IO)
        else io.write_string("open", !IO)
    ),
    io.nl(!IO).

main(!IO) :-
    promise_pure (
        ( if build(1, 3, L1) then show(L1, !IO) else io.write_string("no\n", !IO) ),
        ( if head_is_3(L2) then show(L2, !IO) else io.write_string("no\n", !IO) ),
        ( if not_empty(L3) then show(L3, !IO) else io.write_string("no\n", !IO) ),
        ( if small(L4, 2) then show(L4, !IO) else io.write_string("no\n", !IO) ),
        ( if one_or_two(L5) then show(L5, !IO) else io.write_string("no\n", !IO) ),
        ( if nest(1000, L6), impure from_open(L6, G6) then
            io.write_int(list.foldl(plus, G6, 0), !IO), io.nl(!IO)
        else
            io.write_string("no\n", !IO)
        )
    ).

% build fills the hole; head_is_3 binds the fresh L2 to [3 | _]; L3 = []
% succeeds on the unbound L3, so \+ fails; small; one_or_two undoes [1].
% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:open
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[2]
% OUTPUT-NEXT:[3]
% OUTPUT-NEXT:500500
