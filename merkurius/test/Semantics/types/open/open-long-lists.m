% Lists of open terms written out (design document, "Open terms as built"): each
% cell's twin may take a ground head, so each element doubles the second
% pass's assignments. Those of finished variables, whose goals have all
% been checked, are pruned as they come, as the end would keep the most
% open of them; without it, twelve elements were over the limit of 1000.
%
% RUN: merkc -fc1 -fopen-types -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s

:- module open_long_lists.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open.

:- type t ---> t(int, int, int, int).

:- pred holes(open(list(int))::oa) is det.
holes([_, _, _, _, _, _, _, _, _, _, _, _]).
% CHECK:      clause at [[@LINE-1]]:1 of pred open_long_lists.holes/1

:- pred cells(open(list(t))::oa) is det.
cells([t(_, _, _, _), t(_, 2, _, _), t(_, _, _, _), t(1, _, _, 4),
    t(_, _, _, _), t(_, _, _, _)]).
% CHECK:      clause at [[@LINE-2]]:1 of pred open_long_lists.cells/1
% CHECK-DAG:  [[@LINE-3]]:8 t -> constructor open_long_lists.t/4, open{{$}}
% CHECK-DAG:  [[@LINE-4]]:23 t -> constructor open_long_lists.t/4, open, argument 2 ground{{$}}
% CHECK-DAG:  [[@LINE-5]]:53 t -> constructor open_long_lists.t/4, open, argument 1 ground, argument 4 ground{{$}}
% CHECK-DAG:  [[@LINE-5]]:20 t -> constructor open_long_lists.t/4, open{{$}}

main(!IO) :-
    holes(H), cells(C),
    promise_pure ( if impure from_open(H, _), impure from_open(C, _) then
        io.write_string("ground\n", !IO)
    else
        io.write_string("holes\n", !IO)
    ).
