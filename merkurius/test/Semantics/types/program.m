% Type checking a program that mmc compiles: constructors and fields,
% field update, lambdas and higher-order calls, closures of functions,
% overloaded functions such as int's and float's +, DCG rules, try goals,
% and state variables. Each variable gets its type, and each name the one
% declaration that fits.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s

:- module program.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module float, int, list, string, univ.
:- type point ---> point(x :: int, y :: int).

main(!IO) :-
    P0 = point(1, 2),
    P = P0 ^ x := P0 ^ y + 1,
    F = (func(A) = A * 2),
    L = list.map(F, [P ^ x, P ^ y]),
    io.write_string(string.join_list(", ", list.map(string.from_int, L)), !IO),
    io.nl(!IO),
    digits(Ds, [1, 2, 3], _),
    length(Ds, N),
    io.print_line({Ds, N}, !IO),
    ( try [io(!IO)] io.write_string("x", !IO)
      then true
      catch_any E -> io.print_line(E, !IO)
    ),
    Apply = (pred(X::in, Y::out) is det :- Y = X + 1),
    Apply(1, Two),
    call(Apply, Two, Three),
    Half = float(Three) / 2.0,
    io.format("%d %d %f\n", [i(Two), i(Three), f(Half)], !IO).

:- pred digits(list(int)::out, list(int)::in, list(int)::out) is det.
digits(Ds) -->
    ( [D] -> digits(Ds0), { Ds = [D | Ds0] }
    ; { Ds = [] }
    ).

% CHECK-LABEL: clause at 17:1 of pred program.main/2: pred program.main(io.state, io.state)
% CHECK-NEXT:   IO!0: io.state
% CHECK-NEXT:   P0: program.point
% CHECK-NEXT:   P: program.point
% CHECK-NEXT:   F: (func(int) = int)
% CHECK-NEXT:   A: int
% CHECK-NEXT:   L: list.list(int)
% CHECK:        Ds: list.list(int)
% CHECK:        N: int
% CHECK:        E: univ.univ
% CHECK:        Apply: pred(int, int)
% CHECK:        Two: int
% CHECK-NEXT:   Three: int
% CHECK-NEXT:   Half: float
% CHECK:        18:10 point -> constructor program.point/2
% CHECK-NEXT:   19:24 y -> field program.y/1
% CHECK-NEXT:   19:26 + -> func int.'+'/2
% CHECK-NEXT:   19:14 x -> field update program.'x :='/2
% CHECK-NEXT:   20:22 * -> func int.'*'/2
% CHECK:        21:9 list.map -> func list.map/2
% CHECK:        24:5 digits -> pred program.digits/3
% CHECK-NEXT:   25:5 length -> pred list.length/2
% CHECK:        34:12 float -> func float.float/1
% CHECK-NEXT:   34:25 / -> func float.'/'/2
% CHECK-LABEL: clause at 38:1 of pred program.digits/3: pred program.digits(list.list(int), list.list(int), list.list(int))
% CHECK-NEXT:   Ds: list.list(int)
% CHECK:        D: int
