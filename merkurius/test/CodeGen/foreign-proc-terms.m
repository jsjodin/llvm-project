% Terms and arrays in a foreign_proc (reference manual, "C data passing
% conventions"): a term is an MR_Word and a tuple an MR_Tuple, opaque to C,
% whatever its layout, and an array an MR_ArrayPtr, as mmc passes it. The
% output is mmc's for the same program.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module foreign_proc_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- type colour ---> red ; green ; blue.
:- type temp ---> temp(float).
:- type tree ---> leaf ; node(tree, int, tree).

:- pred pass_colour(colour::in, colour::out) is det.
:- pragma foreign_proc("C", pass_colour(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").

% A notag type of a float, in the word its C type is.
:- pred pass_temp(temp::in, temp::out) is det.
:- pragma foreign_proc("C", pass_temp(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").

:- pred pass_tree(tree::in, tree::out) is det.
:- pragma foreign_proc("C", pass_tree(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").

:- pred pass_tuple({int, string}::in, {int, string}::out) is det.
:- pragma foreign_proc("C", pass_tuple(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").

% Several outputs, through pointers, and a test.
:- pred both(tree::in, int::in, tree::out, int::out) is semidet.
:- pragma foreign_proc("C", both(X::in, N::in, Y::out, M::out),
    [will_not_call_mercury, promise_pure],
    "Y = X; M = N + 1; SUCCESS_INDICATOR = N > 0;").

:- pred pass_array(array(float)::array_ui, int::out) is det.
:- pragma foreign_proc("C", pass_array(A::array_ui, N::out),
    [will_not_call_mercury, promise_pure], "N = sizeof(A);").

:- func total(tree) = int.
total(leaf) = 0.
total(node(L, V, R)) = total(L) + V + total(R).

main(!IO) :-
    pass_colour(blue, C),
    ( if C = blue then io.write_string("blue\n", !IO) else true ),
    pass_temp(temp(2.5), temp(T)),
    io.write_float(T, !IO),
    io.nl(!IO),
    pass_tree(node(leaf, 3, node(leaf, 4, leaf)), Tree),
    io.write_int(total(Tree), !IO),
    io.nl(!IO),
    pass_tuple({7, "seven"}, {I, S}),
    io.write_int(I, !IO),
    io.write_string(S, !IO),
    io.nl(!IO),
    ( if both(Tree, 1, Tree2, M) then
        io.write_int(total(Tree2) + M, !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    array.init(2, 0.5, A),
    pass_array(A, Size),
    io.write_int(Size, !IO),
    io.nl(!IO).

% OUTPUT:blue
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:7
% OUTPUT-NEXT:7seven
% OUTPUT-NEXT:9
% OUTPUT-NEXT:8
