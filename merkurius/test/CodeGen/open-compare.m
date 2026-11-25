% Open types have no order (design document, "Open terms", Operations): a
% comparison that reaches an open(T) value through a type variable ends the
% program with status 1, as one that reaches a closure does. A ground
% closed(T) value compares as its value does.
%
% RUN: merkc -fopen-types %s -o %t && not %t 2>&1 | FileCheck %s --match-full-lines

:- module open_compare.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module open.

:- type big ---> big(int).

:- pred cmp(T::in, T::in, comparison_result::out) is det.
:- pragma no_inline(cmp/3).
cmp(X, Y, R) :- compare(R, X, Y).

main(!IO) :-
    cmp(closed(big(1)) : closed(big), closed(big(2)), R0),
    io.write(R0, !IO), io.nl(!IO),
    cmp(big(1) : open(big), big(2), R1),
    io.write(R1, !IO), io.nl(!IO).
% CHECK:'<'
% CHECK-NEXT:Merkurius runtime: called compare/3 for an open type
