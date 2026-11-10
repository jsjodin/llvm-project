% Constructors whose names are also syntax, which mmc accepts in a data-term:
% -> without an else part, ! and its forms of anything but a variable, and {},
% which overloads a tuple when one is visible. The output is mmc's for the same
% program.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --match-full-lines

:- module constructors_named_as_syntax.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type b ---> p ; q.
:- type a ---> x ; (b -> b) ; (! b) ; (!. b) ; {} ; {b, b}.

:- pred name(b::in, io::di, io::uo) is det.
name(p, !IO) :- io.write_string("p", !IO).
name(q, !IO) :- io.write_string("q", !IO).

:- pred show(a::in, io::di, io::uo) is det.
show(x, !IO) :- io.write_string("x", !IO).
show(A -> B, !IO) :- name(A, !IO), io.write_string(" arrow ", !IO), name(B, !IO).
% !A would be a state variable.
show(! p, !IO) :- io.write_string("bang p", !IO).
show(! q, !IO) :- io.write_string("bang q", !IO).
show(!. p, !IO) :- io.write_string("bangdot p", !IO).
show(!. q, !IO) :- io.write_string("bangdot q", !IO).
show({}, !IO) :- io.write_string("braces", !IO).
show({A, B}, !IO) :-
    io.write_string("pair ", !IO), name(A, !IO), io.write_string(" ", !IO), name(B, !IO).

:- pred line(a::in, io::di, io::uo) is det.
line(A, !IO) :- show(A, !IO), io.nl(!IO).

% A tuple still, where no constructor fits.
:- pred second({int, int}::in, int::out) is det.
second({_, B}, B).

main(!IO) :-
    % CHECK:p arrow q
    line(p -> q, !IO),
    % CHECK-NEXT:bang p
    line(! p, !IO),
    % CHECK-NEXT:bangdot q
    line(!. q, !IO),
    % CHECK-NEXT:braces
    line({}, !IO),
    % CHECK-NEXT:pair p q
    line({p, q}, !IO),
    second({1, 2}, N),
    % CHECK-NEXT:2
    io.write_int(N, !IO),
    io.nl(!IO).
