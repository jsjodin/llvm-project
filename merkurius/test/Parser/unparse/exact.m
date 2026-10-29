% The exact unparser gives back the file, byte for byte: comments, layout,
% line number directives and every spelling of a token come from the tree.
%
% RUN: merkc -fc1 -fdebug-unparse-exact %s | cmp - %s
% RUN: merkc -fc1 -fdebug-unparse-exact-minimal-layout %s | FileCheck %s

:- module exact.        % a comment after an item
:- interface.
:- import_module io.

:- type shape
    --->    circle(float)   /* a block comment */
    ;       square(side :: float).

:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module float, int, list, string.

% The minimal layout keeps each token's spelling, and removes comments and
% line number directives.
% CHECK:      {{^}}:- module exact.{{$}}
% CHECK-NEXT: {{^}}:- interface.{{$}}
% CHECK-NEXT: {{^}}:- import_module io.{{$}}
% CHECK-NEXT: {{^}}:- type shape{{$}}
% CHECK-NEXT: {{^}}---> circle(float){{$}}
% CHECK-NEXT: {{^}}; square(side :: float).{{$}}

main(!IO) :-
    Xs = [0x1F, 0b101, 0'a, 1_000],
    io.write_float(3.0e2, !IO),
    (   if list.member(X, Xs), X > 30
#200
    then io.write_string("big\n\t\"quoted\"", !IO)
    else true
    ),
    S = string.'++'("y", "x"),
    io.write_string(S, !IO).
% CHECK:      {{^}}main(!IO) :-{{$}}
% CHECK-NEXT: {{^}}Xs = [0x1F, 0b101, 0'a, 1_000],{{$}}
% CHECK-NEXT: {{^}}io.write_float(3.0e2, !IO),{{$}}
% CHECK-NEXT: {{^}}( if list.member(X, Xs), X > 30{{$}}
% CHECK-NEXT: {{^}}then io.write_string("big\n\t\"quoted\"", !IO){{$}}
