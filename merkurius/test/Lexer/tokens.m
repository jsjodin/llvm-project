% The tokens of the reference manual (2.4, Tokens), in a module that
% mmc 22.01 also compiles.
%
% RUN: merkc -fc1 -fdebug-unparse-tokens %s | cmp - %s
% RUN: merkc -fc1 -fdebug-dump-tokens %s | FileCheck %s

:- module tokens.
:- interface.
:- import_module float, int, list, string, uint8.

:- func items = list(int).
:- func text = list(string).
:- func ratio = float.
:- func byte = uint8.
:- func big = float.
:- func 'quoted name' = int.
:- func larger(int, int) = int.

:- implementation.

% CHECK: [[@LINE+3]]:1 name items
% CHECK-NEXT: [[@LINE+2]]:7 name =
% CHECK-NEXT: [[@LINE+1]]:9 open_list [
items = [0, 1_000, 0b1010, 0o17, 0xff, 0'a, 0' ].
% CHECK: integer 1_000
% CHECK: integer 0b1010
% CHECK: integer 0o17
% CHECK: integer 0xff
% CHECK: integer 0'a
% CHECK-NEXT: comma ,
% CHECK-NEXT: integer 0'{{ }}
% CHECK-NEXT: close_list ]
% CHECK-NEXT: end .

% Strings may span lines, with or without an escaped newline.
text = ["say ""hi""\n", "two
lines", "cont\
inued"].
% CHECK: string "say ""hi""\n"
% CHECK: string "two\nlines"
% CHECK: string "cont\\ninued"

ratio = 1.5e-3.
% CHECK: float 1.5e-3

byte = 0xff_u8.
% CHECK: integer 0xff_u8

% mmc 22.01.9 accepts an underscore between a fraction and its exponent.
big = 1.5_e10.
% CHECK: float 1.5_e10

'quoted name' = 1.
% CHECK: [[@LINE-1]]:1 name 'quoted name'

% The backquote, open_ct against open, and module qualification.
larger(A, B) = A `int.max` (B).
% CHECK: [[@LINE-1]]:1 name larger
% CHECK-NEXT: open_ct (
% CHECK: backquote `
% CHECK-NEXT: name int
% CHECK-NEXT: name .
% CHECK-NEXT: name max
% CHECK-NEXT: backquote `
% CHECK-NEXT: open (
% CHECK-NEXT: variable B
% CHECK-NEXT: close )
% CHECK-NEXT: end .
% CHECK-NEXT: eof
