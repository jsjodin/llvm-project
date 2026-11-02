% Undefined names, with notes like mmc's: a module not imported, a name of
% a module imported only with use_module written unqualified, and a name
% that exists with another arity. The character type is `character`;
% `char` is a type only where the module char is imported, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck %s

:- module undefined.
:- interface.
:- pred p(int::in) is det.
:- pred c(char::in) is det.
% CHECK: undefined.m:[[@LINE-1]]:11: error: undefined type char/0
:- pred d(character::in) is det.
:- implementation.
:- import_module list.
:- use_module string.

p(N) :-
    nosuch(N),
% CHECK: undefined.m:[[@LINE-1]]:5: error: undefined predicate nosuch/1
    X = nofunc(N),
% CHECK: undefined.m:[[@LINE-1]]:9: error: undefined symbol nofunc/1
    Y = from_int(N),
% CHECK: undefined.m:[[@LINE-1]]:9: error: undefined symbol from_int/1
% CHECK: because: from_int is in module string, which is imported with use_module, so it must be written string.from_int
    Z = map.init,
% CHECK: undefined.m:[[@LINE-1]]:9: error: undefined symbol map.init/0
% CHECK: because: the module map has not been imported
    W = string.nope(N),
% CHECK: undefined.m:[[@LINE-1]]:9: error: undefined symbol string.nope/1
    length([X, Y, Z, W], 2, 3).
% CHECK: undefined.m:[[@LINE-1]]:5: error: undefined predicate length/3
% CHECK: because: there is pred list.length/2
c(_).
d(_).
