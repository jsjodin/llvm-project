% Mercury code gets statepoints and a stack map after the optimizer. Each root
% has a type's index in the stack map, or -1 for one the collector can ignore
% (docs/INPROGRESS.md, Statepoints and stack maps).
%
% RUN: merkc -fc1 -S -O2 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=ASM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% ASM: .section .llvm_stackmaps
% ASM: __LLVM_StackMaps:

:- module statepoints.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pragma no_inline(keep/1).
:- func keep(list(T)) = list(T).
keep(L) = L.

:- pragma no_inline(split/2).
:- pred split(list(int)::in, list(int)::out) is semidet.
split([_ | T], keep(T)).

main(!IO) :-
    L = keep([1, 2, 3]),
    ( if split(L, R) then
        io.write_int(list.length(R) + list.length(L), !IO)
    else
        io.write_int(0, !IO)
    ),
    io.nl(!IO).

% OUTPUT:5
