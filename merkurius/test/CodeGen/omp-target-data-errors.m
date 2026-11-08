% What the target data directive rejects, which a stock compiler reads as comments.
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s

:- module omp_target_data_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred up(array(int)::array_di, array(int)::array_uo) is det.
up(!A) :- !A ^ elem(0) := 1.

:- pred one(array(int)::array_di, array(int)::array_uo) is semidet.
one(A0, A) :- A0 ^ elem(0) = 1, up(A0, A).

% CHECK: :[[@LINE+3]]:5: error: the goal of the target data directive updates A, whose map type does not copy it back; use tofrom or from
:- pred lost(array(int)::array_di, array(int)::array_uo) is det.
lost(!A) :-
    %$omp target data map(to: A)
    up(!A).

% CHECK: :[[@LINE+3]]:5: error: the goal of a target data directive must be det, but it is semidet
:- pred fails(array(int)::array_di, array(int)::array_uo) is semidet.
fails(!A) :-
    %$omp target data map(tofrom: A)
    one(!A).

% CHECK: :[[@LINE+3]]:5: error: B is not a variable or state variable that the goal of the target data directive uses
:- pred unknown(array(int)::array_di, array(int)::array_uo) is det.
unknown(!A) :-
    %$omp target data map(tofrom: B)
    up(!A).

% CHECK: :[[@LINE+3]]:5: error: A is mapped twice
:- pred twice(array(int)::array_di, array(int)::array_uo) is det.
twice(!A) :-
    %$omp target data map(tofrom: A) map(tofrom: A)
    up(!A).

% CHECK: :[[@LINE+3]]:5: error: a target data directive must have a map clause
:- pred no_map(array(int)::array_di, array(int)::array_uo) is det.
no_map(!A) :-
    %$omp target data
    up(!A).

% CHECK: :[[@LINE+3]]:5: error: not yet implemented: code generation for mapping values of type int with the target data directive
:- pred scalar(int::in, int::out) is det.
scalar(X, Y) :-
    %$omp target data map(tofrom: X)
    Y = X + 1.

% CHECK: :[[@LINE+3]]:5: error: not yet implemented: code generation for array sections on the target data directive
:- pred section(array(int)::array_di, array(int)::array_uo) is det.
section(!A) :-
    %$omp target data map(tofrom: A(0 .. 1))
    up(!A).

% CHECK: :[[@LINE+3]]:5: error: not yet implemented: code generation for the if clause of the target data directive
:- pred clause(array(int)::array_di, array(int)::array_uo) is det.
clause(!A) :-
    %$omp target data map(tofrom: A) if(1)
    up(!A).

main(!IO) :- io.nl(!IO).
