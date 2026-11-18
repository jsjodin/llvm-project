% Errors in dyn_groupprivate clauses (docs/INPROGRESS.md, GPU processes that
% run loops, (s5)). The clause is OpenMP 6.1's; it names arrays, with the
% modifiers fallback(default_mem), fallback(abort) and cgroup; fallback(null)
% and the clause on teams distribute parallel do are not yet implemented.
% Both compiles check the clause; only the device's, which makes the arrays
% group private, checks that the loop's body makes each one with
% array.init. Each is in a predicate of its own, as the first that is not
% yet implemented ends a procedure's code.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-version=61 -fdo-loops -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-loops -fopenmp-version=61 -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,DEVICE --implicit-check-not=error:
%
% Before OpenMP 6.1 the clause is not allowed, which the error says.
% RUN: not merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %S/process-group-private.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=VERSION
% VERSION: error: the dyn_groupprivate clause is not allowed on the teams distribute directive before OpenMP 6.1 (-fopenmp-version=61)
%
% A group-private array that the loop's body may give on, here as the
% array the loop threads, is an error of the device's compile.
% RUN: not merkc -fc1 -fprocess-loops -fopenmp-version=61 -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %S/Inputs/process_group_private_escape.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=ESCAPE
% ESCAPE: process_group_private_escape.m:36:9: error: a group-private array must not outlive its iteration of the loop, but the loop's body may give it, or an array updated from it, on

:- module process_group_private_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        array.init(N, 0, A0),
        unmade(N, A0, A1),
        not_variable(N, A1, A2),
        bad_modifier(N, A2, A3),
        bad_fallback(N, A3, A4),
        two_fallbacks(N, A4, A5),
        null_fallback(N, A5, A6),
        two_clauses(N, A6, A7),
        twice(N, A7, A8),
        combined(N, A8, A),
        send(Reply, array.lookup(A, 0), !IO)
    else
        true
    ).

:- pred unmade(int::in, array(int)::array_di, array(int)::array_uo) is det.
unmade(N, !A) :-
% DEVICE: :[[@LINE+1]]:5: error: X is not an array that the loop's body makes with array.init or array.unsafe_noinit
    %$omp teams distribute dyn_groupprivate(G, X)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred not_variable(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
not_variable(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: a dyn_groupprivate clause names arrays that the loop's body makes with array.init or array.unsafe_noinit
    %$omp teams distribute dyn_groupprivate(64)
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := I
    ).

:- pred bad_modifier(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
bad_modifier(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: the modifiers of a dyn_groupprivate clause are fallback(default_mem), fallback(abort) and cgroup
    %$omp teams distribute dyn_groupprivate(strict: G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred bad_fallback(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
bad_fallback(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: the modifiers of a dyn_groupprivate clause are fallback(default_mem), fallback(abort) and cgroup
    %$omp teams distribute dyn_groupprivate(fallback(global): G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred two_fallbacks(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
two_fallbacks(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: a dyn_groupprivate clause can have only one fallback modifier
    %$omp teams distribute dyn_groupprivate(fallback(abort), fallback(default_mem): G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred null_fallback(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
null_fallback(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: not yet implemented: code generation for fallback(null) in a dyn_groupprivate clause
    %$omp teams distribute dyn_groupprivate(cgroup, fallback(null): G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred two_clauses(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
two_clauses(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: a directive can have only one dyn_groupprivate clause
    %$omp teams distribute dyn_groupprivate(G) dyn_groupprivate(H)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        array.init(4, I, H),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0) + H ^ unsafe_elem(0)
    ).

:- pred twice(int::in, array(int)::array_di, array(int)::array_uo) is det.
twice(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: G is named twice in the dyn_groupprivate clause
    %$omp teams distribute dyn_groupprivate(G, G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

:- pred combined(int::in, array(int)::array_di, array(int)::array_uo) is det.
combined(N, !A) :-
% CHECK: :[[@LINE+1]]:5: error: not yet implemented: code generation for the dyn_groupprivate clause of the teams directive
    %$omp teams distribute parallel do dyn_groupprivate(G)
    ( for(I, 0, N - 1) do
        array.init(4, I, G),
        !A ^ unsafe_elem(I) := G ^ unsafe_elem(0)
    ).

main(!IO) :-
    io.write_int(1, !IO),
    io.nl(!IO).
