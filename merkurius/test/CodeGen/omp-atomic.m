% Atomic updates of array elements under %$omp atomic (docs/INPROGRESS.md,
% Atomic updates of array elements), as Flang lowers Fortran's x = x op
% expr: the operand before the atomic operation, then omp.atomic.update on
% the element's address, whose region applies the operator to the old
% value; a capture puts it in omp.atomic.capture with omp.atomic.read
% into a slot, before the update for the old value, after it for the new.
% Run on four threads, the program prints what the sequential program
% prints: the floats added are exact, so the sums do not depend on the
% order, and what the captures give is checked as a set.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-llvm %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_atomic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, float, int, list.

main(!IO) :-
    array.init(10, 0, H0),
    counts(100000, H0, H),
    write_ints(H, !IO),
    array.init(4, 0.0, F0),
    halves(100000, F0, F),
    write_floats(F, !IO),
    array.init(3, 0, B0),
    bits(1000, B0, B),
    write_ints(B, !IO),
    array.init(1, 0, C0),
    array.init(50000, -1, V0),
    odds(100000, C0, C, V0, V),
    check_odds(C, V, !IO),
    array.init(1, 0, D0),
    array.init(1001, 0, M0),
    marks(1000, D0, D, M0, M),
    check_marks(D, M, !IO).
% OUTPUT:10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
% OUTPUT:12500.0 12500.0 12500.0 12500.0
% OUTPUT:1023 -500500 0
% OUTPUT:50000 odds, each once
% OUTPUT:1000 marks, 1 to 1000, each once

    % A count of each digit: the update, by state variable.
:- pred counts(int::in, array(int)::array_di, array(int)::array_uo) is det.
counts(N, !H) :-
    %$omp parallel do
    ( for(I, 0, N - 1) do
        K = I mod 10,
        %$omp atomic
        !H ^ elem(K) := !.H ^ elem(K) + 1
    ).
% MERK-LABEL: merk.proc private @omp_atomic.counts__body_
% MERK:         merk.check_index %[[H:.*]][%[[K:.*]]], "array.set" : <i64>
% MERK-NEXT:    %[[AT:.*]] = merk.array_element_ref %[[H]][%[[K]]] : <i64> -> !llvm.ptr
% MERK-NEXT:    omp.atomic.update %[[AT]] : !llvm.ptr {
% MERK-NEXT:    ^bb0(%[[OLD:.*]]: i64):
% MERK-NEXT:      %[[NEW:.*]] = arith.addi %[[OLD]], %{{.*}} : i64
% MERK-NEXT:      omp.yield(%[[NEW]] : i64)
% MERK-NEXT:    } atomic_control <>
% MERK-NEXT:    merk.succeed %[[H]]
%
% An atomicrmw, at the element after the header word.
% LLVM-LABEL: define {{.*}} @omp_atomic.counts__body_
% LLVM:         %[[P:.*]] = getelementptr i64, ptr addrspace(1) %{{.*}}, i64 %{{.*}}
% LLVM-NEXT:    %[[Q:.*]] = addrspacecast ptr addrspace(1) %[[P]] to ptr
% LLVM-NEXT:    atomicrmw add ptr %[[Q]], i64 1 monotonic

    % A float sum in a predicate the loop calls, OpenMP's orphaned atomic,
    % written with array.set and array.lookup.
:- pred halves(int::in, array(float)::array_di, array(float)::array_uo) is det.
halves(N, !F) :-
    %$omp parallel do
    ( for(I, 0, N - 1) do
        add_half(I mod 4, !F)
    ).

:- pred add_half(int::in, array(float)::array_di, array(float)::array_uo) is det.
add_half(K, !F) :-
    %$omp atomic update
    array.set(K, array.lookup(!.F, K) + 0.5, !F).
% MERK-LABEL: merk.proc private @omp_atomic.add_half_3_0(
% MERK:         omp.atomic.update %{{.*}} : !llvm.ptr {
% MERK-NEXT:    ^bb0(%[[OLD:.*]]: f64):
% MERK-NEXT:      %[[NEW:.*]] = arith.addf %[[OLD]], %{{.*}} : f64
% MERK-NEXT:      omp.yield(%[[NEW]] : f64)
% LLVM-LABEL: define {{.*}} @omp_atomic.add_half_3_0(
% LLVM:         atomicrmw fadd ptr %{{.*}}, double 5.000000e-01 monotonic

    % Bits by \/, a sum by an operand before the element, and xor twice of
    % each, which cancels.
:- pred bits(int::in, array(int)::array_di, array(int)::array_uo) is det.
bits(N, !B) :-
    %$omp parallel do
    ( for(I, 1, N) do
        %$omp atomic
        !B ^ elem(0) := !.B ^ elem(0) \/ (1 << (I mod 10)),
        %$omp atomic
        !B ^ elem(1) := (-I) + !.B ^ elem(1),
        %$omp atomic
        !B ^ elem(2) := !.B ^ elem(2) `xor` (I * 7919),
        %$omp atomic
        !B ^ elem(2) := !.B ^ elem(2) `xor` (I * 7919)
    ).

    % Each odd number appended to V at the index a capture takes from C:
    % the old value.
:- pred odds(int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.
odds(N, !C, !V) :-
    %$omp parallel do
    ( for(I, 0, N - 1) do
        ( if I mod 2 = 1 then
            %$omp atomic capture
            ( At = !.C ^ elem(0), !C ^ elem(0) := At + 1 ),
            !V ^ elem(At) := I
        else
            true
        )
    ).
% MERK-LABEL: merk.proc private @omp_atomic.odds__body_
% MERK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x i64
% MERK:         %[[AT:.*]] = merk.array_element_ref
% MERK-NEXT:    omp.atomic.capture {
% MERK-NEXT:      omp.atomic.read %[[SLOT]] = %[[AT]] : !llvm.ptr, !llvm.ptr, i64
% MERK-NEXT:      omp.atomic.update %[[AT]] : !llvm.ptr {
% MERK:           }
% MERK-NEXT:    }
% MERK-NEXT:    llvm.load %[[SLOT]] : !llvm.ptr -> i64
% LLVM-LABEL: define {{.*}} @omp_atomic.odds__body_
% LLVM:         atomicrmw add ptr %{{.*}}, i64 1 monotonic

    % Marks 1 to N, each at the index a capture of the new value gives.
:- pred marks(int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.
marks(N, !D, !M) :-
    %$omp parallel do
    ( for(_, 1, N) do
        %$omp atomic capture
        ( !D ^ elem(0) := !.D ^ elem(0) + 1, At = !.D ^ elem(0) ),
        !M ^ elem(At) := !.M ^ elem(At) + 1
    ).
% MERK-LABEL: merk.proc private @omp_atomic.marks__body_
% MERK:         omp.atomic.capture {
% MERK-NEXT:      omp.atomic.update
% MERK:           omp.atomic.read

:- pred check_odds(array(int)::in, array(int)::in, io::di, io::uo) is det.
check_odds(C, V, !IO) :-
    array.init(100000, no, Seen0),
    seen(V, 0, Seen0, Seen, yes, Once),
    Count = C ^ elem(0),
    io.write_int(Count, !IO),
    io.write_string(" odds", !IO),
    ( if Once = yes, all_odd_seen(Seen, 0) then
        io.write_string(", each once\n", !IO)
    else
        io.write_string(", not each once\n", !IO)
    ).

:- pred seen(array(int)::in, int::in, array(bool)::array_di,
    array(bool)::array_uo, bool::in, bool::out) is det.
seen(V, J, !Seen, !Once) :-
    ( if J < array.size(V) then
        X = V ^ elem(J),
        ( if X >= 0, X < array.size(!.Seen), !.Seen ^ elem(X) = no then
            !Seen ^ elem(X) := yes
        else
            !:Once = no
        ),
        seen(V, J + 1, !Seen, !Once)
    else
        true
    ).

:- pred all_odd_seen(array(bool)::in, int::in) is semidet.
all_odd_seen(Seen, J) :-
    ( if J < array.size(Seen) then
        Seen ^ elem(J) = ( if J mod 2 = 1 then yes else no ),
        all_odd_seen(Seen, J + 1)
    else
        true
    ).

:- pred check_marks(array(int)::in, array(int)::in, io::di, io::uo) is det.
check_marks(D, M, !IO) :-
    io.write_int(D ^ elem(0), !IO),
    ( if M ^ elem(0) = 0, ones(M, 1) then
        io.write_string(" marks, 1 to 1000, each once\n", !IO)
    else
        io.write_string(" marks, not each once\n", !IO)
    ).

:- pred ones(array(int)::in, int::in) is semidet.
ones(M, J) :-
    ( if J < array.size(M) then
        M ^ elem(J) = 1,
        ones(M, J + 1)
    else
        true
    ).

:- pred write_ints(array(int)::in, io::di, io::uo) is det.
write_ints(A, !IO) :-
    array.to_list(A, L),
    write_list(L, (pred(X::in, !.IO::di, !:IO::uo) is det :- io.write_int(X, !IO)),
        !IO).

:- pred write_floats(array(float)::in, io::di, io::uo) is det.
write_floats(A, !IO) :-
    array.to_list(A, L),
    write_list(L,
        (pred(X::in, !.IO::di, !:IO::uo) is det :- io.write_float(X, !IO)),
        !IO).

:- pred write_list(list(T)::in, pred(T, io, io)::in(pred(in, di, uo) is det),
    io::di, io::uo) is det.
write_list([], _, !IO) :-
    io.nl(!IO).
write_list([X | Xs], Write, !IO) :-
    Write(X, !IO),
    ( if Xs = [] then true else io.write_char(' ', !IO) ),
    write_list(Xs, Write, !IO).
