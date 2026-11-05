% The end of a target data region copies the device's elements to the array that
% came in, so a state variable that leaves as another array is an error at run
% time, rather than a wrong answer later. On the host too, where the mapping does
% nothing, so that the program does not work only without a device.
%
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t
% RUN: %libomp-env not %t 2>&1 | FileCheck %s

:- module omp_target_data_replaced.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred replace(array(int)::array_di, array(int)::array_uo) is det.
replace(A, array.copy(A)).

main(!IO) :-
    some [!A] (
        array.init(4, 1, !:A),
        % CHECK: Uncaught Mercury exception:
        % CHECK-NEXT: Software Error: target data: the array A was replaced in the region, not updated in place
        %$omp target data map(tofrom: A)
        replace(!A),
        io.write_int(!.A ^ elem(0), !IO),
        io.nl(!IO)
    ).
