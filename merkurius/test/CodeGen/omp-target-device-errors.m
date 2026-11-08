% What the device clause does not take yet, or at all.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_device_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

main(!IO) :-
    a(1, _),
    b(1.0, 1, _),
    c(1, _),
    d(1, 1, _),
    e(1, _),
    io.write_int(1, !IO).

:- pred a(int::in, int::out) is det.
a(X, Y) :-
    %$omp target device(Unbound)
    Y = X + 1.
% CHECK: omp-target-device-errors.m:[[@LINE-2]]:5: error: the device number Unbound is not bound before the directive

:- pred b(float::in, int::in, int::out) is det.
b(F, X, Y) :-
    %$omp target device(F)
    Y = X + 1.
% CHECK: omp-target-device-errors.m:[[@LINE-2]]:5: error: the device number F must be an integer

:- pred c(int::in, int::out) is det.
c(X, Y) :-
    %$omp target device(X + 1)
    Y = X + 1.
% CHECK: omp-target-device-errors.m:[[@LINE-2]]:5: error: not yet implemented: {{.*}}a device number that is not an integer or a variable

:- pred d(int::in, int::in, int::out) is det.
d(W, X, Y) :-
    %$omp target device(ancestor: W)
    Y = X + 1.
% CHECK: omp-target-device-errors.m:[[@LINE-2]]:5: error: not yet implemented: {{.*}}modifiers of the device clause

:- pred e(int::in, int::out) is det.
e(X, Y) :-
    %$omp target device(0) device(1)
    Y = X + 1.
% CHECK: omp-target-device-errors.m:[[@LINE-2]]:5: error: {{.*}}
