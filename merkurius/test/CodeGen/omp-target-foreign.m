% Foreign code on a device (design document, Library code on the device):
% the device's compile compiles the C of the foreign_procs that its
% target region reaches, the module's own and the library's, as OpenMP
% compiles C for the device, with the GPU's libc and libm, whose printf goes
% to the host. Only the functions reached are compiled and linked: pid's C
% and host_pid, in foreign_code, call getpid, which the device does not have.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_foreign.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, math, string.
:- pragma foreign_decl("C", "
#include <math.h>
#include <stdio.h>
#include <unistd.h>
long host_pid(void);
").
:- pragma foreign_code("C", "long host_pid(void) { return getpid(); }").

main(!IO) :-
    io.write_string("before\n", !IO),
    %$omp target
    compute(N, X),
    io.write_int(N, !IO),
    io.nl(!IO),
    io.write_float(X, !IO),
    io.nl(!IO),
    ( if pid(0) > 0 then io.write_string("after\n", !IO) else true ).

:- pred compute(int::out, float::out) is det.

compute(N, X) :-
    N = hello(plus_one(40)) + string.length("abc"),
    X = my_sin(0.5) + hypotenuse(3.0, 4.0) + math.ceiling(2.25) +
        math.floor(1.75).

:- func plus_one(int) = int.
:- pragma foreign_proc("C", plus_one(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe], "Y = X + 1;").

:- func hello(int) = int.
:- pragma foreign_proc("C", hello(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    printf(""hello %d\\n"", (int) X);
    fflush(stdout);
    Y = X;
").

:- func my_sin(float) = float.
:- pragma foreign_proc("C", my_sin(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe], "Y = sin(X);").

:- func hypotenuse(float, float) = float.
:- pragma foreign_proc("C", hypotenuse(X::in, Z::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe], "Y = hypot(X, Z);").

:- func pid(int) = int.
:- pragma foreign_proc("C", pid(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe], "Y = X + host_pid();").

% DEVICE-DAG: define internal i64 @merk_foreign_omp__target__foreign_Dplus__one__f1__0(
% DEVICE-DAG: define internal i64 @merk_foreign_omp__target__foreign_Dhello__f1__0(
% DEVICE-DAG: define internal double @merk_foreign_omp__target__foreign_Dmy__sin__f1__0(
% DEVICE-DAG: define internal double @merk_foreign_math_Dceiling__f1__0(
% DEVICE-DAG: declare {{.*}}double @sin(
% DEVICE-DAG: declare {{.*}}i32 @printf(
% DEVICE-NOT: pid
% DEVICE-NOT: merk.device_foreign

% OUTPUT:before
% OUTPUT-NEXT:hello 41
% OUTPUT-NEXT:44
% OUTPUT-NEXT:9.479425538604204
% OUTPUT-NEXT:after
% OUTPUT-NOT:{{.}}
