% C that compiles for the host but not for a device, as OpenMP compiles it
% there: clang reports its errors, and the device's compile names the module
% whose foreign code it is (docs/INPROGRESS.md, Library code on the device,
% step 3).
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fopenmp -emit-llvm %s -o /dev/null
% REQUIRES: amdgpu-registered-target

:- module omp_target_foreign_c_error.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    %$omp target
    N = twice(21),
    io.write_int(N, !IO),
    io.nl(!IO).

:- func twice(int) = int.
:- pragma foreign_proc("C", twice(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
#ifdef __AMDGPU__
    Y = no_such_variable;
#else
    Y = 2 * X;
#endif
").
% CHECK: omp-target-foreign-c-error.m:[[#@LINE-5]]:9: error: use of undeclared identifier 'no_such_variable'
% CHECK: omp-target-foreign-c-error.m:[[#@LINE-10]]:4: error: omp_target_foreign_c_error's foreign code does not compile for the device

% Without the device's GPU, clang would look for one on the machine.
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-GPU --implicit-check-not=error:
% NO-GPU: omp-target-foreign-c-error.m:24:4: error: compiling foreign code for an OpenMP device needs its GPU, as -target-cpu or the triple names it, such as amdgpu10.30-amd-amdhsa for gfx1030
