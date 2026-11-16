% GPU code calls procedures of another module of the program: a GPU process
% and a target region, which call device_helpers' procedures; those call a
% procedure of their own module that its interface leaves out, recursively,
% and the library, and make values of a type of their module. With several
% sources the driver writes each one's merk bytecode first, from which each
% device compile copies what its code reaches, as it copies the library's
% (design document, Library code on the device).
%
% Here at the host; on the GPU in process-other-module-gpu.test.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fopenmp -fprocess-declarations --offload-arch=%offload-archs -I %S/Inputs %libomp-flags %S/Inputs/device_helpers.m %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_other_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module device_helpers, int, list, process, string.

    % What the GPU process sends: each helper's result in turn.
:- process helpers(mailbox(int)) receives int on gpu.

helpers(Report, _Self, !IO) :-
    send(Report, area(rect(6, 7)), !IO),
    send(Report, area(square(5)), !IO),
    send(Report, steps(27), !IO),
    send(Report, top_bits(12345678901234567890u64), !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Where = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Me, !IO),
    spawn(Where, helpers(Me), _, !IO),
    print_results(Me, 4, !IO),
    %$omp target
    io.write_int(steps(97) + area(rect(2, 3)), !IO),
    io.nl(!IO).

:- pred print_results(mailbox(int)::in, int::in, io::di, io::uo) is det.

print_results(Me, N, !IO) :-
    ( if N > 0 then
        receive(Me, any, after(60000), R, !IO),
        ( if R = message(V) then
            io.write_string(string.int_to_string(V) ++ "\n", !IO)
        else
            io.write_string("no answer\n", !IO)
        ),
        print_results(Me, N - 1, !IO)
    else
        true
    ).

% CHECK:42
% CHECK-NEXT:25
% CHECK-NEXT:111
% CHECK-NEXT:11228329
% CHECK-NEXT:124
% FREED-NEXT:Merkurius runtime: 0 messages are neither read nor freed
% FREED-NEXT:Merkurius runtime: 0 chunks of the devices' arenas and 0 rings are allocated
% FREED-NEXT:Merkurius runtime: 0 blocks of messages between GPU processes are allocated
