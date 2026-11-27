:- module process_gpu_select_array.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- process summer(mailbox(int)) receives array(int) on gpu.

summer(Report, Self, !IO) :-
    % A selected array is in what the case gives.
    receive [io(!IO)] (
        select(Self, A) -> send(Report, array.size(A), !IO)
    ).

main(!IO) :-
    new_mailbox(Report, !IO),
    spawn(gpu(0), summer(Report), _, !IO).
