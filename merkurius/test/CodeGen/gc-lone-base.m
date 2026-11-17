% The base of an interior pointer that only the pointer names at a safepoint
% (runtime/roots.cpp, VisitLoneBases). At -O2 LLVM computes the address of
% count At once, for take_credit's lookup, and keeps it for the store of 0
% after the two copies. The counts there are a phi of the array that came in
% and the one send_if_credit returned, so the statepoint rewrite makes them a
% base of their own. It relocates that base at the first copy's safepoint,
% as the second needs it, but not at the second, where the base is named
% only as the address's: the code after it reads the counts from the
% bundle's root. The collector moved the address by its offset from a base
% it had not moved, so the store went to the old array, freed, and the count
% kept its value. Found 2026-11-17 by the Monte Carlo transport's credits,
% whose backlog then doubled at each credit. Every allocation collects.
%
% RUN: merkc -O2 %s -o %t2
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --match-full-lines --strict-whitespace
%
% CHECK:[0]
% CHECK-NEXT:[0, 0]
% CHECK-NEXT:[0, 0, 0, 2, 1, 0, 1]
% CHECK-NEXT:3
% CHECK-NOT:{{.}}

:- module gc_lone_base.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

main(!IO) :-
    Boxes = array.from_list([0, 0]),
    Owners = array.from_list([0, 1, 0, 1]),
    % Counts: a count for each of the 4 buffers, a credit for each of the 2
    % owners, then the number waiting in the backlog.
    Counts0 = array.from_list([2, 1, 0, 2, 1, 0, 3]),
    Buffers0 = array.init(4 * 2, 0),
    Backlog0 = array.from_list([7, 8, 9]),
    take_credit(Boxes, Owners, 1, 2, Buffers0, _, Counts0, Counts1,
        Backlog0, _, 0, Sent, !IO),
    take_credit(Boxes, Owners, 0, 2, array.init(4 * 2, 0), _, Counts1,
        Counts, array.from_list([4, 5, 6]), _, Sent, Sent2, !IO),
    io.write_line(array.to_list(Counts) `with_type` list(int), !IO),
    io.format("%d\n", [i(Sent2)], !IO).

:- pred take_credit(array(int)::in, array(int)::in, int::in,
    int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo, int::in, int::out,
    io::di, io::uo) is det.

take_credit(Boxes, Owners, From, Size, !Buffers, !Counts, !Backlog, !Sent,
        !IO) :-
    C = array.size(Owners) + From,
    array.lookup(!.Counts, C, Credit),
    array.set(C, Credit + 1, !Counts),
    flush_from(Boxes, Owners, From, Size, 0, !Buffers, !Counts, !Sent, !IO),
    At = array.size(Owners) + array.size(Boxes),
    array.lookup(!.Counts, At, Waiting),
    ( if Waiting > 0 then
        Old = array.copy(!.Backlog),
        Older = array.copy(Old),
        array.set(At, 0, !Counts),
        rerun(Old, Older, 1, Waiting, At, !Counts, !Backlog)
    else
        true
    ).

:- pred flush_from(array(int)::in, array(int)::in, int::in,
    int::in, int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo, int::in, int::out,
    io::di, io::uo) is det.

flush_from(Boxes, Owners, Only, Size, Domain, !Buffers, !Counts, !Sent,
        !IO) :-
    ( if Domain < array.size(Owners) then
        array.lookup(!.Counts, Domain, Count),
        ( if Count > 0, array.lookup(Owners, Domain) = Only then
            send_if_credit(Boxes, Owners, Size, Domain, !.Buffers, !Counts,
                !Sent, !IO)
        else
            true
        ),
        flush_from(Boxes, Owners, Only, Size, Domain + 1, !Buffers, !Counts,
            !Sent, !IO)
    else
        true
    ).

:- pred send_if_credit(array(int)::in, array(int)::in,
    int::in, int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo, int::in, int::out,
    io::di, io::uo) is det.

send_if_credit(Boxes, Owners, Size, Domain, Buffers, !Counts, !Sent, !IO) :-
    ( if Domain >= array.size(Owners) then
        % Recursive, so that it stays out of line, as a larger one would.
        send_if_credit(Boxes, Owners, Size, Domain - array.size(Owners),
            Buffers, !Counts, !Sent, !IO),
        !:Sent = !.Sent + 1
    else
        send_one(Boxes, Owners, Size, Domain, Buffers, !Counts, !Sent, !IO)
    ).

:- pred send_one(array(int)::in, array(int)::in,
    int::in, int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo, int::in, int::out,
    io::di, io::uo) is det.

send_one(Boxes, Owners, Size, Domain, Buffers, !Counts, !Sent, !IO) :-
    Owner = array.lookup(Owners, Domain),
    C = array.size(Owners) + Owner,
    array.lookup(!.Counts, C, Credit),
    array.lookup(!.Counts, Domain, Count),
    ( if Count > 0, Credit > 0 then
        Batch = array.init(Count, array.lookup(Buffers, Domain * Size)),
        io.write_line(array.to_list(Batch) `with_type` list(int), !IO),
        !:Sent = !.Sent + Count + array.lookup(Boxes, Owner),
        array.set(Domain, 0, !Counts),
        array.set(C, Credit - 1, !Counts)
    else
        true
    ).

    % Puts each waiting item back, counting it again.
:- pred rerun(array(int)::in, array(int)::in, int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.

rerun(Old, Older, I, N, At, !Counts, !Backlog) :-
    ( if I < N then
        array.set(I, array.lookup(Old, I) + array.lookup(Older, I), !Backlog),
        array.lookup(!.Counts, At, K),
        array.set(At, K + 1, !Counts),
        rerun(Old, Older, I + 1, N, At, !Counts, !Backlog)
    else
        true
    ).
