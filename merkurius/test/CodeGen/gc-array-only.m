% RUN: merkc -O0 %s -o %t0
% RUN: %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% A loop that allocates only through array operations still collects, so that
% the arrays it drops are freed.
:- module gc_array_only.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, int, list, string.

:- pragma foreign_decl("C", "#include <sys/resource.h>").

:- pred peak_megabytes(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", peak_megabytes(M::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    M = usage.ru_maxrss / 1024;
").

:- pragma no_inline(pred(drop/3)).
:- pred drop(list(int)::in, int::in, int::out) is det.
drop(Kept, N, Last) :-
    array.init(40000, Kept, A0),
    array.copy(A0, A1),
    array.resize(40001, [N], A1, A2),
    array.lookup(A2, 40000, L),
    Last = list.foldl(plus, L, 0).

:- pred loop(list(int)::in, int::in, int::in, int::out) is det.
loop(Kept, N, !Sum) :-
    ( if N = 0 then
        true
    else
        drop(Kept, N, Last),
        !:Sum = !.Sum + Last,
        loop(Kept, N - 1, !Sum)
    ).

main(!IO) :-
    Kept = 1 `..` 3,
    loop(Kept, 3000, 0, Sum),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    io.write(Kept, !IO),
    io.nl(!IO),
    peak_megabytes(M, !IO),
    % 3000 rounds of three arrays of 320 KB are 2.9 GB if nothing is freed.
    io.write_string(( if M < 500 then "bounded" else "grew" ), !IO),
    io.nl(!IO).

% OUTPUT:4501500
% OUTPUT-NEXT:{{\[}}1, 2, 3]
% OUTPUT-NEXT:bounded
