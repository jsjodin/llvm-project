% RUN: merkc -O0 %s -o %t0
% RUN: %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=4001 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% An array with a block of its own stays where it is across collections, its
% elements are traced where they are, and it is freed when it is unreached.
:- module gc_large.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, bool, int, list, string.

:- pred address(array(T)::array_ui, int::out) is det.
:- pragma foreign_proc("C",
    address(Array::array_ui, Address::out),
    [will_not_call_mercury, promise_pure],
"
    Address = (MR_Integer) Array;
").

:- pragma no_inline(pred(churn/3)).
:- pred churn(int::in, int::in, int::out) is det.
churn(N, !Sum) :-
    ( if N = 0 then
        true
    else
        !:Sum = !.Sum + list.length(1 `..` 2000),
        churn(N - 1, !Sum)
    ).

:- pred fill_lists(int::in, array(list(int))::array_di,
    array(list(int))::array_uo) is det.
fill_lists(I, !Array) :-
    ( if I < array.size(!.Array) then
        array.set(I, [I, I + 1], !Array),
        fill_lists(I + 1, !Array)
    else
        true
    ).

:- pred drop_big(int::in, int::out) is det.
:- pragma no_inline(pred(drop_big/2)).
drop_big(N, Last) :-
    array.init(40000, N, Array),
    Last = array.lookup(Array, 39999).

:- pred drop_many(int::in, int::in, int::out) is det.
drop_many(N, !Sum) :-
    ( if N = 0 then
        true
    else
        drop_big(N, Last),
        !:Sum = !.Sum + Last,
        drop_many(N - 1, !Sum)
    ).

main(!IO) :-
    array.init(30000, [], Array0),
    fill_lists(0, Array0, Array),
    address(Array, Before),
    churn(3000, 0, Churned),
    address(Array, After),
    io.write_int(Churned, !IO),
    io.nl(!IO),
    io.write_string(( if Before = After then "same address" else "moved" ), !IO),
    io.nl(!IO),
    array.lookup(Array, 29999, Far),
    io.write(Far, !IO),
    io.nl(!IO),
    drop_many(2000, 0, Sum),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    array.lookup(Array, 7, Near),
    io.write(Near, !IO),
    io.nl(!IO).

% OUTPUT:6000000
% OUTPUT-NEXT:same address
% OUTPUT-NEXT:[29999, 30000]
% OUTPUT-NEXT:2001000
% OUTPUT-NEXT:[7, 8]
