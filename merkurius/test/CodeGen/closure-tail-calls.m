% A closure's call that is a procedure's last goal is a tail call, so each of
% these loops, ten million rounds through a closure, runs in constant stack
% at every optimization level (design document, Lowering, "Closures"). The
% typed procedure makes the first round; its closure calls the procedure's
% word copy, which gives every output as a word, as an entry and a lambda
% expression's procedure do, so each call returns what its caller returns.
% mmc's C grades overflow the stack on these; at ten thousand rounds the
% output is mmc's.
%
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module closure_tail_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list.

:- type parity ---> even ; odd.

% Each round goes through a closure, a lambda expression's or a named
% procedure's, whose call is the last goal.
:- pred icount(int::in, int::in, int::out) is det.
icount(N, A, R) :-
    ( if N = 0 then R = A
    else K = (pred(X::in, Y::out) is det :- icount(N - 1, X, Y)), K(A + 1, R) ).

:- pred fcount(int::in, float::in, float::out) is det.
fcount(N, A, R) :-
    ( if N = 0 then R = A
    else K = (pred(X::in, Y::out) is det :- fcount(N - 1, X, Y)), K(A + 0.5, R) ).

:- pred scount(int::in, string::in, string::out) is det.
scount(N, S, R) :-
    ( if N = 0 then R = S
    else K = (pred(X::in, Y::out) is det :- scount(N - 1, X, Y)), K(S, R) ).

:- pred lcount(int::in, list(int)::in, list(int)::out) is det.
lcount(N, L, R) :-
    ( if N = 0 then R = L
    else K = (pred(X::in, Y::out) is det :- lcount(N - 1, X, Y)), K(L, R) ).

% A closure of a named procedure with a curried argument.
:- pred ncount(int::in, float::in, float::out) is det.
ncount(N, A, R) :-
    ( if N = 0 then R = A else K = ncount(N - 1), K(A + 1.0, R) ).

:- pred loop(int::in, io::di, io::uo) is det.
loop(N, !IO) :-
    ( if N = 0 then io.write_string("done\n", !IO)
    else K = (pred(!.IO::di, !:IO::uo) is det :- loop(N - 1, !IO)), K(!IO) ).

:- pred down(int::in) is semidet.
down(N) :-
    ( if N = 0 then true
    else N > 0, K = (pred(M::in) is semidet :- down(M)), K(N - 1) ).

:- func parity(int) = parity.
parity(N) = parity_from(N, even).

:- func parity_from(int, parity) = parity.
parity_from(N, P0) = P :-
    ( if N = 0 then P = P0
    else F = (func(M) = parity_from(M, flip(P0))), P = F(N - 1) ).

:- func flip(parity) = parity.
flip(even) = odd.
flip(odd) = even.

% A function's result through a function closure's call.
:- func fsum(int, float) = float.
fsum(N, A) = ( if N = 0 then A else apply(fsum(N - 1), A + 2.0) ).

main(!IO) :-
    icount(10000000, 0, I), io.write_int(I, !IO), io.nl(!IO),
    fcount(10000000, 0.0, F), io.write_float(F, !IO), io.nl(!IO),
    scount(10000000, "s", S), io.write_string(S, !IO), io.nl(!IO),
    lcount(10000000, [1], L), io.write(L, !IO), io.nl(!IO),
    ncount(10000000, 0.0, NF), io.write_float(NF, !IO), io.nl(!IO),
    loop(10000000, !IO),
    ( if down(10000000) then io.write_string("down\n", !IO) else true ),
    io.write(parity(10000001), !IO), io.nl(!IO),
    io.write_float(fsum(10000000, 0.0), !IO), io.nl(!IO).

% OUTPUT:10000000
% OUTPUT-NEXT:5000000.0
% OUTPUT-NEXT:s
% OUTPUT-NEXT:[1]
% OUTPUT-NEXT:10000000.0
% OUTPUT-NEXT:done
% OUTPUT-NEXT:down
% OUTPUT-NEXT:odd
% OUTPUT-NEXT:20000000.0

% The lambda expression calls the word copy, which calls the closure, which
% calls the lambda expression: each last.
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.icount_3_0$lambda0"(i64 %0, i64 %1)
% LLVM: musttail call tailcc i64 @"closure_tail_calls.icount_3_0$words"(i64 %{{[0-9]+}}, i64 %1)
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.icount_3_0$words"(i64 %0, i64 %1)
% LLVM: musttail call tailcc i64 %{{[0-9]+}}(ptr %{{[0-9]+}}, i64 %{{[0-9]+}})
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.icount_3_0$lambda0$closure1"(ptr %0, i64 %1)
% LLVM: musttail call tailcc i64 @"closure_tail_calls.icount_3_0$lambda0"(i64 %{{[0-9]+}}, i64 %1)
% A closure of the word copy itself, with a curried argument.
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.ncount_3_0$words"(i64 %0, double %1)
% LLVM: musttail call tailcc i64 %{{[0-9]+}}(ptr %{{[0-9]+}}, i64 %{{[0-9]+}})
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.ncount_3_0$words$closure1"(ptr %0, i64 %1)
% LLVM: musttail call tailcc i64 @"closure_tail_calls.ncount_3_0$words"(i64 %{{[0-9]+}}, double %{{[0-9]+}})
% The io state is word 0 in each.
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.loop_3_0$words"(i64 %0)
% LLVM: musttail call tailcc i64 %{{[0-9]+}}(ptr %{{[0-9]+}}, i64 0)
% A semidet closure's success is its result, as a semidet procedure's is.
% LLVM-LABEL: define internal tailcc i1 @"closure_tail_calls.down_1_0$words"(i64 %0)
% LLVM: musttail call tailcc i1 %{{[0-9]+}}(ptr %{{[0-9]+}}, i64 %{{[0-9]+}})
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.parity_from_f2_0$lambda6"(i64 %0, i64 %1)
% LLVM: musttail call tailcc i64 @"closure_tail_calls.parity_from_f2_0$words"(i64 %1, i64 %{{[0-9]+}})
% LLVM-LABEL: define internal tailcc i64 @"closure_tail_calls.fsum_f2_0$words"(i64 %0, double %1)
% LLVM: musttail call tailcc i64 %{{[0-9]+}}(ptr %{{[0-9]+}}, i64 %{{[0-9]+}})
