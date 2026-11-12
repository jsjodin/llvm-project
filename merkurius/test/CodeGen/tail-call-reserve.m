% A tailcc procedure whose tail call passes more stack arguments than it
% received reserves room for them below its return address before it pushes
% the frame pointer, so its return address is not next to the saved frame
% pointer. The collector reads where it is from the procedure's .eh_frame
% rule (runtime/frame-info.h); taking the word next to the frame pointer for
% it, it ended the walk there and missed the roots of the frames above, here
% the list solutions collects and narrow's caller's. At -O0 a statepoint's
% slot in such a frame hit an assertion in X86FrameLowering.
%
% RUN: merkc -fc1 -S %s -o - | FileCheck %s --check-prefix=ASM
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% ASM-LABEL: tail_call_reserve.narrow_f2_0:
% ASM: subq $32, %rsp
% ASM-NEXT: pushq %rbp
% ASM-LABEL: tail_call_reserve.spread_8_0:
% ASM: subq $16, %rsp
% ASM-NEXT: pushq %rbp
%
% CHECK:[0, 10, 20, 30, 4, 3, 2, 1, 0]
% CHECK-NEXT:{{\[\[}}12, 2, 2], [27, 20, 30, 4, 3, 2]]

:- module tail_call_reserve.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

% narrow tail-calls wide, which takes more stack arguments than narrow
% received, so narrow reserves room for them below its return address.
:- func wide(int, int, int, int, int, int, int, int, int, int, list(int))
    = list(int).
wide(A, B, C, D, E, F, G, H, I, J, L) =
    ( if A > 3 then
        L
    else
        [A + B + C + D + E + F + G + H + I + J | narrow(A + 1, L)]
    ).

:- func narrow(int, list(int)) = list(int).
narrow(A, L) = wide(A, A, A, A, A, A, A, A, A, A, [A | L]).

% Each clause of spread tail-calls its continuation with more stack
% arguments than spread received.
:- pred spread(int::in, int::out, int::out, int::out, int::out, int::out,
    int::out, list(int)::out) is multi.
spread(N, N, N + 1, N + 2, N + 3, N + 4, N + 5, narrow(N, [])).
spread(N, N, N, N, N, N, N, [N, N]).

:- func total(int, int, int, int, int, int, list(int)) = list(int).
total(A, B, C, D, E, F, L) = [A + B + C + D + E + F | L].

main(!IO) :-
    io.write_line(narrow(0, []), !IO),
    solutions(
        ( pred(L::out) is multi :-
            spread(2, A, B, C, D, E, F, L0),
            L = total(A, B, C, D, E, F, L0)
        ), Ls),
    io.write_line(Ls, !IO).
