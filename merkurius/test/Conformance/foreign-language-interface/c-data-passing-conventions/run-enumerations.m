% MANUAL: C data passing conventions
% bool is MR_Bool, with MR_YES and MR_NO, comparison_result is
% MR_Comparison_Result, with MR_COMPARE_LESS, MR_COMPARE_EQUAL and
% MR_COMPARE_GREATER, and another enumeration is passed as an MR_Word.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{yes, no, (<), (=), (>)}
% OUTPUT-NEXT:is_yes
% OUTPUT-NEXT:[green, blue, red]
% OUTPUT-NOT:{{.}}

:- module run_enumerations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, list, string.
:- type colour ---> red ; green ; blue.
:- pred names(bool::out, bool::out, comparison_result::out,
    comparison_result::out, comparison_result::out) is det.
:- pragma foreign_proc("C", names(Y::out, N::out, L::out, E::out, G::out),
    [promise_pure, will_not_call_mercury],
"
    MR_Bool yes = MR_YES; MR_Comparison_Result less = MR_COMPARE_LESS;
    Y = yes; N = MR_NO; L = less; E = MR_COMPARE_EQUAL; G = MR_COMPARE_GREATER;
").
:- pred is_yes(bool::in) is semidet.
:- pragma foreign_proc("C", is_yes(B::in),
    [promise_pure, will_not_call_mercury], "SUCCESS_INDICATOR = (B == MR_YES);").
:- func next(colour) = colour.
:- pragma foreign_proc("C", next(C::in) = (D::out),
    [promise_pure, will_not_call_mercury], "D = (C + 1) % 3;").

main(!IO) :-
    names(Y, N, L, E, G), io.write({Y, N, L, E, G}, !IO), io.nl(!IO),
    ( if is_yes(yes), not is_yes(no) then io.write_string("is_yes\n", !IO)
    else io.write_string("not is_yes\n", !IO) ),
    io.write([next(red), next(green), next(blue)], !IO), io.nl(!IO).
