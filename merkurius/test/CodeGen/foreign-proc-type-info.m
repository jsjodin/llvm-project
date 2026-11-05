% TypeInfo_for_T is the type_info that the runtime's walker reads, whichever way
% it was made: this foreign_proc gives each type_info to merk_write, which mmc
% does not have, so this test has no oracle of its own. Its output is what
% io.write writes, which test/CodeGen/io-write*.m compare with mmc's.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module foreign_proc_type_info.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- pragma foreign_decl("C", "extern void merk_write(MR_Word, MR_Word, MR_Word);").

% The order of the type_infos is the variables', not the arguments'.
:- pred both(B::in, A::in, io::di, io::uo) is det.
both(Y, X, !IO) :- both2(X, Y, !IO).

:- pred both2(A::in, B::in, io::di, io::uo) is det.
:- pragma foreign_proc("C", both2(X::in, Y::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure], "
    merk_write(TypeInfo_for_A, X, 2);
    merk_write(TypeInfo_for_B, Y, 2);
").

:- pred made(T::in, io::di, io::uo) is det.
made(X, !IO) :- both2([X], yes({X, X}), !IO).

main(!IO) :-
    % CHECK:1
    % CHECK-NEXT:"s"
    both2(1, "s", !IO),
    % CHECK-NEXT:1.5
    % CHECK-NEXT:'c'
    both('c', 1.5, !IO),
    % CHECK-NEXT:["x"]
    % CHECK-NEXT:yes({"x", "x"})
    made("x", !IO),
    % CHECK-NEXT:{{\[\[}}1u8]]
    % CHECK-NEXT:yes({[1u8], [1u8]})
    made([1u8], !IO).
