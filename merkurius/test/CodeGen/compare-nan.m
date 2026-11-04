% A NaN in the standard ordering. mmc compares a float it knows is one without
% an exception, so a NaN is equal to everything; one that a type variable stands
% for, through a type_info or as an argument declared with a type variable, is
% an error (design document, "mmc and the manual"). KNOWN and the cases of NAN are
% mmc's for the same program, but for mmc's message and "notag", where mmc says
% (=), as for a float it knows, and the manual says to throw.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=KNOWN --match-full-lines
% RUN: not env CASE=t %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=list %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=tuple %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=array %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=instance %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=op %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: not env CASE=notag %t 2>&1 | FileCheck %s --check-prefix=NAN
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=KNOWN --match-full-lines
% RUN: not env CASE=instance %t2 2>&1 | FileCheck %s --check-prefix=NAN

% NAN: Uncaught Mercury exception:
% NAN-NEXT: incomparable floats in compare/3

:- module compare_nan.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, list.

:- type m ---> m2(float).
:- type mm ---> mm2(float, int).
:- type g(T) ---> g1 ; g2(T).
:- type pp(T) ---> pp(T).

:- pred p(comparison_result::in, io::di, io::uo) is det.
p((=), !IO) :- io.write_string("=\n", !IO).
p((<), !IO) :- io.write_string("<\n", !IO).
p((>), !IO) :- io.write_string(">\n", !IO).

:- pred c(T::in, T::in, io::di, io::uo) is det.
c(X, Y, !IO) :- compare(R, X, Y), p(R, !IO).
:- pragma no_inline(c/4).

:- pred lt(T::in, T::in) is semidet.
lt(X, Y) :- X @< Y.
:- pragma no_inline(lt/2).

:- func nan(float) = float.
:- pragma no_inline(nan/1).
nan(Z) = unchecked_quotient(Z, Z).

:- pragma foreign_decl("C", "#include <stdlib.h>\n#include <string.h>").
:- pred is_case(string::in) is semidet.
:- pragma foreign_proc("C", is_case(Name::in),
    [will_not_call_mercury, promise_pure],
    "const char *value = getenv(\"CASE\");
     SUCCESS_INDICATOR = value != NULL && strcmp(value, Name) == 0;").

main(!IO) :-
    N = nan(0.0),
    N2 = nan(0.0),
    ( if is_case("t") then
        c(N, N2, !IO)
    else if is_case("list") then
        c([N], [N2], !IO)
    else if is_case("tuple") then
        compare(R, {N, 1}, {N2, 1}),
        p(R, !IO)
    else if is_case("array") then
        array.init(1, N, A1),
        array.init(1, N2, A2),
        compare(R, A1, A2),
        p(R, !IO)
    else if is_case("instance") then
        compare(R, g2(N), g2(N2)),
        p(R, !IO)
    else if is_case("op") then
        ( if lt(N, N2) then p((<), !IO) else p((=), !IO) )
    else if is_case("notag") then
        compare(R, pp(N), pp(N2)),
        p(R, !IO)
    else
        % KNOWN:=
        compare(R1, N, 1.0), p(R1, !IO),
        % KNOWN-NEXT:=
        compare(R2, 1.0, N), p(R2, !IO),
        % KNOWN-NEXT:=
        compare(R3, N, N2), p(R3, !IO),
        % KNOWN-NEXT:=
        c(m2(N), m2(N2), !IO),
        % KNOWN-NEXT:=
        c(mm2(N, 1), mm2(N2, 1), !IO),
        % KNOWN-NEXT:<
        c(mm2(N, 1), mm2(N2, 2), !IO),
        % KNOWN-NEXT:no
        ( if N @< 1.0 then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) )
    ).
