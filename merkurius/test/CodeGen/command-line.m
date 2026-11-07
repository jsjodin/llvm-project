% io.command_line_arguments gives the words after the program's name, as
% the shell passed them, and io.progname the name the program was run by,
% the command line's first word. The runtime's main keeps argv, whose
% strings live as long as the program, so they are not copied. The output
% is mmc's for the same program and command lines.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=NONE --match-full-lines --strict-whitespace
% RUN: %t0 a "b c" "" 'd"e' | FileCheck %s --check-prefix=SOME --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=NONE --match-full-lines --strict-whitespace
% RUN: %t2 a "b c" "" 'd"e' | FileCheck %s --check-prefix=SOME --match-full-lines --strict-whitespace
%
% NONE:[]
% NONE-NEXT:0
% NONE-NEXT:{{.*}}command-line.m.tmp{{[02]}}
% SOME:["a", "b c", "", "d\"e"]
% SOME-NEXT:4
% SOME-NEXT:{{.*}}command-line.m.tmp{{[02]}}

:- module command_line.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module list.

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    io.write(Arguments, !IO),
    io.nl(!IO),
    io.write_int(list.length(Arguments), !IO),
    io.nl(!IO),
    io.progname("none", Name, !IO),
    io.write_string(Name, !IO),
    io.nl(!IO).
