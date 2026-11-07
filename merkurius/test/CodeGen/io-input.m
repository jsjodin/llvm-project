% io.read_line_as_string, io.flush_output and io.set_exit_status, against
% mmc's output. Inputs/prompt.py reads the prompt, which the program must
% flush before it reads, and then gives it its input. A line keeps its
% newline, and a carriage return, and its bytes when they are not UTF-8,
% which io.write shows as U+FFFD; a NUL is an error that drops the line up
% to it; the last line may lack a newline. The exit status is the number of
% reads before the end.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %python %S/Inputs/prompt.py %t0 'one\ntwo\r\n\xc3\x28 \xc3\xa9\nx\x00y\nab\x00\x00cd\nlast' | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %python %S/Inputs/prompt.py %t0 '' | FileCheck %s --check-prefix=EMPTY --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %python %S/Inputs/prompt.py %t2 'one\ntwo\r\n\xc3\x28 \xc3\xa9\nx\x00y\nab\x00\x00cd\nlast' | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:before the input: prompt
% OUTPUT-NEXT:"one\n" 4
% OUTPUT-NEXT:"two\r\n" 5
% OUTPUT-NEXT:"�( é\n" 6
% OUTPUT-NEXT:error: null character in input
% OUTPUT-NEXT:"y\n" 2
% OUTPUT-NEXT:error: null character in input
% OUTPUT-NEXT:error: null character in input
% OUTPUT-NEXT:"cd\n" 3
% OUTPUT-NEXT:"last" 4
% OUTPUT-NEXT:eof
% OUTPUT-NEXT:lines: 9
% OUTPUT-NEXT:exit status 9
% EMPTY:before the input: prompt
% EMPTY-NEXT:eof
% EMPTY-NEXT:lines: 0
% EMPTY-NEXT:exit status 0

:- module io_input.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, string.

main(!IO) :-
    io.write_string("prompt\n", !IO),
    io.flush_output(!IO),
    lines(0, Count, !IO),
    io.write_string("lines: " ++ int_to_string(Count) ++ "\n", !IO),
    io.set_exit_status(Count, !IO).

:- pred lines(int::in, int::out, io::di, io::uo) is det.
lines(N, Count, !IO) :-
    io.read_line_as_string(Result, !IO),
    (
        Result = ok(Line),
        io.write(Line, !IO),
        io.write_string(" " ++ int_to_string(string.length(Line)) ++ "\n",
            !IO),
        lines(N + 1, Count, !IO)
    ;
        Result = eof,
        io.write_string("eof\n", !IO),
        Count = N
    ;
        Result = error(Error),
        io.write_string("error: " ++ io.error_message(Error) ++ "\n", !IO),
        lines(N + 1, Count, !IO)
    ).
