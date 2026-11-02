% A stub of the standard library module io, for name resolution until the
% library is written, with its output predicates implemented. The
% declarations are those of the library reference manual (22.01), less
% those that need modules not stubbed yet; see README.md.

:- module io.
:- interface.
:- import_module char.
:- import_module list.
:- import_module string.
:- import_module univ.

:- type io.state.
:- type io == io.state.
:- type input_stream.
:- type output_stream.
:- type text_input_stream == input_stream.
:- type text_output_stream == output_stream.
:- type res
    --->    ok
    ;       error(io.error).
:- type res(T)
    --->    ok(T)
    ;       error(io.error).
:- type maybe_partial_res(T)
    --->    ok(T)
    ;       error(T, io.error).
:- inst maybe_partial_res(T) for maybe_partial_res/1
    --->    ok(T)
    ;       error(T, ground).
:- type result
    --->    ok
    ;       eof
    ;       error(io.error).
:- type result(T)
    --->    ok(T)
    ;       eof
    ;       error(io.error).
:- type read_result(T)
    --->    ok(T)
    ;       eof
    ;       error(string, int).
:- type io.error.
:- pred open_input(string::in, io.res(io.text_input_stream)::out,
    io::di, io::uo) is det.
:- pred open_output(string::in, io.res(io.text_output_stream)::out,
    io::di, io::uo) is det.
:- pred close_input(io.text_input_stream::in, io::di, io::uo) is det.
:- pred close_output(io.text_output_stream::in, io::di, io::uo) is det.
:- func stdin_stream = io.text_input_stream.
:- pred stdin_stream(io.text_input_stream::out, io::di, io::uo) is det.
:- func stdout_stream = io.text_output_stream.
:- pred stdout_stream(io.text_output_stream::out, io::di, io::uo) is det.
:- func stderr_stream = io.text_output_stream.
:- pred stderr_stream(io.text_output_stream::out, io::di, io::uo) is det.
:- pred input_stream(io.text_input_stream::out, io::di, io::uo) is det.
:- pred output_stream(io.text_output_stream::out, io::di, io::uo) is det.
:- pred read_char(io.result(char)::out, io::di, io::uo) is det.
:- pred read_char(io.text_input_stream::in, io.result(char)::out,
    io::di, io::uo) is det.
:- pred write_char(char::in, io::di, io::uo) is det.
:- pred write_char(io.text_output_stream::in, char::in, io::di, io::uo)
    is det.
:- pred write_int(int::in, io::di, io::uo) is det.
:- pred write_int(io.text_output_stream::in, int::in, io::di, io::uo) is det.
:- pred write_float(float::in, io::di, io::uo) is det.
:- pred write_float(io.text_output_stream::in, float::in, io::di, io::uo)
    is det.
:- pred write_string(string::in, io::di, io::uo) is det.
:- pred write_string(io.text_output_stream::in, string::in, io::di, io::uo)
    is det.
:- pred nl(io::di, io::uo) is det.
:- pred nl(io.text_output_stream::in, io::di, io::uo) is det.
:- pred read_line(io.result(list(char))::out, io::di, io::uo) is det.
:- pred read_line(io.text_input_stream::in, io.result(list(char))::out,
    io::di, io::uo) is det.
:- pred read_line_as_string(io.result(string)::out, io::di, io::uo) is det.
:- pred read_line_as_string(io.text_input_stream::in, io.result(string)::out,
    io::di, io::uo) is det.
:- pred read(io.read_result(T)::out, io::di, io::uo) is det.
:- pred read(io.text_input_stream::in, io.read_result(T)::out,
    io::di, io::uo) is det.
:- pred print(T::in, io::di, io::uo) is det.
:- pred print(io.text_output_stream::in, T::in, io::di, io::uo) is det.
:- pred print_line(T::in, io::di, io::uo) is det.
:- pred print_line(io.text_output_stream::in, T::in, io::di, io::uo) is det.
:- pred write(T::in, io::di, io::uo) is det.
:- pred write(io.text_output_stream::in, T::in, io::di, io::uo) is det.
:- pred write_line(T::in, io::di, io::uo) is det.
:- pred write_line(io.text_output_stream::in, T::in, io::di, io::uo) is det.
:- pred format(string::in, list(poly_type)::in, io::di, io::uo) is det.
:- pred format(io.text_output_stream::in, string::in, list(poly_type)::in,
    io::di, io::uo) is det.
:- pred write_strings(list(string)::in, io::di, io::uo) is det.
:- pred write_strings(io.text_output_stream::in, list(string)::in,
    io::di, io::uo) is det.
:- pred write_list(list(T), string, pred(T, io, io), io, io).
:- mode write_list(in, in, pred(in, di, uo) is det, di, uo) is det.
:- mode write_list(in, in, pred(in, di, uo) is cc_multi, di, uo)
    is cc_multi.
:- pred write_list(io.text_output_stream, list(T), string,
    pred(T, io, io), io, io).
:- mode write_list(in, in, in, pred(in, di, uo) is det, di, uo) is det.
:- mode write_list(in, in, in, pred(in, di, uo) is cc_multi, di, uo)
    is cc_multi.
:- pred flush_output(io::di, io::uo) is det.
:- pred flush_output(io.text_output_stream::in, io::di, io::uo) is det.
:- pred read_file_as_string(io.maybe_partial_res(string)::out,
    io::di, io::uo) is det.
:- pred read_file_as_string(io.text_input_stream::in,
    io.maybe_partial_res(string)::out, io::di, io::uo) is det.
:- pred progname(string::in, string::out, io::di, io::uo) is det.
:- pred command_line_arguments(list(string)::out, io::di, io::uo) is det.
:- pred set_exit_status(int::in, io::di, io::uo) is det.
:- func make_io_error(string) = io.error.
:- func error_message(io.error) = string.
:- pred error_message(io.error::in, string::out) is det.

:- implementation.

% The predicates implemented so far write to the standard output with C's
% stdio, which the runtime flushes when the program exits.

:- pragma foreign_decl("C", local, "
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
").

:- pragma foreign_proc("C",
    write_string(S::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    fputs(S, stdout);
").

:- pragma foreign_proc("C",
    nl(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    putc('\\n', stdout);
").

:- pragma foreign_proc("C",
    write_int(N::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    printf(\"%\" PRIdPTR, N);
").

% A character is written in UTF-8.
:- pragma foreign_proc("C",
    write_char(C::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    uint32_t c = (uint32_t) C;
    if (c < 0x80) {
        putc((int) c, stdout);
    } else if (c < 0x800) {
        putc(0xC0 | (c >> 6), stdout);
        putc(0x80 | (c & 0x3F), stdout);
    } else if (c < 0x10000) {
        putc(0xE0 | (c >> 12), stdout);
        putc(0x80 | ((c >> 6) & 0x3F), stdout);
        putc(0x80 | (c & 0x3F), stdout);
    } else {
        putc(0xF0 | (c >> 18), stdout);
        putc(0x80 | ((c >> 12) & 0x3F), stdout);
        putc(0x80 | ((c >> 6) & 0x3F), stdout);
        putc(0x80 | (c & 0x3F), stdout);
    }
").

% As mmc writes a float: with the fewest significant digits, from 15 to 17,
% that read back as the same float, and with .0 if that has no fraction or
% exponent.
:- pragma foreign_proc("C",
    write_float(F::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    char buffer[64];
    if (isnan(F)) {
        strcpy(buffer, \"nan\");
    } else if (isinf(F)) {
        strcpy(buffer, F < 0 ? \"-infinity\" : \"infinity\");
    } else {
        for (int precision = 15; precision <= 17; ++precision) {
            snprintf(buffer, sizeof buffer, \"%.*g\", precision, F);
            if (strtod(buffer, NULL) == F) {
                break;
            }
        }
        if (strpbrk(buffer, \".e\") == NULL) {
            strcat(buffer, \".0\");
        }
    }
    fputs(buffer, stdout);
").
