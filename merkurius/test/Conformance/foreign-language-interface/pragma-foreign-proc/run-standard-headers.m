% MANUAL: pragma foreign_proc
% C code that uses the standard headers without including them, as mmc's
% foreign code may: stdio, stdlib, string, ctype, assert, limits, errno,
% time, stdarg and stddef.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:from C
% OUTPUT-NEXT:37
% OUTPUT-NOT:{{.}}

:- module run_standard_headers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pragma foreign_code("C", "
static int sum(int n, ...)
{
    va_list args;
    int total = 0;
    va_start(args, n);
    while (n-- > 0) {
        total += va_arg(args, int);
    }
    va_end(args);
    return total;
}
").

:- pred check(int::out) is det.
:- pragma foreign_proc("C", check(X::out),
    [will_not_call_mercury, promise_pure], "
    char *copy = malloc(8);
    size_t length;
    assert(copy != NULL);
    strcpy(copy, \"abc\");
    length = strlen(copy);
    free(copy);
    errno = 0;
    X = (MR_Integer) length + (isdigit('7') != 0) + (INT_MAX > 0)
        + (time(NULL) > 0) + sum(2, 10, 20) + (errno == 0)
        + (MR_Integer) offsetof(struct { char a; int b; }, a);
    printf(\"from C\\n\");
    fflush(stdout);
").

main(!IO) :-
    check(X),
    io.write_int(X, !IO),
    io.nl(!IO).
