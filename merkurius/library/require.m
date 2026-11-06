% The standard library module require, with the declarations of the library
% reference manual (22.01). Each error throws software_error with mmc's
% message, which until exceptions can be caught ends the program as an
% uncaught exception does: "Software Error: pred p/1: Unexpected: what".

:- module require.
:- interface.

:- pred error(string::in) is erroneous.
:- func func_error(string) = _ is erroneous.
:- pred error(string::in, string::in) is erroneous.
:- func func_error(string, string) = _ is erroneous.
:- func sorry(string, string) = _ is erroneous.
:- pred sorry(string::in, string::in) is erroneous.
:- func sorry(string, string, string) = _ is erroneous.
:- pred sorry(string::in, string::in, string::in) is erroneous.
:- func unexpected(string, string) = _ is erroneous.
:- pred unexpected(string::in, string::in) is erroneous.
:- func unexpected(string, string, string) = _ is erroneous.
:- pred unexpected(string::in, string::in, string::in) is erroneous.
:- pred require((pred)::((pred) is semidet), string::in) is det.
:- pred expect((pred)::((pred) is semidet), string::in, string::in) is det.
:- pred expect((pred)::((pred) is semidet), string::in, string::in,
    string::in) is det.
:- pred expect_not((pred)::((pred) is semidet), string::in, string::in) is det.
:- pred expect_not((pred)::((pred) is semidet), string::in, string::in,
    string::in) is det.
:- pred report_lookup_error(string::in, K::in) is erroneous.
:- pred report_lookup_error(string::in, K::in, V::unused) is erroneous.

:- implementation.

% The message's parts, which it joins: without the string module, which is
% not compiled yet, the parts are joined in C.
:- pred software_error(string::in, string::in, string::in, string::in,
    string::in) is erroneous.

:- pragma foreign_decl("C", local, "
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void merk_uncaught_exception(const char *exception)
    __attribute__((noreturn));
void merk_not_implemented(const char *what) __attribute__((noreturn));
").

:- pragma foreign_proc("C",
    software_error(A::in, B::in, C::in, D::in, E::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    static const char prefix[] = \"Software Error: \";
    const char *parts[] = {A, B, C, D, E};
    size_t size = sizeof prefix;
    for (int i = 0; i < 5; ++i) {
        size += strlen(parts[i]);
    }
    char *message = malloc(size);
    strcpy(message, prefix);
    for (int i = 0; i < 5; ++i) {
        strcat(message, parts[i]);
    }
    merk_uncaught_exception(message);
").

error(Message) :-
    software_error("", "", "", "", Message).
func_error(Message) = _ :-
    error(Message).
error(Pred, Message) :-
    software_error(Pred, ": ", "", "", Message).
func_error(Pred, Message) = _ :-
    error(Pred, Message).

sorry(Module, What) = _ :-
    sorry(Module, What).
sorry(Module, What) :-
    software_error(Module, ": ", "", "Sorry, not implemented: ", What).
sorry(Module, Pred, What) = _ :-
    sorry(Module, Pred, What).
sorry(Module, Pred, What) :-
    software_error(Module, ": ", Pred, ": Sorry, not implemented: ", What).

unexpected(Module, What) = _ :-
    unexpected(Module, What).
unexpected(Module, What) :-
    software_error(Module, ": ", "", "Unexpected: ", What).
unexpected(Module, Pred, What) = _ :-
    unexpected(Module, Pred, What).
unexpected(Module, Pred, What) :-
    software_error(Module, ": ", Pred, ": Unexpected: ", What).

require(Goal, Message) :-
    ( if Goal then true else error(Message) ).
expect(Goal, Module, Message) :-
    ( if Goal then true else unexpected(Module, Message) ).
expect(Goal, Module, Pred, Message) :-
    ( if Goal then true else unexpected(Module, Pred, Message) ).
expect_not(Goal, Module, Message) :-
    ( if Goal then unexpected(Module, Message) else true ).
expect_not(Goal, Module, Pred, Message) :-
    ( if Goal then unexpected(Module, Pred, Message) else true ).

% Temporary (chunk 1 step 6, docs/INPROGRESS.md): mmc's message has the
% key's type and value, "Key Type: int" and "Key Value: 42" on lines of
% their own, which needs type names and a writer to the standard error;
% until then these say they are not implemented.
:- pragma foreign_proc("C",
    report_lookup_error(_Message::in, _Key::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_not_implemented(\"require.report_lookup_error/2\");
").
:- pragma foreign_proc("C",
    report_lookup_error(_Message::in, _Key::in, _Value::unused),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_not_implemented(\"require.report_lookup_error/3\");
").
