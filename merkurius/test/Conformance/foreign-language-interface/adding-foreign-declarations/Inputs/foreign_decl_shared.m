% A module with a foreign_decl for C that is shared with the modules that
% import it, and a local one that is not.
:- module foreign_decl_shared.
:- interface.
:- func twice(int) = int.
:- implementation.
:- pragma foreign_decl("C", "#define SHARED_ONE 4").
:- pragma foreign_decl("C", "#define SHARED_TWO 5").
:- pragma foreign_decl("C", local, "#define LOCAL_ONE 1").
:- pragma foreign_proc("C", twice(X::in) = (Y::out),
    [promise_pure, will_not_call_mercury], "Y = 2 * X * LOCAL_ONE;").
