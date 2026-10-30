% A declaration-only stub of the standard library module require, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

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
