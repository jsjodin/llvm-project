% A declaration-only stub of the standard library module exception, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module exception.
:- interface.
:- import_module io.
:- import_module list.
:- import_module maybe.
:- import_module univ.

:- type software_error
    --->    software_error(string).
:- type domain_error
    --->    domain_error(string).
:- func throw(T) = _ is erroneous.
:- pred throw(T::in) is erroneous.
:- pred rethrow(exception_result(T)).
:- mode rethrow(in(bound(exception(ground)))) is erroneous.
:- func rethrow(exception_result(T)) = _.
:- mode rethrow(in(bound(exception(ground)))) = out is erroneous.
:- type exception_result(T)
    --->    succeeded(T)
    ;       failed
    ;       exception(univ).
:- inst cannot_fail for exception_result/1
    --->    succeeded(ground)
    ;       exception(ground).
:- pred try(pred(T),                exception_result(T)).
:- mode try(pred(out) is det,       out(cannot_fail)) is cc_multi.
:- mode try(pred(out) is semidet,   out)              is cc_multi.
:- mode try(pred(out) is cc_multi,  out(cannot_fail)) is cc_multi.
:- mode try(pred(out) is cc_nondet, out)              is cc_multi.
:- pred try_io(pred(T, io, io), exception_result(T), io, io).
:- mode try_io(pred(out, di, uo) is det,
    out(cannot_fail), di, uo) is cc_multi.
:- mode try_io(pred(out, di, uo) is cc_multi,
    out(cannot_fail), di, uo) is cc_multi.
:- pred try_all(pred(T), maybe(univ), list(T)).
:- mode try_all(pred(out) is det,     out, out(nil_or_singleton_list))
    is cc_multi.
:- mode try_all(pred(out) is semidet, out, out(nil_or_singleton_list))
    is cc_multi.
:- mode try_all(pred(out) is multi,   out, out) is cc_multi.
:- mode try_all(pred(out) is nondet,  out, out) is cc_multi.
:- inst [] for list/1
    --->    [].
:- inst nil_or_singleton_list for list/1
    --->    []
    ;       [ground].
:- pred finally(pred(T, io, io), T, pred(io.res, io, io), io.res, io, io).
:- mode finally(pred(out, di, uo) is det, out,
    pred(out, di, uo) is det, out, di, uo) is det.
:- mode finally(pred(out, di, uo) is cc_multi, out,
    pred(out, di, uo) is cc_multi, out, di, uo) is cc_multi.
:- type near_stack_limits
    --->    near_stack_limits.
:- impure pred throw_if_near_stack_limits is det.
