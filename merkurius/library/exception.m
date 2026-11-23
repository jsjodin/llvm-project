% The standard library module exception. The declarations are those of the
% library reference manual (22.01), less those that need modules not stubbed
% yet; see README.md. A try calls one of the runtime's entries
% (runtime/exception.cpp) as a closure, which keeps a handler in its frame and
% calls the goal; a throw jumps to the innermost handler (docs/INPROGRESS.md,
% Exceptions). throw_if_near_stack_limits is not written yet.

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

:- implementation.

:- import_module solutions.

throw(Exception) = _ :-
    throw(Exception).

throw(Exception) :-
    throw_univ(univ(Exception), Exception).

rethrow(exception(Univ)) :-
    throw_univ(Univ, univ_value(Univ)).

rethrow(Result) = _ :-
    rethrow(Result).

    % Jumps to the innermost try with the univ. The value and its type are
    % for the report if nothing catches it.
:- pred throw_univ(univ::in, T::in) is erroneous.
:- pragma foreign_proc("C",
    throw_univ(Univ::in, Value::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    merk_throw(Univ, TypeInfo_for_T, Value);
").

% Declared cc_multi, as which exception comes out may depend on the order of
% evaluation; this implementation is det.
:- pragma no_determinism_warning((try)/2).
:- pragma no_determinism_warning(try_io/4).
:- pragma promise_equivalent_clauses((try)/2).

try(Goal::(pred(out) is det), Result::out(cannot_fail)) :-
    try_det_entry(Entry),
    Entry(Goal, Value, Exception),
    det_result(Value, Exception, Result).
try(Goal::(pred(out) is semidet), Result::out) :-
    try_semidet_entry(Entry),
    Entry(Goal, Kind, Value, Exception),
    semidet_result(Kind, Value, Exception, Result).
try(Goal::(pred(out) is cc_multi), Result::out(cannot_fail)) :-
    try_cc_multi_entry(Entry),
    Entry(Goal, Value, Exception),
    det_result(Value, Exception, Result).
try(Goal::(pred(out) is cc_nondet), Result::out) :-
    try_cc_nondet_entry(Entry),
    Entry(Goal, Kind, Value, Exception),
    semidet_result(Kind, Value, Exception, Result).

:- pragma promise_equivalent_clauses(try_io/4).

try_io(Goal::(pred(out, di, uo) is det), Result::out(cannot_fail),
        IO0::di, IO::uo) :-
    try_io_entry(Entry),
    Entry(Goal, Value, Exception, IO0, IO),
    det_result(Value, Exception, Result).
try_io(Goal::(pred(out, di, uo) is cc_multi), Result::out(cannot_fail),
        IO0::di, IO::uo) :-
    try_cc_multi_io_entry(Entry),
    Entry(Goal, Value, Exception, IO0, IO),
    det_result(Value, Exception, Result).

% Declared cc_multi, as which exception comes out, and which solutions
% before it, may depend on the order of evaluation.
:- pragma no_determinism_warning(try_all/3).
:- pragma promise_equivalent_clauses(try_all/3).

try_all(Goal::(pred(out) is det), MaybeException::out,
        Solutions::out(nil_or_singleton_list)) :-
    try(Goal, Result),
    (
        Result = succeeded(Solution),
        MaybeException = no,
        Solutions = [Solution]
    ;
        Result = exception(Exception),
        MaybeException = yes(Exception),
        Solutions = []
    ).
try_all(Goal::(pred(out) is semidet), MaybeException::out,
        Solutions::out(nil_or_singleton_list)) :-
    try(Goal, Result),
    (
        Result = succeeded(Solution),
        MaybeException = no,
        Solutions = [Solution]
    ;
        Result = failed,
        MaybeException = no,
        Solutions = []
    ;
        Result = exception(Exception),
        MaybeException = yes(Exception),
        Solutions = []
    ).
try_all(Goal::(pred(out) is multi), MaybeException::out, Solutions::out) :-
    try_multi_entry(Entry),
    collect_all(
        ( pred(Solution::out) is nondet :-
            Entry(Goal, Kind, Value, Exception),
            all_solution(Kind, Value, Exception, Solution)
        ), MaybeException, Solutions).
try_all(Goal::(pred(out) is nondet), MaybeException::out, Solutions::out) :-
    try_nondet_entry(Entry),
    collect_all(
        ( pred(Solution::out) is nondet :-
            Entry(Goal, Kind, Value, Exception),
            all_solution(Kind, Value, Exception, Solution)
        ), MaybeException, Solutions).

    % A solution of a try_all's goal, or its exception, which comes last.
:- type all_solution(T)
    --->    all_succeeded(T)
    ;       all_threw(univ).

    % Kind is 0 for a solution, 1 for the exception.
:- pred all_solution(int::in, T::in, univ::in, all_solution(T)::out) is det.

all_solution(Kind, Value, Exception, Solution) :-
    ( if Kind = 0 then
        Solution = all_succeeded(Value)
    else
        Solution = all_threw(Exception)
    ).

    % The solutions in the order they came, and the exception if one ended
    % them. The step that adds a solution must not throw, as the runtime's
    % entry would catch it as the goal's.
:- pred collect_all(pred(all_solution(T))::in(pred(out) is nondet),
    maybe(univ)::out, list(T)::out) is cc_multi.

collect_all(Generator, MaybeException, Solutions) :-
    unsorted_aggregate2(Generator, add_solution, no, MaybeException,
        [], Reversed),
    list.reverse(Reversed, Solutions).

:- pred add_solution(all_solution(T)::in, maybe(univ)::in, maybe(univ)::out,
    list(T)::in, list(T)::out) is det.

add_solution(all_succeeded(Solution), !MaybeException, !Reversed) :-
    !:Reversed = [Solution | !.Reversed].
add_solution(all_threw(Exception), _, yes(Exception), !Reversed).

% The cleanup runs whether the goal succeeded or threw; then the goal's
% exception is thrown again. An exception of the cleanup's own goes on in
% its place.
:- pragma promise_equivalent_clauses(finally/6).

finally(Goal::(pred(out, di, uo) is det), Value::out,
        Cleanup::(pred(out, di, uo) is det), CleanupResult::out,
        !.IO::di, !:IO::uo) :-
    promise_equivalent_solutions [Result, !:IO] (
        try_io(Goal, Result, !IO)
    ),
    Cleanup(CleanupResult, !IO),
    (
        Result = succeeded(Value)
    ;
        Result = exception(_),
        rethrow(Result)
    ).
finally(Goal::(pred(out, di, uo) is cc_multi), Value::out,
        Cleanup::(pred(out, di, uo) is cc_multi), CleanupResult::out,
        !.IO::di, !:IO::uo) :-
    try_io(Goal, Result, !IO),
    Cleanup(CleanupResult, !IO),
    (
        Result = succeeded(Value)
    ;
        Result = exception(_),
        rethrow(Result)
    ).

    % An entry gives the goal's value, which is 0 if it threw, and the
    % exception, which is 0 if it did not, a word that no univ is. Neither is
    % kept past a test of the exception, before anything is allocated.
:- pred det_result(T::in, univ::in, exception_result(T)::out(cannot_fail))
    is det.

det_result(Value, Exception, Result) :-
    ( if no_exception(Exception) then
        Result = succeeded(Value)
    else
        Result = exception(Exception)
    ).

    % Kind is 0 if the goal succeeded, 1 if it failed, 2 if it threw.
:- pred semidet_result(int::in, T::in, univ::in, exception_result(T)::out)
    is det.

semidet_result(Kind, Value, Exception, Result) :-
    ( if Kind = 0 then
        Result = succeeded(Value)
    else if Kind = 1 then
        Result = failed
    else
        Result = exception(Exception)
    ).

:- pred no_exception(univ::in) is semidet.
:- pragma foreign_proc("C",
    no_exception(Exception::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = Exception == 0;
").

    % The runtime's entries, as closures.
:- pred try_det_entry(pred(pred(T), T, univ)::
    out(pred(in(pred(out) is det), out, out) is det)) is det.
:- pragma foreign_proc("C",
    try_det_entry(Entry::out(pred(in(pred(out) is det), out, out) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_det_closure;
").

:- pred try_cc_multi_entry(pred(pred(T), T, univ)::
    out(pred(in(pred(out) is cc_multi), out, out) is det)) is det.
:- pragma foreign_proc("C",
    try_cc_multi_entry(
        Entry::out(pred(in(pred(out) is cc_multi), out, out) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_det_closure;
").

:- pred try_semidet_entry(pred(pred(T), int, T, univ)::
    out(pred(in(pred(out) is semidet), out, out, out) is det)) is det.
:- pragma foreign_proc("C",
    try_semidet_entry(
        Entry::out(pred(in(pred(out) is semidet), out, out, out) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_semidet_closure;
").

:- pred try_cc_nondet_entry(pred(pred(T), int, T, univ)::
    out(pred(in(pred(out) is cc_nondet), out, out, out) is det)) is det.
:- pragma foreign_proc("C",
    try_cc_nondet_entry(
        Entry::out(pred(in(pred(out) is cc_nondet), out, out, out) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_semidet_closure;
").

:- pred try_io_entry(pred(pred(T, io, io), T, univ, io, io)::
    out(pred(in(pred(out, di, uo) is det), out, out, di, uo) is det)) is det.
:- pragma foreign_proc("C",
    try_io_entry(Entry::out(pred(in(pred(out, di, uo) is det), out, out,
        di, uo) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_io_closure;
").

:- pred try_cc_multi_io_entry(pred(pred(T, io, io), T, univ, io, io)::
    out(pred(in(pred(out, di, uo) is cc_multi), out, out, di, uo) is det))
    is det.
:- pragma foreign_proc("C",
    try_cc_multi_io_entry(Entry::out(pred(in(pred(out, di, uo) is cc_multi),
        out, out, di, uo) is det)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_io_closure;
").

:- pred try_multi_entry(pred(pred(T), int, T, univ)::
    out(pred(in(pred(out) is multi), out, out, out) is nondet)) is det.
:- pragma foreign_proc("C",
    try_multi_entry(
        Entry::out(pred(in(pred(out) is multi), out, out, out) is nondet)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_nondet_closure;
").

:- pred try_nondet_entry(pred(pred(T), int, T, univ)::
    out(pred(in(pred(out) is nondet), out, out, out) is nondet)) is det.
:- pragma foreign_proc("C",
    try_nondet_entry(
        Entry::out(pred(in(pred(out) is nondet), out, out, out) is nondet)),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    (void)TypeInfo_for_T;
    Entry = (MR_Word) merk_try_nondet_closure;
").

:- pragma foreign_code("C", "
/* The descriptors of the library's own errors, for the runtime's
   merk_library_error, which throws them as values. index_out_of_bounds's
   is array's, which naming it links. */
extern const char merk_software_error_type[]
    __asm__(\"merk.type_ctor.exception.software_error/0\");
extern const char merk_domain_error_type[]
    __asm__(\"merk.type_ctor.exception.domain_error/0\");
extern const char merk_index_out_of_bounds_type[]
    __asm__(\"merk.type_ctor.array.index_out_of_bounds/0\");
const MR_Word merk_library_error_types[3] = {
    (MR_Word) merk_software_error_type,
    (MR_Word) merk_domain_error_type,
    (MR_Word) merk_index_out_of_bounds_type,
};
").

:- pragma foreign_decl("C", local, "
void merk_throw(MR_Word univ, MR_Word type_info, MR_Word value)
    __attribute__((noreturn));
extern const MR_Word merk_try_det_closure[];
extern const MR_Word merk_try_semidet_closure[];
extern const MR_Word merk_try_io_closure[];
extern const MR_Word merk_try_nondet_closure[];
").
