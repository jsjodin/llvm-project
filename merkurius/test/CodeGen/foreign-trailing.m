% The trailing interface of foreign C code (reference manual, "Trailing"):
% its types are defined, so that an untrail function compiles, and its
% macros report trailing as not yet implemented where they are used. mmc's
% installed grade, hlc.gc, has no trailing, so the conformance suite cannot
% compare this.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s
% CHECK-DAG: error: not yet implemented: trailing (MR_trail_function)
% CHECK-DAG: error: not yet implemented: trailing (MR_trail_current_value)
% CHECK-DAG: error: not yet implemented: trailing (MR_current_choicepoint_id)
% CHECK-NOT: error: {{.*}}MR_untrail_reason
% CHECK-NOT: error: {{.*}}MR_ChoicepointId

:- module foreign_trailing.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pragma foreign_decl("C", "
static void untrail(void *value, MR_untrail_reason reason)
{
    if (reason == MR_undo || reason == MR_exception) {
        *(MR_Word *) value = 0;
    }
}
static MR_ChoicepointId last_choicepoint;
").

:- pred set(int::in, int::out) is det.
:- pragma foreign_proc("C", set(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"   static MR_Word cell = 0;
    if (MR_current_choicepoint_id() != last_choicepoint) {
        MR_trail_current_value(&cell);
    }
    MR_trail_function(untrail, &cell);
    cell = X;
    Y = cell;
").

main(!IO) :-
    set(4, Y),
    io.write_int(Y, !IO),
    io.nl(!IO).
