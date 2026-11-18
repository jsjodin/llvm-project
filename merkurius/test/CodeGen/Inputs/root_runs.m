% For root-field-types.m: a cell whose fields are arrays of flat records,
% exported concretely.

:- module root_runs.
:- interface.
:- import_module array, list.

:- type sample
    --->    sample(float, float).

:- type run
    --->    run(
                r_before    :: array(sample),
                r_after     :: array(sample),
                r_steps     :: list(int)
            ).

:- func make_run(int) = run.

:- implementation.
:- import_module int.

make_run(N) = run(array.init(N, sample(1.0, 2.0)),
    array.init(N, sample(3.0, 4.0)), 1 .. N).
