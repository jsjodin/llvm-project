// RUN: %clang_cc1 -fopenmp -fclangir %s -verify -emit-cir -o -

void do_things() {
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPCriticalDirective}}
#pragma omp critical
  {}

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPSingleDirective}}
#pragma omp single
  {}

  int i;
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP PARALLEL 'if' clause}}
#pragma omp parallel if(i)
  {}

  // A clause routed through construct decomposition but not yet emittable must
  // still be diagnosed by the leaf emitter's NYI handling.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP TARGET 'private' clause}}
#pragma omp target private(i)
  {}

  // The worksharing loop supports no clauses yet, so every `for`-leaf clause is
  // reported as not-yet-implemented, including on combined directives.
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP FOR 'schedule' clause}}
#pragma omp parallel for schedule(static)
  for (int j = 0; j < 10; j++) {
  }

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP FOR 'collapse' clause}}
#pragma omp parallel for collapse(1)
  for (int j = 0; j < 10; j++) {
  }
}
