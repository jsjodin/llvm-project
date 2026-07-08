// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// A standalone `teams` directive lowers to an omp.teams op whose region holds
// the associated structured block.
void host_teams(int x) {
  // CHECK: cir.func{{.*}}@host_teams
  // CHECK: omp.teams {
  // CHECK: cir.call @use
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp teams
  {
    use(x);
  }
}
