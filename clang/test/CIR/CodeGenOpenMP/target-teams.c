// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void use(int);

// The combined 'target teams' directive decomposes into a 'target' leaf and a
// 'teams' leaf and lowers to an omp.teams nested inside an omp.target, matching
// the equivalent nesting of the separate 'target' and 'teams' directives.
void target_teams(int x) {
  // CIR-HOST: cir.func{{.*}}@target_teams
  // CIR-HOST: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {name = "x"}
  // CIR-HOST: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.teams {
  // CIR-HOST: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST: cir.call @use(%[[LOAD]])
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // CIR-HOST: }

  // CIR-DEVICE: cir.func{{.*}}@target_teams
  // CIR-DEVICE: omp.target kernel_type(generic) {{.*}} {
  // CIR-DEVICE: omp.teams {
  // CIR-DEVICE: cir.call @use
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
#pragma omp target teams map(tofrom : x)
  {
    use(x);
  }
}
