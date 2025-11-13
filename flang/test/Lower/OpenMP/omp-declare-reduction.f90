! This test checks lowering of OpenMP declare reduction Directive.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine declare_red()
  integer :: my_var
  !CHECK: omp.declare_reduction @my_red : i32 init
  !$omp declare reduction (my_red : integer : omp_out = omp_in) initializer (omp_priv = 0)
  my_var = 0
end subroutine declare_red
