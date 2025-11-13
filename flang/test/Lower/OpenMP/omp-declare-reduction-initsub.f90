! This test checks lowering of OpenMP declare reduction Directive, with initialization
! via a subroutine. This functionality is currently not implemented.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine initme(x,n)
  integer x,n
  x=0
end subroutine initme

function func(x, n, init)
  integer func
  integer x(n)
  integer res
  interface
     subroutine initme(x,n)
       integer x,n
     end subroutine initme
  end interface
!CHECK: omp.declare_reduction @red_add : i32 init
!CHECK: fir.call @_QPinitme
!$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,omp_orig))
  res=init
!$omp simd reduction(red_add:res)
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func
