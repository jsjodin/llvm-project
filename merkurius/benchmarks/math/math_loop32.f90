! math_loop32.m's loops in Fortran, in real(4), with OpenMP offload
! (README.md), which Flang lowers through the math dialect and MathToROCDL.
module loops32
  implicit none
contains
  function run_sqrt(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = sqrt(x + c)
      end do
      total = total + x
    end do
  end function

  function run_exp(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = exp(-(x + c))
      end do
      total = total + x
    end do
  end function

  function run_ln(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = log(x + 2.0 + c)
      end do
      total = total + x
    end do
  end function

  function run_sin(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = sin(x + 1.0 + c)
      end do
      total = total + x
    end do
  end function

  function run_cos(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = cos(x + c)
      end do
      total = total + x
    end do
  end function

  function run_pow(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = (x + 1.0 + c) ** 0.75
      end do
      total = total + x
    end do
  end function

  function run_tanh(n, k) result(total)
    integer(8), intent(in) :: n, k
    real(4) :: total, c, x
    integer(8) :: i, j
    total = 0.0
    !$omp target teams distribute parallel do reduction(+:total) private(c, x, j)
    do i = 0, n - 1
      c = real(i, 4) / real(n, 4)
      x = c
      do j = 1, k
        x = tanh(x + 0.5 + c)
      end do
      total = total + x
    end do
  end function

  integer(8) function setting(name, fallback)
    character(*), intent(in) :: name
    integer(8), intent(in) :: fallback
    character(32) :: text
    integer :: length, status
    call get_environment_variable(name, text, length, status)
    if (status == 0 .and. length > 0) then
      read (text, *) setting
    else
      setting = fallback
    end if
  end function

  real(8) function now()
    integer(8) :: count, rate
    call system_clock(count, rate)
    now = dble(count) / dble(rate)
  end function
end module

program math_loop
  use loops32
  implicit none
  integer(8) :: n, k, repeat
  n = setting("MATH_N", 1048576_8)
  k = setting("MATH_K", 1000_8)
  repeat = setting("MATH_REPEAT", 5_8)
  call report("sqrt", real(run_sqrt(n, k), 8), best_sqrt())
  call report("exp", real(run_exp(n, k), 8), best_exp())
  call report("ln", real(run_ln(n, k), 8), best_ln())
  call report("sin", real(run_sin(n, k), 8), best_sin())
  call report("cos", real(run_cos(n, k), 8), best_cos())
  call report("pow", real(run_pow(n, k), 8), best_pow())
  call report("tanh", real(run_tanh(n, k), 8), best_tanh())
contains
  real(8) function best_sqrt()
    real(8) :: t0, sink
    integer(8) :: r
    best_sqrt = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_sqrt(n, k)
      best_sqrt = min(best_sqrt, now() - t0)
    end do
  end function

  real(8) function best_exp()
    real(8) :: t0, sink
    integer(8) :: r
    best_exp = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_exp(n, k)
      best_exp = min(best_exp, now() - t0)
    end do
  end function

  real(8) function best_ln()
    real(8) :: t0, sink
    integer(8) :: r
    best_ln = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_ln(n, k)
      best_ln = min(best_ln, now() - t0)
    end do
  end function

  real(8) function best_sin()
    real(8) :: t0, sink
    integer(8) :: r
    best_sin = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_sin(n, k)
      best_sin = min(best_sin, now() - t0)
    end do
  end function

  real(8) function best_cos()
    real(8) :: t0, sink
    integer(8) :: r
    best_cos = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_cos(n, k)
      best_cos = min(best_cos, now() - t0)
    end do
  end function

  real(8) function best_pow()
    real(8) :: t0, sink
    integer(8) :: r
    best_pow = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_pow(n, k)
      best_pow = min(best_pow, now() - t0)
    end do
  end function

  real(8) function best_tanh()
    real(8) :: t0, sink
    integer(8) :: r
    best_tanh = 1.0d30
    do r = 1, repeat
      t0 = now()
      sink = run_tanh(n, k)
      best_tanh = min(best_tanh, now() - t0)
    end do
  end function

  subroutine report(name, first, best)
    character(*), intent(in) :: name
    real(8), intent(in) :: first, best
    write (*, '(a6, 1x, f10.6, 1x, es25.17)') name, best, first
  end subroutine
end program
