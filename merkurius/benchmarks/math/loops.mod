!mod$ v1 sum:e627878474678512
module loops
contains
function run_sqrt(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_exp(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_ln(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_sin(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_cos(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_pow(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function run_tanh(n,k) result(total)
integer(8),intent(in)::n
integer(8),intent(in)::k
real(8)::total
end
function setting(name,fallback)
character(*,1),intent(in)::name
integer(8),intent(in)::fallback
integer(8)::setting
end
function now()
real(8)::now
end
end
