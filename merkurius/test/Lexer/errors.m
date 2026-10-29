% Lexical errors are reported with their location, and the tokens still
% cover the whole file.
%
% RUN: not merkc -fc1 -fdebug-unparse-tokens %s 2>%t.err | cmp - %s
% RUN: FileCheck %s < %t.err

p = "\e".
% CHECK: errors.m:[[@LINE-1]]:6: error: '\e' is not an escape in Mercury 22.01
% CHECK-NEXT: p = "\e".
% CHECK-NEXT: {{^}}       ^^{{$}}
q = 1.5_e10.
% CHECK: errors.m:[[@LINE-1]]:5: error: an underscore before an exponent is not allowed in Mercury 22.01
r = 0b12.
% CHECK: errors.m:[[@LINE-1]]:8: error: invalid digit in a literal with prefix '0b'
% CHECK: errors.m:[[@LINE+1]]:5: error: unterminated string
s = "open
