% Lexical errors are reported with their location, and the tokens still
% cover the whole file.
%
% RUN: not merkc -fc1 -fdebug-unparse-tokens %s 2>%t.err | cmp - %s
% RUN: FileCheck %s < %t.err
% The reader's message at the end of the file, after the unterminated
% string, is located after its last character.
% RUN: not merkc -fc1 -fdebug-dump-terms %s 2>&1 >/dev/null \
% RUN:   | FileCheck --check-prefix=EOF %s

p = "\e".
% CHECK: errors.m:[[@LINE-1]]:6: error: '\e' is not an escape in Mercury 22.01
% CHECK-NEXT: p = "\e".
% CHECK-NEXT: {{^}}       ^^{{$}}
q = 2_e3.
% CHECK: errors.m:[[@LINE-1]]:5: error: an underscore may come before an exponent only after a fraction
r = 0b12.
% CHECK: errors.m:[[@LINE-1]]:8: error: invalid digit in a literal with prefix '0b'
% CHECK: errors.m:[[@LINE+2]]:5: error: unterminated string
% EOF: errors.m:[[@LINE+1]]:11: error: expected an operator or '.', but found the end of the file
s = "open
